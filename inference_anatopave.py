"""Incremental ANATOPAVE video inference.

FetalCLIP and the temporal FSM run for every processed frame.  A stable FSM
event alone triggers one representative-frame dense extraction, Composer
update, VLM call, and segmentation result.  Historical slot entries retain
compact summaries only; historical images are never re-encoded.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from inference import (
    CLASS_LABELS,
    CLASS_TO_PROMPT,
    DEFAULT_KEYFRAME_PROMPTS,
    CandidateFrame,
    FSMController,
    compute_motion,
    compute_sharpness,
    merge_adjacent_same_label_events,
    pil_from_bgr,
    score_frame,
    set_random_seed,
)
from train_seg_fetalclip_unetr import _extract_state_dict
from utils.anatopave_cache import sha256_file
from utils.anatopave_runtime import (
    build_heads,
    load_frozen_fetalclip,
    load_vlm_and_processor,
)
from utils.anatopave_video import (
    CLASS_TO_SLOT,
    IncrementalSlotCache,
    SLOT_NAMES,
    build_rule_based_summary,
    build_video_diagnosis,
    compact_summary_sha256,
)
from utils.evidence_injection import EvidenceEmbeddingInjector
from utils.prompt_utils import SYSTEM_PROMPT, build_prompt_text_features
from utils.runtime import autocast_ctx
from utils.segmentation_data import PairedSegTransform, inverse_map_array
from utils.vlm_inputs import (
    _get_pad_token_id,
    build_anatopave_generation_inputs,
    pad_and_stack_vlm_inputs,
)
from utils.models import extract_seg_token_feature


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video_path", required=True)
    parser.add_argument("--checkpoint_path", required=True, help="ANATOPAVE joint best.pt")
    parser.add_argument("--out_json", default="")
    parser.add_argument("--save_dir", default="")
    parser.add_argument("--fetalclip_config", default="")
    parser.add_argument("--fetalclip_weights", default="", help="Optional override; checkpoint path is used by default")
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--gen_max_new_tokens", type=int, default=256)
    parser.add_argument("--gen_do_sample", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--suppress_warnings", action=argparse.BooleanOptionalAction, default=True)

    # These defaults define the fold-level temporal FSM protocol.
    parser.add_argument("--ema_alpha", type=float, default=0.5)
    parser.add_argument("--enter_threshold", type=float, default=0.20)
    parser.add_argument("--exit_threshold", type=float, default=0.15)
    parser.add_argument("--margin_threshold", type=float, default=0.02)
    parser.add_argument("--enter_streak", type=int, default=3)
    parser.add_argument("--exit_patience", type=int, default=3)
    parser.add_argument("--cooldown_frames", type=int, default=40)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--min_state_frames", type=int, default=4)
    parser.add_argument("--max_state_frames", type=int, default=36)
    parser.add_argument("--sharpness_weight", type=float, default=0.03)
    parser.add_argument("--margin_weight", type=float, default=0.40)
    parser.add_argument("--motion_weight", type=float, default=0.20)
    parser.add_argument("--palate_cleft_ratio_thr", type=float, default=0.20)
    parser.add_argument(
        "--slot_representative_strategy",
        choices=["highest_clip_score", "latest_stable_event"],
        default="highest_clip_score",
        help="Final representative selection after adjacent-event merging.",
    )
    return parser.parse_args()


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _resolve_from_checkpoint(value: str, project_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _load_bundle(args: argparse.Namespace, device: torch.device, log: Any) -> Dict[str, Any]:
    checkpoint_path = Path(args.checkpoint_path).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") not in {"anatopave-joint-v1", "anatopave-joint-v2"}:
        raise RuntimeError(f"Not an ANATOPAVE joint checkpoint: {checkpoint_path}")
    config = checkpoint["args"]
    paths = checkpoint["paths"]
    project_root = Path(__file__).resolve().parent
    fetalclip_path = Path(args.fetalclip_weights).resolve() if args.fetalclip_weights else Path(paths["fetalclip"])
    if not fetalclip_path.exists():
        raise FileNotFoundError(f"FetalCLIP checkpoint not found: {fetalclip_path}")
    if sha256_file(fetalclip_path) != paths["fetalclip_sha256"]:
        raise RuntimeError("FetalCLIP checkpoint SHA256 does not match the joint checkpoint")
    seg_path = Path(paths["seg_pretrain"])
    if sha256_file(seg_path) != paths["seg_pretrain_sha256"]:
        raise RuntimeError("Segmentation pretrain SHA256 does not match the joint checkpoint")
    config_path = Path(args.fetalclip_config).resolve() if args.fetalclip_config else _resolve_from_checkpoint(
        str(config["fetalclip_config"]), project_root
    )

    if args.local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    fetalclip_model, fetalclip_preprocess, extractor = load_frozen_fetalclip(
        config_path, fetalclip_path, device
    )
    tokenizer = __import__("open_clip").get_tokenizer("FetalCLIP")
    prompt_features = build_prompt_text_features(
        fetalclip_model, tokenizer, device, DEFAULT_KEYFRAME_PROMPTS
    )

    vlm, processor, seg_token_id, evidence_token_id, vlm_dim = load_vlm_and_processor(
        str(config["vlm_model_id"]), device,
        local_files_only=bool(args.local_files_only),
        lora_r=int(config["lora_r"]), lora_alpha=int(config["lora_alpha"]),
        lora_dropout=float(config["lora_dropout"]),
    )
    saved_token_ids = checkpoint.get("special_token_ids", {})
    if saved_token_ids and (
        int(saved_token_ids.get("seg", seg_token_id)) != seg_token_id
        or int(saved_token_ids.get("evidence", evidence_token_id)) != evidence_token_id
    ):
        raise RuntimeError("Special token ids do not match the joint checkpoint")
    if "special_token_embeddings" in checkpoint:
        embedding = vlm.get_input_embeddings().weight
        with torch.no_grad():
            for name, values in checkpoint["special_token_embeddings"].items():
                token_id = seg_token_id if name == "seg" else evidence_token_id
                embedding[token_id].copy_(values.to(device=embedding.device, dtype=embedding.dtype))
    if "vlm_lora" in checkpoint:
        from peft import set_peft_model_state_dict
        set_peft_model_state_dict(vlm, checkpoint["vlm_lora"])
    heads = build_heads(
        summary_dim=768, dense_feature_dim=extractor.width, vlm_dim=vlm_dim,
        composer_dim=int(config["composer_dim"]), grounding_dim=int(config["grounding_dim"]),
        num_heads=int(config["num_heads"]), seg_pretrain_path=seg_path, device=device,
    )
    for name in ("composer", "language_projection", "segmentation_projection", "grounding", "decoder"):
        getattr(heads, name).load_state_dict(checkpoint[name], strict=True)
    fetalclip_model.eval()
    extractor.eval()
    heads.eval()
    vlm.eval()
    log(f"[anatopave] loaded joint checkpoint epoch={checkpoint.get('epoch')} from {checkpoint_path}")
    return {
        "checkpoint": checkpoint,
        "fetalclip_model": fetalclip_model,
        "fetalclip_preprocess": fetalclip_preprocess,
        "extractor": extractor,
        "prompt_features": prompt_features,
        "vlm": vlm,
        "processor": processor,
        "heads": heads,
        "injector": EvidenceEmbeddingInjector(evidence_token_id),
        "seg_token_id": seg_token_id,
    }


def _restore_mask_to_original(probability: np.ndarray, image: Image.Image) -> Tuple[np.ndarray, np.ndarray]:
    transform = PairedSegTransform("clip_center_crop", train=False, augment=False)
    dummy_mask = Image.fromarray(np.ones((image.height, image.width), dtype=np.uint8) * 255)
    _, _, geometry = transform(image, dummy_mask)
    restored_probability = inverse_map_array(probability, geometry, nearest=False)
    restored_mask = (restored_probability >= 0.5).astype(np.uint8) * 255
    return restored_probability, restored_mask


@torch.no_grad()
def _infer_stable_event(
    *,
    candidate: CandidateFrame,
    state_label: str,
    event_index: int,
    cache: IncrementalSlotCache,
    bundle: Mapping[str, Any],
    device: torch.device,
    args: argparse.Namespace,
    save_dir: Optional[Path],
) -> Tuple[Dict[str, Any], float, float]:
    started = time.perf_counter()
    image = candidate.image_pil
    slot = CLASS_TO_SLOT[state_label]
    extractor = bundle["extractor"]
    preprocess = bundle["fetalclip_preprocess"]
    with autocast_ctx(device):
        dense_features, summary = extractor(torch.stack([preprocess(image)]).to(device))
    main_summary = summary[0]
    cache.update(
        label=state_label, summary=main_summary, frame_idx=candidate.frame_idx,
        timestamp_sec=candidate.timestamp_sec, prompt=CLASS_TO_PROMPT[state_label],
        clip_score=candidate.score, clip_margin=candidate.margin,
        class_scores=candidate.scores,
    )
    slot_summaries, slot_available, slot_is_main = cache.composer_inputs(
        main_slot=slot, main_summary=main_summary, device=device, dtype=main_summary.dtype
    )
    composer_started = time.perf_counter()
    with autocast_ctx(device):
        composed = bundle["heads"].compose(slot_summaries, slot_available, slot_is_main)
    _sync(device)
    composer_ms = (time.perf_counter() - composer_started) * 1000.0

    inputs = build_anatopave_generation_inputs(
        bundle["processor"], [image], [CLASS_TO_PROMPT[state_label]], SYSTEM_PROMPT
    )
    kwargs = pad_and_stack_vlm_inputs(
        inputs, device, _get_pad_token_id(bundle["processor"])
    )
    prompt_length = int(kwargs["input_ids"].shape[1])
    with autocast_ctx(device):
        vlm_output = bundle["injector"].forward(
            bundle["vlm"], composed["language_embedding"][:, 0],
            {**kwargs, "output_hidden_states": True, "return_dict": True},
        )
        seg_hidden = extract_seg_token_feature(
            vlm_output.hidden_states[-1], kwargs["input_ids"], bundle["seg_token_id"]
        )
        main_slot_index = slot_is_main.to(torch.long).argmax(dim=1)
        seg_logits = bundle["heads"].segment(
            features=dense_features,
            seg_hidden=seg_hidden,
            segmentation_context=composed["segmentation_context"],
            main_slot_index=main_slot_index,
        )
        generated_ids = bundle["injector"].generate(
            bundle["vlm"], composed["language_embedding"][:, 0], kwargs,
            max_new_tokens=int(args.gen_max_new_tokens),
            do_sample=bool(args.gen_do_sample), use_cache=True,
        )
    generated_ids = generated_ids[:, prompt_length:]
    text = bundle["processor"].tokenizer.decode(
        generated_ids[0].detach().cpu(), skip_special_tokens=True
    ).strip()
    probability_224 = torch.sigmoid(seg_logits[0, 0]).float().cpu().numpy()
    _, pred_mask = _restore_mask_to_original(probability_224, image)
    pred_score = float(probability_224.max())

    record: Dict[str, Any] = {
        "event_index": int(event_index),
        "label": state_label,
        "slot": slot,
        "prompt": CLASS_TO_PROMPT[state_label],
        "frame_idx": candidate.frame_idx,
        "timestamp_sec": candidate.timestamp_sec,
        "clip_score": candidate.score,
        "clip_margin": candidate.margin,
        "sharpness": candidate.sharpness,
        "motion": candidate.motion,
        "text": text,
        "mask_score": pred_score,
        "class_scores": candidate.scores,
        "slot_availability": {
            slot_name: bool(slot_available[0, index].item())
            for index, slot_name in enumerate(SLOT_NAMES)
        },
        "composer_attention_weights": composed["attention_weights"][0].float().cpu().tolist(),
        "composer_context_norm": float(composed["context"].float().norm().item()),
        "compact_summary_sha256": compact_summary_sha256(main_summary),
        "generated_token_count": int(generated_ids.shape[1]),
        "composer_update_ms": composer_ms,
    }
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        stem = f"event_{event_index:02d}_{state_label.lower()}_f{candidate.frame_idx:06d}"
        image_path = save_dir / f"{stem}_img.png"
        mask_path = save_dir / f"{stem}_mask.png"
        overlay_path = save_dir / f"{stem}_overlay.png"
        image_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        overlay = image_rgb.copy()
        red = np.zeros_like(overlay)
        red[..., 0] = 255
        active = pred_mask > 0
        overlay[active] = (overlay[active] * 0.55 + red[active] * 0.45).astype(np.uint8)
        image.save(image_path)
        Image.fromarray(pred_mask).save(mask_path)
        Image.fromarray(overlay).save(overlay_path)
        record.update({
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "overlay_path": str(overlay_path),
        })
    record["event_latency_ms"] = (time.perf_counter() - started) * 1000.0
    cache.attach_event(slot, record)
    return record, composer_ms, float(record["event_latency_ms"])


def run_video(args: argparse.Namespace, bundle: Mapping[str, Any], device: torch.device, log: Any) -> Dict[str, Any]:
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {args.video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    log(f"[video] path={args.video_path} fps={fps:.3f} total_frames={total_frames}")
    fsm = FSMController(args)
    cache = IncrementalSlotCache(summary_dim=768)
    events: List[Dict[str, Any]] = []
    frame_top1_counts: Counter[str] = Counter()
    prev_bgr: Optional[np.ndarray] = None
    frame_idx = -1
    processed = 0
    clip_ms: List[float] = []
    fsm_ms: List[float] = []
    composer_ms: List[float] = []
    event_ms: List[float] = []
    started = time.perf_counter()
    save_dir = Path(args.save_dir) if args.save_dir else None

    def handle_buffers(buffers: List[Any]) -> None:
        for buffer in buffers:
            best = buffer.best()
            if best is None:
                continue
            event, composer_time, event_time = _infer_stable_event(
                candidate=best, state_label=buffer.label, event_index=len(events),
                cache=cache, bundle=bundle, device=device, args=args, save_dir=save_dir,
            )
            event["segment_start_frame"] = buffer.start_frame
            event["segment_start_sec"] = buffer.start_time_sec
            event["segment_observed_frames"] = buffer.observed_frames
            event["segment_best_score"] = buffer.best_score
            events.append(event)
            composer_ms.append(composer_time)
            event_ms.append(event_time)
            log(
                f"[event] idx={event['event_index']} label={event['label']} "
                f"slot={event['slot']} frame={event['frame_idx']} "
                f"t={event['timestamp_sec']:.2f}s clip={event['clip_score']:.4f}"
            )

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_idx += 1
        if int(args.frame_stride) > 1 and frame_idx % int(args.frame_stride) != 0:
            continue
        if int(args.max_frames) > 0 and processed >= int(args.max_frames):
            break
        processed += 1
        frame_pil = pil_from_bgr(frame_bgr)
        sharpness = compute_sharpness(frame_bgr)
        motion = compute_motion(prev_bgr, frame_bgr)
        t0 = time.perf_counter()
        raw_scores = score_frame(
            fetalclip_model=bundle["fetalclip_model"],
            fetalclip_preprocess=bundle["fetalclip_preprocess"],
            prompt_text_features=bundle["prompt_features"],
            device=device, image_pil=frame_pil,
        )
        _sync(device)
        clip_ms.append((time.perf_counter() - t0) * 1000.0)
        top1_label = max(raw_scores.items(), key=lambda item: item[1])[0]
        frame_top1_counts[top1_label] += 1
        t1 = time.perf_counter()
        buffers = fsm.update(
            frame_idx=frame_idx, timestamp_sec=frame_idx / fps,
            frame_pil=frame_pil, raw_scores=raw_scores,
            sharpness=sharpness, motion=motion,
        )
        fsm_ms.append((time.perf_counter() - t1) * 1000.0)
        handle_buffers(buffers)
        prev_bgr = frame_bgr.copy()
    cap.release()
    handle_buffers(fsm.flush())
    events = merge_adjacent_same_label_events(events, gap_frames=int(args.cooldown_frames))
    selected_representatives = cache.reconcile_events(
        events, strategy=str(args.slot_representative_strategy)
    )
    cache.close()
    diagnosis_started = time.perf_counter()
    diagnosis = build_video_diagnosis(
        events=events, frame_top1_counts=frame_top1_counts,
        processed_frames=processed, palate_cleft_ratio_thr=args.palate_cleft_ratio_thr,
    )
    final_summary = build_rule_based_summary(slot_cache=cache, diagnosis=diagnosis)
    diagnosis_ms = (time.perf_counter() - diagnosis_started) * 1000.0
    total_ms = (time.perf_counter() - started) * 1000.0
    peak_memory = (
        float(torch.cuda.max_memory_allocated(device) / 2**20) if device.type == "cuda" else 0.0
    )
    runtime = {
        "architecture": "anatopave_incremental_fsm_v1",
        "clip_frame_latency_ms_mean": float(np.mean(clip_ms)) if clip_ms else 0.0,
        "fsm_frame_latency_ms_mean": float(np.mean(fsm_ms)) if fsm_ms else 0.0,
        "composer_update_latency_ms_mean": float(np.mean(composer_ms)) if composer_ms else 0.0,
        "event_to_vlm_latency_ms_mean": float(np.mean(event_ms)) if event_ms else 0.0,
        "event_to_vlm_latency_ms_p95": float(np.percentile(event_ms, 95)) if event_ms else 0.0,
        "vlm_event_call_count": len(event_ms),
        "vlm_call_ratio": float(len(event_ms) / processed) if processed else 0.0,
        "processing_fps": float(processed / max(total_ms / 1000.0, 1e-9)),
        "peak_gpu_memory_mb": peak_memory,
        "diagnose_latency_ms": diagnosis_ms,
        "total_latency_ms": total_ms,
        "representative_dense_reencode_count": len(event_ms),
        "historical_dense_reencode_count": 0,
        "cached_compact_slot_count": sum(item["state"] == "AVAILABLE" for item in cache.snapshot().values()),
        "slot_representative_strategy": cache.representative_strategy,
    }
    postprocess = {
        "frame_top1_prompt_counts": {
            CLASS_TO_PROMPT[label]: int(frame_top1_counts.get(label, 0))
            for label in CLASS_LABELS
        },
        **{key: value for key, value in diagnosis.items() if key not in {
            "hard_palate_available", "diagnose", "insufficient_evidence_reason",
            "has_intact_hard_palate_event", "has_cleft_hard_palate_event",
        }},
    }
    return {
        "video_path": args.video_path,
        "checkpoint_path": args.checkpoint_path,
        "fetalclip_weights": str(bundle["checkpoint"]["paths"]["fetalclip"]),
        "fps": fps,
        "total_frames": total_frames,
        "processed_frames": processed,
        "frame_stride": int(args.frame_stride),
        "hard_palate_available": diagnosis["hard_palate_available"],
        "diagnose": diagnosis["diagnose"],
        "insufficient_evidence_reason": diagnosis["insufficient_evidence_reason"],
        "has_intact_hard_palate_event": diagnosis["has_intact_hard_palate_event"],
        "has_cleft_hard_palate_event": diagnosis["has_cleft_hard_palate_event"],
        "diagnosis_rule": diagnosis["diagnosis_rule"],
        "diagnosis_event_pattern": diagnosis["diagnosis_event_pattern"],
        "palate_cleft_ratio_used_for_diagnosis": diagnosis[
            "palate_cleft_ratio_used_for_diagnosis"
        ],
        "palate_cleft_ratio": diagnosis["palate_cleft_ratio"],
        "slot_representative_strategy": cache.representative_strategy,
        "selected_slot_representatives": {
            slot: ({"label": value[0], "frame_idx": value[1]} if value is not None else None)
            for slot, value in selected_representatives.items()
        },
        "video_level_postprocess": postprocess,
        "slot_cache": cache.snapshot(),
        "final_examination_summary": final_summary,
        "runtime": runtime,
        "prompts": DEFAULT_KEYFRAME_PROMPTS,
        "events": events,
    }


def main() -> None:
    args = parse_args()
    if args.suppress_warnings:
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    set_random_seed(int(args.seed))
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    bundle = _load_bundle(args, device, print)
    output = run_video(args, bundle, device, print)
    out_json = Path(args.out_json) if args.out_json else Path(args.video_path).with_suffix(".anatopave.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[post] hard_palate_available={output['hard_palate_available']} "
        f"cleft_hard_palate_count={output['video_level_postprocess']['cleft_hard_palate_count']} "
        f"palate_cleft_ratio={output['palate_cleft_ratio']} diagnose={output['diagnose']}"
    )
    print(f"Saved ANATOPAVE event json: {out_json}")
    print(
        f"[done] events={len(output['events'])} "
        f"vlm_calls={output['runtime']['vlm_event_call_count']} "
        f"fps={output['runtime']['processing_fps']:.2f}"
    )


if __name__ == "__main__":
    main()
