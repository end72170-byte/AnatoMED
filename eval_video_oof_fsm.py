"""Fold-specific FetalCLIP + FSM video OOF inference without downstream VLMs."""

from __future__ import annotations

import argparse
import gc
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import cv2
import numpy as np
import torch

from inference import (
    CLASS_LABELS,
    CLASS_TO_PROMPT,
    DEFAULT_KEYFRAME_PROMPTS,
    FSMController,
    compute_motion,
    compute_sharpness,
    load_fetalclip,
    merge_adjacent_same_label_events,
    pil_from_bgr,
    score_frame,
    set_random_seed,
)
from utils.anatomed_cache import sha256_file
from utils.anatomed_video import CLASS_TO_SLOT, build_video_diagnosis
from utils.runtime import _configure_warnings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split_root",
        type=Path,
        default=Path("splits/sam-dataset/group_kfold_k5_seed42_anatomed"),
    )
    parser.add_argument(
        "--video_manifest",
        type=Path,
        required=True,
        help="Generated video manifest.",
    )
    parser.add_argument(
        "--checkpoint_root",
        type=Path,
        default=Path("checkpoints/fetalclip"),
    )
    parser.add_argument(
        "--out_root",
        type=Path,
        default=Path("runs/video_oof"),
    )
    parser.add_argument("--fetalclip_config", default="model/FetalCLIP_config.json")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--expected_videos", type=int, default=74)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--suppress_warnings", action=argparse.BooleanOptionalAction, default=True)
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
    return parser.parse_args()


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_cases(path: Path) -> set[str]:
    return {
        line.strip().split("/")[0]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def _event_from_buffer(buffer: Any, event_index: int) -> Optional[Dict[str, Any]]:
    best = buffer.best()
    if best is None:
        return None
    return {
        "event_index": int(event_index),
        "label": str(buffer.label),
        "slot": CLASS_TO_SLOT[str(buffer.label)],
        "prompt": CLASS_TO_PROMPT[str(buffer.label)],
        "frame_idx": int(best.frame_idx),
        "timestamp_sec": float(best.timestamp_sec),
        "clip_score": float(best.score),
        "clip_margin": float(best.margin),
        "sharpness": float(best.sharpness),
        "motion": float(best.motion),
        "class_scores": {str(key): float(value) for key, value in best.scores.items()},
        "segment_start_frame": int(buffer.start_frame),
        "segment_start_sec": float(buffer.start_time_sec),
        "segment_observed_frames": int(buffer.observed_frames),
        "segment_best_score": float(buffer.best_score),
    }


@torch.no_grad()
def run_video_fsm(
    *,
    args: argparse.Namespace,
    video_path: Path,
    model: Any,
    preprocess: Any,
    prompt_features: torch.Tensor,
    device: torch.device,
) -> Dict[str, Any]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fsm = FSMController(args)
    events: List[Dict[str, Any]] = []
    frame_top1_counts: Counter[str] = Counter()
    previous_bgr: Optional[np.ndarray] = None
    processed = 0
    frame_idx = -1
    started = time.perf_counter()

    def handle(buffers: Sequence[Any]) -> None:
        for buffer in buffers:
            event = _event_from_buffer(buffer, len(events))
            if event is not None:
                events.append(event)

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
        raw_scores = score_frame(
            fetalclip_model=model,
            fetalclip_preprocess=preprocess,
            prompt_text_features=prompt_features,
            device=device,
            image_pil=frame_pil,
        )
        top1_label = max(raw_scores.items(), key=lambda item: item[1])[0]
        frame_top1_counts[top1_label] += 1
        handle(
            fsm.update(
                frame_idx=frame_idx,
                timestamp_sec=frame_idx / fps,
                frame_pil=frame_pil,
                raw_scores=raw_scores,
                sharpness=compute_sharpness(frame_bgr),
                motion=compute_motion(previous_bgr, frame_bgr),
            )
        )
        previous_bgr = frame_bgr.copy()
    cap.release()
    handle(fsm.flush())
    events = merge_adjacent_same_label_events(events, gap_frames=int(args.cooldown_frames))
    diagnosis = build_video_diagnosis(
        events=events,
        frame_top1_counts=frame_top1_counts,
        processed_frames=processed,
        palate_cleft_ratio_thr=float(args.palate_cleft_ratio_thr),
    )
    elapsed = time.perf_counter() - started
    return {
        "video_path": str(video_path.resolve()),
        "fps": fps,
        "total_frames": total_frames,
        "processed_frames": processed,
        "frame_stride": int(args.frame_stride),
        **diagnosis,
        "frame_top1_counts": {
            label: int(frame_top1_counts.get(label, 0)) for label in CLASS_LABELS
        },
        "processing_seconds": elapsed,
        "processing_fps": float(processed / max(elapsed, 1e-9)),
        "events": events,
    }


def _rate(numerator: float, denominator: float) -> Optional[float]:
    return float(numerator / denominator) if denominator else None


def _clopper_pearson(successes: int, total: int, alpha: float = 0.05) -> Dict[str, Any]:
    if total <= 0:
        return {"successes": successes, "total": total, "rate": None, "lower": None, "upper": None}
    from scipy.stats import beta

    lower = 0.0 if successes == 0 else float(beta.ppf(alpha / 2.0, successes, total - successes + 1))
    upper = 1.0 if successes == total else float(beta.ppf(1.0 - alpha / 2.0, successes + 1, total - successes))
    return {
        "successes": int(successes),
        "total": int(total),
        "rate": float(successes / total),
        "lower": lower,
        "upper": upper,
    }


def summarize_results(results: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    available = [item for item in results if item["hard_palate_available"]]
    tp = sum(item["condition"] == "abnormal" and item["diagnose"] == "abnormal" for item in available)
    tn = sum(item["condition"] == "normal" and item["diagnose"] == "normal" for item in available)
    fp = sum(item["condition"] == "normal" and item["diagnose"] == "abnormal" for item in available)
    fn = sum(item["condition"] == "abnormal" and item["diagnose"] == "normal" for item in available)
    sensitivity = _rate(tp, tp + fn)
    specificity = _rate(tn, tn + fp)
    return {
        "n_videos": len(results),
        "ground_truth": dict(Counter(str(item["condition"]) for item in results)),
        "diagnosis": dict(Counter(str(item["diagnose"]) for item in results)),
        "event_patterns": dict(Counter(str(item["diagnosis_event_pattern"]) for item in results)),
        "hard_palate_available": len(available),
        "coverage": _rate(len(available), len(results)),
        "insufficient_evidence": len(results) - len(available),
        "conditional_confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "conditional_accuracy": _rate(tp + tn, len(available)),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": (
            float((sensitivity + specificity) / 2.0)
            if sensitivity is not None and specificity is not None else None
        ),
        "ppv": _rate(tp, tp + fp),
        "npv": _rate(tn, tn + fn),
        "abnormal_f1": _rate(2 * tp, 2 * tp + fp + fn),
        "resolved_correct_rate_all_videos": _rate(tp + tn, len(results)),
        "clopper_pearson_95": {
            "coverage": _clopper_pearson(len(available), len(results)),
            "conditional_accuracy": _clopper_pearson(tp + tn, len(available)),
            "sensitivity": _clopper_pearson(tp, tp + fn),
            "specificity": _clopper_pearson(tn, tn + fp),
            "ppv": _clopper_pearson(tp, tp + fp),
            "npv": _clopper_pearson(tn, tn + fn),
            "resolved_correct_rate_all_videos": _clopper_pearson(tp + tn, len(results)),
        },
        "processed_frames": sum(int(item["processed_frames"]) for item in results),
        "stable_events": sum(len(item["events"]) for item in results),
        "aggregate_processing_fps": _rate(
            sum(int(item["processed_frames"]) for item in results),
            sum(float(item["processing_seconds"]) for item in results),
        ),
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.suppress_warnings:
        _configure_warnings(suppress=True)
    set_random_seed(int(args.seed))
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    manifest_rows = _read_jsonl(args.video_manifest)
    all_results: List[Dict[str, Any]] = []
    fold_case_sets: List[set[str]] = []
    args.out_root.mkdir(parents=True, exist_ok=True)

    for fold in range(int(args.folds)):
        fold_name = f"fold_{fold}"
        fold_dir = args.split_root / fold_name
        val_cases = _read_cases(fold_dir / "val.txt")
        if any(val_cases & previous for previous in fold_case_sets):
            raise RuntimeError(f"Validation case overlap detected at {fold_name}")
        fold_case_sets.append(val_cases)
        rows = sorted(
            (
                row for row in manifest_rows
                if str(row.get("case_id")) in val_cases
                and row.get("inference_candidate", True)
                and str(row.get("seniority", "high")) == "high"
            ),
            key=lambda row: (str(row["case_id"]), str(row.get("condition", ""))),
        )
        covered_cases = {str(row["case_id"]) for row in rows}
        missing_cases = sorted(val_cases - covered_cases)
        checkpoint_path = (
            args.checkpoint_root / fold_name / "default_15ep" / "fetalclip_finetune_best.pt"
        ).resolve()
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint_fold = Path(str(checkpoint.get("args", {}).get("fold_dir", ""))).resolve()
        if checkpoint_fold != fold_dir.resolve():
            raise RuntimeError(
                f"{fold_name} checkpoint split mismatch: {checkpoint_fold} != {fold_dir.resolve()}"
            )
        del checkpoint
        args.fetalclip_weights = str(checkpoint_path)
        model, preprocess, _, prompt_features = load_fetalclip(args, device, print)
        fold_results: List[Dict[str, Any]] = []
        fold_out = args.out_root / fold_name
        print(
            f"[fold] {fold_name} val_cases={len(val_cases)} videos={len(rows)} "
            f"missing={len(missing_cases)} checkpoint={checkpoint_path}"
        )
        for index, row in enumerate(rows, start=1):
            video_path = Path(str(row["video_path"]))
            if not video_path.is_absolute():
                video_path = Path.cwd() / video_path
            output = run_video_fsm(
                args=args,
                video_path=video_path,
                model=model,
                preprocess=preprocess,
                prompt_features=prompt_features,
                device=device,
            )
            output.update(
                {
                    "fold": fold_name,
                    "case_id": str(row["case_id"]),
                    "examination_id": row.get("examination_id"),
                    "condition": str(row.get("condition", "unknown")),
                    "checkpoint_path": str(checkpoint_path),
                }
            )
            fold_results.append(output)
            all_results.append(output)
            _write_json(fold_out / "json_dir" / f"{row['case_id']}_{row.get('condition')}_high.json", output)
            print(
                f"[{fold_name}] {index}/{len(rows)} {row['case_id']} "
                f"pattern={output['diagnosis_event_pattern']} diagnose={output['diagnose']} "
                f"ratio={output['palate_cleft_ratio']} events={len(output['events'])} "
                f"fps={output['processing_fps']:.2f}"
            )
        fold_summary = {
            "format": "anatomed-video-oof-fold-v1",
            "fold": fold_name,
            "fold_dir": str(fold_dir.resolve()),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "validation_cases": sorted(val_cases),
            "missing_video_cases": missing_cases,
            "metrics": summarize_results(fold_results),
            "results": fold_results,
        }
        _write_json(fold_out / f"{fold_name}.json", fold_summary)
        del model, preprocess, prompt_features
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if len(all_results) != int(args.expected_videos):
        raise RuntimeError(
            f"Expected {args.expected_videos} OOF videos, collected {len(all_results)}"
        )
    pooled = {
        "format": "anatomed-video-oof-pooled-v1",
        "split_root": str(args.split_root.resolve()),
        "video_manifest": str(args.video_manifest.resolve()),
        "diagnosis_rule": "stable_event_pattern_then_ratio_v2",
        "fsm_parameters": {
            key: getattr(args, key)
            for key in (
                "ema_alpha", "enter_threshold", "exit_threshold", "margin_threshold",
                "enter_streak", "exit_patience", "cooldown_frames", "topk",
                "min_state_frames", "max_state_frames", "sharpness_weight",
                "margin_weight", "motion_weight", "palate_cleft_ratio_thr",
            )
        },
        "reference_reachability_status": "not_yet_locked",
        "metrics": summarize_results(all_results),
        "results": all_results,
    }
    _write_json(args.out_root / "pooled_oof.json", pooled)
    print(f"[done] pooled videos={len(all_results)} output={args.out_root / 'pooled_oof.json'}")


if __name__ == "__main__":
    main()
