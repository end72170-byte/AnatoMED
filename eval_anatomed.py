"""Evaluate an AnatoMED joint checkpoint on fold validation router conditions."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from train_anatomed import _hard_dice
from utils.anatomed_cache import CompactSummaryCache, sha256_file
from utils.anatomed_runtime import (
    build_heads, forward_joint_batch, generate_one, load_frozen_fetalclip,
    load_vlm_and_processor,
)
from utils.evidence_injection import EvidenceEmbeddingInjector
from utils.examination_dataset import ExaminationDataset, examination_collate
from utils.metrics import compute_bertscore_f1, compute_bleu_score
from utils.segmentation_engine import MetricAccumulator
from utils.segmentation_losses import SegmentationLossConfig, SparseForegroundLoss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out_json", type=Path, default=Path("runs/anatomed_eval.json"))
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--full_text_eval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gen_max_new_tokens", type=int, default=320)
    parser.add_argument("--bertscore_device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--predictions_jsonl", type=Path, default=None)
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["args"]
    paths = checkpoint["paths"]
    for key, expected in (
        ("fetalclip", paths["fetalclip_sha256"]),
        ("router_manifest", paths["router_manifest_sha256"]),
        ("seg_pretrain", paths["seg_pretrain_sha256"]),
    ):
        actual = sha256_file(Path(paths[key]))
        if actual != expected:
            raise RuntimeError(f"SHA256 mismatch for {key}: {paths[key]}")
    cache_path = Path(checkpoint["summary_cache"] )
    if sha256_file(cache_path) != checkpoint["summary_cache_sha256"]:
        raise RuntimeError("Compact-summary cache SHA256 mismatch")
    cache, metadata = CompactSummaryCache.load(cache_path)
    if metadata.get("contains_dense_features"):
        raise RuntimeError("Evaluation cache must not contain dense features")

    dataset = ExaminationDataset(
        Path(config["observations"]), Path(config["fold_dir"]) / "val.txt",
        source_dataset="clip-dataset", condition_source="router",
        router_manifest=Path(paths["router_manifest"]), seed=int(config["seed"]),
        aux_slot_dropout_prob=0.0,
    )
    indices = list(range(len(dataset)))
    if args.max_samples > 0:
        indices = indices[: args.max_samples]
    loader = DataLoader(Subset(dataset, indices), batch_size=1, shuffle=False, collate_fn=examination_collate)

    _, _, extractor = load_frozen_fetalclip(
        Path(config["fetalclip_config"]), Path(paths["fetalclip"]), device
    )
    vlm, processor, seg_token_id, evidence_token_id, vlm_dim = load_vlm_and_processor(
        config["vlm_model_id"], device, local_files_only=args.local_files_only,
        lora_r=int(config["lora_r"]), lora_alpha=int(config["lora_alpha"]),
        lora_dropout=float(config["lora_dropout"]),
    )
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
        num_heads=int(config["num_heads"]), seg_pretrain_path=Path(paths["seg_pretrain"]), device=device,
    )
    for name in ("composer", "language_projection", "segmentation_projection", "grounding", "decoder"):
        getattr(heads, name).load_state_dict(checkpoint[name], strict=True)
    heads.eval(); vlm.eval(); extractor.eval()
    injector = EvidenceEmbeddingInjector(evidence_token_id)
    criterion = SparseForegroundLoss(SegmentationLossConfig(name="dice_focal", focal_alpha=0.85, focal_gamma=2.0))
    accumulator = MetricAccumulator(0.5)
    losses = []
    lm_losses = []
    seg_losses = []
    predictions = []
    references = []
    records = []
    predictions_path = args.predictions_jsonl or args.out_json.with_suffix(".predictions.jsonl")
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    if args.full_text_eval:
        predictions_path.write_text("", encoding="utf-8")
    started = time.perf_counter()
    with torch.no_grad():
        for sample_index, batch in enumerate(loader, start=1):
            output = forward_joint_batch(
                batch=batch, device=device, feature_extractor=extractor, summary_cache=cache,
                heads=heads, vlm=vlm, processor=processor, injector=injector, seg_token_id=seg_token_id,
                segmentation_criterion=criterion, lambda_lm=float(config["lambda_lm"]),
                lambda_seg=float(config["lambda_seg"]),
            )
            probability = torch.sigmoid(output["seg_logits"])[0, 0].float().cpu().numpy()
            target = output["main_masks"][0, 0].float().cpu().numpy()
            accumulator.add(probability, target, batch["main_gt_clip_prompt"][0], batch["main_image_paths"][0])
            losses.append(float(output["loss"].float().item()))
            lm_losses.append(float(output["loss_lm"].float().item()))
            seg_losses.append(float(output["loss_seg"].float().item()))
            if args.full_text_eval:
                generated_ids = generate_one(
                    batch=batch, processor=processor, vlm=vlm, injector=injector,
                    language_embedding=output["composed"]["language_embedding"],
                    device=device, max_new_tokens=int(args.gen_max_new_tokens),
                )
                prediction = processor.tokenizer.decode(
                    generated_ids[0].detach().cpu(), skip_special_tokens=True
                ).strip()
                reference = str(batch["main_text"][0]).strip()
                record = {
                    "index": sample_index - 1,
                    "main_view_id": batch["main_view_id"][0],
                    "examination_id": batch["examination_id"][0],
                    "image_path": batch["main_image_paths"][0],
                    "router_prompt": batch["main_clip_prompt"][0],
                    "gt_prompt": batch["main_gt_clip_prompt"][0],
                    "prediction": prediction,
                    "reference": reference,
                    "generated_token_count": int(generated_ids.shape[1]),
                }
                predictions.append(prediction)
                references.append(reference)
                records.append(record)
                with predictions_path.open("a", encoding="utf-8", buffering=1) as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()
            if sample_index == 1 or sample_index % 10 == 0 or sample_index == len(indices):
                elapsed = time.perf_counter() - started
                print(
                    f"[Eval] {sample_index}/{len(indices)} elapsed={elapsed / 60:.1f}m "
                    f"loss={float(np.mean(losses)):.4f}",
                    flush=True,
                )
    bert_f1 = None
    bleu = None
    if args.full_text_eval:
        bleu = compute_bleu_score(predictions, references)
        bert_f1 = compute_bertscore_f1(
            predictions, references, device=args.bertscore_device, batch_size=8
        )
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "condition_source": "router",
        "gt_fallback": False,
        "n": len(indices),
        "loss": float(np.mean(losses)),
        "lm_loss": float(np.mean(lm_losses)),
        "seg_loss": float(np.mean(seg_losses)),
        "segmentation": accumulator.summary(),
        "text": {
            "evaluated": bool(args.full_text_eval),
            "n": len(predictions),
            "bert_f1": bert_f1,
            "bleu": bleu,
            "gen_max_new_tokens": int(args.gen_max_new_tokens),
            "mean_generated_tokens": (
                float(np.mean([record["generated_token_count"] for record in records]))
                if records else None
            ),
            "truncated_at_limit_count": sum(
                record["generated_token_count"] >= int(args.gen_max_new_tokens)
                for record in records
            ),
            "predictions_jsonl": str(predictions_path.resolve()) if records else None,
        },
        "elapsed_sec": time.perf_counter() - started,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
