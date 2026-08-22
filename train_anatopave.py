"""Train or smoke-test the ANATOPAVE Composer and shared grounding pipeline.

The default mode is deliberately a bounded smoke test.  Full joint training
requires ``--mode full`` and is not started implicitly.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset

from models.seg_unet import assert_no_batch_norm
from utils.anatopave_cache import (
    CompactSummaryCache,
    build_compact_summary_cache,
    dataset_observation_paths,
    sha256_file,
)
from utils.anatopave_runtime import (
    build_heads,
    forward_joint_batch,
    generate_one,
    load_frozen_fetalclip,
    load_vlm_and_processor,
)
from utils.evidence_injection import EvidenceEmbeddingInjector
from utils.examination_dataset import ExaminationDataset, examination_collate
from utils.segmentation_engine import MetricAccumulator
from utils.segmentation_losses import SegmentationLossConfig, SparseForegroundLoss
from utils.runtime import autocast_ctx


@dataclass
class RunPaths:
    fetalclip: str
    router_manifest: str
    seg_pretrain: str
    fetalclip_sha256: str
    router_manifest_sha256: str
    seg_pretrain_sha256: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--observations", type=Path, required=True, help="Generated observation manifest.")
    parser.add_argument("--fold_dir", type=Path, required=True, help="Fold directory containing train.txt and val.txt.")
    parser.add_argument("--router_manifest", type=Path, required=True, help="Frozen static-router manifest.")
    parser.add_argument("--fetalclip_config", type=Path, default=Path("model/FetalCLIP_config.json"))
    parser.add_argument("--fetalclip_weights", type=Path, required=True, help="Frozen FetalCLIP checkpoint.")
    parser.add_argument("--seg_pretrain", type=Path, required=True, help="Segmentation pretraining checkpoint.")
    parser.add_argument("--out_dir", type=Path, default=Path("runs/anatopave"))
    parser.add_argument("--summary_cache", type=Path, default=Path("runs/compact_summaries.pt"))
    parser.add_argument("--vlm_model_id", default="google/medgemma-4b-it")
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--epochs", type=int, default=None, help="Defaults to 1 in smoke mode and 10 in full mode")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--train_samples", type=int, default=20)
    parser.add_argument("--val_samples", type=int, default=8)
    parser.add_argument("--overfit_steps", type=int, default=20)
    parser.add_argument("--text_eval_samples", type=int, default=2, help="Generated validation texts per full-training epoch")
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--resume_checkpoint", type=Path, default=None)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--decoder_lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lambda_lm", type=float, default=0.2)
    parser.add_argument("--lambda_seg", type=float, default=1.0)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--composer_dim", type=int, default=512)
    parser.add_argument("--grounding_dim", type=int, default=512)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def log_factory(path: Path, *, append: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a" if append else "w", encoding="utf-8", buffering=1)

    def log(message: str) -> None:
        print(message, flush=True)
        handle.write(message + "\n")
        handle.flush()

    return log, handle


def _subset(dataset: ExaminationDataset, count: int) -> Sequence[int]:
    if count <= 0 or count >= len(dataset):
        return list(range(len(dataset)))
    return list(range(count))


def _grad_norm(loss: torch.Tensor, parameters: Iterable[nn.Parameter], retain_graph: bool) -> float:
    params = tuple(parameter for parameter in parameters if parameter.requires_grad)
    grads = torch.autograd.grad(loss, params, retain_graph=retain_graph, allow_unused=True)
    return float(sum(float(grad.detach().float().norm().item()) for grad in grads if grad is not None))


def _optimizer(heads: nn.Module, vlm: Any, lr: float, decoder_lr: float, weight_decay: float) -> AdamW:
    decoder_ids = {id(parameter) for parameter in heads.decoder.base_decoder.parameters()}
    decoder = [p for p in heads.parameters() if id(p) in decoder_ids and p.requires_grad]
    new_heads = [p for p in heads.parameters() if id(p) not in decoder_ids and p.requires_grad]
    lora = [p for p in vlm.parameters() if p.requires_grad]
    groups = [
        {"params": new_heads, "lr": lr, "weight_decay": weight_decay, "name": "new_heads"},
        {"params": decoder, "lr": decoder_lr, "weight_decay": weight_decay, "name": "pretrained_decoder"},
    ]
    if lora:
        groups.append({"params": lora, "lr": lr, "weight_decay": 0.0, "name": "vlm_lora"})
    return AdamW(groups)


def _checkpoint(
    *,
    path: Path, heads: Any, vlm: Any, optimizer: Any, args: argparse.Namespace,
    run_paths: RunPaths, cache_path: Path, cache_sha256: str, metrics: Mapping[str, Any],
    special_token_ids: Mapping[str, int], epoch: int = 0, global_step: int = 0,
    best_metrics: Optional[Mapping[str, Any]] = None,
) -> None:
    payload: Dict[str, Any] = {
        "format": "anatopave-joint-v1",
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "composer": heads.composer.state_dict(),
        "language_projection": heads.language_projection.state_dict(),
        "segmentation_projection": heads.segmentation_projection.state_dict(),
        "grounding": heads.grounding.state_dict(),
        "decoder": heads.decoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "paths": asdict(run_paths),
        "summary_cache": str(cache_path),
        "summary_cache_sha256": cache_sha256,
        "metrics": dict(metrics),
        "special_token_ids": dict(special_token_ids),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_metrics": dict(best_metrics or {}),
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    embedding = vlm.get_input_embeddings().weight.detach()
    payload["special_token_embeddings"] = {
        name: embedding[int(token_id)].cpu().clone()
        for name, token_id in special_token_ids.items()
    }
    if hasattr(vlm, "peft_config"):
        from peft import get_peft_model_state_dict
        payload["vlm_lora"] = get_peft_model_state_dict(vlm)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)


def _restore_checkpoint(
    *, path: Path, heads: Any, vlm: Any, optimizer: Any, device: torch.device,
    run_paths: RunPaths, special_token_ids: Mapping[str, int],
) -> Dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") not in {"anatopave-joint-v1", "anatopave-joint-v2"}:
        raise RuntimeError(f"Unsupported ANATOPAVE checkpoint format: {checkpoint.get('format')}")
    saved_paths = checkpoint.get("paths", {})
    for key in ("fetalclip_sha256", "router_manifest_sha256", "seg_pretrain_sha256"):
        if saved_paths.get(key) != getattr(run_paths, key):
            raise RuntimeError(f"Resume checkpoint dependency mismatch: {key}")
    for name in ("composer", "language_projection", "segmentation_projection", "grounding", "decoder"):
        getattr(heads, name).load_state_dict(checkpoint[name], strict=True)
    if "vlm_lora" in checkpoint:
        from peft import set_peft_model_state_dict
        set_peft_model_state_dict(vlm, checkpoint["vlm_lora"])
    embedding = vlm.get_input_embeddings().weight
    with torch.no_grad():
        for name, values in checkpoint.get("special_token_embeddings", {}).items():
            embedding[int(special_token_ids[name])].copy_(
                values.to(device=embedding.device, dtype=embedding.dtype)
            )
    optimizer.load_state_dict(checkpoint["optimizer"])
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)
    rng = checkpoint.get("rng_state")
    if rng:
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        if device.type == "cuda" and rng.get("cuda") is not None:
            torch.cuda.set_rng_state_all(rng["cuda"])
    return checkpoint


def _hard_dice(logits: torch.Tensor, targets: torch.Tensor) -> float:
    prediction = torch.sigmoid(logits) >= 0.5
    target = targets >= 0.5
    intersection = (prediction & target).flatten(1).sum(1).float()
    denominator = prediction.flatten(1).sum(1) + target.flatten(1).sum(1)
    return float(((2 * intersection + 1e-6) / (denominator + 1e-6)).mean().item())


@torch.no_grad()
def _evaluate_subset(
    *,
    loader: DataLoader,
    device: torch.device,
    feature_extractor: Any,
    summary_cache: CompactSummaryCache,
    heads: Any,
    vlm: Any,
    processor: Any,
    injector: EvidenceEmbeddingInjector,
    seg_token_id: int,
    criterion: Any,
    lambda_lm: float,
    lambda_seg: float,
) -> Dict[str, float]:
    heads_was_training = heads.training
    vlm_was_training = vlm.training
    heads.eval()
    vlm.eval()
    totals = {"loss": 0.0, "loss_lm": 0.0, "loss_seg": 0.0, "dice": 0.0}
    samples = 0
    for batch in loader:
        output = forward_joint_batch(
            batch=batch, device=device, feature_extractor=feature_extractor,
            summary_cache=summary_cache, heads=heads, vlm=vlm, processor=processor,
            injector=injector, seg_token_id=seg_token_id, segmentation_criterion=criterion,
            lambda_lm=lambda_lm, lambda_seg=lambda_seg,
        )
        batch_size = int(output["main_masks"].shape[0])
        samples += batch_size
        for key in ("loss", "loss_lm", "loss_seg"):
            totals[key] += float(output[key].float().item()) * batch_size
        totals["dice"] += _hard_dice(output["seg_logits"], output["main_masks"]) * batch_size
    heads.train(heads_was_training)
    vlm.train(vlm_was_training)
    return {key: value / max(samples, 1) for key, value in totals.items()}


@torch.no_grad()
def _validate_full_epoch(
    *, loader: DataLoader, device: torch.device, feature_extractor: Any,
    summary_cache: CompactSummaryCache, heads: Any, vlm: Any, processor: Any,
    injector: EvidenceEmbeddingInjector, seg_token_id: int, criterion: Any,
    lambda_lm: float, lambda_seg: float, text_eval_samples: int,
) -> Dict[str, Any]:
    heads.eval()
    vlm.eval()
    totals = {"loss": 0.0, "loss_lm": 0.0, "loss_seg": 0.0}
    accumulator = MetricAccumulator(0.5)
    generations = []
    samples = 0
    for batch in loader:
        output = forward_joint_batch(
            batch=batch, device=device, feature_extractor=feature_extractor,
            summary_cache=summary_cache, heads=heads, vlm=vlm, processor=processor,
            injector=injector, seg_token_id=seg_token_id, segmentation_criterion=criterion,
            lambda_lm=lambda_lm, lambda_seg=lambda_seg,
        )
        batch_size = int(output["main_masks"].shape[0])
        samples += batch_size
        for key in totals:
            totals[key] += float(output[key].float().item()) * batch_size
        for index in range(batch_size):
            probability = torch.sigmoid(output["seg_logits"])[index, 0].float().cpu().numpy()
            target = output["main_masks"][index, 0].float().cpu().numpy()
            accumulator.add(
                probability, target, batch["main_gt_clip_prompt"][index],
                batch["main_image_paths"][index],
            )
        if len(generations) < text_eval_samples:
            generated_ids = generate_one(
                batch=batch, processor=processor, vlm=vlm, injector=injector,
                language_embedding=output["composed"]["language_embedding"], device=device,
                max_new_tokens=16,
            )
            generations.append({
                "main_view_id": batch["main_view_id"][0],
                "reference": batch["main_text"][0],
                "generated": processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True),
            })
    means = {key: value / max(samples, 1) for key, value in totals.items()}
    return {
        **means,
        "n": samples,
        "segmentation": accumulator.summary(),
        "generated_samples": generations,
    }


def _run_full_training(
    *, args: argparse.Namespace, log: Any, train_dataset: ExaminationDataset,
    validation_dataset: ExaminationDataset, train_loader: DataLoader, val_loader: DataLoader,
    device: torch.device, feature_extractor: Any, cache: CompactSummaryCache, heads: Any,
    vlm: Any, processor: Any, injector: EvidenceEmbeddingInjector, seg_token_id: int,
    evidence_token_id: int, criterion: Any, optimizer: Any, run_paths: RunPaths,
    cache_sha: str, cache_metadata: Mapping[str, Any], validation_fixed: bool,
) -> None:
    special_token_ids = {"seg": seg_token_id, "evidence": evidence_token_id}
    start_epoch = 1
    global_step = 0
    best_metrics: Dict[str, Any] = {"loss": float("inf"), "epoch": 0}
    if args.resume_checkpoint is not None:
        resumed = _restore_checkpoint(
            path=args.resume_checkpoint, heads=heads, vlm=vlm, optimizer=optimizer,
            device=device, run_paths=run_paths, special_token_ids=special_token_ids,
        )
        start_epoch = int(resumed.get("epoch", 0)) + 1
        global_step = int(resumed.get("global_step", 0))
        best_metrics.update(resumed.get("best_metrics", {}))
        log(
            f"[resume] checkpoint={args.resume_checkpoint} start_epoch={start_epoch} "
            f"global_step={global_step} best_loss={float(best_metrics['loss']):.4f}"
        )
    if start_epoch > args.epochs:
        log(f"[done] checkpoint already completed requested epochs={args.epochs}")
        return

    architecture_audit = {
        "fetalclip_all_frozen": not any(p.requires_grad for p in feature_extractor.parameters()),
        "decoder_batchnorm2d_count": sum(isinstance(m, nn.BatchNorm2d) for m in heads.decoder.modules()),
        "validation_condition_source": sorted(
            set(validation_dataset[index]["condition_source"] for index in range(len(validation_dataset)))
        ),
        "validation_auxiliary_paths_fixed": validation_fixed,
        "cache_contains_dense_features": bool(cache_metadata.get("contains_dense_features", False)),
        "lambda_lm_active_from_epoch_1": args.lambda_lm > 0.0,
    }
    if not architecture_audit["fetalclip_all_frozen"]:
        raise AssertionError("FetalCLIP must remain frozen during full joint training")
    if architecture_audit["decoder_batchnorm2d_count"] != 0:
        raise AssertionError("Joint decoder contains BatchNorm2d")
    if architecture_audit["validation_condition_source"] != ["router"]:
        raise AssertionError("Full validation must use router conditions without GT fallback")
    if not architecture_audit["lambda_lm_active_from_epoch_1"]:
        raise ValueError("Full joint training requires lambda_lm > 0 from epoch 1")
    (args.out_dir / "architecture_audit.json").write_text(
        json.dumps(architecture_audit, indent=2) + "\n", encoding="utf-8"
    )

    metrics_path = args.out_dir / "epoch_metrics.jsonl"
    if args.resume_checkpoint is None:
        metrics_path.write_text("", encoding="utf-8")
    trainable = [
        parameter for parameter in list(heads.parameters()) + list(vlm.parameters())
        if parameter.requires_grad
    ]
    log(
        f"[full] train_samples={len(train_dataset)} val_samples={len(validation_dataset)} "
        f"epochs={args.epochs} lambda_lm={args.lambda_lm:.4f} lambda_seg={args.lambda_seg:.4f} "
        f"text_eval_samples={args.text_eval_samples}"
    )
    run_start = time.perf_counter()
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_dataset.set_epoch(epoch - 1)
        heads.train()
        vlm.train()
        totals = {"loss": 0.0, "loss_lm": 0.0, "loss_seg": 0.0}
        samples = 0
        log(
            f"[sched] Epoch {epoch}/{args.epochs}: lambda_lm={args.lambda_lm:.4f}, "
            f"lambda_seg={args.lambda_seg:.4f}"
        )
        for batch_index, batch in enumerate(train_loader, start=1):
            optimizer.zero_grad(set_to_none=True)
            output = forward_joint_batch(
                batch=batch, device=device, feature_extractor=feature_extractor,
                summary_cache=cache, heads=heads, vlm=vlm, processor=processor,
                injector=injector, seg_token_id=seg_token_id, segmentation_criterion=criterion,
                lambda_lm=args.lambda_lm, lambda_seg=args.lambda_seg,
            )
            if not bool(torch.isfinite(output["loss"])):
                raise FloatingPointError(f"Non-finite joint loss at epoch={epoch} batch={batch_index}")
            output["loss"].backward()
            clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            global_step += 1
            batch_size = int(output["main_masks"].shape[0])
            samples += batch_size
            for key in totals:
                totals[key] += float(output[key].detach().float().item()) * batch_size
            if batch_index == 1 or batch_index % max(args.log_interval, 1) == 0:
                log(
                    f"[Train] Epoch {epoch}/{args.epochs} step={batch_index}/{len(train_loader)} "
                    f"loss={float(output['loss'].detach()):.4f} "
                    f"lm={float(output['loss_lm'].detach()):.4f} "
                    f"seg={float(output['loss_seg'].detach()):.4f}"
                )
        train_metrics = {key: value / max(samples, 1) for key, value in totals.items()}
        validation = _validate_full_epoch(
            loader=val_loader, device=device, feature_extractor=feature_extractor,
            summary_cache=cache, heads=heads, vlm=vlm, processor=processor, injector=injector,
            seg_token_id=seg_token_id, criterion=criterion, lambda_lm=args.lambda_lm,
            lambda_seg=args.lambda_seg, text_eval_samples=args.text_eval_samples,
        )
        segmentation = validation["segmentation"]
        epoch_metrics = {
            "epoch": epoch,
            "global_step": global_step,
            "train": train_metrics,
            "validation": validation,
            "epoch_elapsed_sec": time.perf_counter() - epoch_start,
            "run_elapsed_sec": time.perf_counter() - run_start,
            "lambda_lm": args.lambda_lm,
            "lambda_seg": args.lambda_seg,
        }
        with metrics_path.open("a", encoding="utf-8", buffering=1) as handle:
            handle.write(json.dumps(epoch_metrics, ensure_ascii=False) + "\n")
            handle.flush()
        log(
            f"[Val] Epoch {epoch}/{args.epochs} loss={validation['loss']:.4f} "
            f"lm={validation['loss_lm']:.4f} seg={validation['loss_seg']:.4f} "
            f"dice={segmentation['dice']:.4f} bbox_hit_0_5={segmentation['bbox_hit_0_5']:.4f}"
        )
        improved = float(validation["loss"]) < float(best_metrics["loss"])
        if improved:
            best_metrics = {
                "loss": float(validation["loss"]),
                "dice": float(segmentation["dice"]),
                "bbox_hit_0_5": float(segmentation["bbox_hit_0_5"]),
                "epoch": epoch,
            }
        latest_path = args.out_dir / "resume_latest.pt"
        _checkpoint(
            path=latest_path, heads=heads, vlm=vlm, optimizer=optimizer, args=args,
            run_paths=run_paths, cache_path=args.summary_cache, cache_sha256=cache_sha,
            metrics=epoch_metrics, special_token_ids=special_token_ids, epoch=epoch,
            global_step=global_step, best_metrics=best_metrics,
        )
        log(f"Saved resume checkpoint: {latest_path}")
        if improved:
            best_path = args.out_dir / "best.pt"
            temporary_best = best_path.with_suffix(best_path.suffix + ".tmp")
            shutil.copyfile(latest_path, temporary_best)
            os.replace(temporary_best, best_path)
            log(
                f"[Val] New best checkpoint by loss={best_metrics['loss']:.4f}, "
                f"dice={best_metrics['dice']:.4f}, bbox_hit_0_5={best_metrics['bbox_hit_0_5']:.4f}"
            )
            log(f"Saved best checkpoint: {best_path}")
        if any(parameter.grad is not None for parameter in feature_extractor.parameters()):
            raise AssertionError("Frozen FetalCLIP accumulated gradients")
    log(
        f"[done] full training epochs={args.epochs} best_epoch={best_metrics['epoch']} "
        f"best_loss={best_metrics['loss']:.4f}"
    )


def main() -> None:
    args = parse_args()
    if args.epochs is None:
        args.epochs = 10 if args.mode == "full" else 1
    if args.mode == "full" and args.train_samples > 0:
        raise ValueError("Full mode requires --train_samples 0; bounded defaults are smoke-only")
    if args.mode == "full" and args.val_samples > 0:
        raise ValueError("Full mode requires --val_samples 0; validation must cover the full fold")
    if args.mode == "full" and args.overfit_steps != 0:
        raise ValueError("Full mode requires --overfit_steps 0; overfit auditing is smoke-only")
    set_seed(args.seed)
    if args.local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log, log_handle = log_factory(
        args.out_dir / "log.txt", append=args.resume_checkpoint is not None
    )
    run_paths = RunPaths(
        fetalclip=str(args.fetalclip_weights.resolve()),
        router_manifest=str(args.router_manifest.resolve()),
        seg_pretrain=str(args.seg_pretrain.resolve()),
        fetalclip_sha256=sha256_file(args.fetalclip_weights),
        router_manifest_sha256=sha256_file(args.router_manifest),
        seg_pretrain_sha256=sha256_file(args.seg_pretrain),
    )
    try:
        train_dataset = ExaminationDataset(
            args.observations, args.fold_dir / "train.txt",
            source_dataset="sam-dataset", condition_source="gt", seed=args.seed,
        )
        validation_dataset = ExaminationDataset(
            args.observations, args.fold_dir / "val.txt",
            source_dataset="clip-dataset", condition_source="router",
            router_manifest=args.router_manifest, seed=args.seed, aux_slot_dropout_prob=0.0,
        )
        validation_paths_before = [validation_dataset[i]["slot_image_paths"] for i in range(len(validation_dataset))]
        validation_dataset.set_epoch(99)
        validation_paths_after = [validation_dataset[i]["slot_image_paths"] for i in range(len(validation_dataset))]
        validation_fixed = validation_paths_before == validation_paths_after
        if not validation_fixed:
            raise AssertionError("Validation auxiliary slots changed across epochs")

        _, preprocess, feature_extractor = load_frozen_fetalclip(
            args.fetalclip_config, args.fetalclip_weights, device
        )
        if args.summary_cache.exists():
            cache, cache_metadata = CompactSummaryCache.load(args.summary_cache)
            if cache_metadata.get("fetalclip_sha256") != run_paths.fetalclip_sha256:
                raise RuntimeError("Compact cache belongs to a different FetalCLIP checkpoint")
        else:
            cache = build_compact_summary_cache(
                list(dataset_observation_paths(train_dataset)) + list(dataset_observation_paths(validation_dataset)),
                project_root=Path.cwd(), preprocess=preprocess, feature_extractor=feature_extractor,
                device=device, batch_size=16,
            )
            cache_metadata = {
                "format": "anatopave-compact-summary-v1",
                "fetalclip_sha256": run_paths.fetalclip_sha256,
                "feature_dim": 768,
                "contains_dense_features": False,
            }
            cache.save(args.summary_cache, cache_metadata)
        cache_sha = sha256_file(args.summary_cache)
        log(f"[cache] summaries={len(cache)} size_mb={args.summary_cache.stat().st_size / 2**20:.2f}")

        vlm, processor, seg_token_id, evidence_token_id, vlm_dim = load_vlm_and_processor(
            args.vlm_model_id, device, local_files_only=args.local_files_only,
            lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        )
        if hasattr(vlm, "gradient_checkpointing_enable"):
            vlm.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        vlm.train()
        heads = build_heads(
            summary_dim=768, dense_feature_dim=feature_extractor.width, vlm_dim=vlm_dim,
            composer_dim=args.composer_dim, grounding_dim=args.grounding_dim, num_heads=args.num_heads,
            seg_pretrain_path=args.seg_pretrain, device=device,
        )
        heads.train()
        assert_no_batch_norm(heads.decoder)
        injector = EvidenceEmbeddingInjector(evidence_token_id)
        criterion = SparseForegroundLoss(SegmentationLossConfig(name="dice_focal", focal_alpha=0.85, focal_gamma=2.0))
        optimizer = _optimizer(heads, vlm, args.lr, args.decoder_lr, args.weight_decay)

        train_indices = _subset(train_dataset, args.train_samples)
        val_indices = _subset(validation_dataset, args.val_samples)
        train_loader = DataLoader(Subset(train_dataset, train_indices), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=examination_collate)
        train_eval_loader = DataLoader(Subset(train_dataset, train_indices), batch_size=1, shuffle=False, num_workers=0, collate_fn=examination_collate)
        val_loader = DataLoader(Subset(validation_dataset, val_indices), batch_size=1, shuffle=False, num_workers=0, collate_fn=examination_collate)

        if args.mode == "full":
            _run_full_training(
                args=args, log=log, train_dataset=train_dataset,
                validation_dataset=validation_dataset, train_loader=train_loader,
                val_loader=val_loader, device=device, feature_extractor=feature_extractor,
                cache=cache, heads=heads, vlm=vlm, processor=processor, injector=injector,
                seg_token_id=seg_token_id, evidence_token_id=evidence_token_id,
                criterion=criterion, optimizer=optimizer, run_paths=run_paths,
                cache_sha=cache_sha, cache_metadata=cache_metadata,
                validation_fixed=validation_fixed,
            )
            return

        first_batch = next(iter(train_loader))
        cache.requested_paths.clear()
        audit_output = forward_joint_batch(
            batch=first_batch, device=device, feature_extractor=feature_extractor, summary_cache=cache,
            heads=heads, vlm=vlm, processor=processor, injector=injector, seg_token_id=seg_token_id,
            segmentation_criterion=criterion, lambda_lm=args.lambda_lm, lambda_seg=args.lambda_seg,
        )
        expected_auxiliary_requests = int(
            (
                first_batch["slot_available"]
                & ~first_batch["slot_is_main"]
            ).sum().item()
        )
        actual_auxiliary_requests = len(cache.requested_paths)
        if actual_auxiliary_requests != expected_auxiliary_requests:
            raise AssertionError(
                "Compact-cache requests do not match available auxiliary slots: "
                f"expected={expected_auxiliary_requests} actual={actual_auxiliary_requests}"
            )
        gradient_audit = {
            "language_loss_to_composer_grad_norm": _grad_norm(audit_output["loss_lm"], heads.composer.parameters(), True),
            "segmentation_loss_to_composer_grad_norm": _grad_norm(audit_output["loss_seg"], heads.composer.parameters(), True),
            "language_loss_to_language_projection_grad_norm": _grad_norm(audit_output["loss_lm"], heads.language_projection.parameters(), True),
            "segmentation_loss_to_segmentation_projection_grad_norm": _grad_norm(audit_output["loss_seg"], heads.segmentation_projection.parameters(), True),
        }
        if not all(value > 0 for value in gradient_audit.values()):
            raise AssertionError(f"A required gradient path is zero: {gradient_audit}")
        with torch.no_grad(), autocast_ctx(device):
            baseline = heads.decoder.base_decoder(audit_output["dense_features"])
            disabled = heads.decoder(audit_output["dense_features"], None, use_composer=False)
            baseline_max_abs_diff = float((baseline - disabled).abs().max().item())
        if baseline_max_abs_diff != 0.0:
            raise AssertionError("Composer-off path does not exactly recover the baseline decoder")
        audit_output["loss"].backward()
        optimizer.zero_grad(set_to_none=True)
        del audit_output, baseline, disabled

        architecture_audit = {
            "fetalclip_all_frozen": not any(p.requires_grad for p in feature_extractor.parameters()),
            "fetalclip_has_grad": any(p.grad is not None for p in feature_extractor.parameters()),
            "decoder_batchnorm2d_count": sum(isinstance(m, nn.BatchNorm2d) for m in heads.decoder.modules()),
            "seg_pretrain_strict_loaded": True,
            "validation_condition_source": sorted(set(validation_dataset[i]["condition_source"] for i in val_indices)),
            "validation_auxiliary_paths_fixed": validation_fixed,
            "validation_missing_slot_count": int(
                sum(int((~validation_dataset[i]["slot_available"]).sum().item()) for i in val_indices)
            ),
            "composer_off_baseline_max_abs_diff": baseline_max_abs_diff,
            "auxiliary_cache_requests_expected": expected_auxiliary_requests,
            "auxiliary_cache_requests_actual": actual_auxiliary_requests,
            "missing_slots_requested_from_cache": bool(
                actual_auxiliary_requests != expected_auxiliary_requests
            ),
            "cache_contains_dense_features": bool(cache_metadata.get("contains_dense_features", False)),
        }
        (args.out_dir / "gradient_audit.json").write_text(json.dumps(gradient_audit, indent=2) + "\n", encoding="utf-8")
        (args.out_dir / "architecture_audit.json").write_text(json.dumps(architecture_audit, indent=2) + "\n", encoding="utf-8")
        log(f"[audit] gradients={gradient_audit}")
        overfit_before = _evaluate_subset(
            loader=train_eval_loader, device=device, feature_extractor=feature_extractor,
            summary_cache=cache, heads=heads, vlm=vlm, processor=processor, injector=injector,
            seg_token_id=seg_token_id, criterion=criterion, lambda_lm=args.lambda_lm,
            lambda_seg=args.lambda_seg,
        )
        log(f"[overfit-eval-before] {overfit_before}")

        initial_losses: Optional[Dict[str, float]] = None
        last_losses: Dict[str, float] = {}
        step = 0
        iterator = iter(train_loader)
        start = time.perf_counter()
        while step < args.overfit_steps:
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                batch = next(iterator)
            optimizer.zero_grad(set_to_none=True)
            output = forward_joint_batch(
                batch=batch, device=device, feature_extractor=feature_extractor, summary_cache=cache,
                heads=heads, vlm=vlm, processor=processor, injector=injector, seg_token_id=seg_token_id,
                segmentation_criterion=criterion, lambda_lm=args.lambda_lm, lambda_seg=args.lambda_seg,
            )
            output["loss"].backward()
            clip_grad_norm_([p for p in list(heads.parameters()) + list(vlm.parameters()) if p.requires_grad], 1.0)
            optimizer.step()
            step += 1
            last_losses = {key: float(output[key].detach().float().item()) for key in ("loss", "loss_lm", "loss_seg")}
            if initial_losses is None:
                initial_losses = dict(last_losses)
            log(f"[overfit] step={step}/{args.overfit_steps} total={last_losses['loss']:.4f} lm={last_losses['loss_lm']:.4f} seg={last_losses['loss_seg']:.4f}")

        overfit_after = _evaluate_subset(
            loader=train_eval_loader, device=device, feature_extractor=feature_extractor,
            summary_cache=cache, heads=heads, vlm=vlm, processor=processor, injector=injector,
            seg_token_id=seg_token_id, criterion=criterion, lambda_lm=args.lambda_lm,
            lambda_seg=args.lambda_seg,
        )
        log(f"[overfit-eval-after] {overfit_after}")

        # One bounded epoch is executed after the repeated 20-sample overfit loop.
        epoch_totals = {"loss": 0.0, "loss_lm": 0.0, "loss_seg": 0.0}
        for epoch in range(args.epochs):
            train_dataset.set_epoch(epoch)
            for batch in train_loader:
                optimizer.zero_grad(set_to_none=True)
                output = forward_joint_batch(
                    batch=batch, device=device, feature_extractor=feature_extractor, summary_cache=cache,
                    heads=heads, vlm=vlm, processor=processor, injector=injector, seg_token_id=seg_token_id,
                    segmentation_criterion=criterion, lambda_lm=args.lambda_lm, lambda_seg=args.lambda_seg,
                )
                output["loss"].backward()
                clip_grad_norm_([p for p in list(heads.parameters()) + list(vlm.parameters()) if p.requires_grad], 1.0)
                optimizer.step()
                for key in epoch_totals:
                    epoch_totals[key] += float(output[key].detach().float().item())
        epoch_batches = max(len(train_loader) * max(args.epochs, 1), 1)
        epoch_means = {key: value / epoch_batches for key, value in epoch_totals.items()}

        heads.eval(); vlm.eval()
        val_dice = []
        val_losses = []
        generation_succeeded = False
        with torch.no_grad():
            for index, batch in enumerate(val_loader):
                output = forward_joint_batch(
                    batch=batch, device=device, feature_extractor=feature_extractor, summary_cache=cache,
                    heads=heads, vlm=vlm, processor=processor, injector=injector, seg_token_id=seg_token_id,
                    segmentation_criterion=criterion, lambda_lm=args.lambda_lm, lambda_seg=args.lambda_seg,
                )
                val_dice.append(_hard_dice(output["seg_logits"], output["main_masks"]))
                val_losses.append(float(output["loss"].float().item()))
                if index == 0:
                    generated = generate_one(
                        batch=batch, processor=processor, vlm=vlm, injector=injector,
                        language_embedding=output["composed"]["language_embedding"], device=device,
                    )
                    generation_succeeded = bool(generated.numel())

        architecture_audit["fetalclip_has_grad_after_training"] = any(p.grad is not None for p in feature_extractor.parameters())
        architecture_audit["teacher_forcing_and_generate_shared_helper"] = generation_succeeded
        if architecture_audit["fetalclip_has_grad_after_training"]:
            raise AssertionError("Frozen FetalCLIP accumulated gradients")
        (args.out_dir / "architecture_audit.json").write_text(json.dumps(architecture_audit, indent=2) + "\n", encoding="utf-8")
        metrics = {
            "mode": args.mode,
            "overfit_samples": len(train_indices),
            "overfit_steps": args.overfit_steps,
            "overfit_initial": initial_losses,
            "overfit_final": last_losses,
            "overfit_aggregate_before": overfit_before,
            "overfit_aggregate_after": overfit_after,
            "smoke_epochs": args.epochs,
            "smoke_train_epoch_mean": epoch_means,
            "smoke_train_samples": len(train_indices),
            "smoke_val_samples": len(val_indices),
            "validation_loss": float(np.mean(val_losses)),
            "validation_dice": float(np.mean(val_dice)),
            "elapsed_sec": time.perf_counter() - start,
            "gradient_audit": gradient_audit,
            "architecture_audit": architecture_audit,
        }
        (args.out_dir / "smoke_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        _checkpoint(
            path=args.out_dir / "smoke_latest.pt", heads=heads, vlm=vlm, optimizer=optimizer,
            args=args, run_paths=run_paths, cache_path=args.summary_cache, cache_sha256=cache_sha, metrics=metrics,
            special_token_ids={"seg": seg_token_id, "evidence": evidence_token_id},
        )
        log(f"[done] smoke validation_dice={metrics['validation_dice']:.4f} validation_loss={metrics['validation_loss']:.4f}")
    finally:
        log_handle.close()


if __name__ == "__main__":
    main()
