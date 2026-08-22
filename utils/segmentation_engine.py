from __future__ import annotations

import contextlib
import hashlib
import json
import math
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Subset

from models.seg_unet import assert_no_batch_norm
from utils.runtime import autocast_ctx
from utils.segmentation_data import (
    CLIP_MEAN,
    CLIP_STD,
    PREPROCESS_VERSION,
    PROMPT_CLASSES,
    PROMPT_SLUGS,
    UNET_MEAN,
    UNET_STD,
    PairedSegTransform,
    SegmentationDataset,
    class_counts,
    inverse_map_array,
    read_case_set,
    run_geometry_check,
    segmentation_collate,
    select_fixed_visualization_indices,
)
from utils.segmentation_losses import SegmentationLossConfig, SparseForegroundLoss


@dataclass
class SegmentationExperimentConfig:
    experiment_name: str
    model_type: str
    train_dataset_root: str
    val_dataset_root: str
    fold_dir: str
    out_dir: str
    preprocess_mode: str
    lr: float
    epochs: int = 30
    batch_size: int = 1
    accum_steps: int = 4
    weight_decay: float = 1e-4
    num_workers: int = 2
    patience: int = 6
    seed: int = 42
    threshold: float = 0.5
    max_grad_norm: float = 1.0
    augment: bool = False
    cpu: bool = False
    resume_checkpoint: str = ""
    overfit_check: bool = True
    overfit_samples: int = 20
    overfit_max_steps: int = 400
    overfit_target_dice: float = 0.9
    vis_per_class: int = 5
    vis_every: int = 1
    loss_name: str = "dice_focal"
    focal_alpha: float = 0.85
    focal_gamma: float = 2.0
    fetalclip_weights: str = ""


class FlushLogger:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("a", encoding="utf-8", buffering=1)

    def __call__(self, message: str) -> None:
        print(message, flush=True)
        self.handle.write(message + "\n")
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _device(config: SegmentationExperimentConfig) -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() and not config.cpu else "cpu")


def _amp_scaler(device: torch.device):
    use_fp16_scaler = device.type == "cuda" and not torch.cuda.is_bf16_supported()
    return torch.amp.GradScaler("cuda", enabled=use_fp16_scaler)


def _trainable_module(model: nn.Module) -> nn.Module:
    decoder = getattr(model, "decoder", None)
    return decoder if isinstance(decoder, nn.Module) else model


def parameter_counts(model: nn.Module) -> Tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable


def optimizer_parameter_groups(model: nn.Module, weight_decay: float) -> List[Dict[str, Any]]:
    decay: List[nn.Parameter] = []
    no_decay: List[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim <= 1 or name.endswith(".bias") or "norm" in name.lower():
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    if not decay and not no_decay:
        raise RuntimeError("No trainable parameters found.")
    return [
        {"params": decay, "weight_decay": float(weight_decay)},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def build_scheduler(optimizer: torch.optim.Optimizer, steps_per_epoch: int, epochs: int) -> LambdaLR:
    warmup_steps = max(int(steps_per_epoch), 1)
    total_steps = max(int(steps_per_epoch * epochs), warmup_steps + 1)

    def factor(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    return LambdaLR(optimizer, lr_lambda=factor)


def _bbox_iou(prediction: np.ndarray, target: np.ndarray) -> float:
    pred_y, pred_x = np.where(prediction > 0)
    gt_y, gt_x = np.where(target > 0)
    if len(gt_x) == 0:
        return 1.0 if len(pred_x) == 0 else 0.0
    if len(pred_x) == 0:
        return 0.0
    px0, px1, py0, py1 = pred_x.min(), pred_x.max(), pred_y.min(), pred_y.max()
    gx0, gx1, gy0, gy1 = gt_x.min(), gt_x.max(), gt_y.min(), gt_y.max()
    ix0, ix1 = max(px0, gx0), min(px1, gx1)
    iy0, iy1 = max(py0, gy0), min(py1, gy1)
    intersection = max(ix1 - ix0 + 1, 0) * max(iy1 - iy0 + 1, 0)
    pred_area = (px1 - px0 + 1) * (py1 - py0 + 1)
    gt_area = (gx1 - gx0 + 1) * (gy1 - gy0 + 1)
    return float(intersection / max(pred_area + gt_area - intersection, 1))


def _hard_dice(prediction: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    intersection = float(np.logical_and(prediction > 0, target > 0).sum())
    denominator = float((prediction > 0).sum() + (target > 0).sum())
    return float((2.0 * intersection + eps) / (denominator + eps))


class MetricAccumulator:
    def __init__(self, threshold: float) -> None:
        self.threshold = float(threshold)
        self.records: List[Dict[str, Any]] = []

    def add(self, probability: np.ndarray, target: np.ndarray, prompt: str, path: str) -> None:
        prediction = probability >= self.threshold
        empty = not bool(prediction.any())
        bbox_iou = _bbox_iou(prediction, target)
        self.records.append(
            {
                "path": path,
                "prompt": prompt,
                "dice": _hard_dice(prediction, target),
                "bbox_iou": bbox_iou,
                "bbox_hit_0_5": float(bbox_iou >= 0.5),
                "empty": float(empty),
                "foreground_ratio": float(prediction.mean()),
                "probability_max": float(probability.max()),
                "probability_mean": float(probability.mean()),
            }
        )

    @staticmethod
    def _summarize(records: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
        if not records:
            return {
                "n": 0,
                "dice": 0.0,
                "bbox_iou": 0.0,
                "bbox_hit_0_5": 0.0,
                "empty_mask_rate": 0.0,
                "foreground_ratio_mean": 0.0,
                "foreground_ratio_median": 0.0,
                "foreground_ratio_p10": 0.0,
                "foreground_ratio_p90": 0.0,
                "probability_max_mean": 0.0,
                "probability_mean": 0.0,
            }
        foreground = np.asarray([record["foreground_ratio"] for record in records], dtype=np.float64)
        return {
            "n": len(records),
            "dice": float(np.mean([record["dice"] for record in records])),
            "bbox_iou": float(np.mean([record["bbox_iou"] for record in records])),
            "bbox_hit_0_5": float(np.mean([record["bbox_hit_0_5"] for record in records])),
            "empty_mask_rate": float(np.mean([record["empty"] for record in records])),
            "foreground_ratio_mean": float(foreground.mean()),
            "foreground_ratio_median": float(np.median(foreground)),
            "foreground_ratio_p10": float(np.percentile(foreground, 10)),
            "foreground_ratio_p90": float(np.percentile(foreground, 90)),
            "probability_max_mean": float(np.mean([record["probability_max"] for record in records])),
            "probability_mean": float(np.mean([record["probability_mean"] for record in records])),
        }

    def summary(self) -> Dict[str, Any]:
        grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
        for record in self.records:
            grouped[PROMPT_SLUGS[record["prompt"]]].append(record)
        return {
            **self._summarize(self.records),
            "per_class": {
                slug: self._summarize(grouped.get(slug, [])) for slug in PROMPT_SLUGS.values()
            },
        }


def _finite_or_raise(tensor: torch.Tensor, what: str, paths: Sequence[str]) -> None:
    if not bool(torch.isfinite(tensor).all()):
        raise FloatingPointError(f"NaN/Inf in {what}; batch={list(paths)}")


def _gradient_finite_or_raise(model: nn.Module, paths: Sequence[str]) -> None:
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
            raise FloatingPointError(f"NaN/Inf gradient in {name}; batch={list(paths)}")


def _denormalize(tensor: torch.Tensor, mode: str) -> np.ndarray:
    mean, std = (UNET_MEAN, UNET_STD) if mode == "letterbox" else (CLIP_MEAN, CLIP_STD)
    array = tensor.detach().cpu().float().numpy().transpose(1, 2, 0)
    array = array * np.asarray(std, dtype=np.float32) + np.asarray(mean, dtype=np.float32)
    return np.clip(array * 255.0, 0, 255).astype(np.uint8)


def save_preprocess_audit(dataset: SegmentationDataset, out_dir: Path, count: int = 20) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for index in range(min(count, len(dataset))):
        item = dataset[index]
        image = _denormalize(item["image"], dataset.transform.mode)
        mask = item["mask"][0].numpy() > 0.5
        overlay = image.copy()
        overlay[mask] = (0.55 * overlay[mask] + 0.45 * np.array([255, 0, 0])).astype(np.uint8)
        stem = f"{index:02d}_{item['prompt_slug']}"
        Image.fromarray(image).save(out_dir / f"{stem}_image.png")
        Image.fromarray((mask.astype(np.uint8) * 255)).save(out_dir / f"{stem}_mask.png")
        Image.fromarray(overlay).save(out_dir / f"{stem}_overlay.png")


def _visualization_paths(dataset: SegmentationDataset, per_class: int) -> Tuple[set[str], Dict[str, int]]:
    indices = select_fixed_visualization_indices(dataset, per_class=per_class)
    paths = {str(dataset.samples[index].image_path) for selected in indices.values() for index in selected}
    return paths, {slug: len(selected) for slug, selected in indices.items()}


def save_validation_visualizations(
    records: Sequence[Dict[str, Any]],
    out_dir: Path,
    *,
    threshold: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for record in records:
        probability = record["probability"]
        target = record["target"]
        meta = record["geometry"]
        original_image = np.asarray(record["original_image"].convert("RGB"), dtype=np.uint8)
        probability_original = inverse_map_array(probability, meta, nearest=False)
        gt_original = inverse_map_array(target, meta, nearest=True) >= 0.5
        prediction = probability_original >= float(threshold)
        heat = cv2.applyColorMap(np.clip(probability_original * 255.0, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
        heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
        overlay = original_image.copy()
        overlay[prediction] = (
            0.55 * overlay[prediction] + 0.45 * np.array([255, 0, 0], dtype=np.float32)
        ).astype(np.uint8)
        stem = f"{record['prompt_slug']}_{Path(record['path']).stem}"
        np.save(out_dir / f"{stem}_prob.npy", probability_original.astype(np.float32))
        Image.fromarray(original_image).save(out_dir / f"{stem}_image.png")
        Image.fromarray((gt_original.astype(np.uint8) * 255)).save(out_dir / f"{stem}_gt.png")
        Image.fromarray(heat).save(out_dir / f"{stem}_prob_heatmap.png")
        Image.fromarray((prediction.astype(np.uint8) * 255)).save(out_dir / f"{stem}_pred.png")
        Image.fromarray(overlay).save(out_dir / f"{stem}_overlay.png")


def _forward_model(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    logits = model(images)
    if logits.shape != (images.shape[0], 1, 224, 224):
        raise AssertionError(f"Model must return [B,1,224,224], got {tuple(logits.shape)}")
    return logits


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: SparseForegroundLoss,
    device: torch.device,
    *,
    threshold: float,
    visualization_paths: Optional[set[str]] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    model.eval()
    accumulator = MetricAccumulator(threshold)
    losses = defaultdict(float)
    samples = 0
    all_predictions: List[Dict[str, Any]] = []
    visualizations: List[Dict[str, Any]] = []
    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        masks = batch["masks"].to(device, non_blocking=True)
        with autocast_ctx(device):
            logits = _forward_model(model, images)
            loss_parts = criterion(logits, masks)
        _finite_or_raise(logits, "validation logits", batch["image_paths"])
        probabilities = torch.sigmoid(logits).float().cpu().numpy()[:, 0]
        targets = masks.float().cpu().numpy()[:, 0]
        batch_size = len(batch["image_paths"])
        samples += batch_size
        for key, value in loss_parts.items():
            losses[key] += float(value.item()) * batch_size
        for index, (probability, target, prompt, path) in enumerate(
            zip(probabilities, targets, batch["prompts"], batch["image_paths"])
        ):
            accumulator.add(probability, target, prompt, path)
            prediction_record = {
                "probability": probability,
                "target": target,
                "prompt": prompt,
                "prompt_slug": PROMPT_SLUGS[prompt],
                "path": path,
            }
            all_predictions.append(prediction_record)
            if visualization_paths is not None and path in visualization_paths:
                visualizations.append(
                    {
                        **prediction_record,
                        "geometry": batch["geometry"][index],
                        "original_image": batch["original_images"][index],
                    }
                )
    summary = accumulator.summary()
    summary.update(
        {
            "loss": losses["total"] / max(samples, 1),
            "loss_dice": losses["dice"] / max(samples, 1),
            "loss_focal": losses["focal"] / max(samples, 1),
        }
    )
    return summary, all_predictions, visualizations


def _checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    scaler: Any,
    config: SegmentationExperimentConfig,
    *,
    epoch: int,
    global_step: int,
    best_metrics: Optional[Dict[str, Any]],
    patience_counter: int,
) -> Dict[str, Any]:
    return {
        "model": _trainable_module(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_metrics": best_metrics,
        "patience_counter": int(patience_counter),
        "config": asdict(config),
        "loss_config": asdict(
            SegmentationLossConfig(
                name=config.loss_name,
                focal_alpha=config.focal_alpha,
                focal_gamma=config.focal_gamma,
            )
        ),
        "split_paths": {
            "train": str(Path(config.fold_dir) / "train.txt"),
            "val": str(Path(config.fold_dir) / "val.txt"),
        },
        "threshold": float(config.threshold),
        "preprocess_version": PREPROCESS_VERSION,
    }


def _save_checkpoint(path: Path, payload: Dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _load_model_state(model: nn.Module, checkpoint: Mapping[str, Any]) -> None:
    _trainable_module(model).load_state_dict(checkpoint["model"], strict=True)


def _is_better(current: Mapping[str, Any], best: Optional[Mapping[str, Any]]) -> bool:
    if best is None:
        return True
    dice_delta = float(current["dice"]) - float(best["dice"])
    if dice_delta > 1e-4:
        return True
    if abs(dice_delta) <= 1e-4:
        bbox_delta = float(current["bbox_iou"]) - float(best["bbox_iou"])
        if bbox_delta > 1e-8:
            return True
        if abs(bbox_delta) <= 1e-8:
            return float(current["empty_mask_rate"]) < float(best["empty_mask_rate"])
    return False


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: SparseForegroundLoss,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    scaler: Any,
    device: torch.device,
    *,
    accum_steps: int,
    max_grad_norm: float,
) -> Tuple[Dict[str, float], int]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    totals = defaultdict(float)
    samples = 0
    optimizer_steps = 0
    for batch_index, batch in enumerate(loader):
        images = batch["images"].to(device, non_blocking=True)
        masks = batch["masks"].to(device, non_blocking=True)
        with autocast_ctx(device):
            logits = _forward_model(model, images)
            loss_parts = criterion(logits, masks)
            scaled_loss = loss_parts["total"] / float(accum_steps)
        _finite_or_raise(logits, "training logits", batch["image_paths"])
        _finite_or_raise(loss_parts["total"], "training loss", batch["image_paths"])
        scaler.scale(scaled_loss).backward()
        batch_size = len(batch["image_paths"])
        samples += batch_size
        for key, value in loss_parts.items():
            totals[key] += float(value.item()) * batch_size

        should_step = (batch_index + 1) % accum_steps == 0 or (batch_index + 1) == len(loader)
        if should_step:
            scaler.unscale_(optimizer)
            _gradient_finite_or_raise(model, batch["image_paths"])
            clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            optimizer_steps += 1
            if hasattr(model, "assert_encoder_frozen"):
                model.assert_encoder_frozen()
    return {
        "loss": totals["total"] / max(samples, 1),
        "loss_dice": totals["dice"] / max(samples, 1),
        "loss_focal": totals["focal"] / max(samples, 1),
        "lr": float(optimizer.param_groups[0]["lr"]),
    }, optimizer_steps


def run_overfit_check(
    model_factory: Callable[[], nn.Module],
    dataset: SegmentationDataset,
    config: SegmentationExperimentConfig,
    device: torch.device,
    out_dir: Path,
    log: Callable[[str], None],
) -> Dict[str, Any]:
    count = min(int(config.overfit_samples), len(dataset))
    subset = Subset(dataset, list(range(count)))
    loader = DataLoader(
        subset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=segmentation_collate,
    )
    eval_loader = DataLoader(
        subset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=segmentation_collate,
    )
    model = model_factory().to(device)
    assert_no_batch_norm(_trainable_module(model))
    criterion = SparseForegroundLoss(
        SegmentationLossConfig(
            name=config.loss_name,
            focal_alpha=config.focal_alpha,
            focal_gamma=config.focal_gamma,
        )
    )
    optimizer = AdamW(optimizer_parameter_groups(model, config.weight_decay), lr=config.lr)
    scaler = _amp_scaler(device)
    steps = 0
    best_dice = 0.0
    start = time.perf_counter()
    while steps < config.overfit_max_steps and best_dice < config.overfit_target_dice:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(loader):
            images = batch["images"].to(device)
            masks = batch["masks"].to(device)
            with autocast_ctx(device):
                logits = _forward_model(model, images)
                parts = criterion(logits, masks)
                loss = parts["total"] / float(config.accum_steps)
            _finite_or_raise(loss, "overfit loss", batch["image_paths"])
            scaler.scale(loss).backward()
            if (batch_index + 1) % config.accum_steps == 0 or batch_index + 1 == len(loader):
                scaler.unscale_(optimizer)
                _gradient_finite_or_raise(model, batch["image_paths"])
                clip_grad_norm_([p for p in model.parameters() if p.requires_grad], config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                steps += 1
                if hasattr(model, "assert_encoder_frozen"):
                    model.assert_encoder_frozen()
                if steps % 20 == 0 or steps >= config.overfit_max_steps:
                    metrics, _, _ = validate(
                        model,
                        eval_loader,
                        criterion,
                        device,
                        threshold=config.threshold,
                    )
                    best_dice = max(best_dice, float(metrics["dice"]))
                    log(f"[overfit] step={steps} train_dice={metrics['dice']:.4f} empty={metrics['empty_mask_rate']:.4f}")
                if steps >= config.overfit_max_steps or best_dice >= config.overfit_target_dice:
                    break
    result = {
        "samples": count,
        "optimizer_steps": steps,
        "best_train_dice": best_dice,
        "target_dice": config.overfit_target_dice,
        "passed": best_dice >= config.overfit_target_dice,
        "elapsed_sec": time.perf_counter() - start,
    }
    (out_dir / "overfit_check.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if not result["passed"]:
        raise RuntimeError(
            f"20-sample overfit check failed: Dice {best_dice:.4f} < {config.overfit_target_dice:.4f}"
        )
    return result


def threshold_search(predictions: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    thresholds = [round(value, 2) for value in np.arange(0.10, 0.901, 0.05)]
    rows: List[Dict[str, Any]] = []
    per_class_rows: Dict[str, List[Dict[str, Any]]] = {slug: [] for slug in PROMPT_SLUGS.values()}
    for threshold in thresholds:
        global_acc = MetricAccumulator(threshold)
        class_acc = {slug: MetricAccumulator(threshold) for slug in PROMPT_SLUGS.values()}
        for record in predictions:
            global_acc.add(record["probability"], record["target"], record["prompt"], record["path"])
            class_acc[record["prompt_slug"]].add(
                record["probability"], record["target"], record["prompt"], record["path"]
            )
        rows.append({"threshold": threshold, **global_acc.summary()})
        for slug, accumulator in class_acc.items():
            per_class_rows[slug].append({"threshold": threshold, **accumulator.summary()})

    def best_row(candidates: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        return dict(max(candidates, key=lambda row: (row["dice"], row["bbox_iou"], -row["empty_mask_rate"])))

    return {
        "thresholds": rows,
        "best_global": best_row(rows),
        "per_class": {
            slug: {"thresholds": class_rows, "best": best_row(class_rows)}
            for slug, class_rows in per_class_rows.items()
        },
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_experiment(
    config: SegmentationExperimentConfig,
    model_factory: Callable[[], nn.Module],
) -> Dict[str, Any]:
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = FlushLogger(out_dir / "log.txt")
    try:
        set_seed(config.seed)
        device = _device(config)
        fold_dir = Path(config.fold_dir)
        train_split = fold_dir / "train.txt"
        val_split = fold_dir / "val.txt"
        overlap = read_case_set(train_split) & read_case_set(val_split)
        if overlap:
            raise RuntimeError(f"Train/val case overlap: {sorted(overlap)}")

        train_transform = PairedSegTransform(
            config.preprocess_mode,
            train=True,
            augment=config.augment,
        )
        deterministic_train_transform = PairedSegTransform(
            config.preprocess_mode,
            train=False,
            augment=False,
        )
        val_transform = PairedSegTransform(config.preprocess_mode, train=False, augment=False)
        train_dataset = SegmentationDataset(
            Path(config.train_dataset_root), train_split, transform=train_transform
        )
        train_sanity_dataset = SegmentationDataset(
            Path(config.train_dataset_root), train_split, transform=deterministic_train_transform
        )
        val_dataset = SegmentationDataset(
            Path(config.val_dataset_root), val_split, transform=val_transform
        )
        train_counts, val_counts = class_counts(train_dataset), class_counts(val_dataset)
        vis_paths, vis_counts = _visualization_paths(val_dataset, config.vis_per_class)
        preflight = {
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "train_cases": sorted(read_case_set(train_split)),
            "val_cases": sorted(read_case_set(val_split)),
            "case_overlap": [],
            "train_class_counts": train_counts,
            "val_class_counts": val_counts,
            "visualization_counts": vis_counts,
            "visualization_shortage": {
                slug: max(config.vis_per_class - count, 0) for slug, count in vis_counts.items()
            },
        }
        (out_dir / "preflight.json").write_text(json.dumps(preflight, indent=2), encoding="utf-8")
        log(f"[data] train={len(train_dataset)} val={len(val_dataset)} overlap=0")
        log(f"[data] train_classes={train_counts}")
        log(f"[data] val_classes={val_counts} vis_counts={vis_counts}")

        run_geometry_check(
            val_dataset,
            out_dir / "geometry_check.json",
            num_samples=20,
            seed=config.seed,
        )
        save_preprocess_audit(val_dataset, out_dir / "preprocess_check", count=20)

        if config.overfit_check:
            run_overfit_check(
                model_factory,
                train_sanity_dataset,
                config,
                device,
                out_dir,
                log,
            )

        set_seed(config.seed)
        model = model_factory().to(device)
        assert_no_batch_norm(_trainable_module(model))
        total_params, trainable_params = parameter_counts(model)
        log(f"[model] total_params={total_params} trainable_params={trainable_params} device={device}")
        if hasattr(model, "assert_encoder_frozen"):
            model.assert_encoder_frozen()

        loader_generator = torch.Generator().manual_seed(config.seed)
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=device.type == "cuda",
            collate_fn=segmentation_collate,
            generator=loader_generator,
            persistent_workers=config.num_workers > 0,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=device.type == "cuda",
            collate_fn=segmentation_collate,
            persistent_workers=config.num_workers > 0,
        )
        criterion = SparseForegroundLoss(
            SegmentationLossConfig(
                name=config.loss_name,
                focal_alpha=config.focal_alpha,
                focal_gamma=config.focal_gamma,
            )
        )
        optimizer = AdamW(optimizer_parameter_groups(model, config.weight_decay), lr=config.lr)
        optimizer_steps_per_epoch = math.ceil(len(train_loader) / config.accum_steps)
        scheduler = build_scheduler(optimizer, optimizer_steps_per_epoch, config.epochs)
        scaler = _amp_scaler(device)
        start_epoch = 0
        global_step = 0
        best_metrics: Optional[Dict[str, Any]] = None
        patience_counter = 0
        resume_path = Path(config.resume_checkpoint) if config.resume_checkpoint else None
        if resume_path is not None and resume_path.exists():
            checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
            _load_model_state(model, checkpoint)
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            scaler.load_state_dict(checkpoint.get("scaler", {}))
            start_epoch = int(checkpoint["epoch"])
            global_step = int(checkpoint.get("global_step", 0))
            best_metrics = checkpoint.get("best_metrics")
            patience_counter = int(checkpoint.get("patience_counter", 0))
            log(f"[resume] path={resume_path} start_epoch={start_epoch + 1}")

        config_payload = {
            **asdict(config),
            "preprocess_version": PREPROCESS_VERSION,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "device": str(device),
            "torch_version": torch.__version__,
        }
        (out_dir / "config.json").write_text(json.dumps(config_payload, indent=2), encoding="utf-8")
        if config.fetalclip_weights:
            digest = sha256_file(Path(config.fetalclip_weights))
            (out_dir / "fetalclip_sha256.txt").write_text(digest + "\n", encoding="utf-8")

        metrics_path = out_dir / "metrics.jsonl"
        epoch_times: List[float] = []
        for epoch in range(start_epoch, config.epochs):
            epoch_start = time.perf_counter()
            train_metrics, optimizer_steps = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                scheduler,
                scaler,
                device,
                accum_steps=config.accum_steps,
                max_grad_norm=config.max_grad_norm,
            )
            global_step += optimizer_steps
            val_metrics, _, vis_records = validate(
                model,
                val_loader,
                criterion,
                device,
                threshold=config.threshold,
                visualization_paths=vis_paths,
            )
            epoch_time = time.perf_counter() - epoch_start
            epoch_times.append(epoch_time)
            row = {
                "epoch": epoch + 1,
                "epoch_time_sec": epoch_time,
                "global_optimizer_step": global_step,
                "train": train_metrics,
                "val": val_metrics,
            }
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            log(
                f"[epoch {epoch + 1}/{config.epochs}] "
                f"train_loss={train_metrics['loss']:.4f} "
                f"val_loss={val_metrics['loss']:.4f} dice={val_metrics['dice']:.4f} "
                f"bbox_iou={val_metrics['bbox_iou']:.4f} hit50={val_metrics['bbox_hit_0_5']:.4f} "
                f"empty={val_metrics['empty_mask_rate']:.4f} "
                f"prob_max={val_metrics['probability_max_mean']:.4f} "
                f"prob_mean={val_metrics['probability_mean']:.6f} time={epoch_time:.1f}s"
            )
            if config.vis_every > 0 and (epoch + 1) % config.vis_every == 0:
                save_validation_visualizations(
                    vis_records,
                    out_dir / f"val_vis_epoch{epoch + 1}",
                    threshold=config.threshold,
                )

            improved = _is_better(val_metrics, best_metrics)
            if improved:
                best_metrics = val_metrics
                patience_counter = 0
            else:
                patience_counter += 1
            payload = _checkpoint_payload(
                model,
                optimizer,
                scheduler,
                scaler,
                config,
                epoch=epoch + 1,
                global_step=global_step,
                best_metrics=best_metrics,
                patience_counter=patience_counter,
            )
            _save_checkpoint(out_dir / "resume_latest.pt", payload)
            if improved:
                _save_checkpoint(out_dir / "best.pt", payload)
                log(f"[best] epoch={epoch + 1} dice={val_metrics['dice']:.4f} bbox_iou={val_metrics['bbox_iou']:.4f}")
            if patience_counter >= config.patience:
                log(f"[early-stop] no improvement for {patience_counter} epochs")
                break

        best_checkpoint = torch.load(out_dir / "best.pt", map_location="cpu", weights_only=False)
        _load_model_state(model, best_checkpoint)
        fixed_metrics, best_predictions, best_vis = validate(
            model,
            val_loader,
            criterion,
            device,
            threshold=config.threshold,
            visualization_paths=vis_paths,
        )
        search = threshold_search(best_predictions)
        (out_dir / "threshold_search.json").write_text(json.dumps(search, indent=2), encoding="utf-8")
        calibrated_threshold = float(search["best_global"]["threshold"])
        save_validation_visualizations(
            best_vis,
            out_dir / "val_vis_best_threshold",
            threshold=calibrated_threshold,
        )
        summary = {
            "experiment": config.experiment_name,
            "model_type": config.model_type,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "mean_epoch_time_sec": float(np.mean(epoch_times)) if epoch_times else 0.0,
            "fixed_threshold": config.threshold,
            "fixed_metrics": fixed_metrics,
            "best_threshold": calibrated_threshold,
            "calibrated_metrics": search["best_global"],
            "per_class_best_threshold": {
                slug: details["best"] for slug, details in search["per_class"].items()
            },
            "visualization_counts": vis_counts,
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        log(
            f"[done] fixed_dice={fixed_metrics['dice']:.4f} "
            f"best_threshold={calibrated_threshold:.2f} calibrated_dice={search['best_global']['dice']:.4f}"
        )
        return summary
    except Exception as error:
        log(f"[fatal] {type(error).__name__}: {error}")
        raise
    finally:
        log.close()
