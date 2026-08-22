"""Train a VLM-free GroupNorm UNETR decoder over frozen FetalCLIP features.

Example:
python train_seg_fetalclip_unetr.py \
  --train_dataset_root /data/sam-dataset \
  --val_dataset_root /data/clip-dataset \
  --fold_dir /data/splits/fold_0 \
  --fetalclip_weights /checkpoints/fetalclip.pt \
  --out_dir runs/segmentation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import open_clip
import torch
from open_clip.factory import _MODEL_CONFIGS

from models.fetalclip_unetr import FrozenFetalCLIPUNETR
from utils.models import FetalCLIPSegFeatureExtractor
from utils.segmentation_engine import SegmentationExperimentConfig, run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Focused segmentation with frozen FetalCLIP and a UNETR decoder")
    parser.add_argument("--train_dataset_root", required=True)
    parser.add_argument("--val_dataset_root", required=True)
    parser.add_argument("--fold_dir", required=True)
    parser.add_argument("--out_dir", default="runs/segmentation")
    parser.add_argument("--fetalclip_config", default="model/FetalCLIP_config.json")
    parser.add_argument(
        "--fetalclip_weights",
        required=True,
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--accum_steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--overfit_check", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overfit_samples", type=int, default=20)
    parser.add_argument("--overfit_max_steps", type=int, default=400)
    parser.add_argument("--overfit_target_dice", type=float, default=0.9)
    parser.add_argument("--vis_per_class", type=int, default=5)
    parser.add_argument("--vis_every", type=int, default=1)
    parser.add_argument("--loss_name", choices=["dice_focal", "tversky_focal"], default="dice_focal")
    parser.add_argument("--resume_checkpoint", default="")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def _extract_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("model"), dict):
        state = checkpoint["model"]
    elif isinstance(checkpoint, dict) and isinstance(checkpoint.get("state_dict"), dict):
        state = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict):
        state = checkpoint
    else:
        raise TypeError("Unsupported FetalCLIP checkpoint format")
    cleaned: Dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if not isinstance(value, torch.Tensor):
            continue
        clean_key = key
        for prefix in ("module.", "model."):
            if clean_key.startswith(prefix):
                clean_key = clean_key[len(prefix) :]
        cleaned[clean_key] = value
    return cleaned


def make_model_factory(config_path: Path, weights_path: Path):
    def factory() -> FrozenFetalCLIPUNETR:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        _MODEL_CONFIGS["FetalCLIP"] = config
        fetalclip, _, _ = open_clip.create_model_and_transforms("FetalCLIP", pretrained=None)
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
        incompatible = fetalclip.load_state_dict(_extract_state_dict(checkpoint), strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "FetalCLIP checkpoint does not exactly match the configured model: "
                f"missing={incompatible.missing_keys[:10]} unexpected={incompatible.unexpected_keys[:10]}"
            )
        fetalclip.eval()
        for parameter in fetalclip.parameters():
            parameter.requires_grad = False
        extractor = FetalCLIPSegFeatureExtractor(fetalclip.visual)
        return FrozenFetalCLIPUNETR(extractor, feature_width=extractor.width)

    return factory


def main() -> None:
    args = parse_args()
    weights_path = Path(args.fetalclip_weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"FetalCLIP checkpoint not found: {weights_path}")
    config = SegmentationExperimentConfig(
        experiment_name="frozen_fetalclip_unetr",
        model_type="fetalclip_unetr",
        train_dataset_root=args.train_dataset_root,
        val_dataset_root=args.val_dataset_root,
        fold_dir=args.fold_dir,
        out_dir=args.out_dir,
        preprocess_mode="clip_center_crop",
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        accum_steps=args.accum_steps,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        patience=args.patience,
        seed=args.seed,
        threshold=args.threshold,
        max_grad_norm=args.max_grad_norm,
        augment=False,
        cpu=args.cpu,
        resume_checkpoint=args.resume_checkpoint,
        overfit_check=args.overfit_check,
        overfit_samples=args.overfit_samples,
        overfit_max_steps=args.overfit_max_steps,
        overfit_target_dice=args.overfit_target_dice,
        vis_per_class=args.vis_per_class,
        vis_every=args.vis_every,
        loss_name=args.loss_name,
        fetalclip_weights=str(weights_path),
    )
    factory = make_model_factory(Path(args.fetalclip_config), weights_path)
    run_experiment(config, model_factory=factory)


if __name__ == "__main__":
    main()
