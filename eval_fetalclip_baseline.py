"""
Evaluate the original FetalCLIP weights with the same prompt-classification
metrics/format used in finetune_fetalclip_keyframes.py.

Example:
python eval_fetalclip_baseline.py \
  --val_dataset_root /data/clip-dataset \
  --fold_dir /data/splits/fold_0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import open_clip
import torch
from open_clip.factory import _MODEL_CONFIGS
from torch.utils.data import DataLoader

from finetune_fetalclip_keyframes import (
    DEFAULT_KEYFRAME_PROMPTS,
    ClipDatasetKeyframe,
    _collate_clip,
    validate_prompt_classification,
)
from utils.stage2_dataset import resolve_case_dataset_root


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate original FetalCLIP prompt classification")
    p.add_argument("--val_dataset_root", type=str, required=True)
    p.add_argument("--val_split", type=str, default="val")
    p.add_argument(
        "--fold_dir",
        type=str,
        required=True,
        help="Fold directory containing val.txt. When provided, it overrides --val_split.",
    )
    p.add_argument("--fetalclip_config", type=str, default="model/FetalCLIP_config.json")
    p.add_argument("--fetalclip_weights", type=str, default="model/FetalCLIP_weights.pt")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    use_amp = device.type == "cuda"

    with open(args.fetalclip_config, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    _MODEL_CONFIGS["FetalCLIP"] = cfg

    model, _, preprocess = open_clip.create_model_and_transforms(
        "FetalCLIP", pretrained=args.fetalclip_weights
    )
    model.to(device)
    model.eval()
    tokenizer = open_clip.get_tokenizer("FetalCLIP")

    fold_dir = Path(args.fold_dir) if args.fold_dir else None
    val_split_file = fold_dir / "val.txt" if fold_dir is not None else None
    val_dataset_root = Path(args.val_dataset_root)

    if val_split_file is not None:
        val_ds = ClipDatasetKeyframe(
            dataset_root=resolve_case_dataset_root(val_dataset_root),
            split_file=val_split_file,
        )
        print(f"[data] Val fold file '{val_split_file}': {len(val_ds)} samples")
    else:
        val_ds = ClipDatasetKeyframe(
            dataset_root=val_dataset_root,
            split=args.val_split,
        )
        print(f"[data] Val split '{args.val_split}': {len(val_ds)} samples")

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=_collate_clip,
    )

    val_metrics = validate_prompt_classification(
        model, preprocess, tokenizer, val_loader,
        device, DEFAULT_KEYFRAME_PROMPTS, use_amp=use_amp,
    )

    print(
        f"  [Val][5-prompt] acc={val_metrics['accuracy']:.4f}  "
        f"macro_f1={val_metrics['macro_f1']:.4f}  "
        f"macro_prec={val_metrics['macro_precision']:.4f}  "
        f"macro_rec={val_metrics['macro_recall']:.4f}  "
        f"(n={val_metrics['n_samples']}, skipped={val_metrics['n_skipped']})"
    )
    for pname, pmet in val_metrics["per_class"].items():
        print(
            f"    {pname}: "
            f"P={pmet['precision']:.3f} R={pmet['recall']:.3f} F1={pmet['f1']:.3f} "
            f"(TP={pmet['tp']} FP={pmet['fp']} FN={pmet['fn']})"
        )

    print(
        f"  [Val][4-anatomy] acc={val_metrics['anatomy_accuracy']:.4f}  "
        f"macro_f1={val_metrics['anatomy_macro_f1']:.4f}  "
        f"macro_prec={val_metrics['anatomy_macro_precision']:.4f}  "
        f"macro_rec={val_metrics['anatomy_macro_recall']:.4f}"
    )
    for aname, amet in val_metrics["per_anatomy"].items():
        print(
            f"    {aname}: "
            f"P={amet['precision']:.3f} R={amet['recall']:.3f} F1={amet['f1']:.3f} "
            f"(TP={amet['tp']} FP={amet['fp']} FN={amet['fn']})"
        )

    hp_routing = val_metrics["hard_palate_routing"]
    print(
        f"  [Val][hard-palate-routing] "
        f"recall={hp_routing['hard_palate_routing_recall']:.4f}  "
        f"routed={hp_routing['n_routed_hard_palate']}/"
        f"{hp_routing['n_gt_hard_palate']}  "
        f"missed={hp_routing['n_route_missed']}"
    )

    hp = val_metrics["hard_palate_binary"]
    print(
        f"  [Val][hard-palate-binary|routed] acc={hp['accuracy']:.4f}  "
        f"macro_f1={hp['macro_f1']:.4f}  "
        f"normal_f1={hp['normal']['f1']:.4f}  "
        f"abnormal_f1={hp['abnormal']['f1']:.4f}  "
        f"(n={hp['n_samples']})"
    )
    print(
        f"    normal: "
        f"P={hp['normal']['precision']:.3f} R={hp['normal']['recall']:.3f} F1={hp['normal']['f1']:.3f} "
        f"(TP={hp['normal']['tp']} FP={hp['normal']['fp']} FN={hp['normal']['fn']})"
    )
    print(
        f"    abnormal: "
        f"P={hp['abnormal']['precision']:.3f} R={hp['abnormal']['recall']:.3f} F1={hp['abnormal']['f1']:.3f} "
        f"(TP={hp['abnormal']['tp']} FP={hp['abnormal']['fp']} FN={hp['abnormal']['fn']})"
    )
    print(
        f"    confusion: TN={hp['confusion']['tn']} FP={hp['confusion']['fp']} "
        f"FN={hp['confusion']['fn']} TP={hp['confusion']['tp']}"
    )
    hp_e2e = val_metrics["hard_palate_end_to_end"]
    print(
        f"  [Val][hard-palate-end-to-end] "
        f"resolved_correct_rate={hp_e2e['resolved_correct_rate']:.4f}  "
        f"correct={hp_e2e['correct_state']}  "
        f"wrong={hp_e2e['wrong_state']}  "
        f"unresolved={hp_e2e['unresolved_due_to_route_miss']}  "
        f"(n={hp_e2e['n_gt_hard_palate']})"
    )


if __name__ == "__main__":
    main()
