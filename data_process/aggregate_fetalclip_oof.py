"""Aggregate fold-specific FetalCLIP router metrics into pooled OOF results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# Allow direct execution as ``python data_process/aggregate_fetalclip_oof.py``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from finetune_fetalclip_keyframes import (
    DEFAULT_KEYFRAME_PROMPTS,
    PROMPT_TO_ANATOMY,
    _compute_hard_palate_metrics,
    _compute_multiclass_metrics,
)


ANATOMY_NAMES = [
    "fetal eyes",
    "upper alveolar ridge",
    "hard palate",
    "lower alveolar ridge",
]


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def compute_metrics(pred_indices: List[int], gt_indices: List[int]) -> Dict[str, Any]:
    five = _compute_multiclass_metrics(pred_indices, gt_indices, DEFAULT_KEYFRAME_PROMPTS)
    anatomy_to_idx = {name: index for index, name in enumerate(ANATOMY_NAMES)}
    anatomy_preds = [
        anatomy_to_idx[PROMPT_TO_ANATOMY[DEFAULT_KEYFRAME_PROMPTS[index]]]
        for index in pred_indices
    ]
    anatomy_gts = [
        anatomy_to_idx[PROMPT_TO_ANATOMY[DEFAULT_KEYFRAME_PROMPTS[index]]]
        for index in gt_indices
    ]
    anatomy = _compute_multiclass_metrics(anatomy_preds, anatomy_gts, ANATOMY_NAMES)
    hard_palate = _compute_hard_palate_metrics(
        pred_indices,
        gt_indices,
        DEFAULT_KEYFRAME_PROMPTS,
    )
    return {
        "five_class": five,
        "four_anatomy": anatomy,
        "hard_palate_routing": hard_palate["routing"],
        "hard_palate_binary": hard_palate["conditional_binary"],
        "hard_palate_end_to_end": hard_palate["end_to_end"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics_root",
        type=Path,
        default=Path("artifacts/anatopave/phase2"),
    )
    parser.add_argument(
        "--observations",
        type=Path,
        default=Path("artifacts/anatopave/phase0/observations.jsonl"),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    observations = {
        row["observation_id"]: row
        for row in read_jsonl(args.observations)
        if row.get("source_dataset") == "clip-dataset"
    }
    all_preds: List[int] = []
    all_gts: List[int] = []
    fold_summaries: List[Dict[str, Any]] = []

    for fold_index in range(5):
        fold_dir = args.metrics_root / f"fold_{fold_index}"
        metrics_path = fold_dir / "router_metrics.json"
        manifest_path = fold_dir / "router_manifest.jsonl"
        fold_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        manifest_rows = read_jsonl(manifest_path)
        pred_indices: List[int] = []
        gt_indices: List[int] = []
        prompt_to_index = {
            prompt: index for index, prompt in enumerate(DEFAULT_KEYFRAME_PROMPTS)
        }
        for exam in manifest_rows:
            for observation_id, prediction in exam["observation_predictions"].items():
                gt_prompt = observations[observation_id]["clip_prompt"]
                pred_prompt = prediction["prompt"]
                gt_indices.append(prompt_to_index[gt_prompt])
                pred_indices.append(prompt_to_index[pred_prompt])

        all_preds.extend(pred_indices)
        all_gts.extend(gt_indices)
        fold_summaries.append({
            "fold": fold_index,
            "checkpoint": fold_metrics["checkpoint"],
            "checkpoint_epoch": fold_metrics.get("checkpoint_epoch"),
            "n_observations": len(pred_indices),
            "n_examinations": fold_metrics.get("n_examinations"),
            "n_cases": fold_metrics.get("n_cases"),
            "five_class": fold_metrics["five_class"],
            "four_anatomy": fold_metrics["four_anatomy"],
            "hard_palate_routing": fold_metrics["hard_palate_routing"],
            "hard_palate_binary": fold_metrics["hard_palate_binary"],
            "hard_palate_end_to_end": fold_metrics["hard_palate_end_to_end"],
        })

    pooled = compute_metrics(all_preds, all_gts)
    result = {
        "aggregation": "pooled_observation_level_oof",
        "n_folds": 5,
        "n_observations": len(all_preds),
        "folds": fold_summaries,
        **pooled,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "n_folds": result["n_folds"],
        "n_observations": result["n_observations"],
        "five_class_accuracy": pooled["five_class"]["accuracy"],
        "five_class_macro_f1": pooled["five_class"]["macro_f1"],
        "four_anatomy_macro_f1": pooled["four_anatomy"]["macro_f1"],
        "hard_palate_routing_recall": pooled["hard_palate_routing"]["hard_palate_routing_recall"],
        "hard_palate_conditional_macro_f1": pooled["hard_palate_binary"]["macro_f1"],
        "hard_palate_end_to_end_correct_rate": pooled["hard_palate_end_to_end"]["resolved_correct_rate"],
        "output": str(args.output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
