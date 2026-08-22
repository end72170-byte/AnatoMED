"""Export fold-specific FetalCLIP predictions as an ANATOPAVE router manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

import open_clip
import torch
import torch.nn.functional as F
from open_clip.factory import _MODEL_CONFIGS
from PIL import Image

# Allow direct execution as ``python data_process/build_fetalclip_router_manifest.py``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from finetune_fetalclip_keyframes import (  # noqa: E402
    DEFAULT_KEYFRAME_PROMPTS,
    HARD_PALATE_CLEFT_PROMPT,
    HARD_PALATE_NORMAL_PROMPT,
    PROMPT_TO_ANATOMY,
    _compute_hard_palate_metrics,
    _compute_multiclass_metrics,
)
from utils.examination_dataset import FIVE_CLASS_TO_SLOT, SLOT_NAMES  # noqa: E402


PROMPT_TO_CLASS = {
    "fetal ultrasound, fetal eyes.": "EYES",
    "fetal ultrasound, upper alveolar ridge.": "UPPER_ALVEOLAR",
    "fetal ultrasound, lower alveolar ridge.": "LOWER_ALVEOLAR",
    HARD_PALATE_NORMAL_PROMPT: "INTACT_HARD_PALATE",
    HARD_PALATE_CLEFT_PROMPT: "CLEFT_HARD_PALATE",
}
CLASS_TO_PROMPT = {value: key for key, value in PROMPT_TO_CLASS.items()}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temp_path.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_case_ids(split_file: Path) -> set[str]:
    return {
        line.strip().split("/")[0]
        for line in split_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def compute_metrics(pred_indices: List[int], gt_indices: List[int]) -> Dict[str, Any]:
    five = _compute_multiclass_metrics(pred_indices, gt_indices, DEFAULT_KEYFRAME_PROMPTS)
    anatomy_names = ["fetal eyes", "upper alveolar ridge", "hard palate", "lower alveolar ridge"]
    anatomy_to_idx = {name: index for index, name in enumerate(anatomy_names)}
    anatomy_preds = [
        anatomy_to_idx[PROMPT_TO_ANATOMY[DEFAULT_KEYFRAME_PROMPTS[index]]]
        for index in pred_indices
    ]
    anatomy_gts = [
        anatomy_to_idx[PROMPT_TO_ANATOMY[DEFAULT_KEYFRAME_PROMPTS[index]]]
        for index in gt_indices
    ]
    anatomy = _compute_multiclass_metrics(anatomy_preds, anatomy_gts, anatomy_names)
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
        "--observations",
        type=Path,
        default=Path("artifacts/anatopave/phase0/observations.jsonl"),
    )
    parser.add_argument("--fold_dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fetalclip_config", type=Path, default=Path("model/FetalCLIP_config.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics_output", type=Path, default=None)
    parser.add_argument("--source_dataset", default="clip-dataset")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    observations_path = args.observations.resolve()
    fold_dir = args.fold_dir.resolve()
    checkpoint_path = args.checkpoint.resolve()
    output_path = args.output.resolve()
    metrics_path = (
        args.metrics_output.resolve()
        if args.metrics_output is not None
        else output_path.with_name("router_metrics.json")
    )
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    case_ids = read_case_ids(fold_dir / "val.txt")
    observations = [
        row
        for row in read_jsonl(observations_path)
        if row.get("source_dataset") == args.source_dataset and row.get("case_id") in case_ids
    ]
    if not observations:
        raise RuntimeError(f"No {args.source_dataset} validation observations found for {fold_dir}")

    config = json.loads(args.fetalclip_config.read_text(encoding="utf-8"))
    _MODEL_CONFIGS["FetalCLIP"] = config
    model, _, preprocess = open_clip.create_model_and_transforms("FetalCLIP", pretrained=None)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint, strict=True)
    model.eval().to(device)
    tokenizer = open_clip.get_tokenizer("FetalCLIP")

    prompt_tokens = tokenizer(DEFAULT_KEYFRAME_PROMPTS).to(device)
    prompt_features = F.normalize(model.encode_text(prompt_tokens), dim=-1)
    logit_scale = float(model.logit_scale.exp().detach().cpu())
    predictions: Dict[str, Dict[str, Any]] = {}
    pred_indices: List[int] = []
    gt_indices: List[int] = []
    prompt_to_index = {prompt: index for index, prompt in enumerate(DEFAULT_KEYFRAME_PROMPTS)}

    for start in range(0, len(observations), args.batch_size):
        rows = observations[start:start + args.batch_size]
        images = []
        for row in rows:
            image_path = Path(row["image_path"])
            if not image_path.is_absolute():
                image_path = project_root / image_path
            images.append(preprocess(Image.open(image_path).convert("RGB")))
        image_batch = torch.stack(images).to(device)
        image_features = F.normalize(model.encode_image(image_batch), dim=-1)
        cosine_scores = image_features @ prompt_features.t()
        probabilities = (cosine_scores * logit_scale).softmax(dim=-1)
        batch_preds = cosine_scores.argmax(dim=-1).cpu().tolist()
        cosine_rows = cosine_scores.float().cpu().tolist()
        probability_rows = probabilities.float().cpu().tolist()

        for row, pred_index, score_values, probability_values in zip(
            rows,
            batch_preds,
            cosine_rows,
            probability_rows,
        ):
            gt_index = prompt_to_index[row["clip_prompt"]]
            pred_prompt = DEFAULT_KEYFRAME_PROMPTS[pred_index]
            pred_class = PROMPT_TO_CLASS[pred_prompt]
            class_scores = {
                PROMPT_TO_CLASS[prompt]: float(score_values[index])
                for index, prompt in enumerate(DEFAULT_KEYFRAME_PROMPTS)
            }
            class_probabilities = {
                PROMPT_TO_CLASS[prompt]: float(probability_values[index])
                for index, prompt in enumerate(DEFAULT_KEYFRAME_PROMPTS)
            }
            predictions[row["observation_id"]] = {
                "class_name": pred_class,
                "slot_name": FIVE_CLASS_TO_SLOT[pred_class],
                "prompt": pred_prompt,
                "score": float(score_values[pred_index]),
                "probability": float(probability_values[pred_index]),
                "class_scores": class_scores,
                "class_probabilities": class_probabilities,
                "hard_palate_scores": {
                    "INTACT_HARD_PALATE": class_scores["INTACT_HARD_PALATE"],
                    "CLEFT_HARD_PALATE": class_scores["CLEFT_HARD_PALATE"],
                },
            }
            pred_indices.append(pred_index)
            gt_indices.append(gt_index)

    by_exam: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in observations:
        by_exam[row["examination_id"]].append(row)

    checkpoint_hash = sha256_file(checkpoint_path)
    manifest_rows: List[Dict[str, Any]] = []
    for examination_id, exam_observations in sorted(by_exam.items()):
        exam_predictions = {
            row["observation_id"]: predictions[row["observation_id"]]
            for row in sorted(exam_observations, key=lambda value: value["observation_id"])
        }
        slots: Dict[str, Dict[str, Any]] = {}
        for slot_name in SLOT_NAMES:
            candidates = [
                row
                for row in exam_observations
                if predictions[row["observation_id"]]["slot_name"] == slot_name
            ]
            if not candidates:
                slots[slot_name] = {
                    "available": False,
                    "selected_observation_id": None,
                    "predicted_class": None,
                    "score": None,
                }
                continue
            selected = max(
                candidates,
                key=lambda row: predictions[row["observation_id"]]["score"],
            )
            selected_prediction = predictions[selected["observation_id"]]
            slots[slot_name] = {
                "available": True,
                "selected_observation_id": selected["observation_id"],
                "predicted_class": selected_prediction["class_name"],
                "score": selected_prediction["score"],
                "probability": selected_prediction["probability"],
            }
            if slot_name == "HARD_PALATE":
                slots[slot_name]["hard_palate_scores"] = selected_prediction["hard_palate_scores"]

        manifest_rows.append({
            "case_id": exam_observations[0]["case_id"],
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_hash,
            "examination_id": examination_id,
            "manifest_type": "static_keyframe_router",
            "observation_predictions": exam_predictions,
            "slots": slots,
            "source_dataset": args.source_dataset,
            "status": "ready",
        })

    write_jsonl(output_path, manifest_rows)
    metrics = compute_metrics(pred_indices, gt_indices)
    metrics.update({
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
        "checkpoint_sha256": checkpoint_hash,
        "fold_dir": str(fold_dir),
        "source_dataset": args.source_dataset,
        "n_cases": len(case_ids),
        "n_examinations": len(by_exam),
        "n_observations": len(observations),
        "predicted_class_counts": dict(sorted(Counter(
            prediction["class_name"] for prediction in predictions.values()
        ).items())),
        "slot_available_counts": {
            slot: sum(bool(row["slots"][slot]["available"]) for row in manifest_rows)
            for slot in SLOT_NAMES
        },
    })
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[done] observations={len(observations)} examinations={len(by_exam)} cases={len(case_ids)}")
    print(f"[done] router manifest: {output_path}")
    print(f"[done] router metrics: {metrics_path}")


if __name__ == "__main__":
    main()
