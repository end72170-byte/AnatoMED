"""Run phase-1 ExaminationDataset acceptance checks."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

from torch.utils.data import DataLoader

# Allow direct execution as ``python data_process/validate_anatomed_stage1.py``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.examination_dataset import ExaminationDataset, RouterManifestNotReady, examination_collate


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_oracle_router(observations_path: Path, split_file: Path, output_path: Path) -> None:
    case_ids = {
        line.strip().split("/")[0]
        for line in split_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    rows = [
        row for row in read_jsonl(observations_path)
        if row["source_dataset"] == "clip-dataset" and row["case_id"] in case_ids
    ]
    by_exam = defaultdict(list)
    for row in rows:
        by_exam[row["examination_id"]].append(row)
    output = []
    class_to_slot = {
        "EYES": "EYES",
        "UPPER_ALVEOLAR": "UPPER_ALVEOLAR",
        "LOWER_ALVEOLAR": "LOWER_ALVEOLAR",
        "INTACT_HARD_PALATE": "HARD_PALATE",
        "CLEFT_HARD_PALATE": "HARD_PALATE",
    }
    for examination_id, exam_rows in sorted(by_exam.items()):
        selected = {}
        predictions = {}
        for row in sorted(exam_rows, key=lambda value: value["observation_id"]):
            predictions[row["observation_id"]] = {
                "class_name": row["clip_class"],
                "slot_name": class_to_slot[row["clip_class"]],
                "prompt": row["clip_prompt"],
                "score": None,
            }
            selected.setdefault(row["anatomy_slot"], row["observation_id"])
        slots = {
            slot: {
                "available": slot in selected,
                "selected_observation_id": selected.get(slot),
                "predicted_class": None,
                "score": None,
            }
            for slot in ("EYES", "UPPER_ALVEOLAR", "LOWER_ALVEOLAR", "HARD_PALATE")
        }
        output.append({
            "examination_id": examination_id,
            "case_id": exam_rows[0]["case_id"],
            "source_dataset": "clip-dataset",
            "status": "oracle_test",
            "checkpoint": "oracle-from-ground-truth-for-interface-test-only",
            "slots": slots,
            "observation_predictions": predictions,
        })
    output_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest_dir", type=Path, default=Path("artifacts/anatomed/phase0"))
    parser.add_argument("--split_dir", type=Path, default=Path("splits/sam-dataset/group_kfold_k5_seed42_anatomed/fold_0"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/anatomed/phase1/stage1_validation.json"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_dir = args.manifest_dir.resolve()
    split_dir = args.split_dir.resolve()
    output_path = args.output.resolve()
    observations_path = manifest_dir / "observations.jsonl"
    train = ExaminationDataset(
        observations_path,
        split_dir / "train.txt",
        source_dataset="sam-dataset",
        condition_source="gt",
        seed=args.seed,
    )
    with tempfile.TemporaryDirectory(prefix="anatomed_oracle_") as temp_dir:
        oracle_path = Path(temp_dir) / "router_manifest_oracle.jsonl"
        build_oracle_router(observations_path, split_dir / "val.txt", oracle_path)
        validation = ExaminationDataset(
            observations_path,
            split_dir / "val.txt",
            source_dataset="clip-dataset",
            condition_source="router",
            router_manifest=oracle_path,
            seed=args.seed,
        )

        if len(train) == 0 or len(validation) == 0:
            raise AssertionError("Phase-1 Dataset must not be empty")
        train_item = train[0]
        val_item_a = validation[0]
        val_item_b = validation[0]
        if val_item_a["slot_image_paths"] != val_item_b["slot_image_paths"]:
            raise AssertionError("Validation slot paths are not deterministic")
        if int(val_item_a["slot_is_main"].sum()) != 1:
            raise AssertionError("Validation must have exactly one main slot")
        if val_item_a["condition_source"] != "router":
            raise AssertionError("Validation condition source is not router")
        for item in (train_item, val_item_a):
            main_index = int(item["slot_is_main"].nonzero(as_tuple=False)[0].item())
            if item["slot_image_paths"][main_index] != item["main_image_path"]:
                raise AssertionError("Main slot does not use main image")
            available_paths = [path for path in item["slot_image_paths"] if path is not None]
            if len(available_paths) != len(set(available_paths)) and len(available_paths) > 1:
                raise AssertionError("Unexpected duplicate auxiliary image path")

        loader = DataLoader(validation, batch_size=2, shuffle=False, collate_fn=examination_collate)
        batch = next(iter(loader))
        if batch["slot_available"].shape[1] != 4 or batch["slot_is_main"].shape[1] != 4:
            raise AssertionError("Collate did not preserve four slot shape")
        if tuple(batch["main_image"].shape[1:]) != (3, 224, 224):
            raise AssertionError(f"Paired image transform shape is wrong: {tuple(batch['main_image'].shape)}")
        if tuple(batch["main_mask"].shape[1:]) != (1, 224, 224):
            raise AssertionError(f"Paired mask transform shape is wrong: {tuple(batch['main_mask'].shape)}")
        if not bool(batch["mask_valid"].all()):
            raise AssertionError("A main sample has invalid mask")

        result = {
            "passed": True,
            "train_samples": len(train),
            "validation_samples": len(validation),
            "validation_condition_source": "router",
            "validation_manifest": "temporary oracle_test only; not a model result",
            "checks": [
                "train GT condition construction",
                "validation router condition construction",
                "main image bound to main slot",
                "fixed validation auxiliary paths",
                "four-slot collate shape",
                "paired image/mask transform before collate",
                "non-empty main masks",
            ],
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
