"""Build ANATOPAVE phase-0 manifests and anonymized group-k-fold splits.

The source datasets intentionally have different meanings:

* ``sam-dataset`` is the training image source and may contain synthetic
  abnormal examinations for otherwise normal cases.
* ``clip-dataset`` is the validation source and contains the case-level
  diagnosis used for the 83-case/11-positive audit.

The script keeps those notions separate in every observation record.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".avi", ".mp4", ".mov", ".mkv"}
SLOT_NAMES = ("EYES", "UPPER_ALVEOLAR", "LOWER_ALVEOLAR", "HARD_PALATE")
FIVE_CLASS_NAMES = (
    "EYES",
    "UPPER_ALVEOLAR",
    "LOWER_ALVEOLAR",
    "INTACT_HARD_PALATE",
    "CLEFT_HARD_PALATE",
)
PROMPT_TO_CLASS = {
    "fetal ultrasound, fetal eyes.": "EYES",
    "fetal ultrasound, upper alveolar ridge.": "UPPER_ALVEOLAR",
    "fetal ultrasound, lower alveolar ridge.": "LOWER_ALVEOLAR",
    "fetal ultrasound, intact hard palate.": "INTACT_HARD_PALATE",
    "fetal ultrasound, cleft hard palate.": "CLEFT_HARD_PALATE",
}
ANATOMY_TO_SLOT = {
    "fetal ultrasound, fetal eyes": "EYES",
    "fetal ultrasound, upper alveolar ridge": "UPPER_ALVEOLAR",
    "fetal ultrasound, lower alveolar ridge": "LOWER_ALVEOLAR",
    "fetal ultrasound, hard palate": "HARD_PALATE",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_root", type=Path, default=Path("dataset"))
    parser.add_argument(
        "--manifest_dir",
        type=Path,
        default=Path("artifacts/anatopave/phase0"),
    )
    parser.add_argument(
        "--split_root",
        type=Path,
        default=Path("splits/sam-dataset/group_kfold_k5_seed42_anatopave"),
    )
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def as_posix_relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def read_text_lines(path: Path) -> List[str]:
    text = path.read_text(encoding="utf-8-sig")
    return [line.strip() for line in text.splitlines() if line.strip()]


def find_mask(masks_dir: Path, stem: str) -> Optional[Path]:
    for suffix in (".png", ".jpg", ".jpeg", ".bmp", ".webp"):
        candidate = masks_dir / f"{stem}_mask{suffix}"
        if candidate.is_file():
            return candidate
    return None


def stable_observation_id(source_dataset: str, relative_path: str) -> str:
    digest = hashlib.sha1(f"{source_dataset}:{relative_path}".encode("utf-8")).hexdigest()[:16]
    return f"obs_{digest}"


def acquisition_id(condition_dir: Path, seniority_dir: Path) -> str:
    videos = sorted(
        p for p in seniority_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".avi", ".mp4", ".mov", ".mkv"}
    )
    if len(videos) == 1:
        return videos[0].stem
    if len(videos) > 1:
        return "video_" + "_".join(p.stem for p in videos)
    return "acq0"


def build_case_labels(clip_root: Path) -> Dict[str, int]:
    labels: Dict[str, int] = {}
    for case_dir in sorted(p for p in clip_root.iterdir() if p.is_dir()):
        labels[case_dir.name] = int((case_dir / "abnormal").is_dir())
    return labels


def scan_inference_videos(
    videos_root: Path,
    case_labels: Dict[str, int],
    project_root: Path,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Build the explicitly curated high-seniority video inference cohort."""
    rows: List[Dict[str, Any]] = []
    excluded_non_high = 0
    invalid_entries: List[str] = []
    if not videos_root.is_dir():
        raise FileNotFoundError(f"Expected curated video root: {videos_root}")

    for video_path in sorted(path for path in videos_root.rglob("*") if path.is_file()):
        if video_path.suffix.lower() not in VIDEO_EXTS:
            continue
        relative = video_path.relative_to(videos_root)
        if len(relative.parts) != 4:
            invalid_entries.append(relative.as_posix())
            continue
        case_id, condition, seniority, _ = relative.parts
        if seniority != "high":
            excluded_non_high += 1
            continue
        if case_id not in case_labels or condition not in {"normal", "abnormal"}:
            invalid_entries.append(relative.as_posix())
            continue
        project_relative = as_posix_relative(video_path, project_root)
        rows.append(
            {
                "video_id": stable_observation_id("videos", project_relative).replace("obs_", "vid_"),
                "case_id": case_id,
                "condition": condition,
                "condition_label": int(condition == "abnormal"),
                "diagnosis_label": int(case_labels[case_id]),
                "seniority": seniority,
                "acquisition_id": video_path.stem,
                "examination_id": f"{case_id}__{condition}__{seniority}__{video_path.stem}",
                "video_path": project_relative,
                "inference_candidate": True,
                "hard_palate_event_required_for_video_inclusion": False,
                "hard_palate_event_required_for_diagnosis": True,
                "no_hard_palate_event_policy": "insufficient_evidence",
            }
        )

    video_cases = {row["case_id"] for row in rows}
    audit = {
        "video_root": str(videos_root.resolve()),
        "candidate_video_count": len(rows),
        "candidate_case_count": len(video_cases),
        "candidate_positive_cases": sum(case_labels[case_id] for case_id in video_cases),
        "candidate_normal_cases": sum(1 - case_labels[case_id] for case_id in video_cases),
        "cohort_case_count": len(case_labels),
        "candidate_count_less_than_case_cohort": len(video_cases) < len(case_labels),
        "missing_video_case_ids": sorted(set(case_labels) - video_cases),
        "excluded_non_high_video_count": excluded_non_high,
        "invalid_entries": invalid_entries,
        "video_output_count": len(rows),
        "availability_metric_denominator": len(rows),
        "conditional_diagnosis_denominator": None,
        "conditional_diagnosis_denominator_note": (
            "Computed after FetalCLIP/FSM inference as the number of videos with "
            "an accepted intact/cleft hard-palate event."
        ),
        "all_candidates_retained_in_outputs": True,
        "no_hard_palate_event_policy": "insufficient_evidence",
        "diagnosis_labels": ["normal", "abnormal", "insufficient_evidence"],
        "palate_cleft_ratio_when_unavailable": None,
        "event_availability_reported_separately": True,
    }
    return rows, audit


def scan_source(
    source_name: str,
    source_root: Path,
    case_labels: Dict[str, int],
    project_root: Path,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, Any]]:
    observations: List[Dict[str, Any]] = []
    examinations: Dict[str, Dict[str, Any]] = {}
    audit: Dict[str, Any] = {
        "source_dataset": source_name,
        "case_count": 0,
        "examination_count": 0,
        "valid_observation_count": 0,
        "invalid_image_count": 0,
        "missing_mask_count": 0,
        "missing_text_count": 0,
        "empty_mask_count": 0,
        "prompt_class_counts": Counter(),
        "slot_counts": Counter(),
        "condition_counts": Counter(),
        "seniority_counts": Counter(),
        "missing_anatomy_by_examination": Counter(),
        "invalid_examples": [],
    }

    for case_dir in sorted(p for p in source_root.iterdir() if p.is_dir()):
        audit["case_count"] += 1
        case_id = case_dir.name
        for condition_dir in sorted(p for p in case_dir.iterdir() if p.is_dir()):
            condition = condition_dir.name
            if condition not in {"normal", "abnormal"}:
                continue
            for seniority_dir in sorted(p for p in condition_dir.iterdir() if p.is_dir()):
                seniority = seniority_dir.name
                if seniority not in {"high", "low"}:
                    continue
                acq = acquisition_id(condition_dir, seniority_dir)
                examination_id = f"{case_id}__{condition}__{seniority}__{acq}"
                exam = examinations.setdefault(
                    examination_id,
                    {
                        "examination_id": examination_id,
                        "case_id": case_id,
                        "condition": condition,
                        "seniority": seniority,
                        "acquisition_id": acq,
                        "video_path": None,
                        "source_datasets": [],
                        "source_roots": {},
                        "observation_count_by_source": {},
                        "available_slots_by_source": {},
                    },
                )
                source_root_rel = as_posix_relative(seniority_dir, project_root)
                if source_name not in exam["source_datasets"]:
                    exam["source_datasets"].append(source_name)
                exam["source_roots"][source_name] = source_root_rel
                video_candidates = sorted(
                    p for p in seniority_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in {".avi", ".mp4", ".mov", ".mkv"}
                )
                if video_candidates:
                    exam["video_path"] = as_posix_relative(video_candidates[0], project_root)

                source_exam_obs: List[Dict[str, Any]] = []
                for anatomy_dir in sorted(p for p in seniority_dir.iterdir() if p.is_dir()):
                    slot = ANATOMY_TO_SLOT.get(anatomy_dir.name)
                    if slot is None:
                        continue
                    imgs_dir = anatomy_dir / "imgs"
                    masks_dir = anatomy_dir / "masks"
                    if not imgs_dir.is_dir() or not masks_dir.is_dir():
                        continue
                    txt_files = sorted(p for p in anatomy_dir.glob("*.txt") if p.is_file())
                    if not txt_files:
                        audit["missing_text_count"] += sum(
                            1 for p in imgs_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
                        )
                        continue
                    txt_path = txt_files[0]
                    lines = read_text_lines(txt_path)
                    if not lines or lines[0] not in PROMPT_TO_CLASS:
                        audit["missing_text_count"] += 1
                        if len(audit["invalid_examples"]) < 20:
                            audit["invalid_examples"].append({"reason": "invalid_prompt", "path": as_posix_relative(txt_path, project_root)})
                        continue
                    clip_class = PROMPT_TO_CLASS[lines[0]]
                    doctor_text = " ".join(lines[1:3]) if len(lines) >= 2 else lines[0]
                    if not doctor_text:
                        audit["missing_text_count"] += 1
                        continue
                    for image_path in sorted(p for p in imgs_dir.iterdir() if p.is_file()):
                        if image_path.suffix.lower() not in IMAGE_EXTS:
                            continue
                        mask_path = find_mask(masks_dir, image_path.stem)
                        if mask_path is None:
                            audit["missing_mask_count"] += 1
                            continue
                        rel_image = as_posix_relative(image_path, project_root)
                        observation_id = stable_observation_id(source_name, rel_image)
                        observation = {
                            "observation_id": observation_id,
                            "source_dataset": source_name,
                            "case_id": case_id,
                            "examination_id": examination_id,
                            "condition": condition,
                            "seniority": seniority,
                            "acquisition_id": acq,
                            "anatomy": anatomy_dir.name,
                            "anatomy_slot": slot,
                            "clip_class": clip_class,
                            "clip_prompt": lines[0],
                            "main_text": doctor_text,
                            "image_path": rel_image,
                            "mask_path": as_posix_relative(mask_path, project_root),
                            "text_path": as_posix_relative(txt_path, project_root),
                            "case_diagnosis_label": int(case_labels.get(case_id, condition == "abnormal")),
                            "condition_label": int(condition == "abnormal"),
                            "synthetic_condition": bool(source_name == "sam-dataset" and condition == "abnormal" and case_labels.get(case_id, 0) == 0),
                        }
                        observations.append(observation)
                        source_exam_obs.append(observation)
                        audit["valid_observation_count"] += 1
                        audit["prompt_class_counts"][clip_class] += 1
                        audit["slot_counts"][slot] += 1
                        audit["condition_counts"][condition] += 1
                        audit["seniority_counts"][seniority] += 1

                slots = sorted({row["anatomy_slot"] for row in source_exam_obs})
                exam["observation_count_by_source"][source_name] = len(source_exam_obs)
                exam["available_slots_by_source"][source_name] = slots

    audit["examination_count"] = len(examinations)
    audit["prompt_class_counts"] = dict(audit["prompt_class_counts"])
    audit["slot_counts"] = dict(audit["slot_counts"])
    audit["condition_counts"] = dict(audit["condition_counts"])
    audit["seniority_counts"] = dict(audit["seniority_counts"])
    audit["missing_anatomy_by_examination"] = dict(audit["missing_anatomy_by_examination"])
    return observations, examinations, audit


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_lines(path: Path, values: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{value}\n" for value in values), encoding="utf-8")


def make_stratified_folds(case_labels: Dict[str, int], k: int, seed: int) -> List[List[str]]:
    if k < 2:
        raise ValueError("k must be at least 2")
    if len(case_labels) < k:
        raise ValueError("k cannot exceed the number of cases")
    folds: List[List[str]] = [[] for _ in range(k)]
    rng = random.Random(seed)
    for label in (1, 0):
        cases = [case for case, value in case_labels.items() if value == label]
        cases.sort()
        rng.shuffle(cases)
        for index, case in enumerate(cases):
            folds[index % k].append(case)
    for fold in folds:
        fold.sort(key=lambda value: (len(value), value))
    return folds


def write_splits(split_root: Path, case_labels: Dict[str, int], k: int, seed: int, dataset_root: Path) -> Dict[str, Any]:
    folds = make_stratified_folds(case_labels, k, seed)
    all_cases = sorted(case_labels)
    write_lines(split_root / "all_cases.txt", all_cases)
    fold_meta = []
    for index, val_cases in enumerate(folds):
        val_set = set(val_cases)
        train_cases = [case for case in all_cases if case not in val_set]
        fold_dir = split_root / f"fold_{index}"
        write_lines(fold_dir / "train.txt", train_cases)
        write_lines(fold_dir / "val.txt", val_cases)
        fold_meta.append(
            {
                "fold": index,
                "train_cases": len(train_cases),
                "val_cases": len(val_cases),
                "val_positive_cases": sum(case_labels[case] for case in val_cases),
                "val_normal_cases": sum(1 - case_labels[case] for case in val_cases),
            }
        )
    meta = {
        "method": "stratified_group_kfold",
        "group_unit": "case_id",
        "k": k,
        "seed": seed,
        "dataset_name": "sam-dataset",
        "training_dataset_root": str(dataset_root / "sam-dataset" / "all"),
        "validation_dataset_root": str(dataset_root / "clip-dataset" / "all"),
        "num_cases": len(all_cases),
        "positive_cases": sum(case_labels.values()),
        "normal_cases": len(all_cases) - sum(case_labels.values()),
        "fold_sizes": fold_meta,
        "only_train_val_files": True,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (split_root / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return meta


def main() -> None:
    args = parse_args()
    project_root = Path.cwd().resolve()
    dataset_root = args.dataset_root.resolve()
    sam_root = dataset_root / "sam-dataset" / "all"
    clip_root = dataset_root / "clip-dataset" / "all"
    videos_root = dataset_root / "videos"
    if not sam_root.is_dir() or not clip_root.is_dir() or not videos_root.is_dir():
        raise FileNotFoundError(f"Expected dataset roots: {sam_root} and {clip_root}")

    case_labels = build_case_labels(clip_root)
    video_rows, video_audit = scan_inference_videos(videos_root, case_labels, project_root)
    video_case_ids = {row["case_id"] for row in video_rows}
    sam_obs, sam_exams, sam_audit = scan_source("sam-dataset", sam_root, case_labels, project_root)
    clip_obs, clip_exams, clip_audit = scan_source("clip-dataset", clip_root, case_labels, project_root)
    observations = sorted(sam_obs + clip_obs, key=lambda row: row["observation_id"])

    exams: Dict[str, Dict[str, Any]] = {}
    for row in list(sam_exams.values()) + list(clip_exams.values()):
        current = exams.setdefault(row["examination_id"], row)
        for source in row.get("source_datasets", []):
            if source not in current["source_datasets"]:
                current["source_datasets"].append(source)
        current["source_roots"].update(row.get("source_roots", {}))
        current["observation_count_by_source"].update(row.get("observation_count_by_source", {}))
        current["available_slots_by_source"].update(row.get("available_slots_by_source", {}))
        if row.get("video_path"):
            current["video_path"] = row["video_path"]
    examinations = sorted(exams.values(), key=lambda row: row["examination_id"])

    manifest_dir = args.manifest_dir.resolve()
    manifest_dir.mkdir(parents=True, exist_ok=True)
    case_rows = []
    for case_id in sorted(case_labels):
        case_rows.append(
            {
                "case_id": case_id,
                "diagnosis_label": case_labels[case_id],
                "diagnosis": "abnormal" if case_labels[case_id] else "normal",
                "has_inference_video": case_id in video_case_ids,
                "sam_has_synthetic_abnormal": any(
                    row["case_id"] == case_id and row["synthetic_condition"] for row in sam_obs
                ),
            }
        )
    write_jsonl(manifest_dir / "cases.jsonl", case_rows)
    with (manifest_dir / "cases.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(case_rows[0].keys()),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(case_rows)
    write_jsonl(manifest_dir / "examinations.jsonl", examinations)
    write_jsonl(manifest_dir / "observations.jsonl", observations)
    write_jsonl(manifest_dir / "video_manifest.jsonl", video_rows)

    router_rows = []
    for exam in examinations:
        if "clip-dataset" not in exam["source_datasets"]:
            continue
        router_rows.append(
            {
                "examination_id": exam["examination_id"],
                "case_id": exam["case_id"],
                "source_dataset": "clip-dataset",
                "status": "pending_fetalclip_manifest",
                "checkpoint": None,
                "slots": {slot: {"available": False, "selected_observation_id": None} for slot in SLOT_NAMES},
                "observation_predictions": {},
            }
        )
    write_jsonl(manifest_dir / "router_manifest.jsonl", router_rows)
    router_config = {
        "classification_classes": list(FIVE_CLASS_NAMES),
        "slot_names": list(SLOT_NAMES),
        "class_to_slot": {
            "EYES": "EYES",
            "UPPER_ALVEOLAR": "UPPER_ALVEOLAR",
            "LOWER_ALVEOLAR": "LOWER_ALVEOLAR",
            "INTACT_HARD_PALATE": "HARD_PALATE",
            "CLEFT_HARD_PALATE": "HARD_PALATE",
        },
        "train_condition_source": "gt",
        "validation_condition_source": "router_manifest",
        "diagnosis_source": "fetalclip_fsm_video_postprocess",
        "explicit_vlm_clip_diagnosis_field": False,
        "training_strategy": "stagewise_not_end_to_end",
        "inference_strategy": "threshold_fsm_end_to_end",
        "primary_fold": 0,
        "primary_fold_status": "provisional",
        "primary_fold_validation_positive_cases": 3,
        "video_inference_root": as_posix_relative(videos_root, project_root),
        "video_inference_seniority": "high",
        "video_candidate_count": len(video_rows),
        "video_diagnosis_labels": ["normal", "abnormal", "insufficient_evidence"],
        "video_no_hard_palate_event_policy": "insufficient_evidence",
        "video_hard_palate_event_required_for_inclusion": False,
        "video_hard_palate_event_required_for_diagnosis": True,
        "video_palate_cleft_ratio_when_unavailable": None,
    }
    (manifest_dir / "router_config.json").write_text(json.dumps(router_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    split_meta = write_splits(args.split_root.resolve(), case_labels, args.k, args.seed, dataset_root)
    audit = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(dataset_root),
        "case_count": len(case_rows),
        "normal_cases": sum(1 - row["diagnosis_label"] for row in case_rows),
        "abnormal_cases": sum(row["diagnosis_label"] for row in case_rows),
        "source_audit": {"sam-dataset": sam_audit, "clip-dataset": clip_audit},
        "observation_count": {"sam-dataset": len(sam_obs), "clip-dataset": len(clip_obs)},
        "examination_count": len(examinations),
        "video_inference": video_audit,
        "router_manifest_status": "template_only_pending_fetalclip_predictions",
        "split_root": str(args.split_root.resolve()),
        "split_meta": split_meta,
    }
    (manifest_dir / "phase0_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"[phase0] cases={len(case_rows)} normal={audit['normal_cases']} abnormal={audit['abnormal_cases']}")
    print(f"[phase0] observations sam={len(sam_obs)} clip={len(clip_obs)} examinations={len(examinations)}")
    print(
        f"[phase0] inference videos={len(video_rows)} cases={video_audit['candidate_case_count']} "
        f"(< cohort cases={len(case_rows)})"
    )
    print(f"[phase0] manifests={manifest_dir}")
    print(f"[phase0] splits={args.split_root.resolve()}")


if __name__ == "__main__":
    main()
