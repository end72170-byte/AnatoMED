"""Independent-examination multi-view Dataset for ANATOPAVE phase 1."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, get_worker_info

from utils.segmentation_data import PairedSegTransform


SLOT_NAMES = ("EYES", "UPPER_ALVEOLAR", "LOWER_ALVEOLAR", "HARD_PALATE")
FIVE_CLASS_TO_SLOT = {
    "EYES": "EYES",
    "UPPER_ALVEOLAR": "UPPER_ALVEOLAR",
    "LOWER_ALVEOLAR": "LOWER_ALVEOLAR",
    "INTACT_HARD_PALATE": "HARD_PALATE",
    "CLEFT_HARD_PALATE": "HARD_PALATE",
}


class RouterManifestNotReady(RuntimeError):
    """Raised when validation is asked to use a pending prediction manifest."""


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
    return rows


def _read_split_cases(split_file: Path) -> set[str]:
    return {
        line.strip().split("/")[0]
        for line in split_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def _resolve_path(value: str, project_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (project_root / path)


@dataclass(frozen=True)
class Observation:
    raw: Mapping[str, Any]

    @property
    def observation_id(self) -> str:
        return str(self.raw["observation_id"])

    @property
    def examination_id(self) -> str:
        return str(self.raw["examination_id"])

    @property
    def slot(self) -> str:
        return str(self.raw["anatomy_slot"])


class ExaminationDataset(Dataset):
    """One main image per valid observation with four evidence slots.

    ``condition_source='gt'`` is the training mode. ``condition_source='router'``
    is the validation/inference mode and requires a ready router manifest. The
    router manifest selects fixed auxiliary observations per examination and
    maps each possible main observation to a predicted class/slot.
    """

    def __init__(
        self,
        observations_path: Path,
        split_file: Path,
        *,
        source_dataset: str,
        condition_source: str,
        router_manifest: Optional[Path] = None,
        project_root: Optional[Path] = None,
        seed: int = 42,
        aux_slot_dropout_prob: float = 0.0,
        paired_transform: Optional[PairedSegTransform] = None,
    ) -> None:
        if condition_source not in {"gt", "router"}:
            raise ValueError("condition_source must be 'gt' or 'router'")
        if not 0.0 <= aux_slot_dropout_prob <= 1.0:
            raise ValueError("aux_slot_dropout_prob must be in [0, 1]")
        if condition_source == "router" and router_manifest is None:
            raise ValueError("router_manifest is required when condition_source='router'")

        self.observations_path = Path(observations_path).resolve()
        self.split_file = Path(split_file).resolve()
        self.source_dataset = source_dataset
        self.condition_source = condition_source
        self.project_root = (project_root or Path(__file__).resolve().parents[1]).resolve()
        self.seed = int(seed)
        self.aux_slot_dropout_prob = float(aux_slot_dropout_prob)
        self.epoch = 0
        # Apply the baseline CLIP geometry to each image/mask pair before batching.
        self.paired_transform = paired_transform or PairedSegTransform(
            "clip_center_crop",
            train=condition_source == "gt",
            augment=False,
        )

        all_rows = _read_jsonl(self.observations_path)
        case_ids = _read_split_cases(self.split_file)
        selected = [
            row for row in all_rows
            if row.get("source_dataset") == source_dataset and row.get("case_id") in case_ids
        ]
        if not selected:
            raise RuntimeError(f"No observations for {source_dataset} in {self.split_file}")
        self._observations = {row["observation_id"]: Observation(row) for row in selected}
        self._items = sorted(self._observations.values(), key=lambda item: item.observation_id)
        self._by_exam: Dict[str, List[Observation]] = {}
        for item in self._items:
            self._by_exam.setdefault(item.examination_id, []).append(item)

        self._router: Dict[str, Dict[str, Any]] = {}
        if condition_source == "router":
            self._router = {
                row["examination_id"]: row
                for row in _read_jsonl(Path(router_manifest).resolve())
                if row.get("source_dataset", source_dataset) == source_dataset
            }
            for examination_id in self._by_exam:
                row = self._router.get(examination_id)
                if row is None or row.get("status") not in {"ready", "oracle_test"}:
                    raise RouterManifestNotReady(
                        f"Router manifest has no ready row for {examination_id}; "
                        "run FetalCLIP/FSM manifest generation before validation."
                    )

    def __len__(self) -> int:
        return len(self._items)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _rng(self, index: int) -> random.Random:
        worker = get_worker_info()
        worker_seed = int(worker.seed) if worker is not None else self.seed
        mixed = self.seed + 1000003 * self.epoch + 9176 * index + worker_seed
        return random.Random(mixed & ((1 << 63) - 1))

    def _load_observation(self, observation_id: str) -> Observation:
        try:
            return self._observations[observation_id]
        except KeyError as exc:
            raise KeyError(f"Router selected observation outside this split: {observation_id}") from exc

    def _gt_condition(self, main: Observation, index: int) -> Dict[str, Any]:
        candidates = {
            slot: [item for item in self._by_exam[main.examination_id] if item.slot == slot]
            for slot in SLOT_NAMES
        }
        slots: Dict[str, Dict[str, Any]] = {}
        rng = self._rng(index)
        for slot in SLOT_NAMES:
            pool = candidates[slot]
            if slot == main.slot:
                selected = main
            elif pool:
                selected = pool[rng.randrange(len(pool))]
            else:
                selected = None
            slots[slot] = {"observation": selected, "available": selected is not None, "predicted_class": None}
        return {"main_slot": main.slot, "main_prompt": main.raw["clip_prompt"], "slots": slots}

    def _router_condition(self, main: Observation) -> Dict[str, Any]:
        row = self._router[main.examination_id]
        prediction = row.get("observation_predictions", {}).get(main.observation_id)
        if not prediction:
            raise RouterManifestNotReady(
                f"Router manifest has no prediction for main observation {main.observation_id}"
            )
        main_class = str(prediction.get("class_name", ""))
        main_slot = str(prediction.get("slot_name") or FIVE_CLASS_TO_SLOT.get(main_class, ""))
        if main_slot not in SLOT_NAMES:
            raise ValueError(f"Invalid predicted slot for {main.observation_id}: {main_slot}")
        slots: Dict[str, Dict[str, Any]] = {}
        manifest_slots = row.get("slots", {})
        for slot in SLOT_NAMES:
            if slot == main_slot:
                slots[slot] = {
                    "observation": main,
                    "available": True,
                    "predicted_class": main_class,
                    "score": prediction.get("score"),
                }
                continue
            slot_row = manifest_slots.get(slot, {})
            selected_id = slot_row.get("selected_observation_id")
            selected = self._load_observation(selected_id) if selected_id else None
            if selected is not None and selected.examination_id != main.examination_id:
                raise ValueError(f"Router slot {slot} crosses examination_id for {main.observation_id}")
            slots[slot] = {
                "observation": selected,
                "available": bool(slot_row.get("available", selected is not None)) and selected is not None,
                "predicted_class": slot_row.get("predicted_class"),
                "score": slot_row.get("score"),
            }
        return {"main_slot": main_slot, "main_prompt": prediction.get("prompt"), "slots": slots}

    def _apply_dropout(self, condition: Dict[str, Any], index: int) -> None:
        if self.condition_source != "gt" or self.aux_slot_dropout_prob <= 0.0:
            return
        rng = self._rng(index + 7919)
        main_slot = condition["main_slot"]
        for slot, value in condition["slots"].items():
            if slot == main_slot or not value["available"]:
                continue
            if rng.random() < self.aux_slot_dropout_prob:
                value["observation"] = None
                value["available"] = False
                value["dropped"] = True

    def __getitem__(self, index: int) -> Dict[str, Any]:
        main = self._items[index]
        condition = self._gt_condition(main, index) if self.condition_source == "gt" else self._router_condition(main)
        self._apply_dropout(condition, index)

        main_image_path = _resolve_path(main.raw["image_path"], self.project_root)
        main_mask_path = _resolve_path(main.raw["mask_path"], self.project_root)
        main_text = str(main.raw["main_text"])
        raw_main_image = Image.open(main_image_path).convert("RGB")
        raw_main_mask = Image.open(main_mask_path).convert("L")
        mask_np = (np.asarray(raw_main_mask, dtype=np.uint8) > 127).astype(np.float32)
        if not mask_np.any():
            raise RuntimeError(f"Empty main mask: {main_mask_path}")
        main_image_tensor, main_mask_tensor, geometry = self.paired_transform(
            raw_main_image,
            raw_main_mask,
        )
        if not bool(main_mask_tensor.any()):
            raise RuntimeError(f"GT mask became empty after paired transform: {main_mask_path}")

        slot_images: List[Optional[Image.Image]] = []
        slot_paths: List[Optional[str]] = []
        slot_available: List[bool] = []
        slot_is_main: List[bool] = []
        slot_metadata: List[Dict[str, Any]] = []
        for slot in SLOT_NAMES:
            value = condition["slots"][slot]
            selected = value["observation"]
            is_main = slot == condition["main_slot"]
            if is_main:
                selected = main
            if selected is None or not value["available"]:
                slot_images.append(None)
                slot_paths.append(None)
                slot_available.append(False)
                slot_is_main.append(False)
                slot_metadata.append({
                    "slot": slot,
                    "available": False,
                    "is_main": False,
                    "condition_source": self.condition_source,
                    "dropped": bool(value.get("dropped", False)),
                })
                continue
            image_path = _resolve_path(selected.raw["image_path"], self.project_root)
            slot_images.append(Image.open(image_path).convert("RGB"))
            slot_paths.append(selected.raw["image_path"])
            slot_available.append(True)
            slot_is_main.append(is_main)
            slot_metadata.append({
                "slot": slot,
                "available": True,
                "is_main": is_main,
                "condition_source": self.condition_source,
                "observation_id": selected.observation_id,
                "image_path": selected.raw["image_path"],
                "gt_anatomy_slot": selected.slot,
                "gt_clip_class": selected.raw["clip_class"],
                "predicted_class": value.get("predicted_class"),
                "score": value.get("score"),
            })

        if sum(slot_is_main) != 1:
            raise AssertionError(f"Expected exactly one main slot, got {slot_is_main}")
        return {
            "case_id": main.raw["case_id"],
            "examination_id": main.examination_id,
            "condition": main.raw["condition"],
            "seniority": main.raw["seniority"],
            "acquisition_id": main.raw["acquisition_id"],
            "main_image": main_image_tensor,
            "main_image_raw": raw_main_image,
            "main_mask_raw": raw_main_mask,
            "main_view_id": main.observation_id,
            "main_mask": main_mask_tensor,
            "main_text": main_text,
            "main_clip_prompt": condition["main_prompt"],
            "main_gt_clip_prompt": main.raw["clip_prompt"],
            "main_geometry": asdict(geometry),
            "slot_images": slot_images,
            "slot_available": torch.tensor(slot_available, dtype=torch.bool),
            "slot_is_main": torch.tensor(slot_is_main, dtype=torch.bool),
            "slot_image_paths": slot_paths,
            "slot_metadata": slot_metadata,
            "mask_valid": True,
            "diagnosis_label": int(main.raw["case_diagnosis_label"]),
            "condition_label": int(main.raw["condition_label"]),
            "synthetic_condition": bool(main.raw["synthetic_condition"]),
            "condition_source": self.condition_source,
            "main_image_path": main.raw["image_path"],
            "main_mask_path": main.raw["mask_path"],
            "main_text_path": main.raw["text_path"],
        }


def examination_collate(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not batch:
        raise ValueError("Cannot collate an empty examination batch")
    images = [item["main_image"] for item in batch]
    masks = [item["main_mask"] for item in batch]
    image_shape = tuple(images[0].shape)
    mask_shape = tuple(masks[0].shape)
    if any(tuple(image.shape) != image_shape for image in images):
        raise ValueError("main_image tensors must have identical shapes before collate")
    if any(tuple(mask.shape) != mask_shape for mask in masks):
        raise ValueError("main_mask tensors must have identical shapes before collate")
    return {
        "case_id": [item["case_id"] for item in batch],
        "examination_id": [item["examination_id"] for item in batch],
        "main_image": torch.stack(images),
        "main_image_raw": [item["main_image_raw"] for item in batch],
        "main_view_id": [item["main_view_id"] for item in batch],
        "main_mask": torch.stack(masks),
        "main_mask_raw": [item["main_mask_raw"] for item in batch],
        "main_text": [item["main_text"] for item in batch],
        "main_clip_prompt": [item["main_clip_prompt"] for item in batch],
        "main_gt_clip_prompt": [item["main_gt_clip_prompt"] for item in batch],
        "main_geometry": [item["main_geometry"] for item in batch],
        "slot_images": [item["slot_images"] for item in batch],
        "slot_available": torch.stack([item["slot_available"] for item in batch]),
        "slot_is_main": torch.stack([item["slot_is_main"] for item in batch]),
        "slot_image_paths": [item["slot_image_paths"] for item in batch],
        "slot_metadata": [item["slot_metadata"] for item in batch],
        "mask_valid": torch.tensor([item["mask_valid"] for item in batch], dtype=torch.bool),
        "diagnosis_label": torch.tensor([item["diagnosis_label"] for item in batch], dtype=torch.long),
        "condition_label": torch.tensor([item["condition_label"] for item in batch], dtype=torch.long),
        "condition_source": [item["condition_source"] for item in batch],
        "main_image_paths": [item["main_image_path"] for item in batch],
        "main_mask_paths": [item["main_mask_path"] for item in batch],
    }
