from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from utils.stage2_dataset import ClipDatasetKeyframeMask, resolve_case_dataset_root


IMAGE_SIZE = 224
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
UNET_MEAN = (0.5, 0.5, 0.5)
UNET_STD = (0.5, 0.5, 0.5)
PREPROCESS_VERSION = "paired-seg-v1"

PROMPT_CLASSES = (
    "fetal ultrasound, fetal eyes.",
    "fetal ultrasound, upper alveolar ridge.",
    "fetal ultrasound, lower alveolar ridge.",
    "fetal ultrasound, intact hard palate.",
    "fetal ultrasound, cleft hard palate.",
)
PROMPT_SLUGS = {
    PROMPT_CLASSES[0]: "eyes",
    PROMPT_CLASSES[1]: "upper_alveolar",
    PROMPT_CLASSES[2]: "lower_alveolar",
    PROMPT_CLASSES[3]: "intact_hard_palate",
    PROMPT_CLASSES[4]: "cleft_hard_palate",
}


@dataclass(frozen=True)
class GeometryMeta:
    mode: str
    original_width: int
    original_height: int
    canvas_width: int
    canvas_height: int
    resized_width: int
    resized_height: int
    offset_x: int
    offset_y: int
    output_size: int = IMAGE_SIZE


def _pil_to_normalized_tensor(image: Image.Image, mean: Sequence[float], std: Sequence[float]) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    mean_t = torch.tensor(mean, dtype=tensor.dtype).view(3, 1, 1)
    std_t = torch.tensor(std, dtype=tensor.dtype).view(3, 1, 1)
    return (tensor - mean_t) / std_t


def _mask_to_tensor(mask: Image.Image) -> torch.Tensor:
    array = (np.asarray(mask.convert("L"), dtype=np.uint8) > 127).astype(np.float32)
    return torch.from_numpy(array).unsqueeze(0)


def _resize_shortest_size(width: int, height: int, target: int) -> Tuple[int, int]:
    if width <= height:
        return target, int(target * height / width)
    return int(target * width / height), target


class PairedSegTransform:
    """Apply one explicitly shared geometry to an RGB image and binary mask."""

    def __init__(
        self,
        mode: str,
        *,
        train: bool,
        augment: bool = False,
        output_size: int = IMAGE_SIZE,
    ) -> None:
        if mode not in {"letterbox", "clip_center_crop"}:
            raise ValueError(f"Unsupported paired transform mode: {mode}")
        if mode == "clip_center_crop" and augment:
            raise ValueError("CLIP center-crop mode cannot use random augmentation in this experiment.")
        self.mode = mode
        self.train = bool(train)
        self.augment = bool(augment)
        self.output_size = int(output_size)

    def _augment_pair(self, image: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        if not (self.train and self.augment):
            return image, mask
        if random.random() < 0.5:
            image = ImageOps.mirror(image)
            mask = ImageOps.mirror(mask)
        angle = random.uniform(-5.0, 5.0)
        translate = (
            int(round(random.uniform(-0.05, 0.05) * image.width)),
            int(round(random.uniform(-0.05, 0.05) * image.height)),
        )
        image = TF.affine(
            image,
            angle=angle,
            translate=translate,
            scale=1.0,
            shear=(0.0, 0.0),
            interpolation=InterpolationMode.BICUBIC,
            fill=0,
        )
        mask = TF.affine(
            mask,
            angle=angle,
            translate=translate,
            scale=1.0,
            shear=(0.0, 0.0),
            interpolation=InterpolationMode.NEAREST,
            fill=0,
        )
        image = ImageEnhance.Brightness(image).enhance(random.uniform(0.9, 1.1))
        image = ImageEnhance.Contrast(image).enhance(random.uniform(0.9, 1.1))
        return image, mask

    def __call__(self, image: Image.Image, mask: Image.Image) -> Tuple[torch.Tensor, torch.Tensor, GeometryMeta]:
        image = image.convert("RGB")
        mask = mask.convert("L")
        image, mask = self._augment_pair(image, mask)
        width, height = image.size

        if self.mode == "letterbox":
            side = max(width, height)
            left = (side - width) // 2
            top = (side - height) // 2
            right = side - width - left
            bottom = side - height - top
            image_t = ImageOps.expand(image, border=(left, top, right, bottom), fill=0)
            mask_t = ImageOps.expand(mask, border=(left, top, right, bottom), fill=0)
            image_t = image_t.resize((self.output_size, self.output_size), Image.Resampling.BICUBIC)
            mask_t = mask_t.resize((self.output_size, self.output_size), Image.Resampling.NEAREST)
            meta = GeometryMeta(
                mode=self.mode,
                original_width=width,
                original_height=height,
                canvas_width=side,
                canvas_height=side,
                resized_width=self.output_size,
                resized_height=self.output_size,
                offset_x=left,
                offset_y=top,
                output_size=self.output_size,
            )
            image_tensor = _pil_to_normalized_tensor(image_t, UNET_MEAN, UNET_STD)
        else:
            resized_width, resized_height = _resize_shortest_size(width, height, self.output_size)
            image_r = image.resize((resized_width, resized_height), Image.Resampling.BICUBIC)
            mask_r = mask.resize((resized_width, resized_height), Image.Resampling.NEAREST)
            crop_x = max((resized_width - self.output_size) // 2, 0)
            crop_y = max((resized_height - self.output_size) // 2, 0)
            box = (crop_x, crop_y, crop_x + self.output_size, crop_y + self.output_size)
            image_t = image_r.crop(box)
            mask_t = mask_r.crop(box)
            meta = GeometryMeta(
                mode=self.mode,
                original_width=width,
                original_height=height,
                canvas_width=resized_width,
                canvas_height=resized_height,
                resized_width=resized_width,
                resized_height=resized_height,
                offset_x=crop_x,
                offset_y=crop_y,
                output_size=self.output_size,
            )
            image_tensor = _pil_to_normalized_tensor(image_t, CLIP_MEAN, CLIP_STD)

        mask_tensor = _mask_to_tensor(mask_t)
        if image_tensor.shape[-2:] != (self.output_size, self.output_size):
            raise AssertionError(f"Unexpected image shape: {tuple(image_tensor.shape)}")
        if mask_tensor.shape[-2:] != (self.output_size, self.output_size):
            raise AssertionError(f"Unexpected mask shape: {tuple(mask_tensor.shape)}")
        return image_tensor, mask_tensor, meta


def inverse_map_array(array_224: np.ndarray, meta: GeometryMeta, *, nearest: bool) -> np.ndarray:
    """Map a transformed probability/mask array back to original image geometry."""
    interpolation = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    array = np.asarray(array_224, dtype=np.float32)
    if meta.mode == "letterbox":
        canvas = cv2.resize(array, (meta.canvas_width, meta.canvas_height), interpolation=interpolation)
        y0, x0 = meta.offset_y, meta.offset_x
        restored = canvas[y0 : y0 + meta.original_height, x0 : x0 + meta.original_width]
    else:
        canvas = np.zeros((meta.canvas_height, meta.canvas_width), dtype=np.float32)
        y0, x0 = meta.offset_y, meta.offset_x
        canvas[y0 : y0 + meta.output_size, x0 : x0 + meta.output_size] = array
        restored = cv2.resize(
            canvas,
            (meta.original_width, meta.original_height),
            interpolation=interpolation,
        )
    return restored


class SegmentationDataset(Dataset):
    """Transform wrapper around the repository's existing case-aware scanner."""

    def __init__(
        self,
        dataset_root: Path,
        split_file: Path,
        *,
        transform: PairedSegTransform,
    ) -> None:
        root = resolve_case_dataset_root(Path(dataset_root))
        self.base = ClipDatasetKeyframeMask(root, split_file=Path(split_file))
        self.transform = transform
        unknown = sorted({sample.clip_prompt for sample in self.base._items} - set(PROMPT_CLASSES))
        if unknown:
            raise ValueError(f"Unexpected CLIP prompt classes: {unknown}")

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.base._items[index]
        image = Image.open(sample.image_path).convert("RGB")
        mask = Image.open(sample.mask_path).convert("L")
        original_mask = (np.asarray(mask, dtype=np.uint8) > 127).astype(np.uint8)
        if not original_mask.any():
            raise RuntimeError(f"Empty GT mask is not allowed: {sample.mask_path}")
        image_tensor, mask_tensor, geometry = self.transform(image, mask)
        if not bool(mask_tensor.any()):
            raise RuntimeError(f"GT mask became empty after transform: {sample.mask_path}")
        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "original_image": image,
            "original_mask": original_mask,
            "geometry": geometry,
            "image_path": str(sample.image_path),
            "mask_path": str(sample.mask_path),
            "prompt": sample.clip_prompt,
            "prompt_slug": PROMPT_SLUGS[sample.clip_prompt],
        }

    @property
    def samples(self):
        return self.base._items


def segmentation_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "images": torch.stack([item["image"] for item in batch]),
        "masks": torch.stack([item["mask"] for item in batch]),
        "original_images": [item["original_image"] for item in batch],
        "original_masks": [item["original_mask"] for item in batch],
        "geometry": [item["geometry"] for item in batch],
        "image_paths": [item["image_path"] for item in batch],
        "mask_paths": [item["mask_path"] for item in batch],
        "prompts": [item["prompt"] for item in batch],
        "prompt_slugs": [item["prompt_slug"] for item in batch],
    }


def read_case_set(split_file: Path) -> set[str]:
    return {
        line.strip()
        for line in Path(split_file).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def select_fixed_visualization_indices(dataset: SegmentationDataset, per_class: int = 5) -> Dict[str, List[int]]:
    selected: Dict[str, List[int]] = {PROMPT_SLUGS[prompt]: [] for prompt in PROMPT_CLASSES}
    for index, sample in enumerate(dataset.samples):
        slug = PROMPT_SLUGS[sample.clip_prompt]
        if len(selected[slug]) < per_class:
            selected[slug].append(index)
    return selected


def class_counts(dataset: SegmentationDataset) -> Dict[str, int]:
    counts = {slug: 0 for slug in PROMPT_SLUGS.values()}
    for sample in dataset.samples:
        counts[PROMPT_SLUGS[sample.clip_prompt]] += 1
    return counts


def _bbox(mask: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def _bbox_center_size(box: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    x0, y0, x1, y1 = box
    return (x0 + x1) / 2.0, (y0 + y1) / 2.0, x1 - x0 + 1.0, y1 - y0 + 1.0


def _manual_transform_mask(mask: np.ndarray, meta: GeometryMeta) -> np.ndarray:
    if meta.mode == "letterbox":
        canvas = np.zeros((meta.canvas_height, meta.canvas_width), dtype=np.uint8)
        canvas[
            meta.offset_y : meta.offset_y + meta.original_height,
            meta.offset_x : meta.offset_x + meta.original_width,
        ] = mask
        return np.asarray(
            Image.fromarray(canvas).resize(
                (meta.output_size, meta.output_size), Image.Resampling.NEAREST
            ),
            dtype=np.uint8,
        )
    resized = np.asarray(
        Image.fromarray(mask).resize(
            (meta.resized_width, meta.resized_height), Image.Resampling.NEAREST
        ),
        dtype=np.uint8,
    )
    return resized[
        meta.offset_y : meta.offset_y + meta.output_size,
        meta.offset_x : meta.offset_x + meta.output_size,
    ]


def run_geometry_check(dataset: SegmentationDataset, out_path: Path, *, num_samples: int = 20, seed: int = 42) -> Dict[str, Any]:
    if dataset.transform.train and dataset.transform.augment:
        raise ValueError("Geometry checks require deterministic transforms without augmentation.")
    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    indices = indices[: min(num_samples, len(indices))]
    records: List[Dict[str, Any]] = []
    max_error = 0.0
    for index in indices:
        item = dataset[index]
        actual = item["mask"][0].numpy().astype(np.uint8)
        expected = _manual_transform_mask(item["original_mask"], item["geometry"])
        actual_box = _bbox(actual)
        expected_box = _bbox(expected)
        if actual_box is None or expected_box is None:
            raise AssertionError(f"Geometry check produced an empty bbox: {item['mask_path']}")
        actual_css = _bbox_center_size(actual_box)
        expected_css = _bbox_center_size(expected_box)
        errors = [abs(a - b) for a, b in zip(actual_css, expected_css)]
        sample_max = max(errors)
        max_error = max(max_error, sample_max)
        records.append(
            {
                "image_path": item["image_path"],
                "actual_bbox": actual_box,
                "manual_bbox": expected_box,
                "center_size_errors": errors,
                "max_error_px": sample_max,
                "geometry": asdict(item["geometry"]),
            }
        )
    result = {
        "preprocess_version": PREPROCESS_VERSION,
        "mode": dataset.transform.mode,
        "num_samples": len(records),
        "max_center_or_size_error_px": max_error,
        "passed": max_error <= 1.0,
        "records": records,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if not result["passed"]:
        raise AssertionError(f"Paired geometry check failed with max error {max_error:.3f}px")
    return result
