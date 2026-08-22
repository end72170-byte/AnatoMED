"""Compact FetalCLIP slot-summary cache for ANATOPAVE."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import torch
from PIL import Image


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


class CompactSummaryCache:
    """Path-keyed global image vectors; dense maps are never persisted."""

    def __init__(self, summaries: Optional[Mapping[str, torch.Tensor]] = None) -> None:
        self.summaries: Dict[str, torch.Tensor] = {
            str(path): value.detach().cpu().float().contiguous()
            for path, value in (summaries or {}).items()
        }
        self.requested_paths: list[str] = []

    def __len__(self) -> int:
        return len(self.summaries)

    def get(self, path: str, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        self.requested_paths.append(str(path))
        try:
            value = self.summaries[str(path)]
        except KeyError as exc:
            raise KeyError(f"Compact summary is missing for auxiliary image: {path}") from exc
        return value.to(device=device, dtype=dtype)

    def save(self, path: Path, metadata: Mapping[str, Any]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"metadata": dict(metadata), "summaries": self.summaries}, path)

    @classmethod
    def load(cls, path: Path) -> tuple["CompactSummaryCache", Dict[str, Any]]:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        return cls(payload["summaries"]), dict(payload.get("metadata", {}))


@torch.no_grad()
def build_compact_summary_cache(
    image_paths: Iterable[str],
    *,
    project_root: Path,
    preprocess: Any,
    feature_extractor: Any,
    device: torch.device,
    batch_size: int = 16,
) -> CompactSummaryCache:
    unique = sorted(set(str(path) for path in image_paths if path))
    summaries: Dict[str, torch.Tensor] = {}
    feature_extractor.eval()
    for start in range(0, len(unique), batch_size):
        paths = unique[start : start + batch_size]
        tensors = []
        for value in paths:
            path = Path(value)
            path = path if path.is_absolute() else project_root / path
            with Image.open(path) as image:
                tensors.append(preprocess(image.convert("RGB")))
        images = torch.stack(tensors).to(device)
        _, pooled = feature_extractor(images)
        for value, vector in zip(paths, pooled):
            summaries[value] = vector.detach().cpu().float()
    return CompactSummaryCache(summaries)


def dataset_observation_paths(dataset: Any) -> Sequence[str]:
    """Return every candidate path so epoch-wise GT auxiliary sampling is cache-safe."""
    return [str(item.raw["image_path"]) for item in dataset._items]


def assemble_slot_summaries(
    *,
    main_summary: torch.Tensor,
    slot_available: torch.Tensor,
    slot_is_main: torch.Tensor,
    slot_image_paths: Sequence[Sequence[Optional[str]]],
    cache: CompactSummaryCache,
) -> torch.Tensor:
    """Use online main summary and cached auxiliaries; never query missing slots."""
    batch_size, num_slots = slot_available.shape
    output = main_summary.new_zeros((batch_size, num_slots, main_summary.shape[-1]))
    for batch_index in range(batch_size):
        for slot_index in range(num_slots):
            if not bool(slot_available[batch_index, slot_index]):
                continue
            if bool(slot_is_main[batch_index, slot_index]):
                output[batch_index, slot_index] = main_summary[batch_index]
                continue
            path = slot_image_paths[batch_index][slot_index]
            if path is None:
                raise AssertionError("Available auxiliary slot has no image path")
            output[batch_index, slot_index] = cache.get(
                path, device=output.device, dtype=output.dtype
            )
    return output
