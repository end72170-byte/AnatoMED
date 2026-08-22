import argparse
import json
import math
import os
import random
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class SampleEntry:
    source_root: Path
    image_rel: Path
    mask_rel: Path
    txt_rels: Tuple[Path, ...]
    overlay_rel: Optional[Path]

    @property
    def image_path(self) -> Path:
        return self.source_root / self.image_rel

    @property
    def mask_path(self) -> Path:
        return self.source_root / self.mask_rel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate a mixed subdataset with all clip samples plus a sampled ratio of SAM augmentations."
    )
    p.add_argument(
        "--clip_dataset_root",
        type=str,
        required=True,
        help="Path to clip-dataset root or case root. Examples: .../clip-dataset or .../clip-dataset/all",
    )
    p.add_argument(
        "--sam_dataset_root",
        type=str,
        required=True,
        help="Path to sam-dataset root or case root. Examples: .../sam-dataset5-ab or .../sam-dataset5-ab/all",
    )
    p.add_argument(
        "--aug_ratio",
        type=float,
        default=0.2,
        help="Fraction of SAM augmentation candidates to include after removing duplicated normal offset_p00 samples. Range: [0, 1].",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used to sample SAM augmentation candidates.",
    )
    p.add_argument(
        "--out_root",
        type=str,
        default="dataset",
        help="Output parent directory. The generated dataset folder will be created inside it.",
    )
    p.add_argument(
        "--out_name",
        type=str,
        default="",
        help="Optional output dataset folder name. Defaults to an auto-generated name based on ratio and seed.",
    )
    p.add_argument(
        "--mode",
        type=str,
        default="symlink",
        choices=["symlink", "copy"],
        help="Materialization mode for output files. 'symlink' saves disk space.",
    )
    return p.parse_args()


def resolve_case_root(dataset_root: Path) -> Path:
    dataset_root = dataset_root.resolve()
    all_dir = dataset_root / "all"
    if all_dir.exists() and all_dir.is_dir():
        return all_dir
    return dataset_root


def infer_out_name(clip_root: Path, sam_root: Path, aug_ratio: float, seed: int) -> str:
    ratio_tag = int(round(aug_ratio * 100))
    return f"{clip_root.parent.name}_plus_{sam_root.parent.name}_ratio{ratio_tag:02d}_seed{seed}"


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS


def iter_anatomy_dirs(case_root: Path) -> Iterable[Path]:
    for condition_dir in sorted(case_root.iterdir()):
        if not condition_dir.is_dir():
            continue
        for quality_dir in sorted(condition_dir.iterdir()):
            if not quality_dir.is_dir():
                continue
            for anatomy_dir in sorted(quality_dir.iterdir()):
                if anatomy_dir.is_dir():
                    yield anatomy_dir


def find_mask(masks_dir: Path, stem: str) -> Optional[Path]:
    for ext in (".png", ".jpg", ".jpeg", ".bmp", ".webp"):
        candidate = masks_dir / f"{stem}_mask{ext}"
        if candidate.exists():
            return candidate
    return None


def collect_entries(case_root_parent: Path) -> List[SampleEntry]:
    entries: List[SampleEntry] = []
    for case_dir in sorted(p for p in case_root_parent.iterdir() if p.is_dir()):
        for anatomy_dir in iter_anatomy_dirs(case_dir):
            imgs_dir = anatomy_dir / "imgs"
            masks_dir = anatomy_dir / "masks"
            if not imgs_dir.is_dir() or not masks_dir.is_dir():
                continue

            txt_paths = tuple(sorted(p.relative_to(case_root_parent) for p in anatomy_dir.glob("*.txt") if p.is_file()))
            overlay_dir = anatomy_dir / "overlay"

            for img_path in sorted(imgs_dir.iterdir()):
                if not img_path.is_file() or not is_image_file(img_path):
                    continue
                mask_path = find_mask(masks_dir, img_path.stem)
                if mask_path is None:
                    continue

                overlay_rel: Optional[Path] = None
                overlay_path = overlay_dir / img_path.name
                if overlay_path.exists() and overlay_path.is_file():
                    overlay_rel = overlay_path.relative_to(case_root_parent)

                entries.append(
                    SampleEntry(
                        source_root=case_root_parent,
                        image_rel=img_path.relative_to(case_root_parent),
                        mask_rel=mask_path.relative_to(case_root_parent),
                        txt_rels=txt_paths,
                        overlay_rel=overlay_rel,
                    )
                )
    return entries


def build_clip_duplicate_relpaths(clip_entries: Sequence[SampleEntry]) -> Set[Path]:
    duplicate_relpaths: Set[Path] = set()
    for entry in clip_entries:
        image_rel = entry.image_rel
        if image_rel.parent.name != "imgs":
            continue
        duplicate_relpaths.add(image_rel)
        dup_name = f"{image_rel.stem}_offset_p00{image_rel.suffix}"
        duplicate_relpaths.add(image_rel.with_name(dup_name))
    return duplicate_relpaths


def split_sam_entries(
    sam_entries: Sequence[SampleEntry],
    clip_duplicate_relpaths: Set[Path],
) -> Tuple[List[SampleEntry], List[SampleEntry]]:
    duplicate_entries: List[SampleEntry] = []
    aug_entries: List[SampleEntry] = []
    for entry in sam_entries:
        if entry.image_rel in clip_duplicate_relpaths:
            duplicate_entries.append(entry)
        else:
            aug_entries.append(entry)
    return duplicate_entries, aug_entries


def select_aug_entries(entries: Sequence[SampleEntry], aug_ratio: float, seed: int) -> List[SampleEntry]:
    if aug_ratio < 0.0 or aug_ratio > 1.0:
        raise ValueError("--aug_ratio must be in [0, 1]")
    if not entries or aug_ratio == 0.0:
        return []

    count = min(len(entries), max(1, math.ceil(len(entries) * aug_ratio)))
    rng = random.Random(seed)
    pool = list(entries)
    rng.shuffle(pool)
    selected = sorted(pool[:count], key=lambda item: item.image_rel.as_posix())
    return selected


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def materialize_file(src: Path, dst: Path, mode: str) -> None:
    ensure_parent(dst)
    if dst.exists() or dst.is_symlink():
        return
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    os.symlink(src, dst)


def materialize_entries(entries: Sequence[SampleEntry], out_case_root: Path, mode: str) -> None:
    for entry in entries:
        materialize_file(entry.image_path, out_case_root / entry.image_rel, mode)
        materialize_file(entry.mask_path, out_case_root / entry.mask_rel, mode)
        for txt_rel in entry.txt_rels:
            materialize_file(entry.source_root / txt_rel, out_case_root / txt_rel, mode)
        if entry.overlay_rel is not None:
            materialize_file(entry.source_root / entry.overlay_rel, out_case_root / entry.overlay_rel, mode)


def count_by_condition(entries: Sequence[SampleEntry]) -> Dict[str, int]:
    stats: Dict[str, int] = {}
    for entry in entries:
        parts = entry.image_rel.parts
        condition = parts[1] if len(parts) > 1 else "unknown"
        stats[condition] = stats.get(condition, 0) + 1
    return stats


def write_meta(path: Path, meta: Dict[str, object]) -> None:
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    clip_case_root = resolve_case_root(Path(args.clip_dataset_root))
    sam_case_root = resolve_case_root(Path(args.sam_dataset_root))

    if not clip_case_root.exists():
        raise FileNotFoundError(f"Clip dataset root not found: {clip_case_root}")
    if not sam_case_root.exists():
        raise FileNotFoundError(f"SAM dataset root not found: {sam_case_root}")

    out_root = Path(args.out_root).resolve()
    out_name = args.out_name or infer_out_name(clip_case_root, sam_case_root, float(args.aug_ratio), int(args.seed))
    out_dataset_root = out_root / out_name
    out_case_root = out_dataset_root / "all"

    if out_dataset_root.exists():
        raise FileExistsError(f"Output dataset already exists: {out_dataset_root}")

    clip_entries = collect_entries(clip_case_root)
    sam_entries = collect_entries(sam_case_root)
    if not clip_entries:
        raise RuntimeError(f"No clip samples found under {clip_case_root}")
    if not sam_entries:
        raise RuntimeError(f"No SAM samples found under {sam_case_root}")

    duplicate_relpaths = build_clip_duplicate_relpaths(clip_entries)
    sam_duplicate_entries, sam_aug_entries = split_sam_entries(sam_entries, duplicate_relpaths)
    selected_aug_entries = select_aug_entries(sam_aug_entries, float(args.aug_ratio), int(args.seed))

    out_case_root.mkdir(parents=True, exist_ok=True)
    materialize_entries(clip_entries, out_case_root, args.mode)
    materialize_entries(selected_aug_entries, out_case_root, args.mode)

    total_entries = len(clip_entries) + len(selected_aug_entries)
    meta = {
        "method": "clip_plus_sampled_sam_augmentations",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "clip_dataset_root": str(clip_case_root),
        "sam_dataset_root": str(sam_case_root),
        "output_dataset_root": str(out_dataset_root),
        "mode": args.mode,
        "seed": int(args.seed),
        "aug_ratio": float(args.aug_ratio),
        "duplicate_rule": "exclude sam images whose relative path exactly matches a clip image, or matches the clip image with filename suffix '_offset_p00'",
        "clip_samples": len(clip_entries),
        "sam_samples_total": len(sam_entries),
        "sam_samples_excluded_as_clip_duplicates": len(sam_duplicate_entries),
        "sam_augmentation_candidates": len(sam_aug_entries),
        "sam_augmentation_selected": len(selected_aug_entries),
        "output_total_samples": total_entries,
        "clip_condition_counts": count_by_condition(clip_entries),
        "selected_aug_condition_counts": count_by_condition(selected_aug_entries),
    }
    write_meta(out_dataset_root / "meta.json", meta)

    print(f"[done] Output dataset: {out_dataset_root}")
    print(f"[done] Clip samples kept: {len(clip_entries)}")
    print(f"[done] SAM duplicates excluded: {len(sam_duplicate_entries)}")
    print(f"[done] SAM augmentation candidates: {len(sam_aug_entries)}")
    print(f"[done] SAM augmentation selected: {len(selected_aug_entries)}")
    print(f"[done] Total output samples: {total_entries}")


if __name__ == "__main__":
    main()
