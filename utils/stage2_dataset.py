from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image


def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def normalize_text_line(line: str) -> str:
    """Normalize annotation lines while tolerating UTF-8 BOM-prefixed files."""
    return line.strip().lstrip("\ufeff").strip()


def _read_case_entries(split_file: Path) -> List[str]:
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")

    entries: List[str] = []
    for line in split_file.read_text(encoding="utf-8").splitlines():
        entry = line.strip()
        if not entry or entry.startswith("#"):
            continue
        entries.append(entry)
    return entries


def _resolve_case_entry(dataset_root: Path, entry: str) -> Path:
    entry_path = Path(entry)
    if entry_path.is_absolute():
        candidates = [entry_path]
    else:
        candidates = [dataset_root / entry_path]
        parent_candidate = dataset_root.parent / entry_path
        if parent_candidate != candidates[0]:
            candidates.append(parent_candidate)

    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate.resolve()

    raise FileNotFoundError(
        f"Cannot resolve case entry '{entry}' under dataset_root={dataset_root}"
    )


def resolve_case_dataset_root(dataset_root: Path) -> Path:
    """Return the case-directory root used by split files.

    Supports passing either the case root itself (e.g. ``.../all``) or its
    parent dataset directory (e.g. ``.../clip-dataset`` when ``.../clip-dataset/all``
    exists).
    """
    all_dir = dataset_root / "all"
    if all_dir.exists() and all_dir.is_dir():
        return all_dir
    return dataset_root


def resolve_case_dirs(
    dataset_root: Path,
    split: Optional[str] = None,
    split_file: Optional[Path] = None,
) -> List[Path]:
    if split and split_file is not None:
        raise ValueError("Provide either split or split_file, not both.")
    if not split and split_file is None:
        raise ValueError("One of split or split_file must be provided.")

    if split_file is not None:
        entries = _read_case_entries(split_file)
        case_dirs = [_resolve_case_entry(dataset_root, entry) for entry in entries]
        if not case_dirs:
            raise RuntimeError(f"No case entries found in split file: {split_file}")
        return case_dirs

    split_dir = dataset_root / str(split)
    if not split_dir.exists():
        raise FileNotFoundError(f"Split folder not found: {split_dir}")

    case_dirs = [case_dir.resolve() for case_dir in sorted(split_dir.iterdir()) if case_dir.is_dir()]
    if not case_dirs:
        raise RuntimeError(f"No case folders found under {split_dir}")
    return case_dirs


# -----------------------------
# Dataset (clip-dataset layout)
# -----------------------------
# clip-dataset structure:
#   <root>/<case>/<normal|abnormal>/<high|low>/<anatomy_category>/imgs/*.png
#   <root>/<case>/<normal|abnormal>/<high|low>/<anatomy_category>/masks/*_mask.png
#   <root>/<case>/<normal|abnormal>/<high|low>/<anatomy_category>/<stem>.txt
# Masks are read as images from the masks/ subfolder.
# txt format (3 lines):
#   line1: clip prompt (brief image description)
#   line2: detailed description + diagnosis
#   line3: next-step operation suggestion
# doctor_text = line2 + " " + line3

@dataclass(frozen=True)
class ClipSample:
    image_path: Path
    mask_path: Path
    txt_path: Path
    clip_prompt: str
    doctor_text: str


class ClipDatasetKeyframeMask(Dataset):
    """Dataset for the clip-dataset folder layout.

    Hierarchy:
        <dataset_root>/<split>/<case>/<normal|abnormal>/<high|low>/
            <anatomy_folder>/imgs/<img>.png
            <anatomy_folder>/masks/<img>_mask.png
            <anatomy_folder>/<img>.txt
    """

    def __init__(
        self,
        dataset_root: Path,
        split: Optional[str] = None,
        split_file: Optional[Path] = None,
    ) -> None:
        self.dataset_root = dataset_root
        self.split = split
        self.split_file = split_file

        case_dirs = resolve_case_dirs(dataset_root, split=split, split_file=split_file)
        self._items: List[ClipSample] = []
        self._scan(case_dirs)

        if not self._items:
            source = split_file if split_file is not None else dataset_root / str(split)
            raise RuntimeError(f"No samples found under {source}")

    def _scan(self, case_dirs: List[Path]) -> None:
        """Walk: case / (normal|abnormal) / (high|low) / anatomy_folder."""
        for case_dir in case_dirs:
            for condition_dir in sorted(case_dir.iterdir()):  # normal / abnormal
                if not condition_dir.is_dir():
                    continue
                for quality_dir in sorted(condition_dir.iterdir()):  # high / low
                    if not quality_dir.is_dir():
                        continue
                    for anatomy_dir in sorted(quality_dir.iterdir()):
                        if not anatomy_dir.is_dir():
                            continue
                        imgs_dir = anatomy_dir / "imgs"
                        masks_dir = anatomy_dir / "masks"
                        if not imgs_dir.is_dir() or not masks_dir.is_dir():
                            continue
                        self._scan_anatomy_folder(anatomy_dir, imgs_dir, masks_dir)

    def _scan_anatomy_folder(self, anatomy_dir: Path, imgs_dir: Path, masks_dir: Path) -> None:
        for img_path in sorted(imgs_dir.iterdir()):
            if not img_path.is_file() or not is_image_file(img_path):
                continue
            stem = img_path.stem  # e.g. "1-1"

            # mask: <stem>_mask.<ext> in masks/
            mask_path = self._find_mask(masks_dir, stem)
            if mask_path is None:
                continue

            # txt: <stem>.txt in anatomy_dir
            # txt_path = anatomy_dir / f"{stem}.txt"
            # if not txt_path.exists():
            #     continue

            # 假设 anatomy_dir 是一个 Path 对象
            txt_files = list(anatomy_dir.glob("*.txt"))
            if not txt_files:
                continue  # 如果没有找到任何 txt 文件，则跳过
            txt_path = txt_files[0]  # 取第一个 txt 文件

            lines = [
                normalize_text_line(line)
                for line in txt_path.read_text(encoding="utf-8").splitlines()
                if normalize_text_line(line)
            ]
            if not lines:
                continue

            clip_prompt = lines[0]
            if len(lines) >= 3:
                doctor_text = lines[1] + " " + lines[2]
            elif len(lines) == 2:
                doctor_text = lines[1]
            else:
                doctor_text = lines[0]
            if not doctor_text:
                continue

            self._items.append(ClipSample(
                image_path=img_path,
                mask_path=mask_path,
                txt_path=txt_path,
                clip_prompt=clip_prompt,
                doctor_text=doctor_text,
            ))

    @staticmethod
    def _find_mask(masks_dir: Path, stem: str) -> Optional[Path]:
        """Find mask file matching <stem>_mask.* in masks_dir."""
        for ext in (".png", ".jpg", ".jpeg", ".bmp", ".webp"):
            candidate = masks_dir / f"{stem}_mask{ext}"
            if candidate.exists():
                return candidate
        return None

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self._items[idx]
        img = Image.open(s.image_path).convert("RGB")

        # Read mask image as binary {0,1} float tensor
        mask_img = Image.open(s.mask_path).convert("L")
        mask_np = np.array(mask_img, dtype=np.float32)
        mask_np = (mask_np > 127).astype(np.float32)  # binarize

        return {
            "image": img,
            "image_path": str(s.image_path),
            "mask_path": str(s.mask_path),
            "txt_path": str(s.txt_path),
            "clip_prompt": s.clip_prompt,
            "doctor_text": s.doctor_text,
            "mask": torch.from_numpy(mask_np).float(),  # HxW in {0,1}
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    images = [b["image"] for b in batch]
    masks = [b["mask"] for b in batch]
    clip_prompts = [b["clip_prompt"] for b in batch]
    doctor_texts = [b["doctor_text"] for b in batch]
    paths = [b["image_path"] for b in batch]

    # masks: pad to max H/W then later resize to VLM pixel size
    max_h = max(int(m.shape[0]) for m in masks)
    max_w = max(int(m.shape[1]) for m in masks)
    mask_t = torch.zeros((len(masks), max_h, max_w), dtype=torch.float32)
    for i, m in enumerate(masks):
        h, w = int(m.shape[0]), int(m.shape[1])
        mask_t[i, :h, :w] = m

    return {
        "images": images,
        "mask": mask_t,
        "clip_prompt": clip_prompts,
        "doctor_text": doctor_texts,
        "image_paths": paths,
    }
