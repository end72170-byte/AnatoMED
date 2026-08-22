import argparse
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import List

"""
This script generates one fixed validation split and multiple nested training subsets from a given dataset root containing case folders. The splits are written as text files listing the relative paths of the case folders.
The main use case is to create training subsets of varying sizes while keeping the same validation split, which can be useful for learning curve analysis or data efficiency studies.
Usage example:
python data_process/generate_subset_splits.py \
  --dataset_root /data/sam-dataset/all \
  --seed 42

"""

DEFAULT_SUBSET_RATIOS = [0.2, 0.4, 0.6, 0.8, 1.0]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate one fixed validation split and multiple nested training subsets."
    )
    p.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="Dataset root that directly contains case folders.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Shuffle seed used to create the fixed val split and nested train subsets.",
    )
    p.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="Fraction of cases reserved for validation. Default: 0.2",
    )
    p.add_argument(
        "--subset_ratios",
        type=str,
        default="0.2,0.4,0.6,0.8,1.0",
        help="Comma-separated train subset ratios measured on the train pool.",
    )
    p.add_argument(
        "--out_root",
        type=str,
        default="splits",
        help="Root folder where split files will be written.",
    )
    p.add_argument(
        "--dataset_name",
        type=str,
        default="",
        help="Optional dataset name for the output folder. Defaults to parent folder name when dataset_root ends with all.",
    )
    return p.parse_args()


def infer_dataset_name(dataset_root: Path, dataset_name: str) -> str:
    if dataset_name:
        return dataset_name
    if dataset_root.name == "all" and dataset_root.parent.name:
        return dataset_root.parent.name
    return dataset_root.name


def list_case_entries(dataset_root: Path) -> List[str]:
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    if not dataset_root.is_dir():
        raise NotADirectoryError(f"Dataset root is not a directory: {dataset_root}")

    case_dirs = sorted(p for p in dataset_root.iterdir() if p.is_dir())
    if not case_dirs:
        raise RuntimeError(f"No case folders found under {dataset_root}")
    return [p.relative_to(dataset_root).as_posix() for p in case_dirs]


def parse_subset_ratios(raw: str) -> List[float]:
    ratios: List[float] = []
    for part in raw.split(","):
        value = float(part.strip())
        if value <= 0.0 or value > 1.0:
            raise ValueError(f"Subset ratio must be in (0, 1], got {value}")
        ratios.append(value)
    if not ratios:
        raise ValueError("No subset ratios provided")
    if ratios != sorted(ratios):
        raise ValueError("--subset_ratios must be in ascending order")
    return ratios


def subset_name(ratio: float) -> str:
    return f"subset{int(round(ratio * 10)):02d}"


def write_lines(path: Path, lines: List[str]) -> None:
    content = "\n".join(lines)
    if content:
        content += "\n"
    path.write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    dataset_name = infer_dataset_name(dataset_root, args.dataset_name)
    subset_ratios = parse_subset_ratios(args.subset_ratios)

    if args.val_ratio <= 0.0 or args.val_ratio >= 1.0:
        raise ValueError("--val_ratio must be in (0, 1)")

    case_entries = list_case_entries(dataset_root)
    rng = random.Random(args.seed)
    shuffled = list(case_entries)
    rng.shuffle(shuffled)

    num_cases = len(shuffled)
    num_val = max(1, math.ceil(num_cases * args.val_ratio))
    if num_val >= num_cases:
        raise ValueError("Validation split would consume all cases; reduce --val_ratio")

    val_entries = shuffled[:num_val]
    train_pool = shuffled[num_val:]
    if not train_pool:
        raise RuntimeError("Train pool is empty after validation split")

    split_root = (
        Path(args.out_root).resolve()
        / dataset_name
        / f"subset_seed{args.seed}"
    )
    split_root.mkdir(parents=True, exist_ok=True)

    write_lines(split_root / "all_cases.txt", sorted(case_entries))

    subset_sizes = []
    for ratio in subset_ratios:
        train_count = min(len(train_pool), max(1, math.ceil(len(train_pool) * ratio)))
        train_entries = sorted(train_pool[:train_count])
        subset_dir = split_root / subset_name(ratio)
        subset_dir.mkdir(parents=True, exist_ok=True)
        write_lines(subset_dir / "train.txt", train_entries)
        write_lines(subset_dir / "val.txt", sorted(val_entries))
        subset_sizes.append(
            {
                "subset": subset_name(ratio),
                "train_ratio_on_train_pool": ratio,
                "train_cases": len(train_entries),
                "val_cases": len(val_entries),
            }
        )

    meta = {
        "method": "fixed_val_nested_subsets",
        "group_unit": "case_folder",
        "seed": args.seed,
        "dataset_name": dataset_name,
        "dataset_root": str(dataset_root),
        "case_entries_relative_to": str(dataset_root),
        "num_cases": len(case_entries),
        "val_ratio": args.val_ratio,
        "train_pool_cases": len(train_pool),
        "val_cases": len(val_entries),
        "subset_sizes": subset_sizes,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (split_root / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"[done] Wrote subset splits to: {split_root}")
    print(
        f"[done] Total cases: {len(case_entries)} | train pool: {len(train_pool)} | val: {len(val_entries)}"
    )


if __name__ == "__main__":
    main()
