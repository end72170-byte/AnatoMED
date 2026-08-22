import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import List

"""
This script generates group k-fold split files for a dataset organized as case folders under a dataset root. Each fold's train and val splits are written to separate text files listing the relative paths of case folders. A meta.json file is also generated containing metadata about the splits.
Example usage:
python data_process/generate_group_kfold_splits.py \
  --dataset_root /data/sam-dataset/all \
  --k 5 \
  --seed 42

"""

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate offline group k-fold split files from case folders."
    )
    p.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="Dataset root that directly contains case folders.",
    )
    p.add_argument("--k", type=int, required=True, help="Number of folds.")
    p.add_argument("--seed", type=int, default=42, help="Shuffle seed before assigning folds.")
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


def split_evenly(items: List[str], k: int) -> List[List[str]]:
    n = len(items)
    return [items[i * n // k:(i + 1) * n // k] for i in range(k)]


def write_lines(path: Path, lines: List[str]) -> None:
    content = "\n".join(lines)
    if content:
        content += "\n"
    path.write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    if args.k < 2:
        raise ValueError("--k must be >= 2")

    case_entries = list_case_entries(dataset_root)
    if args.k > len(case_entries):
        raise ValueError(
            f"--k={args.k} is larger than the number of cases ({len(case_entries)})"
        )

    shuffled = list(case_entries)
    rng = random.Random(args.seed)
    rng.shuffle(shuffled)
    fold_vals = split_evenly(shuffled, args.k)

    dataset_name = infer_dataset_name(dataset_root, args.dataset_name)
    split_root = (
        Path(args.out_root).resolve()
        / dataset_name
        / f"group_kfold_k{args.k}_seed{args.seed}"
    )
    split_root.mkdir(parents=True, exist_ok=True)

    write_lines(split_root / "all_cases.txt", sorted(case_entries))

    fold_sizes = []
    for fold_idx, val_entries in enumerate(fold_vals):
        val_set = set(val_entries)
        train_entries = [entry for entry in case_entries if entry not in val_set]
        fold_dir = split_root / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        write_lines(fold_dir / "train.txt", train_entries)
        write_lines(fold_dir / "val.txt", sorted(val_entries))
        fold_sizes.append(
            {
                "fold": fold_idx,
                "train_cases": len(train_entries),
                "val_cases": len(val_entries),
            }
        )

    meta = {
        "method": "group_kfold",
        "group_unit": "case_folder",
        "k": args.k,
        "seed": args.seed,
        "dataset_name": dataset_name,
        "dataset_root": str(dataset_root),
        "case_entries_relative_to": str(dataset_root),
        "num_cases": len(case_entries),
        "fold_sizes": fold_sizes,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (split_root / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"[done] Wrote splits to: {split_root}")
    print(f"[done] Total cases: {len(case_entries)}")


if __name__ == "__main__":
    main()
