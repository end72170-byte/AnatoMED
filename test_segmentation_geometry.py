"""Run the required 20-sample paired-geometry checks without training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from utils.segmentation_data import PairedSegTransform, SegmentationDataset, run_geometry_check


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_dataset_root", required=True)
    parser.add_argument("--fold_dir", required=True)
    parser.add_argument("--out_dir", default="runs/geometry_tests")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for mode in ("letterbox", "clip_center_crop"):
        dataset = SegmentationDataset(
            Path(args.val_dataset_root),
            Path(args.fold_dir) / "val.txt",
            transform=PairedSegTransform(mode, train=False),
        )
        results[mode] = run_geometry_check(dataset, out_dir / f"{mode}.json", num_samples=20)
        print(f"[geometry] mode={mode} max_error={results[mode]['max_center_or_size_error_px']:.3f}px")
    (out_dir / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
