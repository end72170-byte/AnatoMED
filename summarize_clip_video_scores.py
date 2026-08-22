"""
Summarize per-video FetalCLIP class distributions on clip-dataset high videos.

Example:
python summarize_clip_video_scores.py \
  --dataset_root /data/clip-dataset \
  --fetalclip_weights /checkpoints/fetalclip.pt \
  --out_jsonl runs/clip_video_score_summary.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
from accelerate import Accelerator

from inference import (
    CLASS_LABELS,
    DEFAULT_KEYFRAME_PROMPTS,
    load_fetalclip,
    pil_from_bgr,
    score_frame,
    set_random_seed,
)
from utils.runtime import _configure_warnings


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize FetalCLIP scores for clip-dataset high videos")
    p.add_argument("--dataset_root", type=str, required=True)
    p.add_argument("--out_jsonl", type=str, default="", help="Optional JSONL output path.")
    p.add_argument("--fetalclip_config", type=str, default="model/FetalCLIP_config.json")
    p.add_argument("--fetalclip_weights", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--suppress_warnings", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--frame_stride", type=int, default=1)
    p.add_argument("--max_frames", type=int, default=0)
    return p.parse_args()


def iter_high_videos(dataset_root: Path) -> List[Path]:
    all_root = dataset_root / "all"
    if not all_root.exists():
        raise FileNotFoundError(f"Expected dataset root with all/ under: {dataset_root}")
    return sorted(all_root.glob("case*/*/high/video.avi"))


def summarize_video(
    *,
    video_path: Path,
    fetalclip_model: Any,
    fetalclip_preprocess: Any,
    prompt_text_features: Any,
    device: Any,
    frame_stride: int,
    max_frames: int,
) -> Dict[str, Any]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0.0:
        fps = 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    frame_idx = -1
    processed = 0
    label_counts = {label: 0 for label in CLASS_LABELS}
    top_scores: List[float] = []
    margins: List[float] = []
    per_label_top_scores: Dict[str, List[float]] = {label: [] for label in CLASS_LABELS}

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_stride > 1 and frame_idx % frame_stride != 0:
            continue
        processed += 1
        if max_frames > 0 and processed > max_frames:
            break

        image_pil = pil_from_bgr(frame_bgr)
        raw_scores = score_frame(
            fetalclip_model=fetalclip_model,
            fetalclip_preprocess=fetalclip_preprocess,
            prompt_text_features=prompt_text_features,
            device=device,
            image_pil=image_pil,
        )
        score_vec = np.array([raw_scores[label] for label in CLASS_LABELS], dtype=np.float32)
        top_idx = int(np.argmax(score_vec))
        sorted_scores = np.sort(score_vec)
        top_score = float(score_vec[top_idx])
        margin = float(sorted_scores[-1] - sorted_scores[-2]) if len(sorted_scores) > 1 else 0.0
        top_label = CLASS_LABELS[top_idx]

        label_counts[top_label] += 1
        top_scores.append(top_score)
        margins.append(margin)
        per_label_top_scores[top_label].append(top_score)

    cap.release()

    def _stats(values: List[float]) -> Dict[str, float]:
        if not values:
            return {"min": 0.0, "mean": 0.0, "max": 0.0}
        return {
            "min": float(np.min(values)),
            "mean": float(np.mean(values)),
            "max": float(np.max(values)),
        }

    per_label: Dict[str, Any] = {}
    for label, prompt in zip(CLASS_LABELS, DEFAULT_KEYFRAME_PROMPTS):
        count = int(label_counts[label])
        ratio = float(count / processed) if processed > 0 else 0.0
        per_label[prompt] = {
            "count": count,
            "ratio": ratio,
            "top_score_stats": _stats(per_label_top_scores[label]),
        }

    cleft_prompt = "fetal ultrasound, cleft hard palate."
    cleft_info = per_label[cleft_prompt]

    return {
        "video_path": str(video_path),
        "case": video_path.parts[-4],
        "condition": video_path.parts[-3],
        "quality": video_path.parts[-2],
        "fps": fps,
        "total_frames": total_frames,
        "processed_frames": processed,
        "frame_stride": frame_stride,
        "overall_top_score_stats": _stats(top_scores),
        "overall_margin_stats": _stats(margins),
        "per_prompt": per_label,
        "cleft_hard_palate_count": int(cleft_info["count"]),
        "cleft_hard_palate_ratio": float(cleft_info["ratio"]),
    }


def main() -> None:
    args = parse_args()
    accelerator = Accelerator(cpu=bool(args.cpu))
    log = accelerator.print
    set_random_seed(int(args.seed))
    _configure_warnings(suppress=bool(args.suppress_warnings))
    device = accelerator.device

    fetalclip_model, fetalclip_preprocess, _, prompt_text_features = load_fetalclip(args, device, log)
    dataset_root = Path(args.dataset_root)
    videos = iter_high_videos(dataset_root)
    log(f"[data] Found {len(videos)} high videos under {dataset_root}")

    summaries: List[Dict[str, Any]] = []
    for video_path in videos:
        summary = summarize_video(
            video_path=video_path,
            fetalclip_model=fetalclip_model,
            fetalclip_preprocess=fetalclip_preprocess,
            prompt_text_features=prompt_text_features,
            device=device,
            frame_stride=int(args.frame_stride),
            max_frames=int(args.max_frames),
        )
        summaries.append(summary)
        cleft_count = int(summary["cleft_hard_palate_count"])
        cleft_ratio = float(summary["cleft_hard_palate_ratio"])
        overall = summary["overall_top_score_stats"]
        log(
            f"{summary['case']} {summary['condition']} high "
            f"frames={summary['processed_frames']} "
            f"cleft_hard_palate={cleft_count} ({cleft_ratio:.3f}) "
            f"top_score min/mean/max={overall['min']:.4f}/{overall['mean']:.4f}/{overall['max']:.4f}"
        )
        for prompt in DEFAULT_KEYFRAME_PROMPTS:
            info = summary["per_prompt"][prompt]
            stats = info["top_score_stats"]
            log(
                f"  {prompt}: {info['count']} ({info['ratio']:.3f})  "
                f"top_score min/mean/max={stats['min']:.4f}/{stats['mean']:.4f}/{stats['max']:.4f}"
            )

    if args.out_jsonl:
        out_path = Path(args.out_jsonl)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for item in summaries:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        log(f"Saved summary JSONL: {out_path}")


if __name__ == "__main__":
    main()
