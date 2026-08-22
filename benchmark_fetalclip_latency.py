"""
Benchmark average per-frame FetalCLIP scoring latency on a video.

Example:
python benchmark_fetalclip_latency.py --video_path /data/example/video.avi
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import json
import open_clip
import torch
from accelerate import Accelerator
from open_clip.factory import _MODEL_CONFIGS

from inference import pil_from_bgr, score_frame, set_random_seed
from utils.prompt_utils import build_prompt_text_features
from utils.runtime import _configure_warnings


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark FetalCLIP per-frame scoring latency")
    p.add_argument(
        "--video_path",
        type=str,
        required=True,
        help="Input video path for benchmarking.",
    )
    p.add_argument("--fetalclip_config", type=str, default="model/FetalCLIP_config.json")
    p.add_argument("--fetalclip_weights", type=str, default="model/FetalCLIP_weights.pt")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--suppress_warnings", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--frame_stride", type=int, default=1)
    p.add_argument("--max_frames", type=int, default=0, help="0 means benchmark all frames.")
    return p.parse_args()


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def load_fetalclip_for_benchmark(args: argparse.Namespace, device: torch.device, log):
    with open(args.fetalclip_config, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    _MODEL_CONFIGS["FetalCLIP"] = cfg

    fetalclip_model, _, fetalclip_preprocess = open_clip.create_model_and_transforms(
        "FetalCLIP", pretrained=None
    )
    log(f"Loading FetalCLIP weights from {args.fetalclip_weights}...")
    checkpoint = torch.load(args.fetalclip_weights, map_location="cpu")
    state_dict = checkpoint
    if isinstance(checkpoint, dict):
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

    if not isinstance(state_dict, dict):
        raise RuntimeError(f"Unsupported FetalCLIP checkpoint format: {type(state_dict)}")

    if any(str(k).startswith("module.") for k in state_dict.keys()):
        state_dict = {str(k).removeprefix("module."): v for k, v in state_dict.items()}

    fetalclip_model.load_state_dict(state_dict, strict=True)
    fetalclip_tokenizer = open_clip.get_tokenizer("FetalCLIP")
    fetalclip_model.eval().to(device)
    for p in fetalclip_model.parameters():
        p.requires_grad = False

    prompt_text_features = build_prompt_text_features(
        fetalclip_model,
        fetalclip_tokenizer,
        device,
        [
            "fetal ultrasound, fetal eyes.",
            "fetal ultrasound, upper alveolar ridge.",
            "fetal ultrasound, intact hard palate.",
            "fetal ultrasound, lower alveolar ridge.",
            "fetal ultrasound, cleft hard palate.",
        ],
    )
    return fetalclip_model, fetalclip_preprocess, prompt_text_features


def main() -> None:
    args = parse_args()
    accelerator = Accelerator(cpu=bool(args.cpu))
    log = accelerator.print
    set_random_seed(int(args.seed))
    _configure_warnings(suppress=bool(args.suppress_warnings))
    device = accelerator.device

    video_path = Path(args.video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    fetalclip_model, fetalclip_preprocess, prompt_text_features = load_fetalclip_for_benchmark(args, device, log)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0.0:
        fps = 25.0

    # Warm up one frame if available, excluded from timing.
    ok, warm_frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError(f"Video has no readable frames: {video_path}")
    warm_pil = pil_from_bgr(warm_frame)
    _ = score_frame(
        fetalclip_model=fetalclip_model,
        fetalclip_preprocess=fetalclip_preprocess,
        prompt_text_features=prompt_text_features,
        device=device,
        image_pil=warm_pil,
    )
    _sync_if_cuda(device)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    processed = 0
    frame_idx = -1

    _sync_if_cuda(device)
    start = time.perf_counter()

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_idx += 1
        if int(args.frame_stride) > 1 and (frame_idx % int(args.frame_stride) != 0):
            continue
        processed += 1
        if int(args.max_frames) > 0 and processed > int(args.max_frames):
            break

        frame_pil = pil_from_bgr(frame_bgr)
        _ = score_frame(
            fetalclip_model=fetalclip_model,
            fetalclip_preprocess=fetalclip_preprocess,
            prompt_text_features=prompt_text_features,
            device=device,
            image_pil=frame_pil,
        )

    _sync_if_cuda(device)
    elapsed = time.perf_counter() - start
    cap.release()

    per_frame = elapsed / processed if processed > 0 else float("nan")

    log(f"[video] path={video_path}")
    log(f"[env] device={device} fps={fps:.3f} total_frames={total_frames}")
    log(f"[bench] processed_frames={processed}")
    log(f"[bench] total_clip_scoring_time_sec={elapsed:.4f}")
    log(f"[bench] avg_time_per_frame_sec={per_frame:.6f}")
    log(f"[bench] avg_time_per_frame_ms={per_frame * 1000.0:.3f}")


if __name__ == "__main__":
    main()
