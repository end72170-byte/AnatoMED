"""Measure Composer overhead independently from image/VLM encoding."""

from __future__ import annotations

import argparse
import time

import torch

from models.anatopave import AvailabilityAwareComposer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--input_dim", type=int, default=1024)
    parser.add_argument("--composer_dim", type=int, default=512)
    args = parser.parse_args()
    device = torch.device(args.device)
    model = AvailabilityAwareComposer(
        input_dim=args.input_dim,
        composer_dim=args.composer_dim,
        num_slots=4,
        num_heads=8,
    ).eval().to(device)
    features = torch.randn(args.batch_size, 4, args.input_dim, device=device)
    available = torch.tensor([[True, True, True, True]], device=device).expand(args.batch_size, -1)
    main = torch.zeros(args.batch_size, 4, dtype=torch.bool, device=device)
    main[:, 0] = True

    with torch.inference_mode():
        for _ in range(args.warmup):
            model(features, available, main)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(args.iterations):
            model(features, available, main)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    print({
        "device": str(device),
        "batch_size": args.batch_size,
        "iterations": args.iterations,
        "composer_ms_per_batch": 1000.0 * elapsed / args.iterations,
        "composer_ms_per_examination": 1000.0 * elapsed / (args.iterations * args.batch_size),
    })


if __name__ == "__main__":
    main()
