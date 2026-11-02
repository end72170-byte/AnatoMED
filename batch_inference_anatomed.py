"""Fold-driven batch wrapper for incremental AnatoMED video inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from inference_anatomed import _load_bundle, run_video, set_random_seed
from utils.runtime import _configure_warnings
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold_dir", type=Path, required=True)
    parser.add_argument("--video_manifest", type=Path, required=True, help="Generated video manifest.")
    parser.add_argument("--videos_root", type=Path, default=Path("dataset/videos"))
    parser.add_argument("--out_root", type=Path, default=Path("runs/anatomed_video"))
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--fetalclip_config", default="")
    parser.add_argument("--fetalclip_weights", default="")
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gen_max_new_tokens", type=int, default=256)
    parser.add_argument("--gen_do_sample", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--max_videos", type=int, default=0, help="0 means all fold-manifest videos")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--suppress_warnings", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ema_alpha", type=float, default=0.5)
    parser.add_argument("--enter_threshold", type=float, default=0.20)
    parser.add_argument("--exit_threshold", type=float, default=0.15)
    parser.add_argument("--margin_threshold", type=float, default=0.02)
    parser.add_argument("--enter_streak", type=int, default=3)
    parser.add_argument("--exit_patience", type=int, default=3)
    parser.add_argument("--cooldown_frames", type=int, default=40)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--min_state_frames", type=int, default=4)
    parser.add_argument("--max_state_frames", type=int, default=36)
    parser.add_argument("--sharpness_weight", type=float, default=0.03)
    parser.add_argument("--margin_weight", type=float, default=0.40)
    parser.add_argument("--motion_weight", type=float, default=0.20)
    parser.add_argument("--palate_cleft_ratio_thr", type=float, default=0.20)
    parser.add_argument(
        "--slot_representative_strategy",
        choices=["highest_clip_score", "latest_stable_event"],
        default="highest_clip_score",
    )
    return parser.parse_args()


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _fold_cases(fold_dir: Path) -> set[str]:
    return {
        line.strip().split("/")[0]
        for line in (fold_dir / "val.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def main() -> None:
    args = parse_args()
    if args.suppress_warnings:
        _configure_warnings(suppress=True)
    set_random_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    fold_name = args.fold_dir.name
    out_dir = args.out_root / fold_name
    events_root = out_dir / "events"
    json_root = out_dir / "json_dir"
    events_root.mkdir(parents=True, exist_ok=True)
    json_root.mkdir(parents=True, exist_ok=True)
    case_ids = _fold_cases(args.fold_dir)
    manifest_rows = _read_jsonl(args.video_manifest)
    manifest_candidates = [
        row for row in manifest_rows
        if row.get("case_id") in case_ids
        and row.get("inference_candidate", True)
        and row.get("seniority", "high") == "high"
    ]
    by_case = {case: [] for case in case_ids}
    for row in manifest_candidates:
        by_case.setdefault(str(row["case_id"]), []).append(row)
    missing_cases = sorted(case for case, values in by_case.items() if not values)
    rows = manifest_candidates
    if args.max_videos > 0:
        rows = rows[: args.max_videos]
    print(f"[data] fold={fold_name} cases={len(case_ids)} videos={len(rows)} missing_cases={len(missing_cases)}")
    bundle = _load_bundle(args, device, print)
    results: List[Dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: (str(item["case_id"]), str(item.get("condition", "")))):
        video_value = Path(str(row["video_path"]))
        video_path = video_value if video_value.is_absolute() else Path.cwd() / video_value
        if not video_path.exists():
            video_path = args.videos_root / str(row["case_id"]) / str(row["condition"]) / "high" / video_value.name
        case = str(row["case_id"])
        condition = str(row.get("condition", "unknown"))
        seniority = str(row.get("seniority", "high"))
        save_dir = events_root / case / f"{condition}_{seniority}"
        out_json = json_root / f"{case}_{condition}_{seniority}.json"
        args.video_path = str(video_path)
        args.save_dir = str(save_dir)
        args.out_json = str(out_json)
        print(f"[case] {case} condition={condition} video={video_path}")
        output = run_video(args, bundle, device, print)
        output["manifest"] = row
        out_json.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        results.append({
            "case_id": case,
            "examination_id": row.get("examination_id"),
            "condition": condition,
            "seniority": seniority,
            "video_path": str(video_path),
            "json_path": str(out_json),
            "events_dir": str(save_dir),
            "hard_palate_available": output["hard_palate_available"],
            "diagnose": output["diagnose"],
            "insufficient_evidence_reason": output["insufficient_evidence_reason"],
            "diagnosis_rule": output["diagnosis_rule"],
            "diagnosis_event_pattern": output["diagnosis_event_pattern"],
            "palate_cleft_ratio_used_for_diagnosis": output[
                "palate_cleft_ratio_used_for_diagnosis"
            ],
            "palate_cleft_ratio": output["palate_cleft_ratio"],
            "has_intact_hard_palate_event": output["has_intact_hard_palate_event"],
            "has_cleft_hard_palate_event": output["has_cleft_hard_palate_event"],
            "runtime": output["runtime"],
        })
    summary = {
        "format": "anatomed-phase5-fold-summary-v1",
        "fold": fold_name,
        "fold_dir": str(args.fold_dir.resolve()),
        "video_manifest": str(args.video_manifest.resolve()),
        "videos_root": str(args.videos_root.resolve()),
        "checkpoint_path": str(Path(args.checkpoint_path).resolve()),
        "cases_in_val": sorted(case_ids),
        "missing_cases": missing_cases,
        "num_videos": len(results),
        "results": results,
    }
    summary_path = out_dir / f"{fold_name}.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved fold summary json: {summary_path}")
    print(f"[done] fold={fold_name} videos={len(results)} missing_cases={len(missing_cases)}")


if __name__ == "__main__":
    main()
