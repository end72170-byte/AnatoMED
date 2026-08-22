"""Pure-state tests for Phase 5 incremental video inference."""

import unittest

import torch

from inference import merge_adjacent_same_label_events
from utils.anatopave_video import (
    IncrementalSlotCache,
    build_rule_based_summary,
    build_video_diagnosis,
    compact_summary_sha256,
)


class AnatoPAVEVideoStateTest(unittest.TestCase):
    @staticmethod
    def _update_hard_palate(
        cache: IncrementalSlotCache,
        *,
        label: str,
        frame_idx: int,
        score: float,
        summary_value: float,
        event_index: int,
    ) -> dict:
        summary = torch.full((4,), summary_value)
        cache.update(
            label=label,
            summary=summary,
            frame_idx=frame_idx,
            timestamp_sec=frame_idx / 25.0,
            prompt=label.lower(),
            clip_score=score,
            clip_margin=score / 10.0,
            class_scores={label: score},
        )
        event = {
            "event_index": event_index,
            "label": label,
            "slot": "HARD_PALATE",
            "frame_idx": frame_idx,
            "timestamp_sec": frame_idx / 25.0,
            "clip_score": score,
            "text": f"text-frame-{frame_idx}",
            "image_path": f"frame-{frame_idx}.png",
            "mask_path": f"frame-{frame_idx}-mask.png",
            "overlay_path": f"frame-{frame_idx}-overlay.png",
            "segment_start_frame": frame_idx - 3,
            "segment_observed_frames": 4,
            "segment_best_score": score,
            "compact_summary_sha256": compact_summary_sha256(summary),
        }
        cache.attach_event("HARD_PALATE", event)
        return event

    def test_pending_closes_as_missing_and_never_reads_missing_features(self) -> None:
        cache = IncrementalSlotCache(summary_dim=4)
        cache.update(
            label="EYES", summary=torch.ones(4), frame_idx=3, timestamp_sec=0.1,
            prompt="eyes", clip_score=0.4, clip_margin=0.1, class_scores={"EYES": 0.4},
        )
        summaries, available, main = cache.composer_inputs(
            main_slot="EYES", main_summary=torch.ones(4),
            device=torch.device("cpu"), dtype=torch.float32,
        )
        self.assertEqual(tuple(summaries.shape), (1, 4, 4))
        self.assertEqual(available.tolist(), [[True, False, False, False]])
        self.assertEqual(main.tolist(), [[True, False, False, False]])
        cache.close()
        self.assertEqual(cache.snapshot()["UPPER_ALVEOLAR"]["state"], "CLOSED_MISSING")
        summary = build_rule_based_summary(
            slot_cache=cache,
            diagnosis=build_video_diagnosis(
                events=[], frame_top1_counts={}, processed_frames=10,
                palate_cleft_ratio_thr=0.2,
            ),
        )
        self.assertEqual(summary["diagnose"], "insufficient_evidence")
        self.assertIsNone(summary["palate_cleft_ratio"])

    def test_hard_palate_diagnosis_requires_event(self) -> None:
        no_event = build_video_diagnosis(
            events=[], frame_top1_counts={"CLEFT_HARD_PALATE": 20},
            processed_frames=30, palate_cleft_ratio_thr=0.2,
        )
        self.assertFalse(no_event["hard_palate_available"])
        self.assertEqual(no_event["diagnose"], "insufficient_evidence")
        self.assertEqual(no_event["diagnosis_event_pattern"], "none")
        self.assertFalse(no_event["palate_cleft_ratio_used_for_diagnosis"])
        self.assertIsNone(no_event["palate_cleft_ratio"])

        intact_only = build_video_diagnosis(
            events=[{"label": "INTACT_HARD_PALATE"}],
            frame_top1_counts={"INTACT_HARD_PALATE": 80, "CLEFT_HARD_PALATE": 20},
            processed_frames=120, palate_cleft_ratio_thr=0.2,
        )
        self.assertTrue(intact_only["hard_palate_available"])
        self.assertEqual(intact_only["diagnosis_event_pattern"], "intact_only")
        self.assertEqual(intact_only["diagnose"], "normal")
        self.assertFalse(intact_only["palate_cleft_ratio_used_for_diagnosis"])
        self.assertAlmostEqual(intact_only["palate_cleft_ratio"], 0.2)

        cleft_only = build_video_diagnosis(
            events=[{"label": "CLEFT_HARD_PALATE"}],
            frame_top1_counts={"INTACT_HARD_PALATE": 95, "CLEFT_HARD_PALATE": 5},
            processed_frames=120, palate_cleft_ratio_thr=0.2,
        )
        self.assertEqual(cleft_only["diagnosis_event_pattern"], "cleft_only")
        self.assertEqual(cleft_only["diagnose"], "abnormal")
        self.assertFalse(cleft_only["palate_cleft_ratio_used_for_diagnosis"])
        self.assertAlmostEqual(cleft_only["palate_cleft_ratio"], 0.05)

        both_normal = build_video_diagnosis(
            events=[
                {"label": "INTACT_HARD_PALATE"},
                {"label": "CLEFT_HARD_PALATE"},
            ],
            frame_top1_counts={"INTACT_HARD_PALATE": 81, "CLEFT_HARD_PALATE": 19},
            processed_frames=120, palate_cleft_ratio_thr=0.2,
        )
        self.assertEqual(both_normal["diagnosis_event_pattern"], "both")
        self.assertEqual(both_normal["diagnose"], "normal")
        self.assertTrue(both_normal["palate_cleft_ratio_used_for_diagnosis"])

        both_abnormal = build_video_diagnosis(
            events=[
                {"label": "INTACT_HARD_PALATE"},
                {"label": "CLEFT_HARD_PALATE"},
            ],
            frame_top1_counts={"INTACT_HARD_PALATE": 80, "CLEFT_HARD_PALATE": 20},
            processed_frames=120, palate_cleft_ratio_thr=0.2,
        )
        self.assertEqual(both_abnormal["diagnose"], "abnormal")
        self.assertTrue(both_abnormal["palate_cleft_ratio_used_for_diagnosis"])

    def test_reconcile_same_slot_after_event_merge_is_atomic(self) -> None:
        cache = IncrementalSlotCache(summary_dim=4)
        events = [
            self._update_hard_palate(
                cache, label="CLEFT_HARD_PALATE", frame_idx=60,
                score=0.90, summary_value=1.0, event_index=0,
            ),
            self._update_hard_palate(
                cache, label="CLEFT_HARD_PALATE", frame_idx=70,
                score=0.80, summary_value=2.0, event_index=1,
            ),
            self._update_hard_palate(
                cache, label="INTACT_HARD_PALATE", frame_idx=201,
                score=0.70, summary_value=3.0, event_index=2,
            ),
        ]
        # The two adjacent cleft events collapse to frame 60. Before final
        # reconciliation, the online cache still correctly points to frame 201.
        self.assertEqual(cache.snapshot()["HARD_PALATE"]["frame_idx"], 201)
        merged = merge_adjacent_same_label_events(events, gap_frames=40)
        selected = cache.reconcile_events(merged, strategy="highest_clip_score")
        self.assertEqual(selected["HARD_PALATE"], ("CLEFT_HARD_PALATE", 60))

        record = cache.records["HARD_PALATE"]
        torch.testing.assert_close(record.summary, torch.ones(4))
        snapshot = cache.snapshot()["HARD_PALATE"]
        self.assertEqual(snapshot["label"], "CLEFT_HARD_PALATE")
        self.assertEqual(snapshot["frame_idx"], 60)
        self.assertEqual(snapshot["event_index"], 0)
        self.assertEqual(snapshot["event_text"], "text-frame-60")
        self.assertEqual(snapshot["event_image_path"], "frame-60.png")
        self.assertEqual(snapshot["event_mask_path"], "frame-60-mask.png")
        self.assertEqual(snapshot["class_scores"], {"CLEFT_HARD_PALATE": 0.9})
        self.assertEqual(snapshot["summary_dim"], 4)
        self.assertEqual(
            snapshot["summary_sha256"],
            next(event["compact_summary_sha256"] for event in events if event["frame_idx"] == 60),
        )

        cache.close()
        summary = build_rule_based_summary(
            slot_cache=cache,
            diagnosis=build_video_diagnosis(
                events=merged,
                frame_top1_counts={"CLEFT_HARD_PALATE": 2, "INTACT_HARD_PALATE": 1},
                processed_frames=10,
                palate_cleft_ratio_thr=0.2,
            ),
        )
        representative = summary["representative_events"]["HARD_PALATE"]
        self.assertEqual(representative["label"], "CLEFT_HARD_PALATE")
        self.assertEqual(representative["frame_idx"], 60)
        self.assertEqual(representative["event_text"], "text-frame-60")


if __name__ == "__main__":
    unittest.main()
