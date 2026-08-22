"""Regression tests for route-aware FetalCLIP hard-palate metrics."""

import unittest

from finetune_fetalclip_keyframes import (
    DEFAULT_KEYFRAME_PROMPTS,
    HARD_PALATE_CLEFT_PROMPT,
    HARD_PALATE_NORMAL_PROMPT,
    _compute_hard_palate_metrics,
)


def _prompt_index(prompt: str) -> int:
    return DEFAULT_KEYFRAME_PROMPTS.index(prompt)


class HardPalateMetricsTest(unittest.TestCase):
    def test_intact_predictions_routed_to_other_anatomies_are_unresolved(self) -> None:
        intact = _prompt_index(HARD_PALATE_NORMAL_PROMPT)
        non_palate_predictions = [
            _prompt_index("fetal ultrasound, fetal eyes."),
            _prompt_index("fetal ultrasound, upper alveolar ridge."),
            _prompt_index("fetal ultrasound, lower alveolar ridge."),
        ]

        metrics = _compute_hard_palate_metrics(
            non_palate_predictions,
            [intact] * len(non_palate_predictions),
            DEFAULT_KEYFRAME_PROMPTS,
        )

        self.assertEqual(metrics["routing"], {
            "n_gt_hard_palate": 3,
            "n_routed_hard_palate": 0,
            "n_route_missed": 3,
            "hard_palate_routing_recall": 0.0,
        })
        self.assertEqual(metrics["conditional_binary"]["n_samples"], 0)
        self.assertEqual(metrics["conditional_binary"]["confusion"], {
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "tp": 0,
        })
        self.assertEqual(metrics["end_to_end"], {
            "n_gt_hard_palate": 3,
            "correct_state": 0,
            "wrong_state": 0,
            "unresolved_due_to_route_miss": 3,
            "resolved_correct_rate": 0.0,
        })

    def test_conditional_binary_excludes_route_misses(self) -> None:
        intact = _prompt_index(HARD_PALATE_NORMAL_PROMPT)
        cleft = _prompt_index(HARD_PALATE_CLEFT_PROMPT)
        upper = _prompt_index("fetal ultrasound, upper alveolar ridge.")
        preds = [intact] * 11 + [upper, intact, cleft, cleft]
        gts = [intact] * 12 + [cleft, cleft, cleft]

        metrics = _compute_hard_palate_metrics(preds, gts, DEFAULT_KEYFRAME_PROMPTS)

        self.assertEqual(metrics["routing"]["n_gt_hard_palate"], 15)
        self.assertEqual(metrics["routing"]["n_routed_hard_palate"], 14)
        self.assertEqual(metrics["routing"]["n_route_missed"], 1)
        self.assertEqual(metrics["routing"]["hard_palate_routing_recall"], 14 / 15)
        self.assertEqual(metrics["conditional_binary"]["n_samples"], 14)
        self.assertEqual(metrics["conditional_binary"]["confusion"], {
            "tn": 11,
            "fp": 0,
            "fn": 1,
            "tp": 2,
        })
        self.assertEqual(metrics["end_to_end"], {
            "n_gt_hard_palate": 15,
            "correct_state": 13,
            "wrong_state": 1,
            "unresolved_due_to_route_miss": 1,
            "resolved_correct_rate": 13 / 15,
        })


if __name__ == "__main__":
    unittest.main()
