"""Regression tests for annotation text normalization."""

import unittest

from utils.stage2_dataset import normalize_text_line


class DatasetTextNormalizationTest(unittest.TestCase):
    def test_removes_utf8_bom_from_prompt(self) -> None:
        self.assertEqual(
            normalize_text_line("\ufefffetal ultrasound, intact hard palate."),
            "fetal ultrasound, intact hard palate.",
        )

    def test_preserves_regular_annotation_text(self) -> None:
        self.assertEqual(normalize_text_line("  fetal ultrasound.  "), "fetal ultrasound.")


if __name__ == "__main__":
    unittest.main()
