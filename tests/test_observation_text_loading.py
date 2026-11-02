"""Verify public path-only text labels remain compatible with legacy inline labels.

Usage: python -m unittest discover -s tests -p test_observation_text_loading.py -v
"""
import tempfile
import unittest
from pathlib import Path

from utils.examination_dataset import _read_observation_text


class ObservationTextLoadingTest(unittest.TestCase):
    """Check compatibility using small isolated TXT inputs without model weights."""

    def test_inline_text_takes_precedence(self):
        """Historical inline labels must not require a present text file."""
        self.assertEqual(_read_observation_text({"main_text": "existing text"}, Path(".")), "existing text")

    def test_release_path_recreates_builder_text(self):
        """Path-only labels must use exactly the builder's line selection."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "label.txt").write_text("prompt\n\n description \n next step \nignored\n", encoding="utf-8-sig")
            self.assertEqual(_read_observation_text({"text_path": "label.txt"}, root), "description next step")

    def test_prompt_only_file_and_empty_file(self):
        """A prompt-only file is valid; an empty file must fail visibly."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "label.txt"
            path.write_text("prompt\n", encoding="utf-8")
            self.assertEqual(_read_observation_text({"text_path": "label.txt"}, root), "prompt")
            path.write_text("\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Empty observation text"):
                _read_observation_text({"text_path": "label.txt"}, root)


if __name__ == "__main__":
    unittest.main()
