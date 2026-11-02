"""Shape, missing-slot, gradient, and restore tests for the stage-3 Composer."""

import tempfile
import unittest
from pathlib import Path

import torch

from models.anatomed import AvailabilityAwareComposer


class AvailabilityAwareComposerTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.model = AvailabilityAwareComposer(
            input_dim=16,
            composer_dim=32,
            num_slots=4,
            num_heads=4,
            dropout=0.0,
        )
        self.features = torch.randn(2, 4, 16)
        self.available = torch.tensor([[True, True, False, True], [True, False, False, False]])
        self.main = torch.tensor([[False, True, False, False], [True, False, False, False]])

    def test_full_and_missing_slots_keep_contract(self) -> None:
        output = self.model(self.features, self.available, self.main)
        self.assertEqual(tuple(output["context"].shape), (2, 32))
        self.assertEqual(tuple(output["slot_tokens"].shape), (2, 4, 32))
        self.assertEqual(tuple(output["attention_weights"].shape), (2, 4))
        torch.testing.assert_close(output["attention_weights"].sum(dim=1), torch.ones(2))

    def test_unavailable_feature_values_do_not_change_output(self) -> None:
        first = self.model(self.features, self.available, self.main)
        changed = self.features.clone()
        changed[0, 2] = torch.randn(16) * 1000.0
        changed[1, 1:] = torch.randn(3, 16) * 1000.0
        second = self.model(changed, self.available, self.main)
        torch.testing.assert_close(first["context"], second["context"])
        torch.testing.assert_close(first["slot_tokens"], second["slot_tokens"])

    def test_checkpoint_restore_and_gradient(self) -> None:
        output = self.model(self.features, self.available, self.main)
        output["context"].square().mean().backward()
        self.assertTrue(any(parameter.grad is not None for parameter in self.model.parameters()))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "composer.pt"
            torch.save(self.model.state_dict(), path)
            restored = AvailabilityAwareComposer(
                input_dim=16,
                composer_dim=32,
                num_slots=4,
                num_heads=4,
                dropout=0.0,
            )
            restored.load_state_dict(torch.load(path, weights_only=True))
            self.model.eval()
            restored.eval()
            expected = self.model(self.features, self.available, self.main)
            actual = restored(self.features, self.available, self.main)
            torch.testing.assert_close(expected["context"], actual["context"])


if __name__ == "__main__":
    unittest.main()
