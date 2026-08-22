"""Phase 3+4 shared-grounding and evidence-injection tests."""

import unittest

import torch
from torch import nn

from models.anatopave import AnatoPAVEHeads
from utils.anatopave_cache import CompactSummaryCache, assemble_slot_summaries
from utils.evidence_injection import EvidenceEmbeddingInjector


class TinyLanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(20, 8)
        self.head = nn.Linear(8, 20)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids):
        return self.head(self.embedding(input_ids))

    def generate(self, input_ids):
        return self.forward(input_ids)


class SharedGroundingTest(unittest.TestCase):
    def test_teacher_and_generate_share_injector(self) -> None:
        model = TinyLanguageModel()
        ids = torch.tensor([[1, 7, 2]])
        evidence = torch.randn(1, 8, requires_grad=True)
        injector = EvidenceEmbeddingInjector(7)
        teacher = injector.forward(model, evidence, {"input_ids": ids})
        generated = injector.generate(model, evidence, {"input_ids": ids})
        torch.testing.assert_close(teacher, generated)
        teacher.sum().backward()
        self.assertIsNotNone(evidence.grad)

    def test_missing_slot_never_queries_cache(self) -> None:
        cache = CompactSummaryCache({"aux": torch.ones(6)})
        output = assemble_slot_summaries(
            main_summary=torch.full((1, 6), 2.0),
            slot_available=torch.tensor([[True, True, False, False]]),
            slot_is_main=torch.tensor([[True, False, False, False]]),
            slot_image_paths=[["main", "aux", None, "must-not-read"]],
            cache=cache,
        )
        self.assertEqual(cache.requested_paths, ["aux"])
        torch.testing.assert_close(output[0, 0], torch.full((6,), 2.0))
        torch.testing.assert_close(output[0, 2:], torch.zeros(2, 6))

    def test_composer_off_recovers_strict_baseline_and_losses_reach_composer(self) -> None:
        torch.manual_seed(3)
        heads = AnatoPAVEHeads(
            summary_dim=12, dense_feature_dim=16, vlm_dim=20,
            composer_dim=16, grounding_dim=16, num_heads=4,
        )
        features = {f"z{i}": torch.randn(1, 16, 2, 2) for i in range(1, 5)}
        baseline = heads.decoder.base_decoder(features)
        disabled = heads.decoder(features, None, use_composer=False)
        torch.testing.assert_close(baseline, disabled, rtol=0, atol=0)

        composed = heads.compose(
            torch.randn(1, 4, 12),
            torch.tensor([[True, True, False, True]]),
            torch.tensor([[True, False, False, False]]),
        )
        language_loss = composed["language_embedding"].square().mean()
        seg_logits = heads.segment(
            features=features,
            seg_hidden=torch.randn(1, 1, 20),
            segmentation_context=composed["segmentation_context"],
            main_slot_index=torch.tensor([0]),
        )
        segmentation_loss = seg_logits.square().mean()
        params = tuple(heads.composer.parameters())
        lm_grad = torch.autograd.grad(language_loss, params, retain_graph=True, allow_unused=True)
        seg_grad = torch.autograd.grad(segmentation_loss, params, allow_unused=True)
        self.assertGreater(sum(float(g.abs().sum()) for g in lm_grad if g is not None), 0.0)
        self.assertGreater(sum(float(g.abs().sum()) for g in seg_grad if g is not None), 0.0)
        self.assertFalse(any(isinstance(module, nn.BatchNorm2d) for module in heads.decoder.modules()))


if __name__ == "__main__":
    unittest.main()
