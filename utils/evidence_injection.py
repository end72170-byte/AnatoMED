"""Shared Composer-context injection for VLM training and generation."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator

import torch


class EvidenceEmbeddingInjector:
    """Replace ``<evidence>`` word embeddings without changing image tokens."""

    def __init__(self, evidence_token_id: int) -> None:
        self.evidence_token_id = int(evidence_token_id)

    @staticmethod
    def _expand_context(context: torch.Tensor, batch_size: int) -> torch.Tensor:
        if context.ndim == 3 and context.shape[1] == 1:
            context = context[:, 0]
        if context.ndim != 2:
            raise ValueError(f"Evidence embedding must be [B,D] or [B,1,D], got {tuple(context.shape)}")
        if context.shape[0] == batch_size:
            return context
        if batch_size % context.shape[0] != 0:
            raise ValueError(f"Cannot expand evidence batch {context.shape[0]} to {batch_size}")
        return context.repeat_interleave(batch_size // context.shape[0], dim=0)

    @contextmanager
    def injected(
        self,
        model: Any,
        evidence_embedding: torch.Tensor,
    ) -> Iterator[None]:
        embedding_layer = model.get_input_embeddings()

        def replace(_module, inputs, output):
            token_ids = inputs[0]
            mask = token_ids.eq(self.evidence_token_id)
            if not bool(mask.any()):
                return output
            per_row = mask.sum(dim=1)
            if not bool(torch.all(per_row == 1)):
                raise AssertionError(
                    f"Expected exactly one <evidence> token per active row, got {per_row.tolist()}"
                )
            context = self._expand_context(evidence_embedding, output.shape[0]).to(
                device=output.device, dtype=output.dtype
            )
            return torch.where(mask.unsqueeze(-1), context.unsqueeze(1), output)

        handle = embedding_layer.register_forward_hook(replace)
        try:
            yield
        finally:
            handle.remove()

    def run(
        self,
        model: Any,
        evidence_embedding: torch.Tensor,
        operation: Callable[..., Any],
        **kwargs: Any,
    ) -> Any:
        """Single injection path shared by forward and generate."""
        with self.injected(model, evidence_embedding):
            return operation(**kwargs)

    def forward(
        self, model: Any, evidence_embedding: torch.Tensor, model_kwargs: Dict[str, Any]
    ) -> Any:
        return self.run(model, evidence_embedding, model, **model_kwargs)

    def generate(
        self, model: Any, evidence_embedding: torch.Tensor, model_kwargs: Dict[str, Any], **generation_kwargs: Any
    ) -> Any:
        return self.run(
            model, evidence_embedding, model.generate, **model_kwargs, **generation_kwargs
        )
