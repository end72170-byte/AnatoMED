"""Availability-aware multi-view Composer building blocks for ANATOPAVE."""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from models.fetalclip_unetr import FetalCLIPUNETRDecoder
from models.seg_unet import assert_no_batch_norm


class ViewFeaturePooler(nn.Module):
    """Pool one view's spatial/token features without destroying the caller's dense features."""

    def __init__(self, mode: str = "mean") -> None:
        super().__init__()
        if mode != "mean":
            raise ValueError("Only deterministic mean pooling is supported in the first Composer version")
        self.mode = mode

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim == 3:
            # [B, views, D] is already pooled.
            return features
        if features.ndim == 4:
            # [B, views, tokens, D].
            return features.mean(dim=2)
        if features.ndim == 5:
            # [B, views, C, H, W].
            return features.mean(dim=(-1, -2))
        raise ValueError(
            "Expected pooled [B,S,D], token [B,S,N,D], or dense [B,S,C,H,W] features; "
            f"got shape {tuple(features.shape)}"
        )


class PrimitiveIdentityEmbedding(nn.Module):
    """Learned anatomy identity q_v for the fixed four Composer slots."""

    def __init__(self, num_slots: int, embedding_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_slots, embedding_dim)

    def forward(self, batch_size: int, *, device: torch.device) -> torch.Tensor:
        slot_ids = torch.arange(self.embedding.num_embeddings, device=device)
        return self.embedding(slot_ids).unsqueeze(0).expand(batch_size, -1, -1)


class UnavailableViewEmbedding(nn.Module):
    """Learned u_v used when a slot is genuinely unavailable."""

    def __init__(self, num_slots: int, embedding_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Parameter(torch.empty(num_slots, embedding_dim))
        nn.init.normal_(self.embedding, mean=0.0, std=0.02)

    def forward(self, batch_size: int, *, device: torch.device) -> torch.Tensor:
        return self.embedding.to(device).unsqueeze(0).expand(batch_size, -1, -1)


class ExaminationAttentionPooler(nn.Module):
    """Attention pool that keeps one output vector for the whole examination."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, 1),
        )

    def forward(self, slot_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.score(slot_tokens).squeeze(-1)
        weights = torch.softmax(logits, dim=-1)
        context = torch.sum(slot_tokens * weights.unsqueeze(-1), dim=1)
        return context, weights


class AvailabilityAwareComposer(nn.Module):
    """Compose four role-preserving slot summaries into one examination state.

    The module receives only compact per-slot summaries.  The main view's dense
    features remain outside this module for the segmentation decoder.
    """

    def __init__(
        self,
        input_dim: int,
        composer_dim: int = 512,
        *,
        num_slots: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_slots < 1:
            raise ValueError("num_slots must be positive")
        if composer_dim % num_heads != 0:
            raise ValueError("composer_dim must be divisible by num_heads")
        self.input_dim = int(input_dim)
        self.composer_dim = int(composer_dim)
        self.num_slots = int(num_slots)
        self.pooler = ViewFeaturePooler()
        self.feature_projection = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, self.composer_dim),
        )
        self.slot_identity = PrimitiveIdentityEmbedding(self.num_slots, self.composer_dim)
        self.unavailable = UnavailableViewEmbedding(self.num_slots, self.composer_dim)
        self.main_identity = nn.Parameter(torch.zeros(self.composer_dim))
        nn.init.normal_(self.main_identity, mean=0.0, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.composer_dim,
            nhead=num_heads,
            dim_feedforward=4 * self.composer_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.self_attention = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.output_norm = nn.LayerNorm(self.composer_dim)
        self.attention_pooler = ExaminationAttentionPooler(self.composer_dim)

    def forward(
        self,
        slot_features: torch.Tensor,
        slot_available: torch.Tensor,
        slot_is_main: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        pooled = self.pooler(slot_features)
        if pooled.ndim != 3:
            raise AssertionError(f"Pooled slot features must be [B,S,D], got {tuple(pooled.shape)}")
        batch_size, num_slots, input_dim = pooled.shape
        if num_slots != self.num_slots or input_dim != self.input_dim:
            raise ValueError(
                f"Expected [B,{self.num_slots},{self.input_dim}] slot features, "
                f"got {tuple(pooled.shape)}"
            )
        available = slot_available.to(device=pooled.device, dtype=torch.bool)
        if tuple(available.shape) != (batch_size, num_slots):
            raise ValueError(
                f"slot_available must be [B,S]=({batch_size},{num_slots}), got {tuple(available.shape)}"
            )
        if not torch.all(available.any(dim=1)):
            raise ValueError("Each examination must have at least one available slot")

        projected = self.feature_projection(pooled)
        identity = self.slot_identity(batch_size, device=pooled.device)
        missing = self.unavailable(batch_size, device=pooled.device)
        slot_tokens = torch.where(available.unsqueeze(-1), projected, missing) + identity

        if slot_is_main is not None:
            main = slot_is_main.to(device=pooled.device, dtype=torch.bool)
            if tuple(main.shape) != (batch_size, num_slots):
                raise ValueError(
                    f"slot_is_main must be [B,S]=({batch_size},{num_slots}), got {tuple(main.shape)}"
                )
            if not torch.all(main.sum(dim=1) == 1):
                raise ValueError("Each examination must have exactly one main slot")
            slot_tokens = slot_tokens + main.unsqueeze(-1).to(slot_tokens.dtype) * self.main_identity

        slot_tokens = self.output_norm(self.self_attention(slot_tokens))
        context, attention_weights = self.attention_pooler(slot_tokens)
        return {
            "context": context,
            "slot_tokens": slot_tokens,
            "attention_weights": attention_weights,
            "slot_available": available,
        }


class LanguageProjection(nn.Module):
    """Project one examination context to one VLM evidence embedding."""

    def __init__(self, composer_dim: int, vlm_dim: int) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.LayerNorm(composer_dim),
            nn.Linear(composer_dim, vlm_dim),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return self.projection(context).unsqueeze(1)


class SegmentationProjection(nn.Module):
    """Independent Composer projection used only by shared grounding."""

    def __init__(self, composer_dim: int, grounding_dim: int) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.LayerNorm(composer_dim),
            nn.Linear(composer_dim, grounding_dim),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return self.projection(context)


class SharedGrounding(nn.Module):
    """Use ``<seg> + context + q_v`` as query over main-view dense features."""

    def __init__(
        self,
        *,
        vlm_dim: int,
        image_dim: int,
        grounding_dim: int = 512,
        num_slots: int = 4,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        if grounding_dim % num_heads != 0:
            raise ValueError("grounding_dim must be divisible by num_heads")
        self.seg_projection = nn.Sequential(nn.LayerNorm(vlm_dim), nn.Linear(vlm_dim, grounding_dim))
        self.slot_identity = nn.Embedding(num_slots, grounding_dim)
        self.query_norm = nn.LayerNorm(grounding_dim)
        self.image_norm = nn.LayerNorm(image_dim)
        self.image_key = nn.Linear(image_dim, grounding_dim)
        self.image_value = nn.Linear(image_dim, grounding_dim)
        self.attention = nn.MultiheadAttention(grounding_dim, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(grounding_dim),
            nn.Linear(grounding_dim, 2 * grounding_dim),
            nn.GELU(),
            nn.Linear(2 * grounding_dim, grounding_dim),
        )
        self.output_norm = nn.LayerNorm(grounding_dim)

    def forward(
        self,
        seg_hidden: torch.Tensor,
        context_seg: torch.Tensor,
        main_slot_index: torch.Tensor,
        main_dense_feature: torch.Tensor,
    ) -> torch.Tensor:
        if seg_hidden.ndim == 3:
            if seg_hidden.shape[1] != 1:
                raise ValueError("seg_hidden must contain exactly one <seg> state")
            seg_hidden = seg_hidden[:, 0]
        if seg_hidden.ndim != 2 or context_seg.ndim != 2:
            raise ValueError("seg_hidden and context_seg must be rank-2 after squeezing")
        slot_identity = self.slot_identity(main_slot_index.to(dtype=torch.long))
        query = self.seg_projection(seg_hidden) + context_seg + slot_identity
        query = self.query_norm(query).unsqueeze(1)
        image_tokens = main_dense_feature.flatten(2).transpose(1, 2)
        image_tokens = self.image_norm(image_tokens)
        attended, _ = self.attention(
            query, self.image_key(image_tokens), self.image_value(image_tokens), need_weights=False
        )
        grounded = query + attended
        grounded = grounded + self.ffn(grounded)
        return self.output_norm(grounded)


class SpatialConditioner(nn.Module):
    """Spatial decoder features remain Query; grounding is Key/Value."""

    def __init__(self, spatial_dim: int, grounding_dim: int, num_heads: int) -> None:
        super().__init__()
        self.spatial_norm = nn.LayerNorm(spatial_dim)
        self.condition_norm = nn.LayerNorm(grounding_dim)
        self.key = nn.Linear(grounding_dim, spatial_dim)
        self.value = nn.Linear(grounding_dim, spatial_dim)
        self.attention = nn.MultiheadAttention(spatial_dim, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(spatial_dim),
            nn.Linear(spatial_dim, 2 * spatial_dim),
            nn.GELU(),
            nn.Linear(2 * spatial_dim, spatial_dim),
        )
        # Small non-zero gates preserve initial segmentation while allowing an
        # immediate smoke-test gradient from mask loss back to Composer.
        self.attention_gate = nn.Parameter(torch.tensor(1e-3))
        self.ffn_gate = nn.Parameter(torch.tensor(1e-3))

    def forward(self, spatial: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = spatial.shape
        tokens = spatial.flatten(2).transpose(1, 2)
        normalized_condition = self.condition_norm(condition)
        attended, _ = self.attention(
            self.spatial_norm(tokens),
            self.key(normalized_condition),
            self.value(normalized_condition),
            need_weights=False,
        )
        tokens = tokens + self.attention_gate * attended
        tokens = tokens + self.ffn_gate * self.ffn(tokens)
        return tokens.transpose(1, 2).reshape(batch, channels, height, width)


class ConditionedFetalCLIPUNETRDecoder(nn.Module):
    """Strictly initialized baseline decoder plus residual shared grounding."""

    def __init__(
        self,
        feature_width: int = 1024,
        grounding_dim: int = 512,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        self.base_decoder = FetalCLIPUNETRDecoder(feature_width=feature_width)
        self.conditioners = nn.ModuleList(
            [
                SpatialConditioner(512, grounding_dim, num_heads),
                SpatialConditioner(256, grounding_dim, num_heads),
                SpatialConditioner(128, grounding_dim, num_heads),
                SpatialConditioner(64, grounding_dim, num_heads),
            ]
        )
        assert_no_batch_norm(self)

    def load_segmentation_pretrain(
        self, checkpoint: Mapping[str, object]
    ) -> None:
        state = checkpoint.get("model")
        if not isinstance(state, Mapping):
            raise KeyError("Segmentation pretrain checkpoint has no model state")
        self.base_decoder.load_state_dict(state, strict=True)

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        grounding: Optional[torch.Tensor],
        *,
        out_hw: Tuple[int, int] = (224, 224),
        use_composer: bool = True,
    ) -> torch.Tensor:
        if not use_composer:
            return self.base_decoder(features, out_hw=out_hw)
        if grounding is None:
            raise ValueError("grounding is required when Composer conditioning is enabled")
        z1, z2, z3, z4 = (features[f"z{i}"] for i in range(1, 5))
        x = self.conditioners[0](self.base_decoder.up1(z4), grounding)
        z3 = F.interpolate(z3, size=x.shape[-2:], mode="bilinear", align_corners=False)
        x = self.conditioners[1](self.base_decoder.up2(torch.cat([x, z3], dim=1)), grounding)
        z2 = F.interpolate(z2, size=x.shape[-2:], mode="bilinear", align_corners=False)
        x = self.conditioners[2](self.base_decoder.up3(torch.cat([x, z2], dim=1)), grounding)
        z1 = F.interpolate(z1, size=x.shape[-2:], mode="bilinear", align_corners=False)
        x = self.conditioners[3](self.base_decoder.up4(torch.cat([x, z1], dim=1)), grounding)
        x = self.base_decoder.final(x)
        if x.shape[-2:] != out_hw:
            x = F.interpolate(x, size=out_hw, mode="bilinear", align_corners=False)
        return x


class AnatoPAVEHeads(nn.Module):
    """Trainable Composer, projections, shared grounding, and decoder."""

    def __init__(
        self,
        *,
        summary_dim: int,
        dense_feature_dim: int,
        vlm_dim: int,
        composer_dim: int = 512,
        grounding_dim: int = 512,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        self.composer = AvailabilityAwareComposer(
            input_dim=summary_dim, composer_dim=composer_dim, num_slots=4, num_heads=num_heads
        )
        self.language_projection = LanguageProjection(composer_dim, vlm_dim)
        self.segmentation_projection = SegmentationProjection(composer_dim, grounding_dim)
        self.grounding = SharedGrounding(
            vlm_dim=vlm_dim,
            image_dim=dense_feature_dim,
            grounding_dim=grounding_dim,
            num_slots=4,
            num_heads=num_heads,
        )
        self.decoder = ConditionedFetalCLIPUNETRDecoder(
            feature_width=dense_feature_dim, grounding_dim=grounding_dim, num_heads=num_heads
        )
        assert_no_batch_norm(self.decoder)

    def compose(
        self,
        slot_summaries: torch.Tensor,
        slot_available: torch.Tensor,
        slot_is_main: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        output = self.composer(slot_summaries, slot_available, slot_is_main)
        context = output["context"]
        output["language_embedding"] = self.language_projection(context)
        output["segmentation_context"] = self.segmentation_projection(context)
        return output

    def segment(
        self,
        *,
        features: Dict[str, torch.Tensor],
        seg_hidden: torch.Tensor,
        segmentation_context: torch.Tensor,
        main_slot_index: torch.Tensor,
        use_composer: bool = True,
    ) -> torch.Tensor:
        if not use_composer:
            return self.decoder(features, None, use_composer=False)
        grounded = self.grounding(
            seg_hidden, segmentation_context, main_slot_index, features["z4"]
        )
        return self.decoder(features, grounded, use_composer=True)
