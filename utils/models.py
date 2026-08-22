from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None

# -----------------------------------------------------------------
# FetalCLIP multi-level feature extractor (hook-based, following
# the official FetalCLIP segmentation approach).
# -----------------------------------------------------------------

class FetalCLIPSegFeatureExtractor(nn.Module):
    """Extract multi-level features from FetalCLIP visual encoder.

    Registers forward hooks at 4 evenly-spaced transformer blocks so that
    intermediate representations can be reshaped to spatial grids and fed
    into a UNETR-style decoder with skip connections.

    Returns
    -------
    features : Dict[str, Tensor]
        Keys ``z1`` .. ``z4``, each ``[B, C, H, W]``.
    pooled : Tensor
        Global (CLS) output ``[B, D]`` from the visual encoder.
    """

    def __init__(self, visual_encoder: nn.Module) -> None:
        super().__init__()
        self.visual = visual_encoder
        self.hooks: List[Any] = []
        self.intermediate_outputs: Dict[str, torch.Tensor] = {}

        self.transformer = visual_encoder.transformer
        self.width: int = self.transformer.width  # 1024 for FetalCLIP
        self.grid_size = visual_encoder.grid_size  # (16, 16) for 224 / patch14

        self._register_hooks()

    def _register_hooks(self) -> None:
        n_blocks = len(self.transformer.resblocks)  # 24 for FetalCLIP
        self.layer_indices = [
            n_blocks // 4 - 1,
            n_blocks // 2 - 1,
            3 * n_blocks // 4 - 1,
            n_blocks - 1,
        ]
        for layer_idx in self.layer_indices:
            layer = self.transformer.resblocks[layer_idx]
            hook = layer.register_forward_hook(self._make_hook(f"layer_{layer_idx}"))
            self.hooks.append(hook)

    def _make_hook(self, name: str):
        def hook(module, input, output):
            self.intermediate_outputs[name] = output
        return hook

    def forward(self, x: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        self.intermediate_outputs = {}

        pooled = self.visual(x)  # triggers hooks; returns CLS/global feature [B, D]

        features: Dict[str, torch.Tensor] = {}
        for i, layer_idx in enumerate(self.layer_indices):
            feat = self.intermediate_outputs[f"layer_{layer_idx}"]  # [B, seq, C]
            feat = feat[:, 1:, :]  # drop CLS token -> [B, num_patches, C]
            feat = feat.permute(0, 2, 1)  # [B, C, N]
            B, C, N = feat.shape
            H = W = int(math.sqrt(N))
            feat = feat.view(B, C, H, W)
            features[f"z{i + 1}"] = feat

        return features, pooled

    def remove_hooks(self) -> None:
        for h in self.hooks:
            h.remove()
        self.hooks = []


# -----------------------------------------------------------------
# Cross-attention conditioner (LDM / Stable-Diffusion style).
# Spatial decoder features (query) attend to VLM hidden-state tokens
# (key/value), so VLM gradients flow through the segmentation path.
# -----------------------------------------------------------------

class CrossAttentionConditioner(nn.Module):
    """LDM-style conditioner: Spatial image features (Q) attend to enriched VLM tokens (K,V).
    This preserves spatial resolution while injecting text-guided attention.
    
    Input
    -----
    x_spatial : [B, C, H, W]  (decoder feature map)
    cond      : [B, N, D_cond] (VLM hidden-state tokens, **with grad**)

    Output
    ------
    x_out : [B, C, H, W]  (same shape, residual-connected)
    """

    def __init__(self, d_spatial: int, d_cond: int, num_heads: int = 8) -> None:
        super().__init__()
        # Image is Q, VLM is K,V
        self.norm_q = nn.LayerNorm(d_spatial)
        self.norm_kv = nn.LayerNorm(d_cond)
        self.q_proj = nn.Linear(d_spatial, d_spatial)
        self.k_proj = nn.Linear(d_cond, d_spatial)
        self.v_proj = nn.Linear(d_cond, d_spatial)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_spatial, num_heads=num_heads, batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_spatial),
            nn.Linear(d_spatial, d_spatial * 2),
            nn.GELU(),
            nn.Linear(d_spatial * 2, d_spatial),
        )

    def forward(self, x_spatial: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x_spatial.shape
        x = x_spatial.flatten(2).permute(0, 2, 1)  # [B, H*W, C] (Image as Query)
        
        # q is spatial, k/v is text
        q = self.q_proj(self.norm_q(x))                  # [B, H*W, d_spatial]
        k = self.k_proj(self.norm_kv(cond))              # [B, N, d_spatial]
        v = self.v_proj(self.norm_kv(cond))              # [B, N, d_spatial]
        attn_out, _ = self.attn(q, k, v, need_weights=False)  # [B, H*W, d_spatial]

        # Element-wise addition strictly preserves spatial mapping
        x = x + attn_out  # residual connection
        x = x + self.ffn(x)

        return x.permute(0, 2, 1).view(B, C, H, W)


class VLMImagePreFusion(nn.Module):
    """Pre-fusion: d_vlm* = d_vlm + CrossAttn(Q=d_vlm, K=F_img, V=F_img).

    Enriches VLM hidden states with FetalCLIP last-layer spatial features
    before LDM-style injection into the UNETR decoder.
    """

    def __init__(self, d_vlm: int, d_img: int, num_heads: int = 8) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(d_vlm)
        self.norm_kv = nn.LayerNorm(d_img)
        self.q_proj = nn.Linear(d_vlm, d_vlm)
        self.k_proj = nn.Linear(d_img, d_vlm)
        self.v_proj = nn.Linear(d_img, d_vlm)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_vlm, num_heads=num_heads, batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_vlm),
            nn.Linear(d_vlm, d_vlm * 2),
            nn.GELU(),
            nn.Linear(d_vlm * 2, d_vlm),
        )

    def forward(self, vlm_tokens: torch.Tensor, img_tokens: torch.Tensor) -> torch.Tensor:
        """
        vlm_tokens : [B, T, D_vlm]  (or [B, 1, D_vlm] for seg-token mode)
        img_tokens : [B, N, D_img]   (flattened last-layer FetalCLIP spatial features)
        Returns    : [B, T, D_vlm]   (same shape, residual-connected)
        """
        q = self.q_proj(self.norm_q(vlm_tokens))
        k = self.k_proj(self.norm_kv(img_tokens))
        v = self.v_proj(self.norm_kv(img_tokens))
        attn_out, _ = self.attn(q, k, v, need_weights=False)
        out = vlm_tokens + attn_out          # residual
        # out = attn_out                       # identity
        out = out + self.ffn(out)            # FFN residual
        return out


# -----------------------------------------------------------------
# VLM-conditioned UNETR segmentation decoder.
#
# Architecture (LDM-style conditioning):
#   - Primary path: FetalCLIP z1-z4 spatial features → UNETR skip-conn
#   - Conditioning: VLM fusion-tower hidden states (last_hidden_state)
#     injected via cross-attention at every decoder level, so seg_loss
#     gradient flows back through cross-attn → VLM hidden → LoRA.
# -----------------------------------------------------------------

class VLMConditionedSegDecoder(nn.Module):
    """UNETR decoder with LDM-style VLM cross-attention conditioning.

    z4 (16×16) → up1 (32) → xattn1 → +z3 → up2 (64) → xattn2
    → +z2 → up3 (128) → xattn3 → +z1 → up4 (~256) → xattn4 → final.
    """

    def __init__(
        self,
        in_channels: int = 1024,
        d_vlm: int = 2048,
        num_classes: int = 1,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels

        # Pre-fusion: d_vlm* = d_vlm + CA(Q=d_vlm, K=F_img_last, V=F_img_last)
        self.pre_fusion = VLMImagePreFusion(d_vlm, in_channels, num_heads)

        self.up1 = self._make_up_block(in_channels, 512)
        self.xattn1 = CrossAttentionConditioner(512, d_vlm, num_heads)

        self.up2 = self._make_up_block(512 + in_channels, 256)
        self.xattn2 = CrossAttentionConditioner(256, d_vlm, num_heads)

        self.up3 = self._make_up_block(256 + in_channels, 128)
        self.xattn3 = CrossAttentionConditioner(128, d_vlm, num_heads)

        self.up4 = self._make_up_block(128 + in_channels, 64)
        self.xattn4 = CrossAttentionConditioner(64, d_vlm, num_heads)

        self.final = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1),
        )

    @staticmethod
    def _make_up_block(in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        vlm_hidden: torch.Tensor,
        out_hw: Tuple[int, int],
    ) -> torch.Tensor:
        """Decode with VLM conditioning.

        Parameters
        ----------
        features : z1..z4 from FetalCLIP  (each [B, C, H, W], frozen)
        vlm_hidden : [B, T, D_vlm]  VLM last hidden state (**has grad**)
        out_hw : target output spatial size
        """
        z1, z2, z3, z4 = features["z1"], features["z2"], features["z3"], features["z4"]

        # Pre-fusion: d_vlm* = d_vlm + CA(Q=d_vlm, K=z4_flat, V=z4_flat)
        z4_tokens = z4.flatten(2).permute(0, 2, 1)    # [B, HW, C]
        vlm_hidden = self.pre_fusion(vlm_hidden, z4_tokens)

        x = self.up1(z4)                              # 16→32
        x = self.xattn1(x, vlm_hidden)                # VLM conditioning

        z3_up = F.interpolate(z3, size=x.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up2(torch.cat([x, z3_up], dim=1))    # 32→64
        x = self.xattn2(x, vlm_hidden)

        z2_up = F.interpolate(z2, size=x.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up3(torch.cat([x, z2_up], dim=1))    # 64→128
        x = self.xattn3(x, vlm_hidden)

        z1_up = F.interpolate(z1, size=x.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up4(torch.cat([x, z1_up], dim=1))    # 128→~256
        x = self.xattn4(x, vlm_hidden)

        x = self.final(x)
        if x.shape[-2:] != out_hw:
            x = F.interpolate(x, size=out_hw, mode="bilinear", align_corners=False)
        return x  # [B, num_classes, H, W]


def dice_loss_with_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    probs = probs.flatten(1)
    targets = targets.flatten(1)
    inter = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2 * inter + eps) / (union + eps)
    return 1.0 - dice.mean()


def extract_vlm_vision_tokens(outputs: Any) -> torch.Tensor:
    """Try to extract vision tokens [B,N,D] from MedGemma outputs.

    MedGemma / transformers versions differ in output field names.
    We try a set of known keys first, then fall back to heuristics.
    """

    def _iter_items(obj: Any):
        if obj is None:
            return
        if isinstance(obj, dict):
            yield from obj.items()
            return
        if hasattr(obj, "items") and callable(getattr(obj, "items")):
            try:
                yield from obj.items()
                return
            except Exception:
                pass
        if hasattr(obj, "__dict__"):
            yield from obj.__dict__.items()

    # 1) Known nested outputs
    for attr in (
        "vision_outputs",
        "vision_model_output",
        "vision_encoder_outputs",
        "image_encoder_outputs",
        "encoder_outputs",
    ):
        if hasattr(outputs, attr):
            vo = getattr(outputs, attr)
            if vo is None:
                continue
            if hasattr(vo, "last_hidden_state") and vo.last_hidden_state is not None:
                x = vo.last_hidden_state
                if isinstance(x, torch.Tensor) and x.dim() == 3:
                    return x

    # 2) Known flat tensor fields
    for attr in (
        "vision_last_hidden_state",
        "encoder_last_hidden_state",
        "image_last_hidden_state",
        "image_hidden_states",
        "vision_hidden_states",
        "image_embeds",
        "image_embeddings",
        "vision_embeds",
    ):
        if not hasattr(outputs, attr):
            continue
        x = getattr(outputs, attr)
        if x is None:
            continue
        # some fields are tuples of hidden states
        if isinstance(x, (list, tuple)) and x:
            x = x[-1]
        if isinstance(x, torch.Tensor):
            if x.dim() == 2:
                return x.unsqueeze(1)
            if x.dim() == 3:
                return x

    # 3) Heuristic: scan outputs for any 3D tensor candidate
    candidates: List[Tuple[str, torch.Tensor]] = []
    for k, v in _iter_items(outputs):
        if isinstance(v, torch.Tensor) and v.dim() == 3:
            candidates.append((str(k), v))
            continue
        if hasattr(v, "last_hidden_state") and isinstance(getattr(v, "last_hidden_state"), torch.Tensor):
            l = getattr(v, "last_hidden_state")
            if l is not None and l.dim() == 3:
                candidates.append((f"{k}.last_hidden_state", l))

    if candidates:
        # Prefer anything explicitly tagged as vision/image; else pick the largest token sequence.
        def _score(item: Tuple[str, torch.Tensor]) -> Tuple[int, int]:
            name, t = item
            name_l = name.lower()
            tag = 2 if ("vision" in name_l or "image" in name_l) else 0
            return (tag, int(t.shape[1]))

        candidates.sort(key=_score, reverse=True)
        return candidates[0][1]

    # 4) Last-resort fallback: use the model last hidden state as tokens.
    for attr in ("last_hidden_state", "hidden_states", "decoder_hidden_states"):
        if not hasattr(outputs, attr):
            continue
        x = getattr(outputs, attr)
        if x is None:
            continue
        if isinstance(x, (list, tuple)) and x:
            x = x[-1]
        if isinstance(x, torch.Tensor):
            if x.dim() == 2:
                return x.unsqueeze(1)
            if x.dim() == 3:
                return x

    raise RuntimeError(
        "Cannot find vision tokens in VLM outputs. "
        "Please update extract_vlm_vision_tokens() for your transformers/medgemma version."
    )


def extract_seg_token_feature(
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor,
    seg_token_id: int,
) -> torch.Tensor:
    """Extract hidden state at the <seg> token position.

    Parameters
    ----------
    hidden_states : [B, T, D]
    input_ids     : [B, T]
    seg_token_id  : token id for ``<seg>``

    Returns
    -------
    [B, 1, D]  hidden state at the (last) <seg> position per sample.
    Falls back to mean-pool if <seg> not found.
    """
    B, T, D = hidden_states.shape
    features = []
    for i in range(B):
        positions = (input_ids[i] == seg_token_id).nonzero(as_tuple=True)[0]
        if len(positions) > 0:
            pos = positions[-1].item()
            features.append(hidden_states[i, pos, :])
        else:
            features.append(hidden_states[i].mean(dim=0))
    return torch.stack(features, dim=0).unsqueeze(1)  # [B, 1, D]