from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Compute per-sample soft Dice loss and then average the batch."""
    probs = torch.sigmoid(logits).flatten(1)
    targets = targets.float().flatten(1)
    intersection = (probs * targets).sum(dim=1)
    denominator = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def sigmoid_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    alpha: float = 0.85,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Numerically stable binary focal loss operating directly on logits."""
    targets = targets.float()
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probs = torch.sigmoid(logits)
    p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    return (alpha_t * (1.0 - p_t).pow(gamma) * bce).mean()


def tversky_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    alpha: float = 0.3,
    beta: float = 0.7,
    eps: float = 1e-6,
) -> torch.Tensor:
    probs = torch.sigmoid(logits).flatten(1)
    targets = targets.float().flatten(1)
    tp = (probs * targets).sum(dim=1)
    fp = (probs * (1.0 - targets)).sum(dim=1)
    fn = ((1.0 - probs) * targets).sum(dim=1)
    score = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    return 1.0 - score.mean()


@dataclass(frozen=True)
class SegmentationLossConfig:
    name: str = "dice_focal"
    dice_weight: float = 0.7
    focal_weight: float = 0.3
    focal_alpha: float = 0.85
    focal_gamma: float = 2.0
    eps: float = 1e-6


class SparseForegroundLoss(nn.Module):
    def __init__(self, config: SegmentationLossConfig | None = None) -> None:
        super().__init__()
        self.config = config or SegmentationLossConfig()
        if self.config.name not in {"dice_focal", "tversky_focal"}:
            raise ValueError(f"Unsupported segmentation loss: {self.config.name}")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        dice = soft_dice_loss(logits, targets, eps=self.config.eps)
        focal = sigmoid_focal_loss(
            logits,
            targets,
            alpha=self.config.focal_alpha,
            gamma=self.config.focal_gamma,
        )
        if self.config.name == "dice_focal":
            total = self.config.dice_weight * dice + self.config.focal_weight * focal
        else:
            tv = tversky_loss(logits, targets, alpha=0.3, beta=0.7, eps=self.config.eps)
            total = 0.5 * tv + 0.5 * focal
        return {"total": total, "dice": dice, "focal": focal}
