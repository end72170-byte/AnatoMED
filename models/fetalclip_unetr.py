from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.seg_unet import assert_no_batch_norm, group_norm


class UpConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, bias=False),
            group_norm(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            group_norm(out_channels),
            nn.ReLU(inplace=True),
        )


class FetalCLIPUNETRDecoder(nn.Module):
    """VLM-free UNETR-style decoder over frozen FetalCLIP z1-z4 features."""

    def __init__(self, feature_width: int = 1024) -> None:
        super().__init__()
        c = int(feature_width)
        self.up1 = UpConvBlock(c, 512)
        self.up2 = UpConvBlock(512 + c, 256)
        self.up3 = UpConvBlock(256 + c, 128)
        self.up4 = UpConvBlock(128 + c, 64)
        self.final = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            group_norm(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
        )
        assert_no_batch_norm(self)

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        out_hw: Tuple[int, int] = (224, 224),
    ) -> torch.Tensor:
        z1, z2, z3, z4 = (features[f"z{i}"] for i in range(1, 5))
        x = self.up1(z4)
        z3 = F.interpolate(z3, size=x.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up2(torch.cat([x, z3], dim=1))
        z2 = F.interpolate(z2, size=x.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up3(torch.cat([x, z2], dim=1))
        z1 = F.interpolate(z1, size=x.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up4(torch.cat([x, z1], dim=1))
        x = self.final(x)
        if x.shape[-2:] != out_hw:
            x = F.interpolate(x, size=out_hw, mode="bilinear", align_corners=False)
        return x


class FrozenFetalCLIPUNETR(nn.Module):
    """Frozen feature extractor plus trainable GroupNorm UNETR decoder."""

    def __init__(self, feature_extractor: nn.Module, feature_width: int = 1024) -> None:
        super().__init__()
        self.feature_extractor = feature_extractor
        self.decoder = FetalCLIPUNETRDecoder(feature_width=feature_width)
        for parameter in self.feature_extractor.parameters():
            parameter.requires_grad = False
        self.feature_extractor.eval()
        assert_no_batch_norm(self.decoder)

    def train(self, mode: bool = True):
        super().train(mode)
        self.feature_extractor.eval()
        return self

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        self.feature_extractor.eval()
        with torch.no_grad():
            features, _ = self.feature_extractor(image)
        return self.decoder(features, out_hw=(224, 224))

    def trainable_parameters(self):
        return self.decoder.parameters()

    def assert_encoder_frozen(self) -> None:
        if any(parameter.requires_grad for parameter in self.feature_extractor.parameters()):
            raise AssertionError("FetalCLIP encoder must remain frozen.")
        if any(parameter.grad is not None for parameter in self.feature_extractor.parameters()):
            raise AssertionError("Frozen FetalCLIP encoder unexpectedly accumulated gradients.")
