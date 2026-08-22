from __future__ import annotations

from typing import Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def group_norm(channels: int) -> nn.GroupNorm:
    preferred = 32 if channels >= 128 else 16 if channels == 64 else 8 if channels == 32 else 4
    groups = min(preferred, max(channels // 4, 1))
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            group_norm(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            group_norm(out_channels),
            nn.ReLU(inplace=True),
        )


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([skip, x], dim=1))


class SegUNet(nn.Module):
    """Classic five-level U-Net with GroupNorm and raw-logit output."""

    def __init__(self, channels: Iterable[int] = (32, 64, 128, 256, 512)) -> None:
        super().__init__()
        c1, c2, c3, c4, c5 = tuple(int(c) for c in channels)
        self.encoder1 = DoubleConv(3, c1)
        self.encoder2 = DoubleConv(c1, c2)
        self.encoder3 = DoubleConv(c2, c3)
        self.encoder4 = DoubleConv(c3, c4)
        self.bottleneck = DoubleConv(c4, c5)
        self.pool = nn.MaxPool2d(2)
        self.decoder4 = UpBlock(c5, c4, c4)
        self.decoder3 = UpBlock(c4, c3, c3)
        self.decoder2 = UpBlock(c3, c2, c2)
        self.decoder1 = UpBlock(c2, c1, c1)
        self.output = nn.Conv2d(c1, 1, kernel_size=1)
        assert_no_batch_norm(self)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        e1 = self.encoder1(image)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))
        x = self.bottleneck(self.pool(e4))
        x = self.decoder4(x, e4)
        x = self.decoder3(x, e3)
        x = self.decoder2(x, e2)
        x = self.decoder1(x, e1)
        return self.output(x)


def assert_no_batch_norm(model: nn.Module) -> None:
    offenders: Tuple[str, ...] = tuple(
        name for name, module in model.named_modules() if isinstance(module, nn.BatchNorm2d)
    )
    if offenders:
        raise AssertionError(f"BatchNorm2d is forbidden in segmentation models: {offenders}")

