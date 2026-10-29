"""Lightweight from-scratch U-Net; no pretrained backbone or download code."""

import torch
from torch import nn
from torch.nn import functional as F

from damage_analytics.labels import CLASS_NAMES
from damage_analytics.training.config import ModelConfig


def _groups(channels: int, requested: int) -> int:
    """Keep at least two channels per group, even at a 1×1 bottleneck and batch size one."""
    groups = min(requested, max(1, channels // 2))
    while channels % groups:
        groups -= 1
    return groups


class ConvBlock(nn.Sequential):
    """Two spatial convolutions with GroupNorm for stable small-batch training."""

    def __init__(self, incoming: int, outgoing: int, groups: int) -> None:
        super().__init__(
            nn.Conv2d(incoming, outgoing, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(outgoing, groups), outgoing),
            nn.ReLU(inplace=True),
            nn.Conv2d(outgoing, outgoing, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(outgoing, groups), outgoing),
            nn.ReLU(inplace=True),
        )


class LightweightUNet(nn.Module):
    """Encoder/decoder with skip concatenation, bilinear upsampling and nine raw logits."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        channels = [config.base_channels * 2**index for index in range(config.depth + 1)]
        self.encoders = nn.ModuleList()
        incoming = 3
        for outgoing in channels:
            self.encoders.append(ConvBlock(incoming, outgoing, config.normalization_groups))
            incoming = outgoing
        self.pool = nn.MaxPool2d(2)
        self.decoders = nn.ModuleList(
            [
                ConvBlock(
                    channels[index + 1] + channels[index],
                    channels[index],
                    config.normalization_groups,
                )
                for index in reversed(range(config.depth))
            ]
        )
        self.output = nn.Conv2d(channels[0], len(CLASS_NAMES), 1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Map N×3×H×W float tensors to N×9×H×W logits (no softmax before CE)."""
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("Expected N×3×H×W RGB input")
        features = []
        value = images
        for index, encoder in enumerate(self.encoders):
            if index:
                value = self.pool(value)
            value = encoder(value)
            features.append(value)
        for decoder, skip in zip(self.decoders, reversed(features[:-1]), strict=True):
            value = F.interpolate(
                value, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
            value = decoder(torch.cat((value, skip), dim=1))
        return self.output(value)
