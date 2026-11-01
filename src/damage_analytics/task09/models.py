"""Original U-Net and lightweight transfer LR-ASPP; no implicit weight downloads."""

import torch
from torch import nn

from damage_analytics.training.config import ModelConfig
from damage_analytics.training.model import LightweightUNet

NAMES = ("unet", "lraspp_mobilenet_v3_large")


class LRASPP(nn.Module):
    """Adapt the two output heads to RC IDs without merging any pretrained source labels."""

    def __init__(self, *, pretrained: bool, freeze_batch_norm: bool) -> None:
        super().__init__()
        from torchvision.models.segmentation import (
            LRASPP_MobileNet_V3_Large_Weights,
            lraspp_mobilenet_v3_large,
        )

        weights = LRASPP_MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        # The torchvision default backbone weights would otherwise download even with weights=None.
        self.network = lraspp_mobilenet_v3_large(weights=weights, weights_backbone=None)
        for name in ("low_classifier", "high_classifier"):
            old = getattr(self.network.classifier, name)
            setattr(self.network.classifier, name, nn.Conv2d(old.in_channels, 9, 1))
        self.freeze_batch_norm = freeze_batch_norm
        self.initial_weights = str(weights) if weights else "none"
        self.initial_weights_url = weights.url if weights else None

    def train(self, mode: bool = True) -> "LRASPP":
        super().train(mode)
        if mode and self.freeze_batch_norm:
            for layer in self.modules():
                if isinstance(layer, nn.BatchNorm2d):
                    layer.eval()  # Train affine parameters, freeze running moments.
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.network(images)["out"]


def build_model(
    name: str,
    architecture: ModelConfig,
    *,
    pretrained: bool = False,
    freeze_batch_norm: bool = True,
) -> nn.Module:
    if name == "unet":
        if pretrained:
            raise ValueError("The existing project U-Net has no pretrained initializer")
        return LightweightUNet(architecture)
    if name == "lraspp_mobilenet_v3_large":
        return LRASPP(pretrained=pretrained, freeze_batch_norm=freeze_batch_norm)
    raise ValueError("Unknown Task 09 segmentation architecture")
