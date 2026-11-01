"""Torchvision transfer learning; weights are requested ONLY by explicit real training."""

import torch
from torch import nn

BACKBONES = ("mobilenet_v3_small", "resnet18")


def build_model(backbone: str, *, pretrained: bool) -> nn.Module:
    """Evaluation always passes pretrained=False before loading saved state; no weight download."""
    if backbone not in BACKBONES:
        raise ValueError("Choose mobilenet_v3_small or resnet18")
    from torchvision import models

    if backbone == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 2)
    else:
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, 2)
    model._task07_weights = str(weights) if weights else "none"
    model._task07_weights_url = weights.url if weights else None
    return model


class TinySmokeNet(nn.Module):
    """Synthetic plumbing fixture only; never a substitute for a real backbone experiment."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 4, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1)
        )
        self.head = nn.Linear(4, 2)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(value).flatten(1))
