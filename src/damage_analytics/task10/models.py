"""Reuse ResNet18 initialization while keeping an isolated four-class checkpoint contract."""

from torch import nn

from damage_analytics.task07.models import TinySmokeNet
from damage_analytics.task07.models import build_model as build_binary


def build_model(*, pretrained: bool, synthetic: bool = False) -> nn.Module:
    model = TinySmokeNet() if synthetic else build_binary("resnet18", pretrained=pretrained)
    if synthetic:
        model.head = nn.Linear(model.head.in_features, 4)
    else:
        model.fc = nn.Linear(model.fc.in_features, 4)
    return model
