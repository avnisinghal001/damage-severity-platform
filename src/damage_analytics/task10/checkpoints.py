"""Restricted four-class checkpoint format; incompatible with earlier crack/segmentation models."""

import json
import os
from pathlib import Path

import torch

from damage_analytics.task10.config import CLASS_NAMES
from damage_analytics.task10.models import build_model


def save(path: Path, model, metadata: dict, epoch: int, score: float) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "task10_format": 1,
            "metadata": json.loads(json.dumps(metadata, allow_nan=False)),
            "epoch": epoch,
            "validation_macro_f1": float(score),
            "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        },
        temporary,
    )
    os.replace(temporary, path)


def load(path: Path, device: torch.device, identity: dict):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        payload.get("task10_format") != 1
        or payload["metadata"]["class_names"] != list(CLASS_NAMES)
        or payload["metadata"]["dataset"] != identity
    ):
        raise ValueError("Not a compatible pinned four-class Phi-Net checkpoint")
    model = build_model(
        pretrained=False, synthetic=payload["metadata"]["purpose"] == "synthetic_smoke"
    )
    model.load_state_dict(payload["model_state"], strict=True)
    return model.to(device).eval(), payload
