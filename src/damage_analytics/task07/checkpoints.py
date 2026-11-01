"""Restricted, portable Task 07 classifier snapshots; incompatible with the serving U-Net."""

import json
import os
from pathlib import Path

import torch

from damage_analytics.task07.models import TinySmokeNet, build_model


def save(path: Path, model, payload: dict) -> None:
    temporary = path.with_suffix(".tmp")
    # Keep metadata primitive-only so restricted weights_only loading remains usable.
    payload = json.loads(json.dumps(payload, allow_nan=False))
    torch.save(
        {
            **payload,
            "task07_format": 1,
            "model_state": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
            "torch_rng_state": torch.get_rng_state(),
        },
        temporary,
    )
    os.replace(temporary, path)


def read_payload(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("task07_format") != 1 or payload.get("task") != "binary_crack_presence":
        raise ValueError("Not a Task 07 binary crack checkpoint")
    return payload


def load(path: Path, device: torch.device, payload: dict | None = None):
    payload = read_payload(path) if payload is None else payload
    if payload["purpose"] == "synthetic_smoke":
        model = TinySmokeNet()
    else:
        model = build_model(payload["backbone"], pretrained=False)
    model.load_state_dict(payload["model_state"], strict=True)
    return model.to(device).eval(), payload
