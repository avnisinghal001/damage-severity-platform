"""Separate Task 09 checkpoints; old serving/training checkpoint formats stay unchanged."""

import json
import os
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

from damage_analytics.labels import CLASS_NAMES
from damage_analytics.task09.models import build_model
from damage_analytics.training.checkpoints import load_checkpoint
from damage_analytics.training.config import DataConfig, ModelConfig


def save(path: Path, model: nn.Module, metadata: dict, epoch: int, score: float) -> None:
    plain = json.loads(json.dumps(metadata, allow_nan=False))
    payload = {
        "task09_format": 1,
        "metadata": plain,
        "epoch": epoch,
        "validation_foreground_miou": score,
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
    }
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load(path: Path, device: torch.device, dataset: dict) -> tuple[nn.Module, dict, DataConfig]:
    """Explicitly weights=None on reload, including pretrained LR-ASPP; never download here."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("task09_format") == 1:
        meta = payload["metadata"]
        if meta["dataset"] != dataset or meta["class_names"] != list(CLASS_NAMES):
            raise ValueError("Task 09 checkpoint dataset/taxonomy differs")
        model = build_model(
            meta["model_name"],
            ModelConfig(**meta["architecture"]),
            pretrained=False,
            freeze_batch_norm=meta["freeze_batch_norm"],
        )
        model.load_state_dict(payload["model_state"], strict=True)
        data = DataConfig(**meta["data"])
    elif payload.get("format_version") == 1:
        model, payload = load_checkpoint(path, device, expected_dataset=dataset)
        data = DataConfig(**payload["metadata"]["config"]["data"])
    else:
        raise ValueError("Not a supported RC segmentation checkpoint")
    return model.to(device).eval(), payload, data


def model_identity(payload: dict, checkpoint_hash: str) -> dict:
    meta = payload["metadata"]
    name = meta["model_name"] if payload.get("task09_format") else "unet"
    data = meta["data"] if payload.get("task09_format") else meta["config"]["data"]
    return {
        "name": name,
        "version": f"checkpoint:{checkpoint_hash[:16]}",
        "checkpoint_sha256": checkpoint_hash,
        "epoch": payload["epoch"],
        "dataset_release_id": meta["dataset"]["release_id"],
        "preprocessing": asdict(DataConfig(**data)),
    }
