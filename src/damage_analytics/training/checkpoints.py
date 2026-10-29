"""Portable, atomic model/optimizer snapshots loaded in restricted weights-only mode."""

import os
from dataclasses import asdict
from pathlib import Path

import torch

from damage_analytics.training.config import ModelConfig
from damage_analytics.training.model import LightweightUNet


def save_checkpoint(
    path: Path,
    model: LightweightUNet,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    score: float,
    metadata: dict,
    architecture: ModelConfig,
) -> None:
    """Store plain metadata and portable model weights without pickled custom classes."""
    payload = {
        "format_version": 1,
        "architecture": asdict(architecture),
        "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "validation_foreground_miou": score,
        "metadata": metadata,
        "torch_rng_state": torch.get_rng_state(),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_checkpoint(
    path: Path, device: torch.device, *, expected_dataset: dict | None = None
) -> tuple[LightweightUNet, dict]:
    """Load a trusted local artifact with explicit weights_only=True and strict state matching."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    if expected_dataset is not None and payload["metadata"]["dataset"] != expected_dataset:
        raise ValueError("Checkpoint is bound to a different dataset release or split hashes")
    model = LightweightUNet(ModelConfig(**payload["architecture"]))
    model.load_state_dict(payload["model_state"], strict=True)
    return model.to(device), payload
