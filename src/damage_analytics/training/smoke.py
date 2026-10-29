"""Tiny synthetic CPU training check, independent of real dataset splits."""

from pathlib import Path

import numpy as np
import torch

from damage_analytics.io import write_json
from damage_analytics.training.checkpoints import load_checkpoint, save_checkpoint
from damage_analytics.training.config import ModelConfig
from damage_analytics.training.losses import SegmentationLoss
from damage_analytics.training.model import LightweightUNet
from damage_analytics.training.runtime import configure_runtime


def synthetic_smoke(output: Path) -> dict:
    """Perform three AdamW steps and a checkpoint roundtrip, with explicit synthetic labeling."""
    device = configure_runtime(42, "cpu", True)
    architecture = ModelConfig(base_channels=4, depth=2, normalization_groups=2)
    model = LightweightUNet(architecture).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    criterion = SegmentationLoss(torch.ones(9), 0.25)
    image = torch.zeros(2, 3, 32, 32)
    image[:, 0, :, 16:] = 1
    target = torch.zeros(2, 32, 32, dtype=torch.long)
    target[:, :, 16:] = 1
    losses = []
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(image), target)
        loss.backward()
        if not all(
            bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
            if parameter.grad is not None
        ):
            raise RuntimeError("Synthetic gradients are not finite")
        optimizer.step()
        losses.append(float(loss.detach()))
    output.mkdir(parents=True, exist_ok=False)
    save_checkpoint(
        output / "synthetic.pt",
        model,
        optimizer,
        epoch=3,
        score=0.0,
        metadata={"purpose": "synthetic_smoke", "dataset": None},
        architecture=architecture,
    )
    restored, _ = load_checkpoint(output / "synthetic.pt", device)
    model.eval()
    restored.eval()
    with torch.inference_mode():
        torch.testing.assert_close(model(image), restored(image), rtol=0, atol=0)
    result = {
        "purpose": "Synthetic plumbing check, not real-data segmentation performance",
        "losses": losses,
        "steps": 3,
        "device": "cpu",
        "checkpoint_roundtrip": True,
        "loss_decreased": losses[-1] < losses[0],
        "test_accessed": False,
    }
    if not np.isfinite(losses).all() or not result["loss_decreased"]:
        raise RuntimeError("Synthetic smoke did not yield finite, decreasing loss")
    write_json(output / "smoke.json", result)
    return result
