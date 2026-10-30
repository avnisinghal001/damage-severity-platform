"""One segmentation pass, with optional synchronized stage timers and native metrics."""

import math
from contextlib import nullcontext

import numpy as np
import torch
from torch.utils.data import DataLoader

from damage_analytics.training.config import IGNORE_INDEX
from damage_analytics.training.losses import SegmentationLoss
from damage_analytics.training.metrics import confusion_matrix, metrics_from_confusion
from damage_analytics.training.model import LightweightUNet
from damage_analytics.training.timing import StageTimer
from damage_analytics.training.transforms import restore_prediction


def run_epoch(
    model: LightweightUNet,
    loader: DataLoader,
    criterion: SegmentationLoss,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    gradient_clip: float = 1.0,
    timer: StageTimer | None = None,
    channels_last: bool = False,
) -> dict:
    """Train or evaluate one pass; pool confusion counts before calculating metrics."""
    training = optimizer is not None
    model.train(training)
    weighted_loss, valid_pixels = 0.0, 0
    matrix = np.zeros((9, 9), dtype=np.int64)
    measure = timer.measure if timer is not None else lambda _name: nullcontext()
    with torch.enable_grad() if training else torch.inference_mode():
        with measure("loader_startup"):
            iterator = iter(loader)
        while True:
            with measure("data_wait"):
                batch = next(iterator, None)
            if batch is None:
                break
            count = int((batch["masks"] != IGNORE_INDEX).sum().item())
            with measure("host_to_device"):
                images, target = batch["images"].to(device), batch["masks"].to(device)
                if channels_last:
                    images = images.contiguous(memory_format=torch.channels_last)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with measure("forward"):
                logits = model(images)
            with measure("loss"):
                loss = criterion(logits, target, supervised_pixels=count)
            with measure("loss_accounting"):
                loss_value = float(loss.detach().cpu())
                if not math.isfinite(loss_value):
                    raise RuntimeError(
                        "Nonfinite loss; refusing to update/save a misleading model"
                    )
                weighted_loss += loss_value * count
                valid_pixels += count
            if training:
                with measure("backward"):
                    loss.backward()
                with measure("gradient_clip"):
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), gradient_clip, error_if_nonfinite=True
                    )
                with measure("optimizer"):
                    optimizer.step()
            if not training:
                with measure("argmax"):
                    if device.type == "cpu":
                        predictions = logits.numpy().argmax(axis=1).astype(np.uint8)
                    else:
                        indices = logits.argmax(dim=1)
                if device.type != "cpu":
                    with measure("device_to_host"):
                        predictions = indices.cpu().numpy().astype(np.uint8)
                with measure("native_metrics"):
                    for index, prediction in enumerate(predictions):
                        native = restore_prediction(prediction, batch["frames"][index])
                        matrix += confusion_matrix(batch["native_masks"][index], native)
    if not valid_pixels:
        raise ValueError("No supervised pixels in loader")
    return {
        "loss": weighted_loss / valid_pixels,
        "loss_pixels": valid_pixels,
        **({"metrics": metrics_from_confusion(matrix)} if not training else {}),
    }
