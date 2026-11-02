"""Four-class passes; binary Task 07 probability slicing and threshold logic are never reused."""

import math

import numpy as np
import torch
from torch import nn

from damage_analytics.task07.timing import timestamp


def pass_epoch(
    model: nn.Module,
    batches,
    weights: torch.Tensor,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict:
    training = optimizer is not None
    model.train(training)
    start, numerator, denominator, samples = timestamp(device), 0.0, 0.0, 0
    targets, probabilities, identifiers = [], [], []
    criterion = nn.CrossEntropyLoss(weight=weights, reduction="sum")
    with torch.enable_grad() if training else torch.inference_mode():
        for images, labels, ids in batches:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            if logits.shape != (len(labels), 4) or not bool(torch.isfinite(logits).all()):
                raise ValueError("Model must return finite N×4 logits")
            mass = weights[labels].sum()
            loss = criterion(logits, labels) / mass
            value, count = float(loss.detach().cpu()), float(mass.detach().cpu())
            if not math.isfinite(value) or count <= 0:
                raise ValueError("Nonfinite/empty weighted CE batch")
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
            numerator += value * count
            denominator += count
            samples += len(labels)
            if not training:
                targets.extend(labels.cpu().tolist())
                probabilities.extend(logits.softmax(1).cpu().numpy().tolist())
                identifiers.extend(ids)
    if denominator <= 0:
        raise ValueError("Empty training/evaluation loader")
    elapsed = timestamp(device) - start
    return {
        "loss": numerator / denominator,
        "targets": targets,
        "probabilities": probabilities,
        "ids": identifiers,
        "runtime": {
            "seconds": elapsed,
            "samples": samples,
            "images_per_second": samples / elapsed,
        },
    }


def prediction_rows(result: dict) -> list[dict]:
    return [
        {
            "sample_id": identifier,
            "target": target,
            "prediction": int(np.argmax(scores)),
            **{f"probability_id_{i}": p for i, p in enumerate(scores)},
        }
        for identifier, target, scores in zip(
            result["ids"], result["targets"], result["probabilities"], strict=True
        )
    ]
