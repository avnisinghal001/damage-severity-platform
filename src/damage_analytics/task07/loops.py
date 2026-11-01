"""Shared classifier passes, deterministic loader construction and prediction tables."""

import math

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from damage_analytics.task07.config import TrainConfig
from damage_analytics.task07.data import CrackDataset, make_loader
from damage_analytics.task07.release import Release


def pass_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict:
    training = optimizer is not None
    model.train(training)
    numerator, denominator, targets, scores, ids = 0.0, 0.0, [], [], []
    with torch.enable_grad() if training else torch.inference_mode():
        for images, labels, sample_ids in loader:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            summed = criterion(logits, labels)
            mass = criterion.weight[labels].sum()
            loss = summed / mass
            value = float(loss.detach().cpu())
            if not math.isfinite(value):
                raise RuntimeError("Nonfinite classifier loss; refusing an optimizer step")
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
            weight = float(mass.cpu())
            numerator += value * weight
            denominator += weight
            if not training:
                probability = logits.softmax(1)[:, 1].cpu().numpy()
                if not np.isfinite(probability).all():
                    raise RuntimeError("Nonfinite validation probabilities")
                targets.extend(labels.cpu().tolist())
                scores.extend(probability.tolist())
                ids.extend(sample_ids)
    if denominator == 0:
        raise ValueError("Empty classification loader")
    return {
        "loss": numerator / denominator,
        "targets": targets,
        "probabilities": scores,
        "ids": ids,
    }


def loader_for(
    release: Release,
    rows: tuple[dict, ...],
    settings: TrainConfig,
    seed: int,
    training: bool = False,
) -> DataLoader:
    dataset = CrackDataset(
        release, rows, settings.image_size, training, seed, settings.num_workers
    )
    return make_loader(dataset, settings.batch_size, settings.num_workers, seed, shuffle=training)


def prediction_rows(result: dict, samples: tuple, threshold: float) -> list[dict]:
    lookup = {r["sample_id"]: r for r in samples}
    return [
        {
            "sample_id": identifier,
            "group_id": lookup[identifier]["group_id"],
            "source_label": lookup[identifier]["source_label"],
            "target": target,
            "probability_crack": probability,
            "prediction": int(probability >= threshold),
        }
        for identifier, target, probability in zip(
            result["ids"], result["targets"], result["probabilities"], strict=True
        )
    ]
