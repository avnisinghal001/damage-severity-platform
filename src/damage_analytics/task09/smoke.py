"""Bounded CPU-only synthetic backward/checkpoint/native-feature checks for both architectures."""

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from damage_analytics.io import write_json
from damage_analytics.labels import CLASS_NAMES
from damage_analytics.task09.checkpoints import save
from damage_analytics.task09.config import FeaturePolicy
from damage_analytics.task09.models import NAMES, build_model
from damage_analytics.task09.pipeline import Segmenter
from damage_analytics.training.config import DataConfig, ModelConfig
from damage_analytics.training.losses import SegmentationLoss
from damage_analytics.training.runtime import configure_runtime
from damage_analytics.training.transforms import prepare_pair


def smoke(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    configure_runtime(42, "cpu", False)
    architecture = ModelConfig(base_channels=2, depth=2, normalization_groups=1)
    data = DataConfig(image_size=(32, 32), crop_probability=0)
    values = np.zeros((24, 40, 3), dtype=np.uint8)
    values[:, 20:, 0] = 255
    image = Image.fromarray(values)
    target = np.zeros((24, 40), dtype=np.uint8)
    target[:, 20:] = 1
    rgb, ids, _ = prepare_pair(
        image, Image.fromarray(target), data, training=False, rng=np.random.default_rng(42)
    )
    inputs, targets = (
        torch.from_numpy(rgb).unsqueeze(0).repeat(2, 1, 1, 1),
        torch.from_numpy(ids).unsqueeze(0).repeat(2, 1, 1),
    )
    dataset = {
        "release_id": "synthetic-task09",
        "checksums_sha256": "synthetic",
        "split_hashes": {},
    }
    results = []
    for name in NAMES:
        model = build_model(name, architecture, pretrained=False, freeze_batch_norm=True).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        criterion = SegmentationLoss(torch.ones(9), 0.25)
        losses = []
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), targets)
            if not bool(torch.isfinite(loss)):
                raise ValueError("Synthetic loss is nonfinite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
        checkpoint = output / f"{name}.pt"
        save(
            checkpoint,
            model,
            {
                "dataset": dataset,
                "class_names": list(CLASS_NAMES),
                "model_name": name,
                "architecture": asdict(architecture),
                "data": asdict(data),
                "freeze_batch_norm": True,
                "purpose": "synthetic_smoke",
            },
            2,
            0.0,
        )
        runner = Segmenter(checkpoint, SimpleNamespace(identity=lambda: dataset), cpu_threads=1)
        report, arrays, oriented = runner.predict(image, FeaturePolicy())
        if arrays["raw"].shape != target.shape or oriented.size != image.size:
            raise ValueError("Native synthetic alignment failed")
        write_json(output / f"{name}_features.json", report)
        Image.fromarray(arrays["raw"]).save(output / f"{name}_ids.png")
        results.append(
            {
                "model": name,
                "parameters": sum(p.numel() for p in model.parameters()),
                "losses": losses,
                "checkpoint_reload": True,
                "native_features_validated": True,
            }
        )
    result = {
        "purpose": "Synthetic plumbing only; not real-data accuracy or speed evidence",
        "device": "cpu",
        "models": results,
        "pretrained_downloaded": False,
        "test_data_accessed": False,
        "real_training_executed": False,
    }
    write_json(output / "smoke.json", result)
    return result
