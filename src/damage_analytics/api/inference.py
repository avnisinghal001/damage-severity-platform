"""Resident checkpoint adapter reusing Task 03/04 geometry, taxonomy and region analytics."""

import math
from dataclasses import asdict, dataclass
from typing import Protocol

import numpy as np
from PIL import Image

from damage_analytics.api.config import Settings


@dataclass(frozen=True)
class Prediction:
    """Native class-ID pixels and the existing Task 04 analytics schema."""

    mask: np.ndarray
    analytics: dict


class Predictor(Protocol):
    """Small serving boundary, also usable with synthetic inference in HTTP tests."""

    metadata: dict

    def predict(self, image: Image.Image) -> Prediction:
        """Predict one already validated, oriented RGB image."""
        ...


class Segmenter:
    """Load trusted pinned weights once per worker; never update weights or read datasets."""

    def __init__(self, settings: Settings) -> None:
        import torch

        from damage_analytics.io import sha256_file
        from damage_analytics.labels import CLASS_NAMES
        from damage_analytics.training.checkpoints import load_checkpoint
        from damage_analytics.training.config import DataConfig
        from damage_analytics.training.runtime import configure_runtime

        if sha256_file(settings.checkpoint) != settings.checkpoint_sha256:
            raise ValueError("Serving checkpoint SHA-256 does not match its pin")
        torch.set_num_threads(settings.cpu_threads)
        self.device = configure_runtime(42, settings.device, True)
        self.model, payload = load_checkpoint(settings.checkpoint, self.device)
        if sha256_file(settings.checkpoint) != settings.checkpoint_sha256:
            raise ValueError("Checkpoint changed during loading")
        if (
            payload["metadata"]["purpose"] != "training"
            or payload["validation_foreground_miou"] is None
        ):
            raise ValueError(
                "Serve a scored training checkpoint, not an exploratory/synthetic run"
            )
        self.data = DataConfig(**payload["metadata"]["config"]["data"])
        self.seed = payload["metadata"]["config"]["seed"]
        self.model.eval().requires_grad_(False)
        self.metadata = {
            "checkpoint_sha256": settings.checkpoint_sha256,
            "checkpoint_epoch": payload["epoch"],
            "dataset": payload["metadata"]["dataset"],
            "architecture": payload["architecture"],
            "preprocessing": asdict(self.data),
            "class_mapping": [
                {"class_id": i, "source_label": name} for i, name in enumerate(CLASS_NAMES)
            ],
            "device": str(self.device),
            "training_code_version": payload["metadata"].get("git_commit"),
            "confidence": "Uncalibrated maximum softmax and normalized entropy only",
            "alignment": "EXIF-oriented source resolution; argmax then nearest restoration",
            "severity_scope": "Source-category spalling labels only; no structural safety score",
        }

    def predict(self, image: Image.Image) -> Prediction:
        """Use the original letterbox/RGB normalization and Task 04 confidence restoration."""
        import torch

        from damage_analytics.analytics.config import RegionPolicy, UncertaintyPolicy
        from damage_analytics.analytics.inference import restore_scalar
        from damage_analytics.analytics.regions import analyze_mask
        from damage_analytics.training.transforms import prepare_pair, restore_prediction

        rgb, _, frame = prepare_pair(
            image,
            Image.new("L", image.size),
            self.data,
            training=False,
            rng=np.random.default_rng(self.seed),
        )
        with torch.inference_mode():
            logits = self.model(torch.from_numpy(rgb).unsqueeze(0).to(self.device))
            probabilities = logits.softmax(1)
            ids = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
            maximum = probabilities.max(1).values[0].cpu().numpy()
            entropy = (
                (-(probabilities * probabilities.clamp_min(1e-12).log()).sum(1)[0] / math.log(9))
                .cpu()
                .numpy()
            )
        mask = restore_prediction(ids, frame)
        report = analyze_mask(
            mask,
            RegionPolicy(),
            UncertaintyPolicy(),
            maximum_softmax=np.clip(restore_scalar(maximum, frame), 0, 1),
            normalized_entropy=np.clip(restore_scalar(entropy, frame), 0, 1),
        )
        return Prediction(mask, report)
