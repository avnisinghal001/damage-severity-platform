"""Load-once, pinned ResNet18 adapters with the original full-image preprocessing."""

import hashlib
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from damage_analytics.io import sha256_file
from damage_analytics.serving.contracts import CrackPrediction, Provenance, SeverityPrediction
from damage_analytics.task07 import checkpoints as crack_checkpoints
from damage_analytics.task07.data import MEAN, STD, prepare_image, validate_preprocessing
from damage_analytics.task10 import checkpoints as severity_checkpoints
from damage_analytics.task10.config import CLASS_NAMES

LIMITATIONS = [
    "Confidence is uncalibrated; no universal image-domain validity or OOD rejection.",
    "No assessment of structural capacity or whole-building safety.",
]


def check_pin(path: Path, digest: str) -> None:
    if sha256_file(path) != digest:
        raise ValueError("Model checkpoint differs from the configured SHA-256 pin")


def probabilities(model, image: Image.Image, size: int, device: torch.device, classes: int):
    values = prepare_image(image, size, training=False, rng=np.random.default_rng(0))
    with torch.inference_mode():
        logits = model(torch.from_numpy(values).unsqueeze(0).to(device))
        if logits.shape != (1, classes) or not bool(torch.isfinite(logits).all()):
            raise ValueError("Classifier returned incompatible or nonfinite logits")
        return logits.softmax(1)[0].cpu().numpy().astype(float)


class CrackClassifier:
    """The frozen validation threshold belongs to this exact Task 07-format checkpoint."""

    def __init__(self, path: Path, digest: str, device: torch.device):
        check_pin(path, digest)
        payload = crack_checkpoints.read_payload(path)
        validate_preprocessing(payload)
        threshold = payload.get("threshold", {})
        if (
            payload["purpose"] != "training"
            or payload["backbone"] != "resnet18"
            or threshold.get("selected_on") != "validation"
            or threshold.get("split_sha256") != payload["dataset"]["splits"]["validation"]
            or not math.isfinite(threshold.get("threshold", math.nan))
            or not 0 <= threshold["threshold"] <= 1.000001
        ):
            raise ValueError("Expected a trained ResNet18 and a pinned validation-only threshold")
        self.model, _ = crack_checkpoints.load(path, device, payload)
        self.model.requires_grad_(False)
        self.device, self.threshold = device, threshold
        self.size = int(payload["settings"]["image_size"])
        self.metadata = Provenance(
            checkpoint_sha256=digest,
            epoch=payload["epoch"],
            dataset=payload["dataset"],
            preprocessing=payload["normalization"],
            device=str(device),
            limitations=LIMITATIONS
            + [
                "No-crack does not mean undamaged concrete.",
                "SDNET patches and whole uploads differ in domain/scale; validity is unverified.",
            ],
        )
        check_pin(path, digest)

    def predict(self, image: Image.Image) -> dict:
        scores = probabilities(self.model, image, self.size, self.device, 2)
        result = int(scores[1] >= self.threshold["threshold"])
        return CrackPrediction(
            class_id=result,
            label=("no_crack", "crack")[result],
            crack_probability=scores[1],
            decision_threshold=self.threshold["threshold"],
            threshold_provenance=self.threshold,
            model=self.metadata,
        ).model_dump(mode="json")


class SeverityClassifier:
    """Four nominal Phi-Net classes; input photographs are RGB, not stored Caffe tensors."""

    def __init__(self, path: Path, digest: str, device: torch.device):
        check_pin(path, digest)
        payload = torch.load(path, map_location="cpu", weights_only=True)
        meta = payload["metadata"]
        expected = {
            "mean": list(MEAN),
            "std": list(STD),
            "resize": "full-image letterbox, bilinear",
        }
        if (
            payload.get("task10_format") != 1
            or meta["purpose"] != "training"
            or meta["class_names"] != list(CLASS_NAMES)
            or meta["normalization"] != expected
            or meta["target"] != "phi_net_task7_visible_component_damage"
        ):
            raise ValueError("Expected the trained four-class Phi-Net checkpoint/preprocessing")
        self.model, _ = severity_checkpoints.load(path, device, meta["dataset"])
        self.model.requires_grad_(False)
        self.device, self.size = device, int(meta["training"]["image_size"])
        self.metadata = Provenance(
            checkpoint_sha256=digest,
            epoch=payload["epoch"],
            dataset=meta["dataset"],
            preprocessing={
                **meta["normalization"],
                "image_size": self.size,
                "training_array_encoding": meta["array_encoding"],
                "input": "EXIF-oriented RGB photograph; no array inverse on an RGB upload",
            },
            device=str(device),
            limitations=LIMITATIONS
            + meta.get("provisional_limitations", [])
            + [
                "Visible component damage level only; no RC1841 paired-feature supervision.",
                "Input component framing, material and domain are not automatically verified.",
            ],
        )
        check_pin(path, digest)

    def predict(self, image: Image.Image) -> SeverityPrediction:
        scores = probabilities(self.model, image, self.size, self.device, 4)
        result = int(scores.argmax())
        entropy = float(-(scores * np.log(np.clip(scores, 1e-12, 1))).sum() / math.log(4))
        return SeverityPrediction(
            class_id=result,
            label=CLASS_NAMES[result],
            confidence=scores[result],
            probabilities=[
                {"class_id": i, "label": name, "probability": scores[i]}
                for i, name in enumerate(CLASS_NAMES)
            ],
            normalized_entropy=min(1.0, max(0.0, entropy)),
            width=image.width,
            height=image.height,
            input_rgb_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
            model=self.metadata,
        )
