"""Reusable load-once segmentation → native masks → validated damage-feature JSON."""

import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from damage_analytics.analytics.inference import restore_scalar
from damage_analytics.io import sha256_file, write_json
from damage_analytics.task09.audit import manifest_rows
from damage_analytics.task09.checkpoints import load, model_identity
from damage_analytics.task09.config import Config, FeaturePolicy
from damage_analytics.task09.features import extract_features
from damage_analytics.task09.schema import validate_report
from damage_analytics.training.manifests import Release
from damage_analytics.training.plots import save_comparison
from damage_analytics.training.runtime import configure_runtime
from damage_analytics.training.transforms import prepare_pair, restore_prediction


def forbid_test_input(release: Release, path: Path) -> None:
    """Hash/membership protection only; never open test pixels or final-test reports."""
    if "test_final" in path.resolve().parts:
        raise ValueError("Task 09 cannot consume final-test artifacts")
    rows = manifest_rows(release)
    forbidden = {
        row[key]
        for row in rows.values()
        if row["split"] == "test"
        for key in ("source_image_sha256", "image_sha256", "mask_sha256")
    }
    if sha256_file(path) in forbidden:
        raise ValueError("Known frozen test artifacts are excluded from Task 09 development")


class Segmenter:
    """Own one model instance; independent of FastAPI, Task 08 and global serving behavior."""

    def __init__(
        self,
        checkpoint: Path,
        release: Release,
        *,
        device: str = "cpu",
        cpu_threads: int = 1,
        maximum_image_pixels: int = 32_000_000,
    ) -> None:
        if device not in ("cpu", "mps") or not 1 <= cpu_threads <= 4:
            raise ValueError("Explicit CPU/MPS and a bounded thread count are required")
        torch.set_num_threads(cpu_threads)
        self.device = configure_runtime(42, device, False)
        self.checkpoint, self.digest = checkpoint, sha256_file(checkpoint)
        self.model, self.payload, self.data = load(checkpoint, self.device, release.identity())
        self.maximum_image_pixels = maximum_image_pixels
        self.identity = model_identity(self.payload, self.digest)
        if (
            len(self.data.mean) != 3
            or len(self.data.std) != 3
            or not all(math.isfinite(v) for v in (*self.data.mean, *self.data.std))
            or min(self.data.std) <= 0
        ):
            raise ValueError("Invalid checkpoint normalization")

    def predict(self, image: Image.Image, policy: FeaturePolicy) -> tuple[dict, dict, Image.Image]:
        """Preserve the existing argmax→unpad→nearest-native rule for all model families."""
        if image.mode != "RGB" or image.width * image.height > self.maximum_image_pixels:
            raise ValueError("Provide a bounded RGB image; no silent alpha/color conversion")
        orientation = image.getexif().get(274, 1)
        if orientation not in range(1, 9):
            raise ValueError("Invalid EXIF orientation")
        oriented = ImageOps.exif_transpose(image).copy()
        rgb, _, frame = prepare_pair(
            oriented,
            Image.new("L", oriented.size),
            self.data,
            training=False,
            rng=np.random.default_rng(42),
        )
        with torch.inference_mode():
            logits = self.model(torch.from_numpy(rgb).unsqueeze(0).to(self.device))
            if logits.shape != (1, 9, *rgb.shape[1:]) or not bool(torch.isfinite(logits).all()):
                raise ValueError(
                    "Segmentation produced invalid/nonfinite aligned nine-class logits"
                )
            probabilities = logits.softmax(1)
            ids = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
            maximum = probabilities.max(1).values[0].cpu().numpy()
            entropy = (
                (-(probabilities * probabilities.clamp_min(1e-12).log()).sum(1)[0] / math.log(9))
                .cpu()
                .numpy()
            )
        raw = restore_prediction(ids, frame)
        maximum, entropy = (
            np.clip(restore_scalar(values, frame), 0, 1) for values in (maximum, entropy)
        )
        report, filtered = extract_features(
            raw,
            policy,
            maximum_softmax=maximum,
            normalized_entropy=entropy,
            model=self.identity,
            coordinate_frame="EXIF_oriented_original_image",
        )
        report["inference"] = {
            "device": str(self.device),
            "source_exif_orientation": orientation,
            "protocol": "native-argmax-nearest-v1",
            "test_evaluated": False,
        }
        validate_report(report)
        if sha256_file(self.checkpoint) != self.digest:
            raise ValueError("Checkpoint changed during inference")
        return (
            report,
            {
                "raw": raw,
                "filtered": filtered,
                "maximum_softmax": maximum,
                "normalized_entropy": entropy,
            },
            oriented,
        )


def predict_file(
    config: Config,
    release: Release,
    checkpoint: Path,
    image_path: Path,
    output: Path,
    *,
    device: str = "cpu",
) -> dict:
    forbid_test_input(release, image_path)
    runner = Segmenter(
        checkpoint,
        release,
        device=device,
        cpu_threads=config.cpu_threads,
        maximum_image_pixels=config.maximum_image_pixels,
    )
    digest = sha256_file(image_path)
    with Image.open(image_path) as image:
        if (
            image.format not in ("JPEG", "PNG")
            or image.width * image.height > config.maximum_image_pixels
        ):
            raise ValueError("Provide a bounded JPEG/PNG image")
        image.load()
        report, values, oriented = runner.predict(image, config.features)
    output.mkdir(parents=True, exist_ok=False)
    for key, name in (("raw", "prediction_ids.png"), ("filtered", "reporting_filtered_ids.png")):
        Image.fromarray(values[key]).save(output / name)
    np.savez_compressed(
        output / "uncalibrated_scores.npz",
        maximum_softmax=values["maximum_softmax"],
        normalized_entropy=values["normalized_entropy"],
    )
    save_comparison(output / "comparison.png", oriented, None, values["raw"], "Task 09 inference")
    report["artifacts"] = {
        "raw_mask": "prediction_ids.png",
        "raw_mask_sha256": sha256_file(output / "prediction_ids.png"),
        "reporting_filtered_mask": "reporting_filtered_ids.png",
        "filtered_mask_sha256": sha256_file(output / "reporting_filtered_ids.png"),
        "uncalibrated_scores": "uncalibrated_scores.npz",
    }
    report["image"]["source_sha256"] = digest
    validate_report(report)
    write_json(output / "damage_features.json", report)
    return report
