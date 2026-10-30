"""Checkpoint-bound, original-resolution structured damage analytics without test reuse."""

import csv
import math
import re
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from damage_analytics.analytics.config import AnalyticsConfig, runtime_training_config
from damage_analytics.analytics.regions import analyze_mask
from damage_analytics.io import sha256_file, write_json
from damage_analytics.training.checkpoints import load_checkpoint
from damage_analytics.training.evaluation import effective_config
from damage_analytics.training.experiments import execution_provenance
from damage_analytics.training.manifests import open_release
from damage_analytics.training.plots import save_comparison
from damage_analytics.training.runtime import configure_runtime
from damage_analytics.training.transforms import Frame, prepare_pair, restore_prediction


def forbid_test_artifact(config: AnalyticsConfig, path: Path) -> None:
    """Reject known test input image/mask hashes, including original image copies.

    This inspects sealed manifest metadata only. It never opens test pixels, final-test
    reports or predictions. It cannot identify deliberately altered/copied unknown data.
    """
    if "test_final" in path.resolve().parts:
        raise ValueError("Task 04 cannot reuse final-test artifacts")
    training = runtime_training_config(config)
    release = open_release(training)
    test_hashes = set()
    with (release.root / "splits/test.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            for key in (
                "image_sha256",
                "mask_sha256",
                "source_image_sha256",
                "source_annotation_sha256",
            ):
                if row.get(key):
                    test_hashes.add(row[key])
    if sha256_file(path) in test_hashes:
        raise ValueError("Known test-set artifacts are excluded from Task 04")


def restore_scalar(values: np.ndarray, frame: Frame) -> np.ndarray:
    """Restore uncalibrated score maps using exactly the prediction's unpad/nearest geometry."""
    if frame.crop_box != (0, 0, *frame.source_size):
        raise ValueError("Scalar restoration requires a full-image frame")
    width, height = frame.resized_size
    cropped = values[frame.top : frame.top + height, frame.left : frame.left + width]
    if cropped.shape != (height, width) or not np.isfinite(cropped).all():
        raise ValueError("Invalid scalar map")
    return np.asarray(
        Image.fromarray(cropped.astype(np.float32)).resize(
            frame.source_size, Image.Resampling.NEAREST
        )
    )


def structured_predict(
    config: AnalyticsConfig, checkpoint: Path, image_path: Path, report_id: str
) -> Path:
    """Save native ID mask, uncalibrated scores and category-scoped region analytics."""
    _safe_id(report_id)
    forbid_test_artifact(config, image_path)
    torch.set_num_threads(config.cpu_threads)
    training = runtime_training_config(config)
    release = open_release(training)
    checkpoint_hash = sha256_file(checkpoint)
    device = configure_runtime(training.seed, config.device, config.deterministic)
    model, payload = load_checkpoint(checkpoint, device, expected_dataset=release.identity())
    effective = effective_config(training, payload)
    with Image.open(image_path) as source:
        source.load()
        if source.mode != "RGB":
            raise ValueError("Provide an RGB image; no silent color/alpha conversion")
        image = ImageOps.exif_transpose(source).copy()
    rgb, _, frame = prepare_pair(
        image,
        Image.new("L", image.size),
        effective.data,
        training=False,
        rng=np.random.default_rng(effective.seed),
    )
    model.eval()
    with torch.inference_mode():
        logits = model(torch.from_numpy(rgb).unsqueeze(0).to(device))
        probabilities = logits.softmax(1)
        ids = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
        maximum = probabilities.max(1).values[0].cpu().numpy()
        entropy = (
            (-(probabilities * probabilities.clamp_min(1e-12).log()).sum(1)[0] / math.log(9))
            .cpu()
            .numpy()
        )
    native = restore_prediction(ids, frame)
    maximum = np.clip(restore_scalar(maximum, frame), 0, 1)
    entropy = np.clip(restore_scalar(entropy, frame), 0, 1)
    result = analyze_mask(
        native,
        config.regions,
        config.uncertainty,
        maximum_softmax=maximum,
        normalized_entropy=entropy,
    )
    if sha256_file(checkpoint) != checkpoint_hash:
        raise ValueError("Checkpoint changed during inference")
    folder = config.output_dir / "inference" / report_id
    folder.mkdir(parents=True, exist_ok=False)
    Image.fromarray(native).save(folder / "prediction_ids.png")
    np.savez_compressed(
        folder / "uncalibrated_scores.npz", maximum_softmax=maximum, normalized_entropy=entropy
    )
    save_comparison(folder / "comparison.png", image, None, native, report_id)
    result.update(
        {
            "provenance": {
                "dataset": release.identity(),
                "checkpoint_sha256": checkpoint_hash,
                "checkpoint_epoch": payload["epoch"],
                "architecture": payload["architecture"],
                "preprocessing": asdict(effective.data),
                "image_sha256": sha256_file(image_path),
                "device": str(device),
                "deterministic_requested": config.deterministic,
                "execution": execution_provenance(),
            },
            "alignment": "Prediction and score maps restored to EXIF-oriented original resolution",
            "masks": {
                "class_ids": "prediction_ids.png",
                "sha256": sha256_file(folder / "prediction_ids.png"),
            },
            "test_evaluated": False,
        }
    )
    write_json(folder / "damage_report.json", result)
    return folder


def analyze_mask_file(
    config: AnalyticsConfig, mask_path: Path, report_id: str, image_path: Path | None = None
) -> Path:
    """Analyze an existing ID mask; model uncertainty stays unknown without logits."""
    _safe_id(report_id)
    forbid_test_artifact(config, mask_path)
    with Image.open(mask_path) as source:
        source.load()
        if source.mode != "L":
            raise ValueError("Expected an L-mode class-ID PNG, not a colored mask")
        values = np.array(source)
    alignment = "Unknown original-image alignment: mask-only input"
    if image_path is not None:
        forbid_test_artifact(config, image_path)
        with Image.open(image_path) as source:
            if ImageOps.exif_transpose(source).size != (values.shape[1], values.shape[0]):
                raise ValueError("Image/mask dimensions differ")
        alignment = (
            "Matches supplied EXIF-oriented image dimensions; annotation correctness is unverified"
        )
    report = analyze_mask(values, config.regions, config.uncertainty)
    report.update(
        {
            "alignment": alignment,
            "input_mask_sha256": sha256_file(mask_path),
            "provenance": {"model_checkpoint": "unknown_for_mask_only_input"},
            "test_evaluated": False,
        }
    )
    folder = config.output_dir / "mask_analytics" / report_id
    folder.mkdir(parents=True, exist_ok=False)
    write_json(folder / "damage_report.json", report)
    return folder


def _safe_id(value: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", value):
        raise ValueError("Provide a safe report ID")
