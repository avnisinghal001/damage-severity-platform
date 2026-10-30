"""Fresh validation-only inference with a second confusion accumulation implementation."""

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from damage_analytics.io import sha256_file, write_csv, write_json
from damage_analytics.training.checkpoints import load_checkpoint
from damage_analytics.training.config import IGNORE_INDEX, TrainingConfig
from damage_analytics.training.dataset import SegmentationDataset, make_loader
from damage_analytics.training.evaluation import effective_config
from damage_analytics.training.experiments import execution_provenance
from damage_analytics.training.losses import SegmentationLoss
from damage_analytics.training.manifests import open_release
from damage_analytics.training.metrics import metrics_from_confusion
from damage_analytics.training.plots import plot_confusion, save_comparison
from damage_analytics.training.runtime import configure_runtime
from damage_analytics.training.transforms import restore_prediction

PROTOCOL = "native-argmax-nearest-v1"


def scalar_foreground_miou(matrix: np.ndarray) -> float | None:
    """Cross-check metric arithmetic using explicit scalar counts instead of saved scores."""
    values = []
    for index in range(1, 9):
        tp = int(matrix[index, index])
        union = (
            sum(int(value) for value in matrix[index])
            + sum(int(value) for value in matrix[:, index])
            - tp
        )
        if union:
            values.append(tp / union)
    return sum(values) / len(values) if values else None


def evaluate_validation(
    config: TrainingConfig,
    checkpoint: Path,
    folder: Path,
    *,
    comparisons: int = 4,
    cpu_threads: int = 4,
) -> dict:
    """Recompute all 243 validation examples; no test branch or test-loader argument exists.

    This reuses the original model/preprocessing but accumulates counts via add.at,
    independently of Task 03's bincount helper. Native predictions are retained for
    reproducible region analytics and targeted qualitative inspection.
    """
    folder.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(cpu_threads)
    device = configure_runtime(config.seed, config.device, config.deterministic)
    release = open_release(config)
    checkpoint_hash = sha256_file(checkpoint)
    model, payload = load_checkpoint(checkpoint, device, expected_dataset=release.identity())
    saved_config = effective_config(config, payload)
    dataset = SegmentationDataset(
        release,
        release.samples("validation"),
        saved_config.data,
        training=False,
        seed=saved_config.seed,
    )
    loader = make_loader(
        dataset,
        saved_config.optimization.batch_size,
        config.num_workers,
        saved_config.seed,
        shuffle=False,
    )
    criterion = SegmentationLoss(
        torch.tensor(payload["metadata"]["class_weights"]["weights"]),
        saved_config.optimization.dice_weight,
    ).to(device)
    model.eval()
    matrices, records, loss_sum, loss_pixels = [], [], 0.0, 0
    (folder / "predictions").mkdir()
    with torch.inference_mode():
        for batch in loader:
            logits = model(batch["images"].to(device))
            targets = batch["masks"].to(device)
            loss = criterion(logits, targets)
            if not bool(torch.isfinite(loss)):
                raise ValueError("Nonfinite validation loss")
            valid = int((targets != IGNORE_INDEX).sum().item())
            loss_sum += float(loss.cpu()) * valid
            loss_pixels += valid
            predictions = logits.argmax(1).cpu().numpy().astype(np.uint8)
            for index, prediction in enumerate(predictions):
                sample = batch["samples"][index]
                native = restore_prediction(prediction, batch["frames"][index])
                target = batch["native_masks"][index]
                matrix = np.zeros((9, 9), dtype=np.int64)
                np.add.at(matrix, (target.ravel(), native.ravel()), 1)
                matrices.append(matrix)
                relative = f"predictions/{len(records):04d}.png"
                Image.fromarray(native).save(folder / relative)
                records.append(
                    {
                        "sample_id": sample.sample_id,
                        "group_id": sample.group_id,
                        "width": sample.width,
                        "height": sample.height,
                        "prediction": relative,
                        "prediction_sha256": sha256_file(folder / relative),
                    }
                )
                if len(records) <= comparisons:
                    with Image.open(release.file(sample.image)) as image:
                        save_comparison(
                            folder / "comparisons" / f"{len(records) - 1:03d}.png",
                            image,
                            target,
                            native,
                            sample.sample_id,
                        )
    matrix = np.sum(matrices, axis=0, dtype=np.int64)
    expected = np.array([row.pixel_counts for row in release.samples("validation")]).sum(axis=0)
    if not np.array_equal(matrix.sum(axis=1), expected):
        raise ValueError("Native ground-truth support differs from the frozen validation manifest")
    metrics = metrics_from_confusion(matrix)
    independent = scalar_foreground_miou(matrix)
    if metrics["foreground_miou"] != independent:
        if independent is None or not np.isclose(
            metrics["foreground_miou"], independent, atol=1e-14, rtol=0
        ):
            raise ValueError("Independent metric arithmetic disagrees")
    if sha256_file(checkpoint) != checkpoint_hash:
        raise ValueError("Checkpoint changed during validation")
    result = {
        "dataset": release.identity(),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_epoch": payload["epoch"],
        "checkpoint_recorded_foreground_miou": payload["validation_foreground_miou"],
        "split": "validation",
        "protocol": PROTOCOL,
        "samples": len(records),
        "metrics": metrics,
        "independent_foreground_miou": independent,
        "loss": loss_sum / loss_pixels,
        "loss_pixels": loss_pixels,
        "data_config": payload["metadata"]["config"]["data"],
        "architecture": payload["architecture"],
        "training_purpose": payload["metadata"]["purpose"],
        "device": str(device),
        "deterministic_requested": config.deterministic,
        "execution": execution_provenance(),
        "test_evaluated": False,
        "loss_comparison": "Different loss-weight variants have different "
        "objectives; do not rank their losses",
    }
    write_json(folder / "metrics.json", result)
    write_csv(folder / "per_class.csv", metrics["classes"], list(metrics["classes"][0]))
    write_csv(folder / "predictions.csv", records, list(records[0]))
    np.save(folder / "confusion_matrix.npy", matrix, allow_pickle=False)
    np.savez_compressed(
        folder / "per_image_confusion.npz",
        matrices=np.stack(matrices),
        sample_ids=np.array([row["sample_id"] for row in records]),
        group_ids=np.array([row["group_id"] for row in records]),
    )
    plot_confusion(folder / "confusion_matrix.png", matrix)
    return result


def read_validation_result(folder: Path) -> dict:
    """Refuse final-test reports even if a user points a comparison command at one."""
    if "test_final" in folder.parts:
        raise ValueError("Final-test artifacts cannot be consumed by Task 04")
    report = json.loads((folder / "metrics.json").read_text())
    if report.get("split") != "validation" or report.get("test_evaluated"):
        raise ValueError("Task 04 comparisons require validation-only results")
    matrix = np.load(folder / "confusion_matrix.npy", allow_pickle=False)
    if not np.array_equal(matrix, report["metrics"]["confusion_matrix"]):
        raise ValueError("Saved confusion matrix differs from the metrics report")
    recomputed = metrics_from_confusion(matrix)
    for key in ("classes", "foreground_miou", "mean_iou", "pixel_accuracy", "evaluated_pixels"):
        if report["metrics"][key] != recomputed[key]:
            raise ValueError("Reported metrics do not follow from the saved confusion counts")
    with np.load(folder / "per_image_confusion.npz", allow_pickle=False) as samples:
        if not np.array_equal(samples["matrices"].sum(axis=0), matrix):
            raise ValueError("Per-image counts do not sum to the reported confusion matrix")
    return report
