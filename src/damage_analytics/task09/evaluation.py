"""Validation-only native-grid evaluation, including authenticated cached prediction replay."""

import csv
import json
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from damage_analytics.analytics.validation import (
    PROTOCOL,
    read_validation_result,
    scalar_foreground_miou,
)
from damage_analytics.io import contained_file, sha256_file, write_csv, write_json
from damage_analytics.task09.checkpoints import load, model_identity
from damage_analytics.task09.config import Baseline, Config
from damage_analytics.task09.visualization import failure_panel
from damage_analytics.training.dataset import SegmentationDataset, make_loader
from damage_analytics.training.manifests import Release, Sample, read_pair
from damage_analytics.training.metrics import confusion_matrix, metrics_from_confusion
from damage_analytics.training.plots import plot_confusion
from damage_analytics.training.runtime import configure_runtime
from damage_analytics.training.transforms import restore_prediction


def evaluate_predictions(
    release: Release,
    predictions: Iterable[tuple[Sample, np.ndarray]],
    output: Path,
    provenance: dict,
    *,
    gallery_samples: int = 2,
) -> dict:
    """Require the exact validation population, integer native masks, and no filtering."""
    output.mkdir(parents=True, exist_ok=False)
    samples = {s.sample_id: s for s in release.samples("validation")}
    matrix, records, class_rows, matrices = np.zeros((9, 9), dtype=np.int64), [], [], []
    seen = set()
    for sample, prediction in predictions:
        if (
            sample.sample_id not in samples
            or sample != samples[sample.sample_id]
            or sample.sample_id in seen
        ):
            raise ValueError("Prediction sample is duplicated or outside frozen validation")
        seen.add(sample.sample_id)
        if (
            prediction.shape != (sample.height, sample.width)
            or not np.issubdtype(prediction.dtype, np.integer)
            or np.any((prediction < 0) | (prediction > 8))
        ):
            raise ValueError("Prediction dimensions/IDs differ from native validation image")
        with Image.open(release.file(sample.mask)) as source:
            target = np.asarray(source)
            if (
                source.mode != "L"
                or source.size != (sample.width, sample.height)
                or tuple(np.bincount(target.ravel(), minlength=9)) != sample.pixel_counts
            ):
                raise ValueError("Native validation ground truth differs from sealed manifest")
            current = confusion_matrix(target, prediction)
        matrix += current
        matrices.append(current)
        relative = f"predictions/{len(records):04d}.png"
        (output / "predictions").mkdir(exist_ok=True)
        Image.fromarray(prediction.astype(np.uint8)).save(output / relative)
        records.append(
            {
                "sample_id": sample.sample_id,
                "group_id": sample.group_id,
                "width": sample.width,
                "height": sample.height,
                "prediction": relative,
                "prediction_sha256": sha256_file(output / relative),
            }
        )
        for row in metrics_from_confusion(current)["classes"][1:]:
            i = row["class_id"]
            class_rows.append(
                {
                    "sample_id": sample.sample_id,
                    **row,
                    "true_positive_pixels": int(current[i, i]),
                    "false_positive_pixels": int(current[:, i].sum() - current[i, i]),
                    "false_negative_pixels": int(current[i].sum() - current[i, i]),
                }
            )
    if seen != set(samples):
        raise ValueError("Incomplete validation predictions; partial metrics are not published")
    if not np.array_equal(
        matrix.sum(axis=1), np.array([s.pixel_counts for s in samples.values()]).sum(0)
    ):
        raise ValueError("Ground truth support differs from frozen validation")
    metrics = metrics_from_confusion(matrix)
    actual, independent = metrics["foreground_miou"], scalar_foreground_miou(matrix)
    if (actual is None) != (independent is None) or (
        actual is not None and not np.isclose(actual, independent, rtol=0, atol=1e-14)
    ):
        raise ValueError("Independent foreground mIoU computation differs")
    result = {
        **provenance,
        "dataset": release.identity(),
        "split": "validation",
        "protocol": PROTOCOL,
        "samples": len(seen),
        "metrics": metrics,
        "test_evaluated": False,
        "evaluation_filtering": "None. Raw model argmax IDs, unpad, nearest-native restoration",
        "loss": None,
        "loss_status": "ID-mask evaluation does not recover logits/loss",
    }
    write_json(output / "metrics.json", result)
    write_csv(output / "predictions.csv", records, list(records[0]))
    write_csv(output / "per_class.csv", metrics["classes"], list(metrics["classes"][0]))
    write_csv(output / "per_image_class_metrics.csv", class_rows, list(class_rows[0]))
    np.save(output / "confusion_matrix.npy", matrix, allow_pickle=False)
    np.savez_compressed(
        output / "per_image_confusion.npz",
        matrices=np.stack(matrices),
        sample_ids=np.array([r["sample_id"] for r in records]),
        group_ids=np.array([r["group_id"] for r in records]),
    )
    plot_confusion(output / "confusion_matrix.png", matrix)
    by_id = {r["sample_id"]: r for r in records}
    selected = []
    for class_id in range(1, 9):
        choices = [r for r in class_rows if r["class_id"] == class_id]
        for error in ("false_positive_pixels", "false_negative_pixels"):
            chosen = sorted(
                [r for r in choices if r[error]], key=lambda r: (-r[error], r["sample_id"])
            )[:gallery_samples]
            for number, row in enumerate(chosen):
                sample = samples[row["sample_id"]]
                image, target = read_pair(release, sample)
                record = by_id[sample.sample_id]
                with Image.open(output / record["prediction"]) as source:
                    prediction = np.array(source)
                relative = f"failures/class_{class_id}/{error}_{number:02d}.png"
                failure_panel(
                    output / relative,
                    image,
                    np.asarray(target),
                    prediction,
                    sample.sample_id,
                    class_id,
                )
                selected.append(
                    {
                        "class_id": class_id,
                        "error": error,
                        "sample_id": sample.sample_id,
                        "error_pixels": row[error],
                        "visualization": relative,
                    }
                )
    write_json(
        output / "failure_selection.json",
        {
            "selection": selected,
            "scope": "Largest absolute FP/FN pixel counts, validation only; illustrative, "
            "not random",
        },
    )
    return result


def replay_baseline(config: Config, release: Release, baseline: Baseline, output: Path) -> dict:
    """Re-evaluate saved native predictions without competing with Task 08 on MPS."""
    if sha256_file(baseline.checkpoint) != baseline.sha256:
        raise ValueError("Pinned baseline checkpoint changed")
    saved = read_validation_result(baseline.validation)
    if (
        saved["dataset"] != release.identity()
        or saved["checkpoint_sha256"] != baseline.sha256
        or saved["protocol"] != PROTOCOL
    ):
        raise ValueError("Saved validation checkpoint/dataset/protocol differs")
    model, payload, data = load(baseline.checkpoint, torch.device("cpu"), release.identity())
    del model  # CPU-only checkpoint shape validation; no forward pass or weight download.
    if json.loads(json.dumps(asdict(data))) != saved["data_config"]:
        raise ValueError("Cached preprocessing differs from its checkpoint")
    with (baseline.validation / "predictions.csv").open() as stream:
        records = list(csv.DictReader(stream))
    samples = {s.sample_id: s for s in release.samples("validation")}
    if len(records) != len(samples) or {r["sample_id"] for r in records} != set(samples):
        raise ValueError("Cached prediction IDs differ from exact validation population")

    def predictions():
        for row in records:
            sample = samples[row["sample_id"]]
            path = contained_file(baseline.validation, row["prediction"])
            if (
                sha256_file(path) != row["prediction_sha256"]
                or row["group_id"] != sample.group_id
                or (int(row["width"]), int(row["height"])) != (sample.width, sample.height)
            ):
                raise ValueError("Cached prediction hash/group differs")
            with Image.open(path) as image:
                if image.mode != "L":
                    raise ValueError("Cached predictions must be class-ID PNGs")
                yield sample, np.array(image)

    result = evaluate_predictions(
        release,
        predictions(),
        output,
        {
            "checkpoint_sha256": baseline.sha256,
            "checkpoint_epoch": payload["epoch"],
            "model": model_identity(payload, baseline.sha256),
            "data_config": asdict(data),
            "mode": "authenticated_native_prediction_replay",
            "fresh_model_forward": False,
            "source_report_sha256": sha256_file(baseline.validation / "metrics.json"),
            "source_predictions_csv_sha256": sha256_file(baseline.validation / "predictions.csv"),
        },
        gallery_samples=config.gallery_samples_per_class,
    )
    if result["metrics"]["confusion_matrix"] != saved["metrics"]["confusion_matrix"]:
        raise ValueError("Recomputed native predictions disagree with historical confusion matrix")
    return result


def fresh_evaluation(
    config: Config, release: Release, checkpoint: Path, output: Path, device_name: str = "cpu"
) -> dict:
    """Manual full-validation forward pass; never has a test-split branch."""
    torch.set_num_threads(config.cpu_threads)
    device = configure_runtime(config.seed, device_name, False)
    digest = sha256_file(checkpoint)
    model, payload, data = load(checkpoint, device, release.identity())
    dataset = SegmentationDataset(
        release, release.samples("validation"), data, training=False, seed=config.seed
    )
    loader = make_loader(dataset, config.optimization.batch_size, 0, config.seed, shuffle=False)

    def predictions():
        with torch.inference_mode():
            for batch in loader:
                logits = model(batch["images"].to(device))
                if logits.shape[1] != 9 or not bool(torch.isfinite(logits).all()):
                    raise ValueError("Model produced invalid nine-class logits")
                values = logits.argmax(1).cpu().numpy().astype(np.uint8)
                for index, prediction in enumerate(values):
                    yield (
                        batch["samples"][index],
                        restore_prediction(prediction, batch["frames"][index]),
                    )

    result = evaluate_predictions(
        release,
        predictions(),
        output,
        {
            "checkpoint_sha256": digest,
            "checkpoint_epoch": payload["epoch"],
            "model": model_identity(payload, digest),
            "data_config": asdict(data),
            "mode": "fresh_validation_forward",
            "fresh_model_forward": True,
            "device": str(device),
        },
        gallery_samples=config.gallery_samples_per_class,
    )
    if sha256_file(checkpoint) != digest:
        raise ValueError("Checkpoint changed during evaluation")
    return result
