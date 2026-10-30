"""Targeted validation error galleries derived from the fresh audited predictions."""

import csv
from pathlib import Path

import numpy as np
from PIL import Image

from damage_analytics.analytics.baseline import audit_baseline
from damage_analytics.analytics.config import AnalyticsConfig, runtime_training_config
from damage_analytics.io import contained_file, sha256_file, write_json
from damage_analytics.labels import CLASS_NAMES
from damage_analytics.training.manifests import open_release, read_pair
from damage_analytics.training.plots import save_comparison


def targeted_review(config: AnalyticsConfig) -> Path:
    """Select explicitly bad cases, not attractive examples or an unbiased metric sample."""
    audit, _ = audit_baseline(config)
    validation = audit / "validation"
    with np.load(validation / "per_image_confusion.npz", allow_pickle=False) as source:
        matrices, ids = source["matrices"], source["sample_ids"].tolist()
    with (validation / "predictions.csv").open(newline="", encoding="utf-8") as stream:
        predictions = {row["sample_id"]: row for row in csv.DictReader(stream)}
    release = open_release(runtime_training_config(config))
    samples = {row.sample_id: row for row in release.samples("validation")}
    choices = []
    for class_id, kind in (
        (3, "false_positive_pixels"),
        (7, "false_negative_fraction"),
        (4, "false_negative_fraction"),
        (8, "false_positive_pixels"),
    ):
        candidates = []
        for index, matrix in enumerate(matrices):
            tp = int(matrix[class_id, class_id])
            support, predicted = int(matrix[class_id].sum()), int(matrix[:, class_id].sum())
            if kind == "false_negative_fraction" and not support:
                continue
            score = (
                (support - tp) / support if kind == "false_negative_fraction" else predicted - tp
            )
            candidates.append((score, ids[index], index))
        if not candidates:
            continue
        score, sample_id, index = max(candidates)
        choices.append(
            {
                "sample_id": sample_id,
                "class_id": class_id,
                "source_label": CLASS_NAMES[class_id],
                "selection": kind,
                "score": score,
                "confusion": matrices[index].tolist(),
            }
        )
    destination = config.output_dir / "targeted_validation_review"
    destination.mkdir(parents=True, exist_ok=False)
    for index, row in enumerate(choices[: config.comparison_samples]):
        sample = samples[row["sample_id"]]
        image, target = read_pair(release, sample)
        record = predictions[sample.sample_id]
        path = contained_file(validation, record["prediction"])
        if sha256_file(path) != record["prediction_sha256"]:
            raise ValueError("Audited prediction changed")
        with Image.open(path) as source:
            predicted = np.array(source)
        save_comparison(
            destination / f"{index:03d}.png",
            image,
            np.asarray(target),
            predicted,
            f"TARGETED: {sample.sample_id} | {row['source_label']}",
        )
    write_json(
        destination / "selection.json",
        {
            "checkpoint_sha256": config.baseline_sha256,
            "split": "validation",
            "examples": choices,
            "selection_bias": "Explicit worst cases for diagnosis, not unbiased "
            "model-performance examples",
            "test_evaluated": False,
        },
    )
    return destination
