"""Authenticate checkpoints and saved validation predictions before drawing conclusions."""

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from damage_analytics.io import sha256_file
from damage_analytics.task07.checkpoints import read_payload
from damage_analytics.task07.data import validate_preprocessing
from damage_analytics.task07.metrics import classification_metrics
from damage_analytics.task07.release import Release, validate_development_ids


@dataclass(frozen=True)
class Run:
    root: Path
    checkpoint_sha256: str
    payload: dict
    history: list[dict]


@dataclass(frozen=True)
class Validation:
    run: Run
    release: Release
    report: dict
    predictions: tuple[dict, ...]
    metrics: dict
    report_path: Path


def load_run(root: Path, source: Release) -> Run:
    """CPU-load existing tensors/metadata, never initialize/download weights or run inference."""
    root = root.resolve()
    payload = read_payload(root / "best.pt")
    run = json.loads((root / "run.json").read_text())
    status = json.loads((root / "status.json").read_text())
    history = json.loads((root / "history.json").read_text())
    if (
        status["status"] != "completed"
        or payload["dataset"] != source.identity()
        or run["dataset"] != source.identity()
        or payload["epoch"] != status["best_epoch"]
        or payload["backbone"] != run["backbone"]
        or payload["settings"] != run["settings"]
        or payload["purpose"] != "training"
    ):
        raise ValueError("Run/checkpoint/source identity or completion status disagrees")
    validate_preprocessing(payload)
    validate_development_ids(payload, source)
    threshold = payload["threshold"]
    if (
        threshold["selected_on"] != "validation"
        or threshold["split_sha256"] != source.identity()["splits"]["validation"]
        or threshold["prediction_csv_sha256"] != sha256_file(root / "validation_predictions.csv")
    ):
        raise ValueError("Checkpoint threshold provenance changed")
    best = next(r for r in history if r["epoch"] == payload["epoch"])
    if not np.isclose(best["validation_roc_auc"], payload["selection_score"], atol=1e-12, rtol=0):
        raise ValueError("Checkpoint selection score differs from history")
    for tensor in payload["model_state"].values():
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise ValueError("Checkpoint contains nonfinite weights")
    payload = {k: v for k, v in payload.items() if k not in ("model_state", "torch_rng_state")}
    return Run(root, sha256_file(root / "best.pt"), payload, history)


def validate_predictions(
    samples: tuple[dict, ...], records: list[dict], threshold: float
) -> tuple[dict, ...]:
    """Require exactly the frozen validation population and source targets, with no test rows."""
    if any(r["split"] != "validation" or r["task"] != "classification" for r in samples):
        raise ValueError("Prediction analysis consumes classification validation only")
    expected = {r["sample_id"]: r for r in samples}
    if len(records) != len(expected) or {r["sample_id"] for r in records} != set(expected):
        raise ValueError("Prediction population missing, duplicated or outside validation")
    validated = []
    for record in records:
        row = expected[record["sample_id"]]
        probability, target = float(record["probability_crack"]), int(record["target"])
        prediction = int(record["prediction"])
        if (
            target != row["label"]
            or record["group_id"] != row["group_id"]
            or record["source_label"] != row["source_label"]
            or not np.isfinite(probability)
            or not 0 <= probability <= 1
            or prediction != int(probability >= threshold)
        ):
            raise ValueError("Prediction label/group/score/threshold integrity failed")
        validated.append(
            {
                "sample_id": row["sample_id"],
                "target": target,
                "probability_crack": probability,
                "prediction": prediction,
                "row": row,
            }
        )
    return tuple(validated)


def load_validation(run: Run, source: Release, release: Release) -> Validation:
    """Read only the hard-coded validation folder; independently recompute saved metrics."""
    folder = (
        run.root.parent.parent
        / "evaluations"
        / run.checkpoint_sha256[:16]
        / release.metadata["release_id"]
        / "validation"
    )
    path = folder / "metrics.json"
    report = json.loads(path.read_text())
    if (
        report["split"] != "validation"
        or report["test_evaluated"]
        or report["dataset"] != release.identity()
        or report["source_dataset"] != source.identity()
        or report["checkpoint_sha256"] != run.checkpoint_sha256
        or report["threshold_provenance"] != run.payload["threshold"]
    ):
        raise ValueError("Saved evaluation scope, checkpoint or threshold provenance differs")
    with (folder / "predictions.csv").open() as stream:
        records = list(csv.DictReader(stream))
    predictions = validate_predictions(
        release.samples("validation"), records, run.payload["threshold"]["threshold"]
    )
    metrics = classification_metrics(
        [r["target"] for r in predictions],
        [r["probability_crack"] for r in predictions],
        run.payload["threshold"]["threshold"],
    )
    if metrics["confusion_matrix"] != report["metrics"]["confusion_matrix"]:
        raise ValueError("Saved confusion matrix cannot be reproduced from predictions")
    for key in (
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "roc_auc",
        "pr_auc",
        "average_precision",
    ):
        actual, saved = metrics[key], report["metrics"][key]
        if (actual is None) != (saved is None) or (
            actual is not None and not np.isclose(actual, saved, atol=1e-10, rtol=0)
        ):
            raise ValueError(f"Saved {key} differs from recomputed prediction metric")
    for actual, saved in zip(metrics["classes"], report["metrics"]["classes"], strict=True):
        for key in ("precision", "recall", "f1"):
            if (actual[key] is None) != (saved[key] is None) or (
                actual[key] is not None and not np.isclose(actual[key], saved[key], atol=1e-10)
            ):
                raise ValueError("Saved class metrics differ from predictions")
    return Validation(run, release, report, predictions, metrics, path)
