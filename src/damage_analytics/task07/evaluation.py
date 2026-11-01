"""Frozen-threshold validation and explicit, non-overwritable final test evaluation."""

import json
from dataclasses import replace
from pathlib import Path

import torch
from torch import nn

from damage_analytics.io import sha256_file, write_csv, write_json
from damage_analytics.task07 import checkpoints
from damage_analytics.task07.config import TrainConfig
from damage_analytics.task07.crosscheck import overlap
from damage_analytics.task07.loops import loader_for, pass_epoch, prediction_rows
from damage_analytics.task07.metrics import classification_metrics
from damage_analytics.task07.plots import confusion
from damage_analytics.task07.release import Release, open_release
from damage_analytics.training.experiments import execution_provenance
from damage_analytics.training.runtime import configure_runtime


def evaluate(
    checkpoint: Path,
    release: Release,
    output: Path,
    device_name: str = "cpu",
    *,
    final_test: bool = False,
    confirm_compatible_labels: bool = False,
    source_release: Path | None = None,
) -> Path:
    """Never select a threshold here; external-domain validation is explicitly exploratory."""
    torch.set_num_threads(4)
    device = configure_runtime(42, device_name, False)
    payload = checkpoints.read_payload(checkpoint)
    if not payload.get("threshold") or payload["threshold"]["selected_on"] != "validation":
        raise ValueError("Checkpoint lacks a frozen validation-selected threshold")
    source = (
        release
        if release.identity() == payload["dataset"]
        else open_release(source_release or Path(payload["release_dir"]))
    )
    if source.identity() != payload["dataset"]:
        raise ValueError("Checkpoint source release/splits changed")
    if payload["threshold"]["split_sha256"] != source.identity()["splits"]["validation"]:
        raise ValueError("Threshold does not reference the checkpoint's validation split")
    external = release.identity() != source.identity()
    overlap_evidence = None
    split = "test" if final_test else "validation"
    if final_test:
        status = json.loads((checkpoint.parent / "status.json").read_text())
        if (
            payload["purpose"] != "training"
            or status["status"] != "completed"
            or sha256_file(checkpoint.parent / "best.pt") != sha256_file(checkpoint)
            or payload["epoch"] != status["best_epoch"]
            or not release.quality["final_evaluation_eligible"]
            or not source.quality["final_evaluation_eligible"]
        ):
            raise ValueError(
                "Final evaluation requires completed best checkpoint and verified "
                "source groups/resolved screening; provisional/synthetic runs cannot test"
            )
    rows = release.samples(split, final_test=final_test)
    if not rows:
        raise ValueError("No classification samples in requested split")
    if external:
        if not confirm_compatible_labels:
            raise ValueError(
                "Cross-dataset evaluation needs --confirm-compatible-labels after source review"
            )
        if (
            source.metadata["inputs"]["taxonomy"]["classification"]
            != release.metadata["inputs"]["taxonomy"]["classification"]
        ):
            raise ValueError("Classification label definitions differ")
        development_ids = set(payload["development_ids"])
        development = tuple(r for r in source.rows if r["sample_id"] in development_ids)
        source_inputs, target_inputs = source.metadata["inputs"], release.metadata["inputs"]
        evidence = overlap(
            development,
            rows,
            radius=max(source_inputs["phash_distance"], target_inputs["phash_distance"]),
            limit=max(source_inputs["max_near_candidates"], target_inputs["max_near_candidates"]),
        )
        overlap_evidence = evidence
        if (
            evidence["exact_pairs"]
            or evidence["near_candidates"]
            or evidence["screening_truncated"]
        ):
            raise ValueError(
                "Cross-dataset overlap candidates require resolution/quarantine before evaluation"
            )
    model, payload = checkpoints.load(checkpoint, device, payload)
    folder = (
        output
        / "evaluations"
        / sha256_file(checkpoint)[:16]
        / release.metadata["release_id"]
        / split
    )
    folder.mkdir(parents=True, exist_ok=False)
    settings = replace(TrainConfig(**payload["settings"]), device=str(device))
    loader = loader_for(release, rows, settings, payload["seed"])
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(payload["class_weights"], dtype=torch.float32, device=device),
        reduction="sum",
    )
    try:
        result = pass_epoch(model, loader, criterion, device)
        threshold = payload["threshold"]["threshold"]
        metrics = classification_metrics(result["targets"], result["probabilities"], threshold)
        predictions = prediction_rows(result, rows, threshold)
        write_csv(folder / "predictions.csv", predictions, list(predictions[0]))
        write_json(
            folder / "metrics.json",
            {
                "metrics": metrics,
                "loss": result["loss"],
                "dataset": release.identity(),
                "source_dataset": source.identity(),
                "split": split,
                "checkpoint_sha256": sha256_file(checkpoint),
                "threshold_provenance": payload["threshold"],
                "external_domain": external,
                "external_overlap_screen": overlap_evidence,
                "test_evaluated": final_test,
                "execution": execution_provenance(),
                "device": str(device),
                "training_config": payload["settings"],
                "training_seed": payload["seed"],
                "backbone": payload["backbone"],
                "pretrained_weights": payload["weights"],
                "purpose": "final_test" if final_test else "exploratory_validation",
                "training_purpose": payload["purpose"],
                "synthetic": payload["purpose"] == "synthetic_smoke",
                "grouping": release.quality,
                "scope": "Crack/no-crack only; no general damage, OOD rejection or severity claim",
            },
        )
        confusion(
            folder / "confusion_matrix.png", metrics["confusion_matrix"], ["No crack", "Crack"]
        )
    except Exception as exc:
        write_json(folder / "failure.json", {"error": str(exc), "test_attempted": final_test})
        raise
    return folder


def compare(paths: list[Path], output: Path) -> dict:
    """Compare validation only; reject final-test feedback and incomparable validation cohorts."""
    if any("test" in path.parts or "test_final" in path.parts for path in paths):
        raise ValueError("Comparison rejects test paths before reading their contents")
    reports = [json.loads(path.read_text()) for path in paths]
    if not reports or any(r["split"] != "validation" or r["test_evaluated"] for r in reports):
        raise ValueError("Comparison consumes validation results only, never test results")
    if any(r["dataset"] != reports[0]["dataset"] for r in reports):
        raise ValueError(
            "Compare the same evaluation dataset/split; external results need separate tables"
        )
    result = {
        "models": [
            {
                "path": str(path),
                "checkpoint": r["checkpoint_sha256"],
                "backbone": r["backbone"],
                "training_seed": r["training_seed"],
                "training_config": r["training_config"],
                "source_dataset": r["source_dataset"],
                **r["metrics"],
            }
            for path, r in zip(paths, reports, strict=True)
        ],
        "limitations": "Validation comparison is exploratory; training seed/budget/domain "
        "must be considered. No unbiased generalization or statistical significance claim.",
    }
    if output.exists():
        raise FileExistsError("Comparison ID already exists")
    write_json(output, result)
    return result
