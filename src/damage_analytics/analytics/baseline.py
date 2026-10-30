"""Immutable baseline audit: fresh validation, historical consistency, and error diagnosis."""

import csv
import json
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np

from damage_analytics.analytics.config import AnalyticsConfig, runtime_training_config
from damage_analytics.analytics.validation import evaluate_validation, scalar_foreground_miou
from damage_analytics.curation.verification import verify_checksums, write_checksums
from damage_analytics.io import sha256_file, write_csv, write_json
from damage_analytics.labels import CLASS_NAMES


def baseline_inputs(config: AnalyticsConfig) -> dict:
    """Fingerprint read-only artifacts; intentionally never open the Task 03 test-final folder."""
    if sha256_file(config.baseline_checkpoint) != config.baseline_sha256:
        raise ValueError("The original baseline checkpoint does not match its pinned SHA-256")
    run = config.baseline_checkpoint.parent
    paths = {
        "checkpoint": config.baseline_checkpoint,
        "history": run / "history.csv",
        "status": run / "status.json",
        "run": run / "run.json",
        "saved_validation": config.baseline_validation,
        "saved_confusion": config.baseline_validation.parent / "confusion_matrix.npy",
    }
    paths.update(
        {
            f"gallery_{path.name}": path
            for path in sorted((config.baseline_validation.parent / "comparisons").glob("*.png"))
        }
    )
    if (run / "learning_curves.png").exists():
        paths["learning_curves"] = run / "learning_curves.png"
    return {name: sha256_file(path) for name, path in paths.items()}


def audit_baseline(config: AnalyticsConfig) -> tuple[Path, dict]:
    """Run short validation inference only; verify/reuse an immutable completed audit."""
    inputs = baseline_inputs(config)
    destination = config.output_dir / "baseline_audit"
    if destination.exists():
        verify_checksums(destination)
        report = json.loads((destination / "audit.json").read_text())
        if report["input_hashes"] != inputs or report.get("verification_policy") != asdict(
            config.verification
        ):
            raise ValueError("Baseline evidence changed; choose a new Task 04 output directory")
        return destination, report
    saved = json.loads(config.baseline_validation.read_text())
    if (
        saved.get("split") != "validation"
        or saved.get("test_evaluated")
        or saved["checkpoint_sha256"] != config.baseline_sha256
    ):
        raise ValueError("The saved report is not validation evidence for the pinned checkpoint")
    with (config.baseline_checkpoint.parent / "history.csv").open(newline="") as stream:
        history = list(csv.DictReader(stream))
    status = json.loads((config.baseline_checkpoint.parent / "status.json").read_text())
    config.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".audit-", dir=config.output_dir) as temporary:
        stage = Path(temporary) / "audit"
        stage.mkdir()
        fresh = evaluate_validation(
            runtime_training_config(config),
            config.baseline_checkpoint,
            stage / "validation",
            comparisons=config.comparison_samples,
            cpu_threads=config.cpu_threads,
        )
        before = np.load(
            config.baseline_validation.parent / "confusion_matrix.npy", allow_pickle=False
        )
        saved_matrix_consistent = np.array_equal(before, saved["metrics"]["confusion_matrix"])
        independent_saved = scalar_foreground_miou(before)
        saved_score_consistent = (
            abs(independent_saved - saved["metrics"]["foreground_miou"]) < 1e-12
        )
        checkpoint_score_consistent = (
            abs(fresh["checkpoint_recorded_foreground_miou"] - saved["metrics"]["foreground_miou"])
            < 1e-12
        )
        after = np.array(fresh["metrics"]["confusion_matrix"], dtype=np.int64)
        support_consistent = np.array_equal(before.sum(axis=1), after.sum(axis=1))
        provenance_consistent = (
            saved.get("dataset") == fresh["dataset"]
            and saved.get("checkpoint_epoch") == fresh["checkpoint_epoch"]
        )
        changed_pixels = int(np.abs(before - after).sum() // 2)
        score_delta = fresh["metrics"]["foreground_miou"] - saved["metrics"]["foreground_miou"]
        loss_delta = fresh["loss"] - saved["loss"]
        best = max(history, key=lambda row: float(row["validation_foreground_miou"]))
        epoch_consistent = (
            int(best["epoch"]) == int(status["best_epoch"]) == fresh["checkpoint_epoch"]
            and abs(
                float(best["validation_foreground_miou"]) - saved["metrics"]["foreground_miou"]
            )
            < 1e-12
        )
        verified = (
            saved_matrix_consistent
            and saved_score_consistent
            and checkpoint_score_consistent
            and support_consistent
            and provenance_consistent
            and status.get("status") == "completed"
            and epoch_consistent
            and abs(score_delta) <= config.verification.metric_absolute_tolerance
            and changed_pixels / after.sum()
            <= config.verification.confusion_l1_half_fraction_tolerance
        )
        diagnosis = diagnose_confusion(after)
        report = {
            "status": "verified" if verified else "mismatch_requires_investigation",
            "input_hashes": inputs,
            "verification_policy": asdict(config.verification),
            "checkpoint_sha256": config.baseline_sha256,
            "historical_foreground_miou": saved["metrics"]["foreground_miou"],
            "fresh_foreground_miou": fresh["metrics"]["foreground_miou"],
            "foreground_miou_delta": score_delta,
            "validation_loss_delta": loss_delta,
            "confusion_reassignment_lower_bound_pixels": changed_pixels,
            "confusion_l1_half_fraction": changed_pixels / int(after.sum()),
            "confusion_difference_interpretation": "Half the count-matrix L1 difference is a "
            "lower bound, not an aligned per-pixel prediction disagreement count",
            "saved_matrix_consistent": saved_matrix_consistent,
            "saved_scalar_metric_consistent": saved_score_consistent,
            "checkpoint_recorded_score_consistent": checkpoint_score_consistent,
            "native_target_support_consistent": support_consistent,
            "saved_provenance_consistent": provenance_consistent,
            "best_epoch_consistent": epoch_consistent,
            "best_epoch": fresh["checkpoint_epoch"],
            "epochs_completed": len(history),
            "historical_device": saved["device"],
            "fresh_device": fresh["device"],
            "diagnosis": diagnosis,
            "test_evaluated_or_test_metrics_consumed": False,
            "interpretation": "A recomputation, not a model improvement; backend "
            "drift is reported explicitly",
        }
        write_json(stage / "audit.json", report)
        write_csv(
            stage / "class_errors.csv",
            diagnosis["class_errors"],
            list(diagnosis["class_errors"][0]),
        )
        write_checksums(stage)
        if baseline_inputs(config) != inputs:
            raise ValueError("Original baseline evidence changed during audit")
        stage.rename(destination)
    return destination, report


def diagnose_confusion(matrix: np.ndarray) -> dict:
    """Quantify false positives/negatives and major confusions; causes remain hypotheses."""
    errors, transitions = [], []
    for class_id, label in enumerate(CLASS_NAMES):
        tp, support, predicted = (
            int(matrix[class_id, class_id]),
            int(matrix[class_id].sum()),
            int(matrix[:, class_id].sum()),
        )
        errors.append(
            {
                "class_id": class_id,
                "source_label": label,
                "true_positive_pixels": tp,
                "false_positive_pixels": predicted - tp,
                "false_negative_pixels": support - tp,
                "target_pixels": support,
                "predicted_pixels": predicted,
                "predicted_to_target_area_ratio": predicted / support if support else None,
            }
        )
        for other in range(9):
            if other != class_id and matrix[class_id, other]:
                transitions.append(
                    {
                        "target": label,
                        "predicted": CLASS_NAMES[other],
                        "pixels": int(matrix[class_id, other]),
                        "fraction_of_target": int(matrix[class_id, other]) / support,
                    }
                )
    return {
        "class_errors": errors,
        "largest_error_transitions": sorted(
            transitions, key=lambda row: row["pixels"], reverse=True
        )[:15],
        "hypotheses": [
            "Rare-class support is limited; image and pixel frequency are different.",
            "Strong rare-pixel weights may trade precision for "
            "recall; ablate rather than assume causation.",
            "Spalling/corrosion/deformation texture overlap and missing context may contribute.",
            "256-pixel resizing can remove boundary detail; higher "
            "resolution costs more computation.",
            "Single seed and prior validation selection prevent "
            "significance/generalization claims.",
        ],
    }
