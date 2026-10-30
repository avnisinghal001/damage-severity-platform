"""Controlled, validation-only comparison of manually trained Task 04 checkpoints."""

import json
import re
import tempfile
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from damage_analytics.analytics.baseline import audit_baseline
from damage_analytics.analytics.config import AnalyticsConfig, runtime_training_config
from damage_analytics.analytics.experiments import experiment_config, scientific_settings
from damage_analytics.analytics.validation import (
    PROTOCOL,
    evaluate_validation,
    read_validation_result,
    scalar_foreground_miou,
)
from damage_analytics.curation.verification import verify_checksums, write_checksums
from damage_analytics.io import sha256_file, write_csv, write_json
from damage_analytics.labels import CLASS_NAMES
from damage_analytics.training.config import PerformanceConfig
from damage_analytics.training.evaluation import effective_config


def evaluate_experiment(config: AnalyticsConfig, run_id: str) -> tuple[Path, dict]:
    """Validate only a completed registered run's best checkpoint, with actual-setting checks."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", run_id):
        raise ValueError("Invalid run ID")
    run = config.output_dir / "experiments" / run_id
    design = json.loads((run / "task04_experiment.json").read_text())
    status = json.loads((run / "status.json").read_text())
    checkpoint = run / "best.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    expected = experiment_config(config, design["name"])
    actual = effective_config(runtime_training_config(config), payload)
    recorded = deepcopy(design["settings"])
    # Historical files predate configurable cadence; their behavior was every epoch.
    recorded["optimization"].setdefault("validation_every", 1)
    exploratory = design.get("exploratory", False)
    if exploratory:
        epochs = recorded["optimization"]["epochs"]
        if type(epochs) is not int or not 1 <= epochs < expected.optimization.epochs:
            raise ValueError("Invalid exploratory epoch budget")
        expected = replace(expected, optimization=replace(expected.optimization, epochs=epochs))
    if (
        status["status"] != "completed"
        or payload["epoch"] != status["best_epoch"]
        or payload["metadata"]["purpose"]
        != ("exploratory_training" if exploratory else "training")
        or design["baseline_sha256"] != config.baseline_sha256
        or scientific_settings(actual) != scientific_settings(expected)
        or recorded != scientific_settings(expected)
    ):
        raise ValueError(
            "Run is incomplete, diagnostic, or differs from its registered scientific settings"
        )
    digest = sha256_file(checkpoint)
    details = {
        "training_selection_schedule": {
            key: getattr(actual.optimization, key)
            for key in ("epochs", "batch_size", "validation_every", "patience", "min_delta")
        },
        "training_performance": {
            **payload["metadata"]["config"].get("performance", asdict(PerformanceConfig())),
            "num_workers": payload["metadata"]["config"].get("num_workers", 0),
        },
        "exploratory": exploratory,
    }
    destination = config.output_dir / "evaluations" / digest[:16] / "validation"
    if destination.exists():
        verify_checksums(destination)
        return destination, {**read_validation_result(destination), **details}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".validation-", dir=destination.parent) as temp:
        stage = Path(temp) / "validation"
        result = evaluate_validation(
            runtime_training_config(config),
            checkpoint,
            stage,
            comparisons=config.comparison_samples,
            cpu_threads=config.cpu_threads,
        )
        result["experiment"] = design["name"]
        result["run_id"] = run_id
        result["training_device"] = payload["metadata"]["device"]
        result["training_environment"] = payload["metadata"]["environment"]
        result["training_deterministic"] = payload["metadata"]["config"]["deterministic"]
        result["training_cpu_threads"] = design["cpu_threads"]
        result["training_code"] = payload["metadata"]["code_sha256"]
        result.update(details)
        write_json(stage / "metrics.json", result)
        write_checksums(stage)
        stage.rename(destination)
    return destination, result


def paired_group_bootstrap(reference: Path, candidate: Path, repeats: int, seed: int) -> dict:
    """Resample known validation components, conditional on the already trained models.

    This is descriptive validation uncertainty, not calibrated prediction confidence,
    independent testing, seed variance, or a multiple-comparison-corrected significance test.
    """
    if repeats == 0:
        return {"available": False, "reason": "Disabled by configuration"}
    with np.load(reference / "per_image_confusion.npz", allow_pickle=False) as left:
        ids, groups, a = left["sample_ids"], left["group_ids"], left["matrices"]
    with np.load(candidate / "per_image_confusion.npz", allow_pickle=False) as right:
        if not np.array_equal(ids, right["sample_ids"]) or not np.array_equal(
            groups, right["group_ids"]
        ):
            raise ValueError(
                "Paired comparison requires identical frozen validation samples/groups"
            )
        b = right["matrices"]
    if a.shape != b.shape or not np.array_equal(a.sum(axis=2), b.sum(axis=2)):
        raise ValueError("Native ground-truth support differs between paired results")
    unique = sorted(set(groups.tolist()))
    grouped_a = np.array([a[groups == group].sum(axis=0) for group in unique])
    grouped_b = np.array([b[groups == group].sum(axis=0) for group in unique])
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(repeats):
        draw = rng.integers(0, len(unique), size=len(unique))
        first, second = (
            scalar_foreground_miou(grouped_a[draw].sum(0)),
            scalar_foreground_miou(grouped_b[draw].sum(0)),
        )
        if first is not None and second is not None:
            deltas.append(second - first)
    return {
        "available": bool(deltas),
        "repeats_requested": repeats,
        "valid_resamples": len(deltas),
        "known_validation_groups": len(unique),
        "seed": seed,
        "descriptive_percentile_interval_95": np.quantile(deltas, [0.025, 0.975]).tolist()
        if deltas
        else None,
        "limitations": "Conditional on fixed selected models and incomplete source groups; "
        "does not account for training seeds, validation adaptation, or multiple comparisons",
    }


def compare_experiments(config: AnalyticsConfig, run_ids: list[str], comparison_id: str) -> Path:
    """Compare completed runs using validation only; exclude the already-used test set."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", comparison_id) or len(
        set(run_ids)
    ) != len(run_ids):
        raise ValueError("Provide a safe comparison ID and unique run IDs")
    baseline_folder, audit = audit_baseline(config)
    if audit["status"] != "verified":
        raise ValueError("Resolve baseline verification discrepancies before comparing models")
    baseline_path = baseline_folder / "validation"
    baseline = read_validation_result(baseline_path)
    results = [("historical_baseline_recomputed", baseline_path, baseline)]
    for run_id in run_ids:
        folder, report = evaluate_experiment(config, run_id)
        if (
            report["dataset"] != baseline["dataset"]
            or report["protocol"] != PROTOCOL
            or report["architecture"] != baseline["architecture"]
        ):
            raise ValueError("Dataset, architecture or evaluation protocol differs")
        results.append((run_id, folder, report))
    controls = [item for item in results[1:] if item[2]["experiment"] == "control"]
    if len(controls) > 1:
        raise ValueError("Choose one matched control per comparison")
    reference = controls[0] if controls else results[0]
    for _, _, item in results[1:]:
        if controls and (
            item["training_device"] != reference[2]["training_device"]
            or item["training_environment"]["torch"]
            != reference[2]["training_environment"]["torch"]
        ):
            raise ValueError(
                "Matched control/variants must use the same training backend and torch version"
            )
        if controls:
            shared = reference[2]
            versions = ("Pillow", "numpy", "python")
            critical = (
                "training/model.py",
                "training/transforms.py",
                "training/dataset.py",
                "training/engine.py",
                "training/losses.py",
                "training/weights.py",
                "training/epochs.py",
                "training/cache.py",
            )
            if (
                item["training_deterministic"] != shared["training_deterministic"]
                or item["training_cpu_threads"] != shared["training_cpu_threads"]
                or any(
                    item["training_environment"][key] != shared["training_environment"][key]
                    for key in versions
                )
                or any(
                    item["training_code"].get(key) != shared["training_code"].get(key)
                    for key in critical
                )
                or item.get("training_selection_schedule")
                != shared.get("training_selection_schedule")
                or item.get("training_performance") != shared.get("training_performance")
            ):
                raise ValueError(
                    "Matched control/variants have incompatible training runtime/code or "
                    "selection schedules. Keep the old control as history; use a matched cohort."
                )
    destination = config.output_dir / "comparisons" / comparison_id
    destination.mkdir(parents=True, exist_ok=False)
    rows, per_class, bootstraps = [], [], {}
    for name, path, result in results:
        rows.append(
            {
                "run": name,
                "experiment": result.get("experiment", "historical_baseline"),
                "foreground_miou": result["metrics"]["foreground_miou"],
                "delta_vs_historical_baseline": result["metrics"]["foreground_miou"]
                - baseline["metrics"]["foreground_miou"],
                "delta_vs_reference": result["metrics"]["foreground_miou"]
                - reference[2]["metrics"]["foreground_miou"],
                "checkpoint_sha256": result["checkpoint_sha256"],
                "evaluation_path": str(path),
            }
        )
        for item in result["metrics"]["classes"]:
            per_class.append({"run": name, **item})
        if name != reference[0]:
            bootstraps[name] = paired_group_bootstrap(
                reference[1],
                path,
                config.comparison.bootstrap_repeats,
                config.comparison.bootstrap_seed,
            )
    with np.load(reference[1] / "per_image_confusion.npz", allow_pickle=False) as source:
        support = source["matrices"].sum(axis=2)
        groups = source["group_ids"]
        validation_support = [
            {
                "class_id": index,
                "source_label": label,
                "images": int(np.count_nonzero(support[:, index])),
                "known_groups": len(set(groups[support[:, index] > 0].tolist())),
            }
            for index, label in enumerate(CLASS_NAMES)
            if index
        ]
    report = {
        "models": rows,
        "reference": reference[0],
        "matched_control_available": bool(controls),
        "paired_group_bootstrap": bootstraps,
        "test_evaluated_or_used_for_selection": False,
        "interpretation": "Exploratory validation comparison; no improvement is presumed and no "
        "claim of unbiased post-selection test generalization is made",
        "validation_class_support": validation_support,
        "rare_class_warning": "Inspect actual validation image/group support; "
        "fragmented regions are not extra independent samples",
        "already_used_test": "Excluded; future confirmatory claims require genuinely "
        "new independent data",
    }
    write_json(destination / "comparison.json", report)
    write_csv(destination / "models.csv", rows, list(rows[0]))
    write_csv(destination / "per_class.csv", per_class, list(per_class[0]))
    write_checksums(destination)
    return destination
