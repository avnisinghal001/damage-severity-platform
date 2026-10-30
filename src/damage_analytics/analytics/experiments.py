"""Predeclared single-factor experiments; training is launched only by an explicit command."""

import json
from dataclasses import asdict, replace
from pathlib import Path

import torch

from damage_analytics.analytics.baseline import baseline_inputs
from damage_analytics.analytics.config import AnalyticsConfig, runtime_training_config
from damage_analytics.io import sha256_file, write_json
from damage_analytics.training.audit import audit_resizing
from damage_analytics.training.config import TrainingConfig, validate_config
from damage_analytics.training.engine import train_model
from damage_analytics.training.evaluation import effective_config
from damage_analytics.training.manifests import open_release


def scientific_settings(config: TrainingConfig) -> dict:
    """Canonical model/data/optimizer settings, excluding machine-specific paths."""
    return json.loads(
        json.dumps(
            {
                "seed": config.seed,
                "data": asdict(config.data),
                "model": asdict(config.model),
                "optimization": asdict(config.optimization),
            }
        )
    )


def experiment_config(config: AnalyticsConfig, name: str) -> TrainingConfig:
    """Reconstruct saved baseline settings and apply exactly the registered single factor."""
    if sha256_file(config.baseline_checkpoint) != config.baseline_sha256:
        raise ValueError("Original baseline checkpoint changed")
    selected = next((item for item in config.experiments if item.name == name), None)
    if selected is None:
        raise ValueError(f"Unregistered experiment: {name}")
    payload = torch.load(config.baseline_checkpoint, map_location="cpu", weights_only=True)
    base = effective_config(runtime_training_config(config), payload)
    base = replace(
        base,
        optimization=replace(
            base.optimization,
            epochs=config.schedule.epochs or base.optimization.epochs,
            validation_every=config.schedule.validation_every,
            batch_size=config.schedule.batch_size or base.optimization.batch_size,
        ),
    )
    if payload["metadata"]["dataset"] != open_release(base).identity():
        raise ValueError("Baseline dataset/split provenance differs")
    if selected.image_size is not None:
        base = replace(base, data=replace(base.data, image_size=tuple(selected.image_size)))
    if selected.class_weighting is not None:
        base = replace(
            base, optimization=replace(base.optimization, class_weighting=selected.class_weighting)
        )
    if selected.max_class_weight is not None:
        base = replace(
            base,
            optimization=replace(base.optimization, max_class_weight=selected.max_class_weight),
        )
    validate_config(base)
    return base


def plan_experiments(config: AnalyticsConfig) -> dict:
    """Save designs/configuration snapshots only; never call the training loop."""
    baseline_inputs(config)
    payload = torch.load(config.baseline_checkpoint, map_location="cpu", weights_only=True)
    original = effective_config(runtime_training_config(config), payload)
    plan = []
    old_pixels = original.data.image_size[0] * original.data.image_size[1]
    for item in config.experiments:
        effective = experiment_config(config, item.name)
        plan.append(
            {
                "name": item.name,
                "hypothesis": item.hypothesis,
                "settings": scientific_settings(effective),
                "nominal_input_pixel_ratio": effective.data.image_size[0]
                * effective.data.image_size[1]
                / old_pixels,
                "training_executed_by_plan": False,
            }
        )
    result = {
        "baseline_checkpoint_sha256": config.baseline_sha256,
        "dataset": open_release(original).identity(),
        "historical_training_device": payload["metadata"]["device"],
        "current_requested_device": config.device,
        "experiments": plan,
        "selection_metric": "Pooled native-resolution validation foreground mIoU",
        "controls": [
            "Same frozen train/validation samples and group assignments",
            "Same architecture, seed, batch size, optimizer, epoch cap and early-stopping rule",
            "Same letterbox/argmax/nearest-native evaluation rules; "
            "resolution is an explicit intervention",
            "Retrain the matched control on the same backend as "
            "variants before causal comparisons",
        ],
        "limitations": [
            "Historical MPS vs current CPU training may differ even with the same seed.",
            "One seed cannot establish training-variance significance.",
            "Validation-driven improvements are exploratory; the already-used test is excluded.",
            "Different losses have different numeric scales; do not compare their loss values.",
        ],
        "test_evaluated_or_used_for_selection": False,
    }
    write_json(config.output_dir / "experiment_plan.json", result)
    for item in plan:
        write_json(config.output_dir / "plans" / f"{item['name']}.json", item)
    return result


def run_experiment(
    config: AnalyticsConfig, name: str, run_id: str, *, exploratory_epochs: int | None = None
) -> Path:
    """Explicit manual training entrypoint; all earlier task files remain read-only."""
    effective = experiment_config(config, name)
    if exploratory_epochs is not None:
        if not 1 <= exploratory_epochs < effective.optimization.epochs:
            raise ValueError("Exploratory epochs must be positive and below the full run budget")
        effective = replace(
            effective, optimization=replace(effective.optimization, epochs=exploratory_epochs)
        )
    torch.set_num_threads(config.cpu_threads)
    original_hash = sha256_file(config.baseline_checkpoint)
    folder = train_model(effective, run_id, exploratory=exploratory_epochs is not None)
    if sha256_file(config.baseline_checkpoint) != original_hash:
        raise ValueError("Original baseline changed unexpectedly")
    write_json(
        folder / "task04_experiment.json",
        {
            "name": name,
            "baseline_sha256": original_hash,
            "settings": scientific_settings(effective),
            "trained_from_scratch": True,
            "test_selection_allowed": False,
            "cpu_threads": config.cpu_threads,
            "exploratory": exploratory_epochs is not None,
            "performance": asdict(effective.performance),
        },
    )
    return folder


def audit_resolutions(config: AnalyticsConfig) -> dict:
    """Measure mask resampling for registered spatial settings; no model training or test data."""
    reports = {}
    selected = [
        item
        for item in config.experiments
        if item.name == "control" or item.image_size is not None
    ]
    for item in selected:
        effective = experiment_config(config, item.name)
        height, width = effective.data.image_size
        key = f"{height}x{width}"
        if key in reports:
            continue
        effective = replace(effective, output_dir=config.output_dir / "resolution_audit" / key)
        report = audit_resizing(effective, open_release(effective))
        reports[key] = {
            "meaning": "Ground-truth resampling only, not model accuracy",
            "train_classes": report["splits"]["train"]["roundtrip"]["classes"],
            "validation_classes": report["splits"]["validation"]["roundtrip"]["classes"],
        }
    summary = {
        "resolutions": reports,
        "model_training_executed": False,
        "test_masks_decoded": False,
    }
    write_json(config.output_dir / "resolution_audit/summary.json", summary)
    return summary
