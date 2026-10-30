"""Synthetic checkpoint integration; no real experiment training or real test data is used."""

# Torch-dependent imports must occur after the optional-dependency skip.
# ruff: noqa: E402

import json
from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from damage_analytics.analytics.baseline import audit_baseline
from damage_analytics.analytics.config import (
    AnalyticsConfig,
    ComparisonPolicy,
    Experiment,
    RegionPolicy,
    SupervisionPolicy,
    UncertaintyPolicy,
    VerificationPolicy,
    runtime_training_config,
    validate_analytics_config,
)
from damage_analytics.analytics.experiments import (
    audit_resolutions,
    experiment_config,
    plan_experiments,
)
from damage_analytics.analytics.inference import (
    analyze_mask_file,
    forbid_test_artifact,
    structured_predict,
)
from damage_analytics.analytics.supervision import assess_supervision
from damage_analytics.analytics.validation import evaluate_validation
from damage_analytics.io import sha256_file, write_csv, write_json
from damage_analytics.training.checkpoints import save_checkpoint
from damage_analytics.training.experiments import execution_provenance
from damage_analytics.training.manifests import open_release
from damage_analytics.training.model import LightweightUNet
from damage_analytics.training.weights import training_weights


@pytest.fixture
def analytics_config(training_release, tmp_path):
    """An untrained tiny synthetic checkpoint with honest fresh validation evidence."""
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(7)
    training = training_release
    release = open_release(training)
    model = LightweightUNet(training.model)
    optimizer = torch.optim.AdamW(model.parameters())
    _, weights = training_weights(release.samples("train"), training.optimization)
    folder = tmp_path / "prior_task03"
    folder.mkdir()
    checkpoint = folder / "best.pt"
    metadata = {
        "purpose": "training",
        "dataset": release.identity(),
        "config": training.snapshot(),
        "class_weights": weights,
        "device": "cpu",
        **execution_provenance(),
    }
    save_checkpoint(
        checkpoint,
        model,
        optimizer,
        epoch=1,
        score=0.0,
        metadata=metadata,
        architecture=training.model,
    )
    result = evaluate_validation(
        training, checkpoint, folder / "validation", comparisons=1, cpu_threads=1
    )
    score = result["metrics"]["foreground_miou"]
    save_checkpoint(
        checkpoint,
        model,
        optimizer,
        epoch=1,
        score=score,
        metadata=metadata,
        architecture=training.model,
    )
    result["checkpoint_sha256"] = sha256_file(checkpoint)
    result["checkpoint_recorded_foreground_miou"] = score
    write_json(folder / "validation/metrics.json", result)
    write_csv(
        folder / "history.csv",
        [{"epoch": 1, "validation_foreground_miou": score}],
        ["epoch", "validation_foreground_miou"],
    )
    write_json(folder / "status.json", {"status": "completed", "best_epoch": 1})
    write_json(folder / "run.json", metadata)
    config = AnalyticsConfig(
        training,
        checkpoint,
        sha256_file(checkpoint),
        folder / "validation/metrics.json",
        tmp_path / "task04",
        RegionPolicy(),
        UncertaintyPolicy(),
        VerificationPolicy(),
        SupervisionPolicy(),
        ComparisonPolicy(10, 42),
        (
            Experiment("control", "Matched control"),
            Experiment("capped_weights", "Cap rare weights", max_class_weight=1.0),
            Experiment("resolution_32", "Increase resolution", image_size=(32, 32)),
        ),
        device="cpu",
        cpu_threads=1,
        comparison_samples=1,
    )
    yield config
    torch.set_num_threads(before)


def test_baseline_is_freshly_verified_and_prior_files_preserved(analytics_config):
    config = analytics_config
    before = sha256_file(config.baseline_checkpoint)
    folder, report = audit_baseline(config)
    assert report["status"] == "verified"
    assert report["checkpoint_recorded_score_consistent"]
    assert report["confusion_reassignment_lower_bound_pixels"] == 0
    assert report["test_evaluated_or_test_metrics_consumed"] is False
    repeated, second = audit_baseline(config)
    assert repeated == folder and second == report
    assert sha256_file(config.baseline_checkpoint) == before
    (folder / "audit.json").write_text("tampered")
    with pytest.raises(ValueError, match="checksum"):
        audit_baseline(config)


def test_exploratory_run_is_explicit_and_can_be_evaluated(analytics_config):
    from damage_analytics.analytics.comparison import evaluate_experiment
    from damage_analytics.analytics.experiments import run_experiment

    folder = run_experiment(analytics_config, "control", "pilot", exploratory_epochs=1)
    design = json.loads((folder / "task04_experiment.json").read_text())
    assert design["exploratory"] and design["settings"]["optimization"]["epochs"] == 1
    output, report = evaluate_experiment(analytics_config, "pilot")
    assert report["exploratory"] and report["training_selection_schedule"]["epochs"] == 1
    assert (output / "confusion_matrix.npy").exists()


def test_legacy_control_design_without_cadence_remains_evaluable(analytics_config):
    from damage_analytics.analytics.comparison import evaluate_experiment
    from damage_analytics.analytics.experiments import run_experiment

    folder = run_experiment(analytics_config, "control", "legacy_compatible")
    path = folder / "task04_experiment.json"
    design = json.loads(path.read_text())
    del design["settings"]["optimization"]["validation_every"]
    del design["performance"]
    del design["exploratory"]
    write_json(path, design)
    _, report = evaluate_experiment(analytics_config, "legacy_compatible")
    assert not report["exploratory"]
    assert report["training_selection_schedule"]["validation_every"] == 1


def test_plan_never_trains_and_each_variant_changes_only_its_factor(analytics_config, monkeypatch):
    import damage_analytics.analytics.experiments as experiments

    def forbidden(*args, **kwargs):
        raise AssertionError("Plan must never launch training")

    monkeypatch.setattr(experiments, "train_model", forbidden)
    report = plan_experiments(analytics_config)
    assert all(not item["training_executed_by_plan"] for item in report["experiments"])
    control = experiment_config(analytics_config, "control")
    capped = experiment_config(analytics_config, "capped_weights")
    resolution = experiment_config(analytics_config, "resolution_32")
    assert capped.data == control.data
    assert capped.optimization.max_class_weight == 1.0
    assert resolution.optimization == control.optimization and resolution.model == control.model
    assert resolution.data.image_size == (32, 32)
    assert control.output_dir == analytics_config.output_dir
    resolution_audit = audit_resolutions(analytics_config)
    assert set(resolution_audit["resolutions"]) == {"16x16", "32x32"}
    assert (
        not resolution_audit["model_training_executed"]
        and not resolution_audit["test_masks_decoded"]
    )


def test_structured_native_inference_and_test_firewall(analytics_config):
    config = analytics_config
    release = open_release(runtime_training_config(config))
    with pytest.raises(ValueError, match="test-set"):
        forbid_test_artifact(config, release.root / release.rows["test"][0].image)
    folder = structured_predict(
        config,
        config.baseline_checkpoint,
        release.root / release.samples("validation")[0].image,
        "synthetic",
    )
    report = json.loads((folder / "damage_report.json").read_text())
    assert report["image"] == {"width": 12, "height": 8, "area_pixels": 96}
    assert sum(row["area_pixels"] for row in report["classes"]) == 96
    assert report["provenance"]["checkpoint_sha256"] == config.baseline_sha256
    assert report["uncertainty"]["calibrated"] is False
    assert report["overall_structural_severity"]["score"] is None
    with np.load(folder / "uncalibrated_scores.npz", allow_pickle=False) as scores:
        assert scores["maximum_softmax"].shape == scores["normalized_entropy"].shape == (8, 12)


def test_supervision_excludes_test_and_never_forces_a_classifier(analytics_config, monkeypatch):
    from damage_analytics.training.manifests import Release

    original = Release.samples

    def guard(self, split, **kwargs):
        assert split != "test"
        return original(self, split, **kwargs)

    monkeypatch.setattr(Release, "samples", guard)
    report = assess_supervision(analytics_config)
    assert not report["independent_region_severity_classifier"]["implemented_or_trained"]
    assert not report["planning_group_gate"]["passed"]
    assert report["test_masks_decoded"] is False


def test_mask_only_output_keeps_uncertainty_unknown(analytics_config, tmp_path):
    from PIL import Image

    path = tmp_path / "custom.png"
    Image.fromarray(np.full((3, 5), 3, dtype=np.uint8)).save(path)
    folder = analyze_mask_file(analytics_config, path, "mask_only")
    report = json.loads((folder / "damage_report.json").read_text())
    assert report["uncertainty"]["status"] == "unknown_no_model_scores"
    assert report["regions"][0]["spalling_assessment"]["label"] is None


def test_invalid_experiments_and_test_report_inputs_are_rejected(analytics_config):
    config = analytics_config
    with pytest.raises(ValueError, match="one factor"):
        validate_analytics_config(
            replace(
                config,
                experiments=(
                    Experiment("invalid", "Two changes", image_size=(32, 32), max_class_weight=1),
                ),
            )
        )
    with pytest.raises(ValueError, match="final-test"):
        validate_analytics_config(
            replace(config, baseline_validation=config.output_dir / "test_final/metrics.json")
        )
    with pytest.raises(ValueError, match="Connectivity"):
        validate_analytics_config(replace(config, regions=replace(config.regions, connectivity=5)))


def test_task04_exposes_no_final_test_command():
    from damage_analytics.analytics.cli import main

    with pytest.raises(SystemExit) as exc:
        main(["final-test"])
    assert exc.value.code == 2
