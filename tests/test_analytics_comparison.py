"""Independent metric arithmetic, paired grouping and strict exclusion of final-test results."""

# Torch-dependent imports must occur after the optional-dependency skip.
# ruff: noqa: E402

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("torch")

from damage_analytics.analytics.baseline import diagnose_confusion
from damage_analytics.analytics.comparison import compare_experiments, paired_group_bootstrap
from damage_analytics.analytics.config import ComparisonPolicy
from damage_analytics.analytics.validation import read_validation_result, scalar_foreground_miou
from damage_analytics.io import write_json
from damage_analytics.training.metrics import metrics_from_confusion


def paired_files(tmp_path):
    paths = [tmp_path / "a", tmp_path / "b"]
    for path, perfect in zip(paths, (False, True), strict=True):
        path.mkdir()
        matrices = np.zeros((3, 9, 9), dtype=np.int64)
        matrices[:, 1, 1] = 2 if perfect else 1
        if not perfect:
            matrices[:, 1, 0] = 1
        np.savez_compressed(
            path / "per_image_confusion.npz",
            matrices=matrices,
            sample_ids=np.array(["one", "two", "three"]),
            group_ids=np.array(["group_a", "group_a", "group_b"]),
        )
    return paths


def test_paired_group_bootstrap_is_seeded_and_preserves_native_support(tmp_path):
    left, right = paired_files(tmp_path)
    result = paired_group_bootstrap(left, right, 20, 42)
    assert result == paired_group_bootstrap(left, right, 20, 42)
    assert result["known_validation_groups"] == 2
    assert result["descriptive_percentile_interval_95"] == [0.5, 0.5]
    assert not paired_group_bootstrap(left, right, 0, 42)["available"]


def test_mismatched_sample_identity_is_not_a_paired_comparison(tmp_path):
    left, right = paired_files(tmp_path)
    with np.load(right / "per_image_confusion.npz", allow_pickle=False) as source:
        matrices = source["matrices"]
    np.savez_compressed(
        right / "per_image_confusion.npz",
        matrices=matrices,
        sample_ids=np.array(["changed", "two", "three"]),
        group_ids=np.array(["group_a", "group_a", "group_b"]),
    )
    with pytest.raises(ValueError, match="identical"):
        paired_group_bootstrap(left, right, 20, 42)


def test_final_test_reports_are_refused_before_comparison(tmp_path):
    forbidden = tmp_path / "test_final"
    with pytest.raises(ValueError, match="Final-test"):
        read_validation_result(forbidden)
    disguised = tmp_path / "disguised"
    write_json(disguised / "metrics.json", {"split": "test", "test_evaluated": True})
    with pytest.raises(ValueError, match="validation-only"):
        read_validation_result(disguised)


def test_independent_scalar_formula_and_false_negative_diagnosis():
    matrix = np.zeros((9, 9), dtype=np.int64)
    matrix[7, 7], matrix[7, 0], matrix[0, 7] = 1, 3, 1
    assert scalar_foreground_miou(matrix) == 0.2
    diagnosis = diagnose_confusion(matrix)
    crush = diagnosis["class_errors"][7]
    assert crush["false_negative_pixels"] == 3 and crush["false_positive_pixels"] == 1
    assert scalar_foreground_miou(np.zeros((9, 9), dtype=np.int64)) is None


def comparison_fixture(tmp_path, monkeypatch, *, incompatible=False):
    import damage_analytics.analytics.comparison as comparison

    left, right = paired_files(tmp_path)
    reports = []
    for index, path in enumerate((left, right)):
        with np.load(path / "per_image_confusion.npz", allow_pickle=False) as source:
            matrix = source["matrices"].sum(0)
        report = {
            "dataset": {"release_id": "synthetic"},
            "protocol": "native-argmax-nearest-v1",
            "architecture": {"base_channels": 4},
            "metrics": metrics_from_confusion(matrix),
            "checkpoint_sha256": str(index) * 64,
            "experiment": "control" if index == 0 else "capped_weights",
            "split": "validation",
            "test_evaluated": False,
            "training_device": "cpu",
            "training_environment": {
                "torch": "same",
                "python": "same",
                "numpy": "same",
                "Pillow": "same",
            },
            "training_deterministic": True,
            "training_cpu_threads": 1,
            "training_code": {
                f"training/{name}.py": "same"
                for name in ("model", "transforms", "dataset", "engine", "losses", "weights")
            },
        }
        reports.append(report)
        np.save(path / "confusion_matrix.npy", matrix, allow_pickle=False)
        write_json(path / "metrics.json", report)
    if incompatible:
        reports[1]["training_environment"]["torch"] = "different"
    root = tmp_path / "audit"
    root.mkdir()
    monkeypatch.setattr(
        comparison, "audit_baseline", lambda config: (root, {"status": "verified"})
    )
    monkeypatch.setattr(comparison, "read_validation_result", lambda path: reports[0])
    monkeypatch.setattr(
        comparison,
        "evaluate_experiment",
        lambda config, name: (left, reports[0]) if name == "control" else (right, reports[1]),
    )
    # The baseline uses the same synthetic counts as the matched control.
    (root / "validation").mkdir()
    with np.load(left / "per_image_confusion.npz", allow_pickle=False) as source:
        np.savez_compressed(
            root / "validation/per_image_confusion.npz",
            **{key: source[key] for key in source.files},
        )
    return SimpleNamespace(output_dir=tmp_path / "outputs", comparison=ComparisonPolicy(10, 42))


def test_controlled_comparison_exports_metrics_without_training(tmp_path, monkeypatch):
    config = comparison_fixture(tmp_path, monkeypatch)
    folder = compare_experiments(config, ["control", "variant"], "synthetic_comparison")
    import json

    report = json.loads((folder / "comparison.json").read_text())
    assert report["matched_control_available"] and report["reference"] == "control"
    assert report["models"][2]["delta_vs_reference"] == 0.5
    assert not report["test_evaluated_or_used_for_selection"]


def test_comparison_rejects_mismatched_training_environment(tmp_path, monkeypatch):
    config = comparison_fixture(tmp_path, monkeypatch, incompatible=True)
    with pytest.raises(ValueError, match="backend and torch"):
        compare_experiments(config, ["control", "variant"], "invalid")


@pytest.mark.parametrize("field", ["training_selection_schedule", "training_performance"])
def test_comparison_rejects_different_budget_or_performance_policy(tmp_path, monkeypatch, field):
    import damage_analytics.analytics.comparison as comparison

    config = comparison_fixture(tmp_path, monkeypatch)
    original = comparison.evaluate_experiment

    def evaluate(config, name):
        path, report = original(config, name)
        report[field] = {"policy": "full" if name == "control" else "different"}
        return path, report

    monkeypatch.setattr(comparison, "evaluate_experiment", evaluate)
    with pytest.raises(ValueError, match="selection schedules"):
        compare_experiments(config, ["control", "variant"], "incompatible_schedule")


def test_scalar_metric_tampering_is_detected_from_counts(tmp_path):
    left, _ = paired_files(tmp_path)
    with np.load(left / "per_image_confusion.npz", allow_pickle=False) as source:
        matrix = source["matrices"].sum(0)
    np.save(left / "confusion_matrix.npy", matrix, allow_pickle=False)
    metrics = metrics_from_confusion(matrix)
    metrics["foreground_miou"] = 0.99
    write_json(
        left / "metrics.json", {"split": "validation", "test_evaluated": False, "metrics": metrics}
    )
    with pytest.raises(ValueError, match="do not follow"):
        read_validation_result(left)
