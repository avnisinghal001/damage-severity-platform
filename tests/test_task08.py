"""Synthetic regression checks for search reporting and validation-only reliability tools."""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from damage_analytics.io import sha256_file
from damage_analytics.task07.config import load_config as load_task07
from damage_analytics.task07.crosscheck import overlap
from damage_analytics.task07.data import MEAN, STD, prepare_image, validate_preprocessing
from damage_analytics.task07.duplicates import HammingTree
from damage_analytics.task07.metrics import ranking_metrics
from damage_analytics.task07.release import Release
from damage_analytics.task08.diagnostics import curve_points, threshold_sensitivity
from damage_analytics.task08.galleries import card, pair_sheets
from damage_analytics.task08.inputs import validate_predictions
from damage_analytics.task08.integrity import candidate_components, investigate_pairs


def fingerprints(identifier, phash, dhash=None):
    return {
        "sample_id": identifier,
        "phash": phash,
        "dhash": phash if dhash is None else dhash,
        "image_sha256": identifier + "-raw",
        "decoded_sha256": identifier + "-decoded",
    }


def test_no_radius_hits_still_reports_real_search_work():
    left = (fingerprints("a", 0), fingerprints("b", 1))
    right = (fingerprints("c", (1 << 63) - 1),)
    report = overlap(left, right, radius=1)
    assert report["near_candidate_count"] == report["phash_candidate_hits_consumed"] == 0
    assert report["screened_comparisons"] == 1  # The other node is correctly pruned.
    assert report["possible_cartesian_pairs"] == 2
    assert report["phash_search"]["queries"] == report["queried_records"] == 1
    assert report["dhash_comparisons"] == 0 and not report["screening_truncated"]


def test_search_telemetry_and_duplicate_hash_nodes():
    tree, stats = HammingTree(), {}
    for value, identifier in ((0, "a"), (0, "b"), (3, "c")):
        tree.add(value, identifier)
    assert set(tree.query(0, 0, stats=stats)) == {"a", "b"}
    assert stats == {"queries": 1, "node_distance_evaluations": 1, "radius_hits": 2}


def test_indexed_overlap_matches_independent_brute_force():
    rng = np.random.default_rng(21)
    left = tuple(fingerprints(str(i), int(v)) for i, v in enumerate(rng.integers(0, 2**62, 60)))
    right = tuple(
        fingerprints(f"r{i}", r["phash"] ^ 3, r["dhash"] ^ 1) for i, r in enumerate(left[:12])
    )
    expected = {
        (a["sample_id"], b["sample_id"])
        for a in left
        for b in right
        if (a["phash"] ^ b["phash"]).bit_count() <= 4
        and (a["dhash"] ^ b["dhash"]).bit_count() <= 6
    }
    report = overlap(left, right)
    assert {(p["left"], p["right"]) for p in report["near_candidates"]} == expected
    assert report["phash_search"]["queries"] == len(right)
    assert report["screened_comparisons"] > len(report["near_candidates"])


@pytest.fixture
def review_release(tmp_path):
    rows = []
    for index, (task, split) in enumerate(
        (("classification", "validation"), ("detection", "train"), ("segmentation", "test"))
    ):
        rows.append(
            {
                **fingerprints(str(index), index),
                "task": task,
                "split": split,
                "label": 1 if task == "classification" else None,
                "source_label": "source",
                "group_id": str(index),
                "source_member": f"{index}.jpg",
            }
        )
    near = [
        {"left": "0", "right": "1", "decision": "unresolved"},
        {"left": "0", "right": "2", "decision": "unresolved"},
    ]
    path = tmp_path / "near_candidates.json"
    path.write_text(json.dumps(near))
    return Release(tmp_path, {}, {}, tuple(rows), {path.name: sha256_file(path)})


def test_view_investigation_leaves_decisions_blank_and_redacts_test(
    review_release, monkeypatch, tmp_path
):
    pairs, affected, summary = investigate_pairs(review_release)
    assert summary["unresolved_cross_split_pairs"] == 2 and len(affected) == 3
    assert summary["affected_classification_samples"] == 1
    assert all(p["decision"] == p["evidence"] == "" for p in pairs)
    opened = []

    def image(self, row):
        assert row["split"] != "test", "Test pixels must never open"
        opened.append(row["sample_id"])
        return Image.new("RGB", (16, 16))

    monkeypatch.setattr(Release, "image", image)
    pair_sheets(review_release, pairs, tmp_path / "pages", 2)
    assert (tmp_path / "pages/pairs_001.png").exists()
    assert set(opened) == {"0", "1"}
    card(review_release, review_release.rows[2], ["test"])


def test_indirect_classification_connections_are_reported_as_candidates(review_release):
    rows = (
        *review_release.rows,
        dict(review_release.rows[0], sample_id="3", group_id="3", split="train"),
    )
    near = [
        {"left": "0", "right": "1", "decision": "unresolved"},
        {"left": "1", "right": "3", "decision": "unresolved"},
    ]
    summary, records = candidate_components(rows, near)
    assert summary["potential_classification_cross_split_components"] == 1
    assert {r["sample_id"] for r in records} == {"0", "1", "3"}


def records_fixture():
    samples = tuple(
        {
            "sample_id": str(i),
            "label": i,
            "group_id": str(i),
            "source_label": str(i),
            "task": "classification",
            "split": "validation",
            "domain": "D",
        }
        for i in (0, 1)
    )
    records = [
        {
            "sample_id": str(i),
            "target": str(i),
            "group_id": str(i),
            "source_label": str(i),
            "probability_crack": str(0.1 if i == 0 else 0.8),
            "prediction": str(i),
        }
        for i in (0, 1)
    ]
    return samples, records


@pytest.mark.parametrize(
    "field,value",
    [
        ("target", "1"),
        ("group_id", "wrong"),
        ("probability_crack", "nan"),
        ("probability_crack", "1.1"),
        ("prediction", "1"),
        ("source_label", "other"),
    ],
)
def test_saved_predictions_must_match_labels_groups_scores_and_threshold(field, value):
    samples, records = records_fixture()
    assert len(validate_predictions(samples, records, 0.5)) == 2
    records[0][field] = value
    with pytest.raises(ValueError):
        validate_predictions(samples, records, 0.5)


def test_saved_population_cannot_include_test_or_duplicate_ids():
    samples, records = records_fixture()
    with pytest.raises(ValueError, match="population"):
        validate_predictions(samples, [records[0], records[0]], 0.5)
    with pytest.raises(ValueError, match="validation only"):
        validate_predictions((dict(samples[0], split="test"), samples[1]), records, 0.5)


def test_threshold_sensitivity_refuses_external_data_without_tuning():
    samples, records = records_fixture()
    identity = {"release_id": "source"}
    source = SimpleNamespace(metadata={"dataset": "sdnet2018"}, identity=lambda: identity)
    validation = SimpleNamespace(
        release=source,
        run=SimpleNamespace(payload={"dataset": identity, "threshold": {"threshold": 0.6}}),
        predictions=validate_predictions(samples, records, 0.6),
    )
    table = threshold_sensitivity(validation)
    assert any(r["frozen_checkpoint_threshold"] for r in table)
    external = SimpleNamespace(metadata={"dataset": "damsegment"}, identity=lambda: identity)
    validation.release = external
    with pytest.raises(ValueError, match="SDNET source validation"):
        threshold_sensitivity(validation)


def test_tied_curves_integrate_to_existing_metrics():
    y, p = [0, 1, 0, 1], [0.5, 0.5, 0.1, 0.9]
    points = curve_points(y, p)

    def area(y_values, x_values):
        y_values, x_values = np.asarray(y_values), np.asarray(x_values)
        return float(np.sum(np.diff(x_values) * (y_values[:-1] + y_values[1:]) / 2))

    auc = area(
        [r["true_positive_rate"] for r in points], [r["false_positive_rate"] for r in points]
    )
    pr = area([r["precision"] for r in points], [r["recall"] for r in points])
    expected = ranking_metrics(y, p)
    assert auc == pytest.approx(expected["roc_auc"]) and pr == pytest.approx(expected["pr_auc"])
    assert points[0]["threshold"] is None and len(points) == 4


def test_preprocessing_guard_and_orientation_are_validation_invariant():
    payload = {
        "normalization": {
            "mean": list(MEAN),
            "std": list(STD),
            "resize": "full-image letterbox, bilinear",
        }
    }
    validate_preprocessing(payload)
    payload["normalization"]["mean"] = [0, 0, 0]
    with pytest.raises(ValueError, match="preprocessing"):
        validate_preprocessing(payload)
    values = np.zeros((16, 16, 3), dtype=np.uint8)
    values[:2, :, :] = 255
    image = Image.fromarray(values)
    baseline = prepare_image(image, 16, training=False, rng=np.random.default_rng(1))
    rotated = prepare_image(
        image, 16, training=False, rng=np.random.default_rng(9), augmentation="orientation"
    )
    np.testing.assert_array_equal(baseline, rotated)
    a = prepare_image(
        image, 16, training=True, rng=np.random.default_rng(3), augmentation="orientation"
    )
    b = prepare_image(
        image, 16, training=True, rng=np.random.default_rng(3), augmentation="orientation"
    )
    np.testing.assert_array_equal(a, b)
    restored = a[0] * STD[0] + MEAN[0]
    assert (
        restored > 0.5
    ).sum() == 32  # No crop or arbitrary-angle interpolation loses foreground.


def test_unweighted_pass_updates_parameters_and_reports_runtime():
    import torch
    from torch.utils.data import DataLoader

    from damage_analytics.task07.loops import pass_epoch
    from damage_analytics.task07.models import TinySmokeNet

    model = TinySmokeNet()
    before = model.head.weight.detach().clone()
    data = [(torch.ones(3, 8, 8) * i, i % 2, str(i)) for i in range(4)]
    loader = DataLoader(data, batch_size=2)
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    result = pass_epoch(model, loader, criterion, torch.device("cpu"), optimizer)
    assert not torch.equal(before, model.head.weight)
    assert result["runtime"]["samples"] == 4 and result["runtime"]["seconds"] > 0
    result = pass_epoch(model, loader, criterion, torch.device("cpu"))
    assert len(result["probabilities"]) == 4
    with pytest.raises(ValueError, match="sum-reduction"):
        pass_epoch(model, loader, torch.nn.CrossEntropyLoss(), torch.device("cpu"))


def test_incomplete_development_metadata_cannot_bypass_overlap_screening():
    from damage_analytics.task07.release import validate_development_ids

    source = SimpleNamespace(
        rows=(
            {"sample_id": "a", "task": "classification", "split": "train"},
            {"sample_id": "b", "task": "classification", "split": "validation"},
            {"sample_id": "c", "task": "classification", "split": "test"},
        )
    )
    validate_development_ids({"development_ids": ["a", "b"]}, source)
    for invalid in ([], ["a"], ["a", "c"], ["a", "a", "b"]):
        with pytest.raises(ValueError, match="development IDs"):
            validate_development_ids({"development_ids": invalid}, source)


def test_timing_barriers_only_for_mps_boundaries(monkeypatch):
    import torch

    from damage_analytics.task07.timing import timestamp

    calls = []
    monkeypatch.setattr(torch.mps, "synchronize", lambda: calls.append("barrier"))
    assert timestamp(torch.device("cpu")) > 0 and not calls
    assert timestamp(torch.device("mps")) > 0 and calls == ["barrier"]


def test_matched_comparison_rejects_mismatched_budgets_before_reading_evaluations(tmp_path):
    from damage_analytics.task08.comparison import compare_runs

    source = SimpleNamespace(
        identity=lambda: {"release_id": "source"}, metadata={"release_id": "source"}
    )
    paths = []
    for index in range(2):
        path = tmp_path / str(index)
        path.mkdir()
        metadata = {
            "dataset": source.identity(),
            "settings": {"epochs": 20 + index},
            "seed": 42,
            "normalization": {},
            "selection": "val_auc",
            "class_weights": [1, 1],
        }
        (path / "run.json").write_text(json.dumps(metadata))
        (path / "status.json").write_text(json.dumps({"status": "completed"}))
        (path / "best.pt").write_bytes(b"placeholder hash only; must not load a model")
        paths.append(path)
    with pytest.raises(ValueError, match="identical source settings"):
        compare_runs(paths, source, tmp_path / "comparison.json")


def test_small_synthetic_smoke_records_runtime_and_no_real_training(tmp_path):
    from damage_analytics.task07.smoke import smoke

    config = replace(
        load_task07(Path("config/task07.toml")),
        training=replace(
            load_task07(Path("config/task07.toml")).training, augmentation="orientation"
        ),
    )
    report = smoke(config, tmp_path / "smoke")
    assert report["completed"] and not report["real_dataset_training"]
    runtime = json.loads((tmp_path / "smoke/experiments/tiny_synthetic/runtime.json").read_text())
    assert runtime["epochs_executed"] == 2 and runtime["train_pass_seconds"] > 0
    history = json.loads((tmp_path / "smoke/experiments/tiny_synthetic/history.json").read_text())
    assert "validation_crack_recall_at_0_5" in history[0]
