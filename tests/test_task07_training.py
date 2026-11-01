"""Known-answer metrics and tiny synthetic pipeline; pretrained downloads never occur."""

import importlib.util
import json
from pathlib import Path

import pytest

from damage_analytics.task07.config import load_config
from damage_analytics.task07.metrics import (
    classification_metrics,
    ranking_metrics,
    select_threshold,
)


def test_rank_ties_and_threshold_known_answers():
    y, p = [0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]
    metrics = classification_metrics(y, p, 0.5)
    assert metrics["accuracy"] == 0.75 and metrics["roc_auc"] == 0.75
    assert metrics["average_precision"] == pytest.approx(5 / 6)
    assert ranking_metrics([0, 1], [0.5, 0.5])["roc_auc"] == 0.5
    assert ranking_metrics([0, 1], [0.5, 0.5])["average_precision"] == 0.5
    assert ranking_metrics([0, 0], [0.2, 0.3])["roc_auc"] is None
    selected = select_threshold(y, p)
    assert selected["threshold"] == 0.8 and selected["selected_on"] == "validation"
    assert selected["validation_objective"] == 0.75


def test_invalid_metrics_and_one_class_threshold_refused():
    with pytest.raises(ValueError):
        classification_metrics([0, 1], [float("nan"), 0.5], 0.5)
    with pytest.raises(ValueError):
        select_threshold([0, 0], [0.1, 0.2])


def test_tiny_smoke_trains_and_evaluates_without_torchvision(tmp_path):
    pytest.importorskip("torch")
    from damage_analytics.task07.smoke import smoke

    config = load_config(Path("config/task07.toml"))
    report = smoke(config, tmp_path / "smoke")
    assert report["completed"] and not report["pretrained_downloaded"]
    assert not report["test_evaluated"]
    assert (tmp_path / "smoke/experiments/tiny_synthetic/best.pt").exists()
    from damage_analytics.task07.evaluation import compare

    path = Path(report["validation_result"]) / "metrics.json"
    comparison = compare([path], tmp_path / "comparison.json")
    assert comparison["models"][0]["backbone"] == "tiny_synthetic"
    result = json.loads(path.read_text())
    assert result["threshold_provenance"]["selected_on"] == "validation"
    assert result["synthetic"] and result["split"] == "validation"


@pytest.mark.skipif(
    importlib.util.find_spec("torchvision") is None,
    reason="torchvision absent; no installation allowed",
)
@pytest.mark.parametrize("backbone", ["mobilenet_v3_small", "resnet18"])
def test_actual_backbones_without_download(backbone):
    import torch

    from damage_analytics.task07.models import build_model

    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        model = build_model(backbone, pretrained=False).eval()
        with torch.inference_mode():
            assert model(torch.zeros(2, 3, 224, 224)).shape == (2, 2)
    finally:
        torch.set_num_threads(previous)


def test_comparison_rejects_test_path_without_opening_it(tmp_path):
    pytest.importorskip("torch")
    from damage_analytics.task07.evaluation import compare

    with pytest.raises(ValueError, match="test paths"):
        compare([tmp_path / "test/metrics.json"], tmp_path / "comparison.json")


def test_full_image_preprocessing_validation_determinism_and_seeded_augmentation():
    import numpy as np
    from PIL import Image

    from damage_analytics.task07.data import prepare_image

    values = np.arange(8 * 16 * 3, dtype=np.uint8).reshape(8, 16, 3)
    image = Image.fromarray(values)
    first = prepare_image(image, 32, training=False, rng=np.random.default_rng(1))
    second = prepare_image(image, 32, training=False, rng=np.random.default_rng(999))
    assert first.shape == (3, 32, 32) and first.dtype == np.float32
    np.testing.assert_array_equal(first, second)
    a = prepare_image(image, 32, training=True, rng=np.random.default_rng(7))
    b = prepare_image(image, 32, training=True, rng=np.random.default_rng(7))
    np.testing.assert_array_equal(a, b)
    assert np.isfinite(a).all() and not np.array_equal(a, first)
