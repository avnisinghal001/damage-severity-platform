"""Hand-computed metric examples, pooling, absent classes, and ignored pixels."""

import numpy as np
import pytest

from damage_analytics.training.metrics import confusion_matrix, metrics_from_confusion


def test_known_confusion_and_foreground_metrics():
    target = np.array([0, 0, 1, 1, 2, 255])
    prediction = np.array([0, 1, 1, 0, 2, 8])
    matrix = confusion_matrix(target, prediction)
    assert matrix[0, 0] == matrix[0, 1] == matrix[1, 0] == matrix[1, 1] == matrix[2, 2] == 1
    result = metrics_from_confusion(matrix)
    one = result["classes"][1]
    assert one["iou"] == pytest.approx(1 / 3)
    assert one["dice"] == one["precision"] == one["recall"] == 0.5
    assert result["foreground_miou"] == pytest.approx(2 / 3)
    assert result["pixel_accuracy"] == 0.6 and result["evaluated_pixels"] == 5
    assert result["classes"][8]["iou"] is None


def test_absent_class_false_positives_count_as_zero_not_excluded():
    matrix = confusion_matrix(np.array([0, 0]), np.array([0, 8]))
    result = metrics_from_confusion(matrix)
    assert result["foreground_miou"] == 0
    assert result["classes"][8]["iou"] == result["classes"][8]["precision"] == 0
    assert result["classes"][8]["recall"] is None
    assert metrics_from_confusion(np.zeros((9, 9), dtype=np.int64))["foreground_miou"] is None


def test_confusion_counts_are_pooled_before_macro_metrics():
    a = confusion_matrix(np.array([1]), np.array([1]))
    b = confusion_matrix(np.ones(9, dtype=int), np.zeros(9, dtype=int))
    assert metrics_from_confusion(a + b)["foreground_miou"] == 0.1


@pytest.mark.parametrize(
    "truth,prediction",
    [
        (np.array([9]), np.array([0])),
        (np.array([0]), np.array([255])),
        (np.array([-1]), np.array([0])),
        (np.array([0.5]), np.array([0])),
    ],
)
def test_invalid_ids_and_float_targets_are_rejected(truth, prediction):
    with pytest.raises(ValueError):
        confusion_matrix(truth, prediction)
