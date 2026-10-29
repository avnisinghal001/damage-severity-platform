"""Dataset-level confusion metrics with explicit absent-class and ignore conventions."""

import numpy as np

from damage_analytics.labels import CLASS_NAMES
from damage_analytics.training.config import IGNORE_INDEX


def confusion_matrix(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """Return int64 counts with ground truth on rows and predictions on columns."""
    if target.shape != prediction.shape:
        raise ValueError("Target/prediction shapes differ")
    valid = target != IGNORE_INDEX
    truth, predicted = target[valid], prediction[valid]
    if (
        not np.issubdtype(truth.dtype, np.integer)
        or not np.issubdtype(predicted.dtype, np.integer)
        or np.any((truth < 0) | (truth > 8))
        or np.any((predicted < 0) | (predicted > 8))
    ):
        raise ValueError("Metrics require integer IDs 0..8 (target padding may be ignored)")
    return np.bincount(
        truth.astype(np.int64) * 9 + predicted.astype(np.int64), minlength=81
    ).reshape(9, 9)


def metrics_from_confusion(matrix: np.ndarray) -> dict:
    """Compute pooled pixel metrics; undefined ratios serialize as null, never NaN.

    IoU/Dice include false-positive-only classes as zero. A class absent in both
    target and prediction has undefined overlap and is excluded from macro means.
    Precision/recall are independently undefined when their denominator is zero.
    Foreground mIoU includes class IDs 1..8 and excludes background, never its errors.
    """
    if matrix.shape != (9, 9) or not np.issubdtype(matrix.dtype, np.integer) or np.any(matrix < 0):
        raise ValueError("Expected a nonnegative 9×9 confusion matrix")
    classes = []
    for index, label in enumerate(CLASS_NAMES):
        tp = int(matrix[index, index])
        support, predicted = int(matrix[index].sum()), int(matrix[:, index].sum())
        union = support + predicted - tp
        classes.append(
            {
                "class_id": index,
                "source_label": label,
                "support_pixels": support,
                "predicted_pixels": predicted,
                "iou": tp / union if union else None,
                "dice": 2 * tp / (support + predicted) if support + predicted else None,
                "precision": tp / predicted if predicted else None,
                "recall": tp / support if support else None,
            }
        )
    foreground = [row["iou"] for row in classes[1:] if row["iou"] is not None]
    all_ious = [row["iou"] for row in classes if row["iou"] is not None]
    total = int(matrix.sum())
    return {
        "classes": classes,
        "foreground_miou": float(np.mean(foreground)) if foreground else None,
        "mean_iou": float(np.mean(all_ious)) if all_ious else None,
        "pixel_accuracy": float(np.trace(matrix) / total) if total else None,
        "evaluated_pixels": total,
        "confusion_matrix": matrix.tolist(),
        "confusion_axes": "rows=ground truth, columns=prediction",
        "macro_policy": "Exclude absent-in-both classes; false-positive-only IoU/Dice=0",
        "severity": "No severity or structural safety score is inferred",
    }
