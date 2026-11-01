"""NumPy metrics, tied-score ranking and validation-only threshold selection."""

import numpy as np


def _arrays(target, probability):
    y, p = np.asarray(target), np.asarray(probability, dtype=float)
    if (
        y.ndim != 1
        or p.shape != y.shape
        or not len(y)
        or not np.isin(y, [0, 1]).all()
        or not np.isfinite(p).all()
        or np.any((p < 0) | (p > 1))
    ):
        raise ValueError("Expected binary labels and finite probabilities in [0,1]")
    return y.astype(np.int64), p


def from_confusion(matrix: np.ndarray) -> dict:
    """Arbitrary native taxonomy; GT on rows, predictions on columns, undefined ratios null."""
    matrix = np.asarray(matrix)
    if (
        matrix.ndim != 2
        or matrix.shape[0] != matrix.shape[1]
        or matrix.shape[0] < 2
        or not np.issubdtype(matrix.dtype, np.integer)
        or np.any(matrix < 0)
    ):
        raise ValueError("Expected nonnegative integer square confusion matrix")
    rows = []
    for i in range(len(matrix)):
        tp, support, predicted = int(matrix[i, i]), int(matrix[i].sum()), int(matrix[:, i].sum())
        denominator, union = support + predicted, support + predicted - tp
        rows.append(
            {
                "class_id": i,
                "support": support,
                "predicted": predicted,
                "precision": tp / predicted if predicted else None,
                "recall": tp / support if support else None,
                "f1": 2 * tp / denominator if denominator else None,
                "dice": 2 * tp / denominator if denominator else None,
                "iou": tp / union if union else None,
            }
        )
    recalls = [r["recall"] for r in rows]
    f1 = [r["f1"] for r in rows if r["f1"] is not None]
    foreground = [r["iou"] for r in rows[1:] if r["iou"] is not None]
    return {
        "confusion_matrix": matrix.tolist(),
        "classes": rows,
        "accuracy": float(matrix.trace() / matrix.sum()) if matrix.sum() else None,
        "balanced_accuracy": float(np.mean(recalls))
        if all(r is not None for r in recalls)
        else None,
        "macro_f1": float(np.mean(f1)) if f1 else None,
        "foreground_miou": float(np.mean(foreground)) if foreground else None,
        "undefined_policy": "Null undefined ratios; absent-in-both overlap excluded. "
        "Balanced accuracy requires every class present.",
    }


def ranking_metrics(target, probability) -> dict:
    """Group tied scores. PR-AUC is trapezoidal; AP is also reported separately."""
    y, p = _arrays(target, probability)
    order = np.argsort(-p, kind="stable")
    y, p = y[order], p[order]
    ends = np.r_[np.flatnonzero(p[:-1] != p[1:]), len(p) - 1]
    tp = np.r_[0, np.cumsum(y)[ends]].astype(float)
    fp = np.r_[0, (np.arange(1, len(y) + 1) - np.cumsum(y))[ends]].astype(float)
    positives, negatives = int(y.sum()), int(len(y) - y.sum())

    def area(x, z):
        return float(np.sum(np.diff(x) * (z[:-1] + z[1:]) / 2))

    recall = tp / positives if positives else np.zeros_like(tp)
    precision = np.divide(tp, tp + fp, out=np.ones_like(tp), where=tp + fp > 0)
    return {
        "roc_auc": area(fp / negatives, recall) if positives and negatives else None,
        "pr_auc": area(recall, precision) if positives else None,
        "average_precision": float(np.sum(np.diff(recall) * precision[1:])) if positives else None,
        "ranking_policy": "Class 1 = crack; tied thresholds grouped. PR-AUC trapezoidal, "
        "average precision step-integrated; one-class ROC undefined.",
    }


def classification_metrics(target, probability, threshold: float) -> dict:
    y, p = _arrays(target, probability)
    if not np.isfinite(threshold) or not 0 <= threshold <= np.nextafter(1.0, 2.0):
        raise ValueError("Invalid classification threshold")
    prediction = (p >= threshold).astype(int)
    matrix = np.bincount(2 * y + prediction, minlength=4).reshape(2, 2)
    return {
        **from_confusion(matrix),
        **ranking_metrics(y, p),
        "threshold": float(threshold),
        "decision_rule": "P(crack) >= threshold; scores are not calibrated safety probabilities",
    }


def select_threshold(target, probability, objective: str = "balanced_accuracy") -> dict:
    """O(N log N) grouped sweep on VALIDATION only; choose largest threshold on ties."""
    y, p = _arrays(target, probability)
    if len(set(y.tolist())) != 2 or objective not in ("balanced_accuracy", "macro_f1"):
        raise ValueError(
            "Threshold selection requires both validation classes and a known objective"
        )
    order = np.argsort(-p, kind="stable")
    y, p = y[order], p[order]
    positives, negatives = int(y.sum()), len(y) - int(y.sum())
    tp, fp = 0, 0
    candidates = [(float(np.nextafter(1.0, 2.0)), 0, 0)]
    for start in np.r_[0, np.flatnonzero(p[:-1] != p[1:]) + 1]:
        end = int(np.searchsorted(-p, -p[start], side="right"))
        tp += int(y[start:end].sum())
        fp += end - start - int(y[start:end].sum())
        candidates.append((float(p[start]), tp, fp))
    best = (-1.0, -1.0)
    for threshold, tp, fp in candidates:
        fn, tn = positives - tp, negatives - fp
        if objective == "balanced_accuracy":
            score = (tp / positives + tn / negatives) / 2
        else:
            score = (2 * tp / (2 * tp + fp + fn) + 2 * tn / (2 * tn + fp + fn)) / 2
        best = max(best, (score, threshold))
    return {
        "threshold": best[1],
        "objective": objective,
        "validation_objective": float(best[0]),
        "selected_on": "validation",
        "tie_break": "largest threshold",
        "note": "1+ULP is an explicit predict-none sentinel, not a probability",
    }
