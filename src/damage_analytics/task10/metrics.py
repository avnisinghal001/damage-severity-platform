"""Nominal four-class metrics: Heavy ID 0 is included, never treated as background."""

import numpy as np

from damage_analytics.task10.config import CLASS_NAMES


def metrics(targets, probabilities) -> dict:
    y, p = np.asarray(targets), np.asarray(probabilities, dtype=float)
    if (
        y.ndim != 1
        or not len(y)
        or p.shape != (len(y), 4)
        or not np.isin(y, range(4)).all()
        or not np.isfinite(p).all()
        or np.any((p < 0) | (p > 1))
        or not np.allclose(p.sum(1), 1, atol=1e-6, rtol=0)
    ):
        raise ValueError("Expected four-class labels and finite normalized N×4 probabilities")
    prediction = p.argmax(1)
    matrix = np.bincount(4 * y.astype(int) + prediction, minlength=16).reshape(4, 4)
    rows = []
    for i, name in enumerate(CLASS_NAMES):
        tp, support, predicted = int(matrix[i, i]), int(matrix[i].sum()), int(matrix[:, i].sum())
        rows.append(
            {
                "class_id": i,
                "label": name,
                "support": support,
                "predicted": predicted,
                "precision": tp / predicted if predicted else None,
                "recall": tp / support if support else None,
                "f1": 2 * tp / (support + predicted) if support + predicted else 0.0,
            }
        )
    return {
        "accuracy": float(matrix.trace() / matrix.sum()),
        "macro_f1": float(np.mean([r["f1"] for r in rows])),
        "balanced_accuracy": float(np.mean([r["recall"] for r in rows]))
        if all(r["support"] for r in rows)
        else None,
        "classes": rows,
        "confusion_matrix": matrix.tolist(),
        "axes": "rows=source ground truth, columns=predicted class",
        "macro_policy": "Fixed four classes; absent-in-both F1=0, undefined precision/recall=null",
        "confidence_calibrated": False,
        "meaning": "Visible component damage classes, not whole-building severity/safety "
        "or ordinal class-ID distances",
    }
