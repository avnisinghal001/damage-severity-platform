"""Native training-only class statistics for explicit, capped imbalance weighting."""

import numpy as np

from damage_analytics.training.config import OptimizationConfig
from damage_analytics.training.manifests import Sample


def training_weights(
    samples: tuple[Sample, ...], config: OptimizationConfig
) -> tuple[np.ndarray, dict]:
    """Estimate weights only from the selected training manifest's native pixel counts."""
    counts = np.array([sample.pixel_counts for sample in samples], dtype=np.int64).sum(axis=0)
    present = counts > 0
    weights = np.ones(9, dtype=np.float64)
    if config.class_weighting == "inverse_sqrt":
        weights[:] = 0
        weights[present] = 1 / np.sqrt(counts[present] / counts.sum())
        weights[present] /= weights[present].mean()
        weights = np.minimum(weights, config.max_class_weight)
    return weights.astype(np.float32), {
        "policy": config.class_weighting,
        "native_training_pixel_counts": counts.tolist(),
        "weights": weights.tolist(),
        "population": "Selected training rows only; no validation/test counts",
        "note": "Native-pixel weighting is approximate after cropping/resizing; "
        "weights are capped",
    }
