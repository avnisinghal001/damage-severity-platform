"""Numerically checked Caffe-style inverse; channel semantics remain a documented inference."""

import numpy as np

CAFFE_OFFSETS = np.array([103.939, 116.779, 123.68], dtype=np.float32)


def recovered_channels(values: np.ndarray, tolerance: float = 0.001) -> tuple[np.ndarray, float]:
    """Validate an integer-grid inverse without clipping unsupported or nonfinite pixels."""
    if (
        values.ndim != 3
        or values.shape[-1] != 3
        or values.dtype != np.float32
        or not np.isfinite(values).all()
    ):
        raise ValueError("Image row is not finite HWC float32 with three channels")
    recovered = values + CAFFE_OFFSETS
    rounded = np.rint(recovered)
    error = float(np.max(np.abs(recovered - rounded)))
    if error > tolerance or rounded.min() < 0 or rounded.max() > 255:
        raise ValueError(
            "Row does not support the configured integer-grid mean-subtraction inverse"
        )
    return rounded.astype(np.uint8), error


def rgb_from_array(values: np.ndarray, tolerance: float = 0.001) -> np.ndarray:
    """Caffe/BGR hypothesis: add offsets, reverse channels; normalization happens later."""
    channels, _ = recovered_channels(values, tolerance)
    return np.ascontiguousarray(channels[..., ::-1])
