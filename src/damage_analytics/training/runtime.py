"""Device selection and best-effort reproducibility; no hidden backend fallback."""

import random

import numpy as np
import torch


def configure_runtime(seed: int, requested: str, deterministic: bool) -> torch.device:
    """Choose available MPS or CPU; strict deterministic operations can reject unsupported ops."""
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic)
    available = torch.backends.mps.is_available()
    if requested == "mps" and not available:
        raise ValueError("Apple MPS was requested but is unavailable in this PyTorch/environment")
    return torch.device(
        "mps" if requested == "mps" or (requested == "auto" and available) else "cpu"
    )
