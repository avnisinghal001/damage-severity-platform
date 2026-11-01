"""Wall-clock timing with barriers only at measured MPS boundaries."""

from time import perf_counter

import torch


def timestamp(device: torch.device) -> float:
    """Wait for queued MPS work before reading a pass boundary; CPU needs no barrier."""
    if device.type == "mps":
        torch.mps.synchronize()
    return perf_counter()
