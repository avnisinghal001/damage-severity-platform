"""Opt-in wall timers; accelerator barriers exist only during stage profiling."""

from collections import defaultdict
from contextlib import contextmanager
from time import perf_counter

import numpy as np
import torch


def synchronize(device: torch.device) -> None:
    """Wait for outstanding MPS work only when a benchmark requires a boundary."""
    if device.type == "mps":
        torch.mps.synchronize()


class StageTimer:
    """Disjoint stage timings; synchronization perturbs overlap, so also time whole passes."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.seconds: dict[str, list[float]] = defaultdict(list)

    @contextmanager
    def measure(self, name: str):
        """Synchronize stage boundaries on MPS, never in ordinary training."""
        synchronize(self.device)
        start = perf_counter()
        try:
            yield
        finally:
            synchronize(self.device)
            self.seconds[name].append(perf_counter() - start)

    def report(self) -> dict:
        """Return seconds and distribution summaries rather than misleading timer precision."""
        return {
            name: {
                "calls": len(values),
                "total_seconds": sum(values),
                "median_seconds": float(np.median(values)),
                "p95_seconds": float(np.quantile(values, 0.95)),
            }
            for name, values in sorted(self.seconds.items())
        }
