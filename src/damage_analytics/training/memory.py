"""Best-effort sampled process-tree RSS without an additional dependency."""

import os
import resource
import subprocess
import sys
import threading

import torch


class MemoryMonitor:
    """Include loader workers in sampled RSS; shared pages can be counted more than once."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.peak_tree_bytes: int | None = None
        self.samples = 0
        self.mps_peak_allocated = self.mps_peak_driver = 0
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def _sample(self) -> None:
        while not self.stop.is_set():
            if self.device.type == "mps":
                self.mps_peak_allocated = max(
                    self.mps_peak_allocated, torch.mps.current_allocated_memory()
                )
                self.mps_peak_driver = max(
                    self.mps_peak_driver, torch.mps.driver_allocated_memory()
                )
            try:
                result = subprocess.run(
                    ["ps", "-axo", "pid=,ppid=,rss="],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=2,
                )
                rows = [tuple(map(int, line.split())) for line in result.stdout.splitlines()]
                descendants = {os.getpid()}
                for _ in range(len(rows)):
                    found = {pid for pid, parent, _ in rows if parent in descendants}
                    if found <= descendants:
                        break
                    descendants |= found
                total = sum(rss * 1024 for pid, _, rss in rows if pid in descendants)
                self.peak_tree_bytes = max(self.peak_tree_bytes or 0, total)
                self.samples += 1
            except (OSError, ValueError, subprocess.SubprocessError):
                return
            self.stop.wait(0.25)

    def __exit__(self, *_args) -> None:
        self.stop.set()
        self.thread.join(timeout=3)

    def report(self) -> dict:
        """Distinguish sampled RSS from process-lifetime high-water and MPS allocation."""
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report = {
            "sampled_process_tree_peak_bytes": self.peak_tree_bytes,
            "rss_samples": self.samples,
            "parent_lifetime_peak_bytes": peak if sys.platform == "darwin" else peak * 1024,
            "limitations": "RSS is sampled, includes shared-page double counting and monitoring "
            "overhead; parent high-water spans previous cases. Not an allocator peak.",
        }
        if self.device.type == "mps":
            report["mps_sampled_peak_allocated_bytes"] = self.mps_peak_allocated
            report["mps_sampled_peak_driver_bytes"] = self.mps_peak_driver
            report["mps_current_allocated_bytes"] = torch.mps.current_allocated_memory()
            report["mps_driver_allocated_bytes"] = torch.mps.driver_allocated_memory()
            report["mps_note"] = "MPS and CPU use unified memory; do not add RSS and driver bytes."
        return report
