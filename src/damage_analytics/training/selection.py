"""Validation scheduling and early stopping never infer scores for unvalidated epochs."""

from dataclasses import dataclass


def validation_due(epoch: int, epochs: int, every: int) -> bool:
    """Always validate first/final epochs, plus the declared cadence."""
    return epoch == 1 or epoch == epochs or epoch % every == 0


@dataclass
class SelectionState:
    """Patience counts completed validation checks, not skipped training epochs."""

    best: float = -1.0
    significant: float = -1.0
    stale: int = 0
    best_epoch: int = 0
    checks: int = 0

    def observe(self, epoch: int, score: float, min_delta: float) -> bool:
        """Select every strictly best score; stop only on declared significant stagnation."""
        self.checks += 1
        improved = score > self.best
        if improved:
            self.best, self.best_epoch = score, epoch
        if score > self.significant + min_delta:
            self.significant, self.stale = score, 0
        else:
            self.stale += 1
        return improved
