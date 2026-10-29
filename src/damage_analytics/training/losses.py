"""Train-only class weighting, ignored-padding CE, and optional foreground soft Dice."""

import torch
from torch import nn
from torch.nn import functional as F

from damage_analytics.training.config import IGNORE_INDEX


class SegmentationLoss(nn.Module):
    """Weighted cross entropy + optional soft Dice over present foreground classes."""

    def __init__(self, weights: torch.Tensor, dice_weight: float) -> None:
        super().__init__()
        self.register_buffer("weights", weights.float())
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Ignore padding in both losses; reject batches without any supervised pixels."""
        valid = target != IGNORE_INDEX
        if not bool(valid.any()):
            raise ValueError("Batch contains no supervised pixels")
        ce = F.cross_entropy(logits, target, weight=self.weights, ignore_index=IGNORE_INDEX)
        if not self.dice_weight:
            return ce
        safe_target = target.masked_fill(~valid, 0)
        truth = F.one_hot(safe_target, num_classes=9).permute(0, 3, 1, 2).float()
        allowed = valid.unsqueeze(1)
        truth = truth * allowed
        probabilities = logits.softmax(dim=1) * allowed
        intersection = (probabilities * truth).sum(dim=(0, 2, 3))
        denominator = (probabilities + truth).sum(dim=(0, 2, 3))
        present = truth.sum(dim=(0, 2, 3)) > 0
        present[0] = False
        dice = 1 - (2 * intersection[present] + 1e-6) / (denominator[present] + 1e-6)
        return ce + self.dice_weight * (dice.mean() if bool(present.any()) else logits.sum() * 0)
