"""PyTorch Dataset/DataLoader adapters around shared, auditable paired preprocessing."""

import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from damage_analytics.training.config import DataConfig
from damage_analytics.training.manifests import Release, Sample, read_pair
from damage_analytics.training.transforms import prepare_pair


class SegmentationDataset(Dataset):
    """Return float32 CHW RGB, int64 HW labels, geometry, and optional native targets."""

    def __init__(
        self,
        release: Release,
        samples: tuple[Sample, ...],
        config: DataConfig,
        *,
        training: bool,
        seed: int,
    ) -> None:
        self.release, self.samples, self.config = release, samples, config
        self.training, self.seed, self.epoch = training, seed, 0

    def __len__(self) -> int:
        return len(self.samples)

    def set_epoch(self, epoch: int) -> None:
        """Derive deterministic epoch-specific augmentation independently of worker order."""
        if epoch < 0:
            raise ValueError("Epoch must be nonnegative")
        self.epoch = epoch

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        image, mask = read_pair(self.release, sample)
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index]))
        rgb, labels, frame = prepare_pair(
            image, mask, self.config, training=self.training, rng=rng
        )
        return {
            "image": torch.from_numpy(rgb),
            "mask": torch.from_numpy(labels),
            "sample": sample,
            "frame": frame,
            "native_mask": None if self.training else np.array(mask),
        }


def collate_samples(items: list[dict]) -> dict:
    """Stack fixed-size tensors; keep variable-resolution native masks as a CPU list."""
    return {
        "images": torch.stack([item["image"] for item in items]),
        "masks": torch.stack([item["mask"] for item in items]),
        "samples": [item["sample"] for item in items],
        "frames": [item["frame"] for item in items],
        "native_masks": [item["native_mask"] for item in items],
    }


def seed_worker(_worker_id: int) -> None:
    """Seed third-party random use in spawned workers; paired transforms have their own RNG."""
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def make_loader(
    dataset: SegmentationDataset, batch_size: int, workers: int, seed: int, *, shuffle: bool
) -> DataLoader:
    """Use explicit shuffling state, no dropped samples and no persistent stale-epoch workers."""
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=collate_samples,
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=False,
        pin_memory=False,
        drop_last=False,
    )
