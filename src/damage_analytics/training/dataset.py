"""PyTorch Dataset/DataLoader adapters around shared, auditable paired preprocessing."""

import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, Sampler

from damage_analytics.training.cache import PairCache
from damage_analytics.training.config import DataConfig
from damage_analytics.training.manifests import Release, Sample
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
        cache_mb: int = 0,
    ) -> None:
        self.release, self.samples, self.config = release, samples, config
        self.training, self.seed, self._epoch = training, seed, 0
        self._shared_epoch: torch.Tensor | None = None
        self.cache = PairCache(cache_mb)

    @property
    def epoch(self) -> int:
        """Read the parent's epoch in persistent spawned workers as well as the main process."""
        return self._epoch if self._shared_epoch is None else int(self._shared_epoch.item())

    def share_epoch(self) -> None:
        """Allocate a tiny shared CPU scalar before workers spawn, without a manager process."""
        if self._shared_epoch is None:
            self._shared_epoch = torch.tensor(self._epoch, dtype=torch.int64).share_memory_()

    def __len__(self) -> int:
        return len(self.samples)

    def set_epoch(self, epoch: int) -> None:
        """Derive deterministic epoch-specific augmentation independently of worker order."""
        if epoch < 0:
            raise ValueError("Epoch must be nonnegative")
        self._epoch = epoch
        if self._shared_epoch is not None:
            self._shared_epoch.fill_(epoch)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        image, mask = self.cache.read(self.release, sample)
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


class ReplaySampler(Sampler[int]):
    """Decouple shuffling from worker lifetime while replaying the old full-epoch order.

    The old loader used one generator for its base seed and RandomSampler. Consume
    the same base-seed draw before each full permutation, even when workers persist.
    This preserves completed-epoch ordering for the existing PyTorch implementation.
    """

    def __init__(self, dataset: Dataset, seed: int) -> None:
        self.generator = torch.Generator().manual_seed(seed)
        self.sampler = RandomSampler(dataset, generator=self.generator)

    def __iter__(self):
        torch.empty((), dtype=torch.int64).random_(generator=self.generator)
        yield from self.sampler

    def __len__(self) -> int:
        return len(self.sampler)


def make_loader(
    dataset: SegmentationDataset,
    batch_size: int,
    workers: int,
    seed: int,
    *,
    shuffle: bool,
    persistent_workers: bool = False,
    prefetch_factor: int = 2,
) -> DataLoader:
    """Preserve all samples and epoch transforms, including across persistent spawn workers."""
    if workers < 0 or (persistent_workers and workers == 0) or prefetch_factor < 1:
        raise ValueError("Invalid workers, persistence or prefetch settings")
    if persistent_workers:
        dataset.share_epoch()
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=ReplaySampler(dataset, seed) if shuffle else None,
        num_workers=workers,
        collate_fn=collate_samples,
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=persistent_workers,
        pin_memory=False,
        drop_last=False,
        **(
            {"multiprocessing_context": "spawn", "prefetch_factor": prefetch_factor}
            if workers
            else {}
        ),
    )
