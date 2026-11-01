"""Full-patch classification preprocessing and deterministic paired worker epoch state."""

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import DataLoader, Dataset

from damage_analytics.task07.release import Release
from damage_analytics.training.dataset import ReplaySampler, seed_worker

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def prepare_image(
    image: Image.Image, size: int, *, training: bool, rng: np.random.Generator
) -> np.ndarray:
    """Keep the whole patch: random cropping could remove a crack but retain its positive label."""
    image = image.convert("RGB")
    if training:
        if rng.random() < 0.5:
            image = ImageOps.mirror(image)
        image = ImageEnhance.Brightness(image).enhance(float(rng.uniform(0.9, 1.1)))
        image = ImageEnhance.Contrast(image).enhance(float(rng.uniform(0.9, 1.1)))
    resized = ImageOps.contain(image, (size, size), Image.Resampling.BILINEAR)
    padded = Image.new("RGB", (size, size), tuple(round(x * 255) for x in MEAN))
    padded.paste(resized, ((size - resized.width) // 2, (size - resized.height) // 2))
    values = np.asarray(padded, dtype=np.float32) / 255
    values = (values - np.asarray(MEAN, np.float32)) / np.asarray(STD, np.float32)
    return np.ascontiguousarray(values.transpose(2, 0, 1))


class CrackDataset(Dataset):
    def __init__(
        self,
        release: Release,
        rows: tuple[dict, ...],
        size: int,
        training: bool,
        seed: int,
        workers: int = 0,
    ) -> None:
        self.release, self.rows, self.size = release, rows, size
        self.training, self.seed = training, seed
        self.epoch = torch.tensor(0, dtype=torch.int64)
        if workers:
            self.epoch.share_memory_()

    def __len__(self) -> int:
        return len(self.rows)

    def set_epoch(self, value: int) -> None:
        self.epoch.fill_(value)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        row = self.rows[index]
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, int(self.epoch), index]))
        rgb = prepare_image(self.release.image(row), self.size, training=self.training, rng=rng)
        return torch.from_numpy(rgb), int(row["label"]), row["sample_id"]


def make_loader(
    dataset: CrackDataset, batch_size: int, workers: int, seed: int, *, shuffle: bool
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=workers,
        drop_last=False,
        sampler=ReplaySampler(dataset, seed) if shuffle else None,
        generator=torch.Generator().manual_seed(seed),
        worker_init_fn=seed_worker,
        persistent_workers=workers > 0,
        pin_memory=False,
        **({"multiprocessing_context": "spawn", "prefetch_factor": 2} if workers else {}),
    )
