"""Shared decoding, loader and whole-pass probes for bounded training profiles."""

from time import perf_counter

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from damage_analytics.training.config import TrainingConfig
from damage_analytics.training.dataset import SegmentationDataset, make_loader
from damage_analytics.training.engine import run_epoch
from damage_analytics.training.losses import SegmentationLoss
from damage_analytics.training.manifests import Release, Sample, read_pair
from damage_analytics.training.model import LightweightUNet
from damage_analytics.training.timing import StageTimer, synchronize
from damage_analytics.training.transforms import prepare_pair


def decoding_profile(
    release: Release, samples: tuple[Sample, ...], config: TrainingConfig
) -> dict:
    """Separate verified pair IO, hashing, codec materialization and paired transforms.

    Separate probes reread the same files with warm OS caches. They overlap conceptually
    and must not be added to the measured complete read_pair time.
    """
    timer = StageTimer(torch.device("cpu"))
    sizes = []
    for index, sample in enumerate(samples):
        with timer.measure("verified_read_pair"):
            image, mask = read_pair(release, sample)
        with timer.measure("hash_image_and_mask"):
            paths = [release.file(sample.image), release.file(sample.mask)]
        for name, path in zip(("image_decode", "mask_decode"), paths, strict=True):
            with timer.measure(name), Image.open(path) as source:
                source.load()
                source.copy()
        with timer.measure("paired_transform"):
            prepare_pair(
                image,
                mask,
                config.data,
                training=True,
                rng=np.random.default_rng(np.random.SeedSequence([config.seed, 1, index])),
            )
        sizes.append(sample.width * sample.height)
    return {"stages": timer.report(), "sample_pixels": sizes, "os_cache": "warm/uncontrolled"}


def profile_loader(
    dataset: SegmentationDataset,
    batch_size: int,
    workers: int,
    persistent: bool,
    prefetch_factor: int = 2,
) -> DataLoader:
    """Build the measured loader without changing the frozen sample order."""
    return make_loader(
        dataset,
        batch_size,
        workers,
        dataset.seed,
        shuffle=False,
        persistent_workers=persistent,
        prefetch_factor=prefetch_factor,
    )


def profile_pass(
    model: LightweightUNet,
    loader: DataLoader,
    criterion: SegmentationLoss,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    *,
    detailed: bool,
    channels_last: bool = False,
    gradient_clip: float = 1.0,
) -> dict:
    """Measure actual work; only whole-pass boundaries synchronize in throughput mode."""
    timer = StageTimer(device) if detailed else None
    synchronize(device)
    start = perf_counter()
    result = run_epoch(
        model,
        loader,
        criterion,
        device,
        optimizer,
        timer=timer,
        channels_last=channels_last,
        gradient_clip=gradient_clip,
    )
    synchronize(device)
    elapsed = perf_counter() - start
    return {
        "seconds": elapsed,
        "images": len(loader.dataset),
        "images_per_second": len(loader.dataset) / elapsed,
        "loss": result["loss"],
        "metrics": result.get("metrics"),
        "stages": timer.report() if timer else None,
    }
