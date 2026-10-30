"""Bounded real-data benchmarks: temporary model updates, no checkpoints or test access."""

import gc
import re
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from damage_analytics.io import write_json
from damage_analytics.training.config import TrainingConfig
from damage_analytics.training.dataset import SegmentationDataset
from damage_analytics.training.engine import run_epoch
from damage_analytics.training.experiments import execution_provenance
from damage_analytics.training.losses import SegmentationLoss
from damage_analytics.training.manifests import open_release
from damage_analytics.training.memory import MemoryMonitor
from damage_analytics.training.model import LightweightUNet
from damage_analytics.training.profile_steps import decoding_profile, profile_loader, profile_pass
from damage_analytics.training.runtime import configure_runtime
from damage_analytics.training.weights import training_weights


def spaced_samples(samples: tuple, count: int) -> tuple:
    """Use an evenly spaced frozen-order subset, not an unrepresentative filename prefix."""
    if count < 1:
        raise ValueError("Profile sample count must be positive")
    return tuple(
        samples[i] for i in np.linspace(0, len(samples) - 1, min(count, len(samples)), dtype=int)
    )


def profile_pipeline(
    config: TrainingConfig,
    folder: Path,
    *,
    cpu_threads: int = 4,
    suite: str = "stages",
    samples: int = 64,
    repeats: int = 3,
    warmup_batches: int = 2,
    workers: int = 0,
    persistent: bool = False,
    cache_mb: int = 0,
    batch_size: int | None = None,
) -> Path:
    """Profile selected train/validation subsets only; never create a real training run.

    Stages add barriers; throughput passes only synchronize at whole-pass boundaries.
    Loader cases include startup and a second epoch to expose persistence/cache effects.
    """
    if (
        suite not in ("stages", "loaders", "batches")
        or not 1 <= repeats <= 10
        or not 1 <= samples <= (2048 if suite == "loaders" else 512)
        or not 0 <= warmup_batches <= 8
        or workers not in (0, 2, 4)
        or cache_mb < 0
        or cpu_threads < 1
    ):
        raise ValueError("Invalid bounded benchmark settings")
    if persistent and workers == 0:
        raise ValueError("Persistent workers require num_workers > 0")
    batch_size = config.optimization.batch_size if batch_size is None else batch_size
    if not 1 <= batch_size <= 32:
        raise ValueError("Benchmark batch size must be in 1..32")
    torch.set_num_threads(cpu_threads)
    device = configure_runtime(config.seed, config.device, config.deterministic)
    release = open_release(config)
    selected = {
        split: spaced_samples(release.samples(split), samples) for split in ("train", "validation")
    }
    folder.mkdir(parents=True, exist_ok=False)
    report = {
        "dataset": release.identity(),
        "config": config.snapshot(),
        "execution": execution_provenance(),
        "device": str(device),
        "cpu_threads": cpu_threads,
        "suite": suite,
        "repeats": repeats,
        "channels_last": config.performance.channels_last,
        "sample_ids": {
            key: [sample.sample_id for sample in rows] for key, rows in selected.items()
        },
        "warmup_batches": warmup_batches,
        "test_accessed": False,
        "temporary_training_updates_only": True,
        "checkpoints_written": False,
        "timing_policy": "Stage barriers perturb overlap; use end_to_end passes for throughput. "
        "Warm OS caches; startup/cold and steady data passes reported separately.",
        "cases": [],
        "status": "in_progress",
        "case_settings_override_template": "Worker/cache/batch settings in each case override "
        "the template config; all cases retain its model/data/loss and split identity.",
        "native_pair_payload_bytes": {
            "full_train": sum(row.width * row.height * 4 for row in release.samples("train")),
            "selected_train": sum(row.width * row.height * 4 for row in selected["train"]),
            "note": "PIL/worker/prefetch/tensor overhead is additional. Small subsets can "
            "overstate cache hits; full-loader sweeps are supported without model training.",
        },
    }
    write_json(folder / "profile.json", report)
    report["decoding"] = decoding_profile(release, selected["train"][:16], config)
    cases = [(workers, persistent, cache_mb, batch_size)]
    if suite == "loaders":
        cases = [
            (w, p, c, batch_size)
            for c in (0, 128)
            for w in (0, 2, 4)
            for p in ((False,) if w == 0 else (False, True))
        ]
    elif suite == "batches":
        cases = [(workers, persistent, cache_mb, b) for b in (2, 4, 8)]
    for worker_count, keep, cache, size in cases:
        configure_runtime(config.seed, config.device, config.deterministic)
        datasets = {
            key: SegmentationDataset(
                release,
                rows,
                config.data,
                training=key == "train",
                seed=config.seed,
                cache_mb=cache,
            )
            for key, rows in selected.items()
        }
        loaders = {
            key: profile_loader(ds, size, worker_count, keep, config.performance.prefetch_factor)
            for key, ds in datasets.items()
        }
        case = {
            "workers": worker_count,
            "persistent_workers": keep,
            "cache_mb_per_process": cache,
            "batch_size": size,
            "prefetch_factor": config.performance.prefetch_factor if worker_count else None,
            "passes": [],
        }
        print(f"Profiling {case}", flush=True)
        with MemoryMonitor(device) as memory:
            if suite == "loaders":
                for epoch in range(1, repeats + 1):
                    datasets["train"].set_epoch(epoch)
                    start = perf_counter()
                    count = sum(len(batch["samples"]) for batch in loaders["train"])
                    seconds = perf_counter() - start
                    case["passes"].append(
                        {
                            "epoch": epoch,
                            "seconds": seconds,
                            "images": count,
                            "images_per_second": count / seconds,
                        }
                    )
            else:
                model = LightweightUNet(config.model).to(device)
                if config.performance.channels_last:
                    model = model.to(memory_format=torch.channels_last)
                weights, _ = training_weights(release.samples("train"), config.optimization)
                criterion = SegmentationLoss(
                    torch.tensor(weights), config.optimization.dice_weight
                )
                criterion = criterion.to(device)
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=config.optimization.learning_rate,
                    weight_decay=config.optimization.weight_decay,
                )
                if warmup_batches:
                    warm = SegmentationDataset(
                        release,
                        selected["train"][: size * warmup_batches],
                        config.data,
                        training=True,
                        seed=config.seed,
                    )
                    run_epoch(
                        model,
                        profile_loader(warm, size, 0, False),
                        criterion,
                        device,
                        optimizer,
                        channels_last=config.performance.channels_last,
                        gradient_clip=config.optimization.gradient_clip,
                    )
                for repeat in range(repeats):
                    datasets["train"].set_epoch(repeat + 1)
                    case["passes"].append(
                        {
                            "repeat": repeat + 1,
                            "train": profile_pass(
                                model,
                                loaders["train"],
                                criterion,
                                device,
                                optimizer,
                                detailed=False,
                                channels_last=config.performance.channels_last,
                                gradient_clip=config.optimization.gradient_clip,
                            ),
                            "validation": profile_pass(
                                model,
                                loaders["validation"],
                                criterion,
                                device,
                                None,
                                detailed=False,
                                channels_last=config.performance.channels_last,
                                gradient_clip=config.optimization.gradient_clip,
                            ),
                        }
                    )
                if suite == "stages":
                    case["stages"] = {
                        "train": profile_pass(
                            model,
                            loaders["train"],
                            criterion,
                            device,
                            optimizer,
                            detailed=True,
                            channels_last=config.performance.channels_last,
                            gradient_clip=config.optimization.gradient_clip,
                        ),
                        "validation": profile_pass(
                            model,
                            loaders["validation"],
                            criterion,
                            device,
                            None,
                            detailed=True,
                            channels_last=config.performance.channels_last,
                            gradient_clip=config.optimization.gradient_clip,
                        ),
                    }
                del model, optimizer, criterion
        case["memory"] = memory.report()
        if worker_count == 0:
            case["parent_cache"] = {
                key: {
                    "hits": ds.cache.hits,
                    "misses": ds.cache.misses,
                    "payload_bytes": ds.cache.bytes,
                }
                for key, ds in datasets.items()
            }
        report["cases"].append(case)
        write_json(folder / "profile.json", report)
        del loaders, datasets
        gc.collect()
    report["status"] = "completed"
    write_json(folder / "profile.json", report)
    return folder


def profile_destination(output: Path, report_id: str) -> Path:
    """Keep immutable profiling reports separate from existing model experiments."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", report_id):
        raise ValueError("Use a safe new profiling report ID")
    return output / "performance" / report_id
