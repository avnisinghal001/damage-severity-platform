"""Separate validation and explicit, single-result final-test evaluation commands."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from damage_analytics.io import sha256_file, write_csv, write_json
from damage_analytics.training.checkpoints import load_checkpoint
from damage_analytics.training.config import (
    DataConfig,
    ModelConfig,
    OptimizationConfig,
    TrainingConfig,
)
from damage_analytics.training.dataset import SegmentationDataset, make_loader
from damage_analytics.training.engine import run_epoch
from damage_analytics.training.experiments import execution_provenance
from damage_analytics.training.losses import SegmentationLoss
from damage_analytics.training.manifests import Release, open_release
from damage_analytics.training.plots import plot_confusion, save_comparison
from damage_analytics.training.runtime import configure_runtime
from damage_analytics.training.transforms import restore_prediction


def effective_config(config: TrainingConfig, payload: dict) -> TrainingConfig:
    """Use saved preprocessing/loss settings independently of current hyperparameter edits."""
    saved = payload["metadata"]["config"]
    return replace(
        config,
        seed=saved["seed"],
        data=DataConfig(**saved["data"]),
        model=ModelConfig(**payload["architecture"]),
        optimization=OptimizationConfig(**saved["optimization"]),
    )


def evaluate_checkpoint(
    config: TrainingConfig, checkpoint: Path, *, final_test: bool = False
) -> Path:
    """Evaluate validation normally; only this explicit flag opens test pixels/targets.

    Final test requires a completed normal training run's best checkpoint. It creates
    one non-overwritable result directory, never feeds early stopping or model selection,
    and refuses synthetic/overfit checkpoints. It cannot prevent deliberate test tuning
    across separately named experiments; that remains a research protocol obligation.
    """
    runtime_seed = config.seed
    device = configure_runtime(runtime_seed, config.device, config.deterministic)
    release = open_release(config)
    model, payload = load_checkpoint(checkpoint, device, expected_dataset=release.identity())
    config = effective_config(config, payload)
    checkpoint_hash = sha256_file(checkpoint)
    if final_test:
        import json

        status = json.loads((checkpoint.parent / "status.json").read_text())
        if (
            payload["metadata"]["purpose"] != "training"
            or status["status"] != "completed"
            or sha256_file(checkpoint.parent / "best.pt") != checkpoint_hash
            or payload["epoch"] != status["best_epoch"]
        ):
            raise ValueError(
                "Final test requires the completed run's validation-selected best checkpoint"
            )
    split = "test" if final_test else "validation"
    folder = (
        config.output_dir
        / "evaluations"
        / checkpoint_hash[:16]
        / ("test_final" if final_test else "validation")
    )
    folder.mkdir(parents=True, exist_ok=False)
    samples = release.samples(split, final_test=final_test)
    dataset = SegmentationDataset(release, samples, config.data, training=False, seed=config.seed)
    loader = make_loader(
        dataset, config.optimization.batch_size, config.num_workers, config.seed, shuffle=False
    )
    weights = payload["metadata"]["class_weights"]["weights"]
    criterion = SegmentationLoss(torch.tensor(weights), config.optimization.dice_weight).to(device)
    try:
        result = run_epoch(model, loader, criterion, device)
        result.update(
            {
                "dataset": release.identity(),
                "split": split,
                "checkpoint_sha256": checkpoint_hash,
                "checkpoint_epoch": payload["epoch"],
                "device": str(device),
                "runtime_seed": runtime_seed,
                "deterministic_requested": config.deterministic,
                "execution": execution_provenance(),
                "data_config": payload["metadata"]["config"]["data"],
                "metric_grid": "Native ground truth vs unpadded argmax IDs "
                "nearest-restored to native size",
                "test_evaluated": final_test,
                "purpose": "Final frozen test evaluation"
                if final_test
                else "Validation evaluation; permitted for model selection",
            }
        )
        write_json(folder / "metrics.json", result)
        write_csv(
            folder / "per_class.csv",
            result["metrics"]["classes"],
            list(result["metrics"]["classes"][0]),
        )
        matrix = np.asarray(result["metrics"]["confusion_matrix"], dtype=np.int64)
        np.save(folder / "confusion_matrix.npy", matrix, allow_pickle=False)
        plot_confusion(folder / "confusion_matrix.png", matrix)
        _comparisons(model, loader, release, device, folder, config.comparison_samples)
    except Exception as exc:
        write_json(
            folder / "failure.json",
            {
                "error": str(exc),
                "split": split,
                "note": "Result directory is preserved to record an attempted evaluation",
            },
        )
        raise
    return folder


def _comparisons(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    release: Release,
    device: torch.device,
    folder: Path,
    limit: int,
) -> None:
    """Create a fixed prefix of split examples; never choose examples by model score."""
    model.eval()
    saved = 0
    with torch.inference_mode():
        for batch in loader:
            values = model(batch["images"].to(device)).argmax(1).cpu().numpy().astype(np.uint8)
            for index, prediction in enumerate(values):
                sample = batch["samples"][index]
                native = restore_prediction(prediction, batch["frames"][index])
                with Image.open(release.file(sample.image)) as image:
                    save_comparison(
                        folder / "comparisons" / f"{saved:03d}.png",
                        image,
                        batch["native_masks"][index],
                        native,
                        sample.sample_id,
                    )
                saved += 1
                if saved >= limit:
                    return
