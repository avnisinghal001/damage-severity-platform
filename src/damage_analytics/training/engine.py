"""Small training/evaluation loops; validation metrics use full native-resolution targets."""

from pathlib import Path
from time import perf_counter

import torch

from damage_analytics.io import write_csv, write_json
from damage_analytics.training.checkpoints import save_checkpoint
from damage_analytics.training.config import TrainingConfig
from damage_analytics.training.dataset import SegmentationDataset, make_loader
from damage_analytics.training.epochs import run_epoch as run_epoch
from damage_analytics.training.experiments import create_experiment
from damage_analytics.training.losses import SegmentationLoss
from damage_analytics.training.manifests import open_release
from damage_analytics.training.model import LightweightUNet
from damage_analytics.training.plots import plot_history
from damage_analytics.training.runtime import configure_runtime
from damage_analytics.training.selection import SelectionState, validation_due
from damage_analytics.training.timing import synchronize
from damage_analytics.training.weights import training_weights


def train_model(
    config: TrainingConfig, run_id: str, *, overfit: bool = False, exploratory: bool = False
) -> Path:
    """Train with frozen splits; optional sanity mode evaluates only its tiny training subset."""
    device = configure_runtime(config.seed, config.device, config.deterministic)
    release = open_release(config)
    train_samples = release.samples("train")
    validation_samples = release.samples("validation")
    if overfit:
        train_samples = train_samples[: config.sanity.overfit_samples]
        validation_samples = train_samples
    training = SegmentationDataset(
        release,
        train_samples,
        config.data,
        training=not overfit,
        seed=config.seed,
        cache_mb=config.performance.cache_mb,
    )
    validation = SegmentationDataset(
        release,
        validation_samples,
        config.data,
        training=False,
        seed=config.seed,
        cache_mb=config.performance.cache_mb,
    )
    train_loader = make_loader(
        training,
        config.optimization.batch_size,
        config.num_workers,
        config.seed,
        shuffle=not overfit,
        persistent_workers=config.performance.persistent_workers,
        prefetch_factor=config.performance.prefetch_factor,
    )
    validation_loader = make_loader(
        validation,
        config.optimization.batch_size,
        config.num_workers,
        config.seed,
        shuffle=False,
        persistent_workers=config.performance.persistent_workers,
        prefetch_factor=config.performance.prefetch_factor,
    )
    model = LightweightUNet(config.model).to(device)
    if config.performance.channels_last:
        model = model.to(memory_format=torch.channels_last)
    weights, weight_report = training_weights(train_samples, config.optimization)
    criterion = SegmentationLoss(torch.tensor(weights), config.optimization.dice_weight).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.optimization.learning_rate,
        weight_decay=config.optimization.weight_decay,
    )
    folder, metadata = create_experiment(
        config,
        release,
        run_id,
        str(device),
        purpose="training_subset_overfit"
        if overfit
        else ("exploratory_training" if exploratory else "training"),
    )
    metadata["training_sample_ids"] = [row.sample_id for row in train_samples]
    metadata["validation_sample_ids"] = [row.sample_id for row in validation_samples]
    metadata["class_weights"] = weight_report
    metadata["parameter_count"] = sum(parameter.numel() for parameter in model.parameters())
    if overfit:
        metadata["selection"] = "Training subset overfit diagnostic; NOT validation performance"
    write_json(folder / "class_weights.json", weight_report)
    history, selection = [], SelectionState()
    metadata["validation_schedule"] = {
        "every": 1 if overfit else config.optimization.validation_every,
        "first_and_final": True,
        "patience_units": "completed validation checks",
        "note": "A sparser schedule changes selectable epochs; compare identical schedules.",
    }
    write_json(folder / "run.json", metadata)
    epochs = config.sanity.overfit_epochs if overfit else config.optimization.epochs
    try:
        for epoch in range(1, epochs + 1):
            training.set_epoch(epoch)
            synchronize(device)
            start = perf_counter()
            trained = run_epoch(
                model,
                train_loader,
                criterion,
                device,
                optimizer,
                gradient_clip=config.optimization.gradient_clip,
                channels_last=config.performance.channels_last,
            )
            synchronize(device)
            training_seconds = perf_counter() - start
            start = perf_counter()
            due = validation_due(
                epoch, epochs, 1 if overfit else config.optimization.validation_every
            )
            validated = (
                run_epoch(
                    model,
                    validation_loader,
                    criterion,
                    device,
                    channels_last=config.performance.channels_last,
                )
                if due
                else None
            )
            if due:
                synchronize(device)
            validation_seconds = perf_counter() - start if due else 0.0
            score = validated["metrics"]["foreground_miou"] if validated else None
            if due and score is None:
                raise ValueError("No defined foreground mIoU; validation cannot select a model")
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": trained["loss"],
                    "validation_loss": validated["loss"] if validated else None,
                    "validation_foreground_miou": score,
                    "validated": due,
                    "training_seconds": training_seconds,
                    "validation_seconds": validation_seconds,
                }
            )
            artifact_start = perf_counter()
            improved = (
                selection.observe(epoch, score, config.optimization.min_delta) if due else False
            )
            if improved:
                save_checkpoint(
                    folder / "best.pt",
                    model,
                    optimizer,
                    epoch=epoch,
                    score=score,
                    metadata=metadata,
                    architecture=config.model,
                )
                write_json(folder / "best_validation_metrics.json", validated["metrics"])
            save_checkpoint(
                folder / "last.pt",
                model,
                optimizer,
                epoch=epoch,
                score=score,
                metadata=metadata,
                architecture=config.model,
            )
            stopping = due and not overfit and selection.stale >= config.optimization.patience
            if epoch % config.performance.plot_every == 0 or epoch == epochs or stopping:
                plot_history(folder, history)
            history[-1]["artifact_seconds"] = perf_counter() - artifact_start
            write_csv(folder / "history.csv", history, list(history[0]))
            write_json(folder / "history.json", history)
            detail = (
                f"validation={validated['loss']:.5f} foreground_mIoU={score:.5f}"
                if due
                else "validation=not_scheduled"
            )
            print(f"Epoch {epoch}: train={trained['loss']:.5f} {detail}", flush=True)
            if stopping:
                break
        status = {
            "status": "completed",
            "epochs_completed": len(history),
            "best_epoch": selection.best_epoch,
            "best_foreground_miou": selection.best,
            "validation_checks": selection.checks,
            "exploratory": exploratory,
            "test_evaluated": False,
        }
        if overfit:
            status.update(
                {
                    "diagnostic_only": True,
                    "target": config.sanity.overfit_target_miou,
                    "overfit_target_met": selection.best >= config.sanity.overfit_target_miou,
                }
            )
        write_json(folder / "status.json", status)
    except Exception as exc:
        write_json(
            folder / "status.json",
            {"status": "failed", "error": str(exc), "test_evaluated": False},
        )
        raise
    return folder
