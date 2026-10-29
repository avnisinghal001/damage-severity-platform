"""Small training/evaluation loops; validation metrics use full native-resolution targets."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from damage_analytics.io import write_csv, write_json
from damage_analytics.training.checkpoints import save_checkpoint
from damage_analytics.training.config import IGNORE_INDEX, TrainingConfig
from damage_analytics.training.dataset import SegmentationDataset, make_loader
from damage_analytics.training.experiments import create_experiment
from damage_analytics.training.losses import SegmentationLoss
from damage_analytics.training.manifests import open_release
from damage_analytics.training.metrics import confusion_matrix, metrics_from_confusion
from damage_analytics.training.model import LightweightUNet
from damage_analytics.training.plots import plot_history
from damage_analytics.training.runtime import configure_runtime
from damage_analytics.training.transforms import restore_prediction
from damage_analytics.training.weights import training_weights


def run_epoch(
    model: LightweightUNet,
    loader: DataLoader,
    criterion: SegmentationLoss,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    gradient_clip: float = 1.0,
) -> dict:
    """Train or evaluate one pass; pool confusion counts before calculating metrics."""
    training = optimizer is not None
    model.train(training)
    weighted_loss, valid_pixels = 0.0, 0
    matrix = np.zeros((9, 9), dtype=np.int64)
    with torch.set_grad_enabled(training):
        for batch in loader:
            images, target = batch["images"].to(device), batch["masks"].to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, target)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Nonfinite loss; refusing to update/save a misleading model")
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), gradient_clip, error_if_nonfinite=True
                )
                optimizer.step()
            count = int((target != IGNORE_INDEX).sum().item())
            weighted_loss += float(loss.detach().cpu()) * count
            valid_pixels += count
            if not training:
                predictions = logits.argmax(dim=1).detach().cpu().numpy().astype(np.uint8)
                for index, prediction in enumerate(predictions):
                    native = restore_prediction(prediction, batch["frames"][index])
                    matrix += confusion_matrix(batch["native_masks"][index], native)
    if not valid_pixels:
        raise ValueError("No supervised pixels in loader")
    return {
        "loss": weighted_loss / valid_pixels,
        "loss_pixels": valid_pixels,
        **({"metrics": metrics_from_confusion(matrix)} if not training else {}),
    }


def train_model(config: TrainingConfig, run_id: str, *, overfit: bool = False) -> Path:
    """Train with frozen splits; optional sanity mode evaluates only its tiny training subset."""
    device = configure_runtime(config.seed, config.device, config.deterministic)
    release = open_release(config)
    train_samples = release.samples("train")
    validation_samples = release.samples("validation")
    if overfit:
        train_samples = train_samples[: config.sanity.overfit_samples]
        validation_samples = train_samples
    training = SegmentationDataset(
        release, train_samples, config.data, training=not overfit, seed=config.seed
    )
    validation = SegmentationDataset(
        release, validation_samples, config.data, training=False, seed=config.seed
    )
    train_loader = make_loader(
        training,
        config.optimization.batch_size,
        config.num_workers,
        config.seed,
        shuffle=not overfit,
    )
    validation_loader = make_loader(
        validation, config.optimization.batch_size, config.num_workers, config.seed, shuffle=False
    )
    model = LightweightUNet(config.model).to(device)
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
        purpose="training_subset_overfit" if overfit else "training",
    )
    metadata["training_sample_ids"] = [row.sample_id for row in train_samples]
    metadata["validation_sample_ids"] = [row.sample_id for row in validation_samples]
    metadata["class_weights"] = weight_report
    metadata["parameter_count"] = sum(parameter.numel() for parameter in model.parameters())
    if overfit:
        metadata["selection"] = "Training subset overfit diagnostic; NOT validation performance"
    write_json(folder / "run.json", metadata)
    write_json(folder / "class_weights.json", weight_report)
    history, best, significant, stale, best_epoch = [], -1.0, -1.0, 0, 0
    epochs = config.sanity.overfit_epochs if overfit else config.optimization.epochs
    try:
        for epoch in range(1, epochs + 1):
            training.set_epoch(epoch)
            trained = run_epoch(
                model,
                train_loader,
                criterion,
                device,
                optimizer,
                gradient_clip=config.optimization.gradient_clip,
            )
            validated = run_epoch(model, validation_loader, criterion, device)
            score = validated["metrics"]["foreground_miou"]
            if score is None:
                raise ValueError("No defined foreground mIoU; validation cannot select a model")
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": trained["loss"],
                    "validation_loss": validated["loss"],
                    "validation_foreground_miou": score,
                }
            )
            improved = score > best
            if improved:
                best, best_epoch = score, epoch
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
            if score > significant + config.optimization.min_delta:
                significant, stale = score, 0
            else:
                stale += 1
            save_checkpoint(
                folder / "last.pt",
                model,
                optimizer,
                epoch=epoch,
                score=score,
                metadata=metadata,
                architecture=config.model,
            )
            write_csv(folder / "history.csv", history, list(history[0]))
            write_json(folder / "history.json", history)
            plot_history(folder, history)
            print(
                f"Epoch {epoch}: train={trained['loss']:.5f} validation={validated['loss']:.5f} "
                f"foreground_mIoU={score:.5f}",
                flush=True,
            )
            if not overfit and stale >= config.optimization.patience:
                break
        status = {
            "status": "completed",
            "epochs_completed": len(history),
            "best_epoch": best_epoch,
            "best_foreground_miou": best,
            "test_evaluated": False,
        }
        if overfit:
            status.update(
                {
                    "diagnostic_only": True,
                    "target": config.sanity.overfit_target_miou,
                    "overfit_target_met": best >= config.sanity.overfit_target_miou,
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
