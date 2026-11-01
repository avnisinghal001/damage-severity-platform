"""Transfer learning with validation ROC-AUC selection and a frozen decision threshold."""

import importlib.metadata
import re
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn

from damage_analytics.io import sha256_file, write_csv, write_json
from damage_analytics.task07 import checkpoints
from damage_analytics.task07.config import TrainConfig, validate_training
from damage_analytics.task07.data import MEAN, STD
from damage_analytics.task07.loops import loader_for, pass_epoch, prediction_rows
from damage_analytics.task07.metrics import classification_metrics, select_threshold
from damage_analytics.task07.models import TinySmokeNet, build_model
from damage_analytics.task07.plots import curves
from damage_analytics.task07.release import Release
from damage_analytics.training.experiments import execution_provenance
from damage_analytics.training.runtime import configure_runtime
from damage_analytics.training.selection import SelectionState


def train(
    release: Release,
    settings: TrainConfig,
    output: Path,
    run_id: str,
    backbone: str,
    seed: int = 42,
    *,
    allow_provisional: bool = False,
    synthetic: bool = False,
) -> Path:
    """Real training explicitly requests pretrained weights. No implicit smoke downloads."""
    validate_training(settings)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", run_id):
        raise ValueError("Unsafe run ID")
    folder = output / "experiments" / run_id
    if folder.exists():
        raise FileExistsError("Run ID already exists")
    if not release.quality["final_evaluation_eligible"] and not allow_provisional:
        raise ValueError(
            "Unverified source grouping/near matches. Supply evidence or explicitly "
            "use --allow-provisional for exploratory training (no final test)."
        )
    train_rows, validation_rows = release.samples("train"), release.samples("validation")
    if any({r["label"] for r in rows} != {0, 1} for rows in (train_rows, validation_rows)):
        raise ValueError("Train and validation each require both binary classes")
    torch.set_num_threads(settings.cpu_threads)
    device = configure_runtime(seed, settings.device, settings.deterministic)
    # The only production codepath that can request ImageNet network downloads.
    model = (TinySmokeNet() if synthetic else build_model(backbone, pretrained=True)).to(device)
    counts = np.bincount([r["label"] for r in train_rows], minlength=2)
    weights = (
        len(train_rows) / (2 * counts) if settings.class_weighting == "balanced" else np.ones(2)
    )
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device), reduction="sum"
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay
    )
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, settings.epochs)
        if settings.scheduler == "cosine"
        else None
    )
    folder.mkdir(parents=True, exist_ok=False)
    metadata = {
        "task": "binary_crack_presence",
        "purpose": "synthetic_smoke" if synthetic else "training",
        "backbone": "tiny_synthetic" if synthetic else backbone,
        "seed": seed,
        "device": str(device),
        "dataset": release.identity(),
        "release_dir": str(release.root),
        "settings": asdict(settings),
        "normalization": {"mean": MEAN, "std": STD, "resize": "full-image letterbox, bilinear"},
        "pretrained": not synthetic,
        "weights": getattr(model, "_task07_weights", "none_synthetic"),
        "weights_url": getattr(model, "_task07_weights_url", None),
        "class_weights": weights.tolist(),
        "selection": "validation ROC-AUC; threshold chosen ONCE on selected validation checkpoint",
        "quality": release.quality,
        "execution": execution_provenance(),
        "development_ids": [r["sample_id"] for r in (*train_rows, *validation_rows)],
        "threshold": None,
        "test_evaluated": False,
        "limitations": "Crack recognition only; no OOD, general-damage or severity claim.",
    }
    try:
        metadata["torchvision_version"] = importlib.metadata.version("torchvision")
    except importlib.metadata.PackageNotFoundError:
        metadata["torchvision_version"] = None
    write_json(folder / "run.json", metadata)
    train_loader = loader_for(release, train_rows, settings, seed, True)
    validation_loader = loader_for(release, validation_rows, settings, seed)
    state, history = SelectionState(), []
    started = perf_counter()
    try:
        for epoch in range(1, settings.epochs + 1):
            train_loader.dataset.set_epoch(epoch)
            trained = pass_epoch(model, train_loader, criterion, device, optimizer)
            validated = pass_epoch(model, validation_loader, criterion, device)
            metrics = classification_metrics(validated["targets"], validated["probabilities"], 0.5)
            score = metrics["roc_auc"]
            if score is None:
                raise ValueError("Validation ROC-AUC undefined; cannot select a checkpoint")
            improved = state.observe(epoch, score, settings.min_delta)
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": trained["loss"],
                    "validation_loss": validated["loss"],
                    "validation_roc_auc": score,
                    "validation_accuracy_at_0_5": metrics["accuracy"],
                    "validation_macro_f1_at_0_5": metrics["macro_f1"],
                    "validation_balanced_accuracy_at_0_5": metrics["balanced_accuracy"],
                    "validation_crack_precision_at_0_5": metrics["classes"][1]["precision"],
                    "validation_crack_recall_at_0_5": metrics["classes"][1]["recall"],
                    "validation_crack_f1_at_0_5": metrics["classes"][1]["f1"],
                    "validation_pr_auc": metrics["pr_auc"],
                    "train_seconds": trained["runtime"]["seconds"],
                    "validation_seconds": validated["runtime"]["seconds"],
                    "train_images_per_second": trained["runtime"]["images_per_second"],
                    "validation_images_per_second": validated["runtime"]["images_per_second"],
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )
            payload = {**metadata, "epoch": epoch, "selection_score": score}
            if improved:
                checkpoints.save(folder / "best.pt", model, payload)
            checkpoints.save(folder / "last.pt", model, payload)
            write_json(folder / "history.json", history)
            write_csv(folder / "history.csv", history, list(history[0]))
            print(
                f"Epoch {epoch}: loss={trained['loss']:.5f}, val_ROC_AUC={score:.5f}", flush=True
            )
            if scheduler:
                scheduler.step()
            if state.stale >= settings.patience:
                break
        selected, payload = checkpoints.load(folder / "best.pt", device)
        validation = pass_epoch(selected, validation_loader, criterion, device)
        threshold = select_threshold(
            validation["targets"], validation["probabilities"], settings.threshold_objective
        )
        selected_metrics = classification_metrics(
            validation["targets"], validation["probabilities"], threshold["threshold"]
        )
        rows = prediction_rows(validation, validation_rows, threshold["threshold"])
        write_csv(folder / "validation_predictions.csv", rows, list(rows[0]))
        threshold["prediction_csv_sha256"] = sha256_file(folder / "validation_predictions.csv")
        threshold["split_sha256"] = release.identity()["splits"]["validation"]
        write_json(folder / "threshold.json", threshold)
        checkpoints.save(
            folder / "best.pt",
            selected,
            {
                **{
                    k: v for k, v in payload.items() if k not in ("model_state", "torch_rng_state")
                },
                "threshold": threshold,
            },
        )
        write_json(folder / "validation_metrics.json", selected_metrics)
        curves(folder, history)
        write_json(
            folder / "runtime.json",
            {
                "run_wall_seconds": perf_counter() - started,
                "train_pass_seconds": sum(r["train_seconds"] for r in history),
                "validation_pass_seconds": sum(r["validation_seconds"] for r in history),
                "selected_validation_pass": validation["runtime"],
                "device": str(device),
                "epochs_executed": len(history),
                "scope": "Run wall time includes epoch checkpointing, thresholding and plots; "
                "excludes model initialization/weight download. Not model-only latency.",
            },
        )
        write_json(
            folder / "status.json",
            {
                "status": "completed",
                "epochs": len(history),
                "best_epoch": state.best_epoch,
                "best_validation_roc_auc": state.best,
                "test_evaluated": False,
            },
        )
    except Exception as exc:
        write_json(
            folder / "status.json",
            {"status": "failed", "error": str(exc), "test_evaluated": False},
        )
        raise
    return folder
