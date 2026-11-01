"""Matched manual training with existing data/loss/epoch utilities and isolated checkpoints."""

import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import torch

from damage_analytics.io import sha256_file, write_csv, write_json
from damage_analytics.labels import CLASS_NAMES
from damage_analytics.task09.checkpoints import save
from damage_analytics.task09.config import Config, safe_id
from damage_analytics.task09.models import NAMES, build_model
from damage_analytics.training.dataset import SegmentationDataset, make_loader
from damage_analytics.training.epochs import run_epoch
from damage_analytics.training.experiments import create_experiment
from damage_analytics.training.losses import SegmentationLoss
from damage_analytics.training.manifests import Release
from damage_analytics.training.plots import plot_history
from damage_analytics.training.runtime import configure_runtime
from damage_analytics.training.selection import SelectionState, validation_due
from damage_analytics.training.timing import synchronize
from damage_analytics.training.weights import training_weights


def plan(config: Config, release: Release, output: Path) -> dict:
    settings = config.training()
    result = {
        "dataset": release.identity(),
        "models": list(NAMES),
        "config": settings.snapshot(),
        "controls": "Identical frozen splits, normalization, geometry, seed, batch size, CE+Dice, "
        "training-only weights, AdamW, epoch cap, patience and native evaluation",
        "intervention": "Model family + pretrained initialization; not an isolated "
        "pretraining ablation",
        "pretraining": "LR-ASPP COCO_WITH_VOC_LABELS_V1; both classification heads replaced "
        "with nine random RC heads",
        "freeze_pretrained_batch_norm": config.freeze_pretrained_batch_norm,
        "limits": [
            "Historical U-Net used different normalization/LR/50-epoch budget; contextual "
            "baseline only",
            "LR-ASPP's coarse features may hurt thin cracks; lightweight does not imply "
            "faster than the tiny U-Net",
            "Image-space geometry is not physical damage size or structural severity",
            "Source identity and source-label semantics remain unresolved",
        ],
        "run_only_after_task08_finishes": True,
        "training_executed": False,
        "test_used": False,
    }
    if output.exists():
        raise FileExistsError("Plan already exists")
    write_json(output, result)
    return result


def train(
    config: Config,
    release: Release,
    name: str,
    run_id: str,
    *,
    device_name: str = "cpu",
    synthetic: bool = False,
) -> Path:
    """An explicit command is the only code path that can request pretrained weights."""
    if name not in NAMES:
        raise ValueError("Unknown architecture")
    safe_id(run_id)
    settings = config.training(device=device_name)
    folder = settings.output_dir / "experiments" / run_id
    if folder.exists():
        raise FileExistsError("Experiment ID already exists")
    torch.set_num_threads(config.cpu_threads)
    device = configure_runtime(settings.seed, settings.device, settings.deterministic)
    model = build_model(
        name,
        settings.model,
        pretrained=name != "unet" and not synthetic,
        freeze_batch_norm=config.freeze_pretrained_batch_norm,
    ).to(device)
    train_samples, validation_samples = release.samples("train"), release.samples("validation")
    weights, weight_report = training_weights(train_samples, settings.optimization)
    criterion = SegmentationLoss(torch.tensor(weights), settings.optimization.dice_weight).to(
        device
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings.optimization.learning_rate,
        weight_decay=settings.optimization.weight_decay,
    )
    training = SegmentationDataset(
        release, train_samples, settings.data, training=True, seed=settings.seed
    )
    validation = SegmentationDataset(
        release, validation_samples, settings.data, training=False, seed=settings.seed
    )
    train_loader = make_loader(
        training, settings.optimization.batch_size, 0, settings.seed, shuffle=True
    )
    validation_loader = make_loader(
        validation, settings.optimization.batch_size, 0, settings.seed, shuffle=False
    )
    folder, metadata = create_experiment(
        settings,
        release,
        run_id,
        str(device),
        purpose="synthetic_smoke" if synthetic else "training",
    )
    metadata.update(
        {
            "model_name": name,
            "architecture": asdict(settings.model),
            "class_names": list(CLASS_NAMES),
            "data": asdict(settings.data),
            "optimization": asdict(settings.optimization),
            "class_weights": weight_report,
            "freeze_batch_norm": config.freeze_pretrained_batch_norm if name != "unet" else False,
            "initial_weights": getattr(model, "initial_weights", "none_from_scratch"),
            "initial_weights_url": getattr(model, "initial_weights_url", None),
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "training_sample_ids": [s.sample_id for s in train_samples],
            "validation_sample_ids": [s.sample_id for s in validation_samples],
            "reproducibility": "Seeded; float32; deterministic=False for MPS compatibility. "
            "No cross-device bitwise equivalence claim.",
        }
    )
    write_json(folder / "run.json", metadata)
    state, history, started = SelectionState(), [], perf_counter()
    try:
        for epoch in range(1, settings.optimization.epochs + 1):
            training.set_epoch(epoch)
            synchronize(device)
            before = perf_counter()
            trained = run_epoch(
                model,
                train_loader,
                criterion,
                device,
                optimizer,
                gradient_clip=settings.optimization.gradient_clip,
            )
            synchronize(device)
            train_seconds = perf_counter() - before
            before = perf_counter()
            due = validation_due(
                epoch, settings.optimization.epochs, settings.optimization.validation_every
            )
            validated = run_epoch(model, validation_loader, criterion, device) if due else None
            synchronize(device)
            score = validated["metrics"]["foreground_miou"] if due else None
            if due and score is None:
                raise ValueError("Undefined validation foreground mIoU cannot select a checkpoint")
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": trained["loss"],
                    "validation_loss": validated["loss"] if due else None,
                    "validation_foreground_miou": score,
                    "validated": due,
                    "training_seconds": train_seconds,
                    "validation_seconds": perf_counter() - before if due else None,
                }
            )
            if due and state.observe(epoch, score, settings.optimization.min_delta):
                save(folder / "best.pt", model, metadata, epoch, score)
                write_json(folder / "best_validation_metrics.json", validated["metrics"])
            save(folder / "last.pt", model, metadata, epoch, score)
            write_json(folder / "history.json", history)
            write_csv(folder / "history.csv", history, list(history[0]))
            print(
                f"Epoch {epoch}: train={trained['loss']:.5f}, validation_foreground_mIoU={score}",
                flush=True,
            )
            if due and state.stale >= settings.optimization.patience:
                break
        plot_history(folder, history)
        write_json(
            folder / "status.json",
            {
                "status": "completed",
                "epochs_completed": len(history),
                "best_epoch": state.best_epoch,
                "best_foreground_miou": state.best,
                "runtime_seconds": perf_counter() - started,
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


def compare_runs(config: Config, release: Release, run_ids: list[str], output: Path) -> dict:
    """Require the same registered cohort settings before comparing raw validation metrics."""
    if len(set(run_ids)) < 2 or len(set(run_ids)) != len(run_ids):
        raise ValueError("Compare at least two unique Task 09 runs")
    reports, controls, inputs = [], [], []
    for name in run_ids:
        folder = config.output_dir / "experiments" / safe_id(name)
        meta = json.loads((folder / "run.json").read_text())
        status = json.loads((folder / "status.json").read_text())
        if meta["dataset"] != release.identity() or status["status"] != "completed":
            raise ValueError("Comparison requires completed runs on the pinned release")
        controls.append(
            {k: meta[k] for k in ("dataset", "data", "optimization", "seed", "device")}
        )
        inputs.append((name, folder, meta, status))
    if any(c != controls[0] for c in controls[1:]):
        raise ValueError("Comparison data/optimizer/budget/seed/backend settings differ")
    for name, folder, meta, status in inputs:
        digest = sha256_file(folder / "best.pt")
        evaluation = config.output_dir / "evaluations" / digest[:16] / "validation"
        report = read_validation(evaluation)
        if report["checkpoint_sha256"] != digest or report["dataset"] != release.identity():
            raise ValueError("Validation checkpoint identity differs")
        reports.append(
            {
                "run_id": name,
                "model_name": meta["model_name"],
                "checkpoint_sha256": digest,
                "metrics": report["metrics"],
                "initial_weights": meta["initial_weights"],
                "epochs_completed": status["epochs_completed"],
            }
        )
    if output.exists():
        raise FileExistsError("Comparison already exists")
    result = {
        "controls": controls[0],
        "models": reports,
        "scope": "Validation research comparison; no significance, test or severity claim",
    }
    write_json(output, result)
    return result


def read_validation(folder: Path) -> dict:
    from damage_analytics.analytics.validation import read_validation_result

    return read_validation_result(folder)
