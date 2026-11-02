"""Reproducible ResNet18 transfer learning and validation-only four-class evaluation."""

from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from damage_analytics.io import sha256_file, write_csv, write_json
from damage_analytics.task07.data import MEAN, STD
from damage_analytics.task07.plots import confusion
from damage_analytics.task10 import checkpoints
from damage_analytics.task10.config import CLASS_NAMES, Config, TrainConfig, safe_id
from damage_analytics.task10.data import loader
from damage_analytics.task10.learning import pass_epoch, prediction_rows
from damage_analytics.task10.metrics import metrics
from damage_analytics.task10.models import build_model
from damage_analytics.task10.plots import history_plot
from damage_analytics.task10.release import Release
from damage_analytics.training.experiments import execution_provenance
from damage_analytics.training.runtime import configure_runtime
from damage_analytics.training.selection import SelectionState


def train(
    config: Config,
    release: Release,
    run_id: str,
    *,
    device_name: str = "cpu",
    allow_provisional: bool = False,
    synthetic: bool = False,
) -> Path:
    safe_id(run_id)
    folder = config.output_dir / "experiments" / run_id
    if folder.exists():
        raise FileExistsError("Run ID already exists")
    if not release.quality["technical_training_ready"]:
        raise ValueError(
            "Dataset has hard readiness blockers; --allow-provisional cannot bypass them"
        )
    unresolved = []
    if not release.quality["encoding_verified"]:
        unresolved.append(
            "Caffe/BGR reconstruction is numerical inference without publisher verification"
        )
    if not release.quality["provided_source_evidence_complete"]:
        unresolved.append(
            "Original photo/component/structure identities are unavailable or incomplete"
        )
    if unresolved and not allow_provisional:
        raise ValueError(
            "Provisional training requires explicit --allow-provisional: " + "; ".join(unresolved)
        )
    settings = config.training
    settings.validate()
    torch.set_num_threads(settings.cpu_threads)
    device = configure_runtime(config.seed, device_name, settings.deterministic)
    counts = np.bincount([r["class_id"] for r in release.samples("train")], minlength=4)
    if np.any(counts == 0) or {r["class_id"] for r in release.samples("validation")} != set(
        range(4)
    ):
        raise ValueError("All four classes must be represented in train and validation")
    weights = (
        len(release.samples("train")) / (4 * counts)
        if settings.class_weighting == "balanced"
        else np.ones(4)
    )
    # Only this explicit real-training branch may request ImageNet weights.
    model = build_model(pretrained=not synthetic, synthetic=synthetic).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay
    )
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, settings.epochs)
        if settings.scheduler == "cosine"
        else None
    )
    weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    batches = loader(release, "train", settings, config.seed, provisional=allow_provisional)
    validation = loader(
        release, "validation", settings, config.seed, provisional=allow_provisional
    )
    folder.mkdir(parents=True, exist_ok=False)
    metadata = {
        "purpose": "synthetic_smoke" if synthetic else "training",
        "target": "phi_net_task7_visible_component_damage",
        "class_names": list(CLASS_NAMES),
        "dataset": release.identity(),
        "release_dir": str(release.root),
        "seed": config.seed,
        "training": asdict(settings),
        "device": str(device),
        "class_weights": weights.tolist(),
        "training_class_counts": counts.tolist(),
        "weights_population": "Selected training manifest only",
        "initial_weights": getattr(model, "_task07_weights", "none_synthetic"),
        "initial_weights_url": getattr(model, "_task07_weights_url", None),
        "normalization": {
            "mean": list(MEAN),
            "std": list(STD),
            "resize": "full-image letterbox, bilinear",
        },
        "array_encoding": release.metadata["inputs"]["settings"],
        "allow_inferred_encoding": allow_provisional,
        "provisional_limitations": unresolved,
        "selection": "Best validation macro-F1 over fixed source IDs 0..3",
        "execution": execution_provenance(),
        "test_evaluated": False,
        "scope": "No RC1841 label pairing, whole-building safety or calibrated confidence",
    }
    write_json(folder / "run.json", metadata)
    state, history, started = SelectionState(), [], perf_counter()
    try:
        for epoch in range(1, settings.epochs + 1):
            batches.dataset.set_epoch(epoch)
            trained = pass_epoch(model, batches, weight_tensor, device, optimizer)
            validated = pass_epoch(model, validation, weight_tensor, device)
            result = metrics(validated["targets"], validated["probabilities"])
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": trained["loss"],
                    "validation_loss": validated["loss"],
                    "validation_accuracy": result["accuracy"],
                    "validation_macro_f1": result["macro_f1"],
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "training_seconds": trained["runtime"]["seconds"],
                    "validation_seconds": validated["runtime"]["seconds"],
                }
            )
            if state.observe(epoch, result["macro_f1"], settings.min_delta):
                checkpoints.save(folder / "best.pt", model, metadata, epoch, result["macro_f1"])
            checkpoints.save(folder / "last.pt", model, metadata, epoch, result["macro_f1"])
            write_json(folder / "history.json", history)
            write_csv(folder / "history.csv", history, list(history[0]))
            print(
                f"Epoch {epoch}: train={trained['loss']:.5f}, "
                f"val_macro_F1={result['macro_f1']:.5f}",
                flush=True,
            )
            if scheduler:
                scheduler.step()
            if state.stale >= settings.patience:
                break
        selected, payload = checkpoints.load(folder / "best.pt", device, release.identity())
        selected_result = pass_epoch(selected, validation, weight_tensor, device)
        selected_metrics = metrics(selected_result["targets"], selected_result["probabilities"])
        write_json(folder / "validation_metrics.json", selected_metrics)
        predictions = prediction_rows(selected_result)
        write_csv(folder / "validation_predictions.csv", predictions, list(predictions[0]))
        write_csv(
            folder / "per_class.csv",
            selected_metrics["classes"],
            list(selected_metrics["classes"][0]),
        )
        confusion(
            folder / "confusion_matrix.png",
            selected_metrics["confusion_matrix"],
            list(CLASS_NAMES),
        )
        history_plot(folder, history)
        write_json(
            folder / "status.json",
            {
                "status": "completed",
                "epochs_completed": len(history),
                "best_epoch": state.best_epoch,
                "best_validation_macro_f1": state.best,
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


def evaluate(
    config: Config, release: Release, checkpoint: Path, *, device_name: str = "cpu"
) -> Path:
    """Validation only; no threshold tuning or final-test branch exists."""
    torch.set_num_threads(config.training.cpu_threads)
    digest = sha256_file(checkpoint)
    device = configure_runtime(config.seed, device_name, False)
    model, payload = checkpoints.load(checkpoint, device, release.identity())
    metadata = payload["metadata"]
    settings = TrainConfig(**metadata["training"])
    settings.validate()
    if metadata["array_encoding"] != release.metadata["inputs"]["settings"]:
        raise ValueError("Checkpoint array decoder/configuration differs from pinned release")
    expected = {"mean": list(MEAN), "std": list(STD), "resize": "full-image letterbox, bilinear"}
    if metadata["normalization"] != expected:
        raise ValueError("Checkpoint normalization differs from implemented preprocessing")
    output = config.output_dir / "evaluations" / digest[:16] / "validation"
    output.mkdir(parents=True, exist_ok=False)
    batches = loader(
        release,
        "validation",
        settings,
        metadata["seed"],
        provisional=metadata["allow_inferred_encoding"],
    )
    result = pass_epoch(
        model,
        batches,
        torch.tensor(metadata["class_weights"], dtype=torch.float32, device=device),
        device,
    )
    report = {
        "metrics": metrics(result["targets"], result["probabilities"]),
        "loss": result["loss"],
        "runtime": result["runtime"],
        "checkpoint_sha256": digest,
        "checkpoint_epoch": payload["epoch"],
        "dataset": release.identity(),
        "split": "validation",
        "test_evaluated": False,
        "class_names": list(CLASS_NAMES),
        "device": str(device),
        "seed": metadata["seed"],
        "training_purpose": metadata["purpose"],
        "provisional_limitations": metadata["provisional_limitations"],
        "execution": execution_provenance(),
    }
    if sha256_file(checkpoint) != digest:
        raise ValueError("Checkpoint changed during validation")
    write_json(output / "metrics.json", report)
    records = prediction_rows(result)
    write_csv(output / "predictions.csv", records, list(records[0]))
    write_csv(
        output / "per_class.csv",
        report["metrics"]["classes"],
        list(report["metrics"]["classes"][0]),
    )
    confusion(
        output / "confusion_matrix.png", report["metrics"]["confusion_matrix"], list(CLASS_NAMES)
    )
    return output
