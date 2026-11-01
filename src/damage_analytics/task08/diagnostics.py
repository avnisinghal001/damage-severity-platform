"""Known-definition validation diagnostics; no external-domain threshold optimization."""

from pathlib import Path

import numpy as np

from damage_analytics.task07.metrics import classification_metrics, from_confusion
from damage_analytics.task08.inputs import Validation


def flat_metrics(metrics: dict) -> dict:
    crack = metrics["classes"][1]
    return {
        **{
            k: metrics[k]
            for k in (
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "roc_auc",
                "pr_auc",
                "average_precision",
            )
        },
        "crack_precision": crack["precision"],
        "crack_recall": crack["recall"],
        "crack_f1": crack["f1"],
    }


def curve_points(target: list[int], probability: list[float]) -> list[dict]:
    """Tie-grouped empirical ROC/PR points, with an explicit origin and undefined ratios null."""
    y, p = np.asarray(target, dtype=int), np.asarray(probability, dtype=float)
    if (
        not len(y)
        or p.shape != y.shape
        or not np.isfinite(p).all()
        or not np.isin(y, [0, 1]).all()
    ):
        raise ValueError("Curves need finite binary predictions")
    order = np.argsort(-p, kind="stable")
    y, p = y[order], p[order]
    ends = np.r_[np.flatnonzero(p[:-1] != p[1:]), len(y) - 1]
    tp = np.r_[0, np.cumsum(y)[ends]]
    fp = np.r_[0, (np.arange(1, len(y) + 1) - np.cumsum(y))[ends]]
    thresholds = [None, *p[ends].tolist()]
    positives, negatives = int(y.sum()), int(len(y) - y.sum())
    return [
        {
            "threshold": threshold,
            "true_positive_rate": int(t) / positives if positives else None,
            "false_positive_rate": int(f) / negatives if negatives else None,
            "recall": int(t) / positives if positives else None,
            "precision": int(t) / int(t + f) if t + f else 1.0,
        }
        for threshold, t, f in zip(thresholds, tp, fp, strict=True)
    ]


def threshold_sensitivity(validation: Validation) -> list[dict]:
    """SDNET source validation only. This table neither chooses nor persists a new threshold."""
    if (
        validation.release.metadata["dataset"] != "sdnet2018"
        or validation.release.identity() != validation.run.payload["dataset"]
    ):
        raise ValueError("Threshold sensitivity is restricted to SDNET source validation")
    y = np.array([r["target"] for r in validation.predictions])
    p = np.array([r["probability_crack"] for r in validation.predictions])
    frozen = validation.run.payload["threshold"]["threshold"]
    thresholds = sorted(set(np.linspace(0, 1, 101).tolist() + [frozen]))
    results = []
    for threshold in thresholds:
        predicted = (p >= threshold).astype(int)
        metrics = from_confusion(np.bincount(2 * y + predicted, minlength=4).reshape(2, 2))
        results.append(
            {
                "threshold": threshold,
                "frozen_checkpoint_threshold": threshold == frozen,
                "crack_precision": metrics["classes"][1]["precision"],
                "crack_recall": metrics["classes"][1]["recall"],
                "crack_f1": metrics["classes"][1]["f1"],
                "macro_f1": metrics["macro_f1"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "false_positives": metrics["confusion_matrix"][0][1],
                "false_negatives": metrics["confusion_matrix"][1][0],
            }
        )
    return results


def score_summary(records: tuple[dict, ...]) -> dict:
    result = {}
    for label in (0, 1):
        group = [r for r in records if r["target"] == label]
        values = [r["probability_crack"] for r in group]
        result[str(label)] = {
            "samples": len(group),
            "score_quantiles_05_25_50_75_95": np.quantile(
                values, [0.05, 0.25, 0.5, 0.75, 0.95]
            ).tolist()
            if values
            else None,
            "errors": sum(r["prediction"] != label for r in group),
        }
    return result


def material_breakdown(validation: Validation) -> list[dict]:
    """Patch-level metrics per supplied material tag; group counts remain provisional."""
    if validation.release.metadata["dataset"] != "sdnet2018":
        raise ValueError("Material tags only exist for SDNET")
    results = []
    for domain in sorted({r["row"]["domain"] for r in validation.predictions}):
        subset = [r for r in validation.predictions if r["row"]["domain"] == domain]
        metrics = classification_metrics(
            [r["target"] for r in subset],
            [r["probability_crack"] for r in subset],
            validation.run.payload["threshold"]["threshold"],
        )
        results.append(
            {
                "material_tag": domain,
                "samples": len(subset),
                "crack_samples": sum(r["target"] for r in subset),
                "candidate_components": len({r["row"]["group_id"] for r in subset}),
                **flat_metrics(metrics),
            }
        )
    return results


def figures(
    output: Path, validation: Validation, points: list[dict], sensitivity: list[dict] | None
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    axes[0].plot(
        [r["false_positive_rate"] for r in points], [r["true_positive_rate"] for r in points]
    )
    axes[0].plot([0, 1], [0, 1], "--", color="gray")
    matrix = validation.metrics["confusion_matrix"]
    crack = validation.metrics["classes"][1]
    if sum(matrix[0]) and crack["recall"] is not None:
        axes[0].scatter([matrix[0][1] / sum(matrix[0])], [crack["recall"]], color="black", s=18)
    axes[0].set(
        xlabel="False positive rate", ylabel="Crack recall", title="ROC", xlim=(0, 1), ylim=(0, 1)
    )
    axes[1].plot([r["recall"] for r in points], [r["precision"] for r in points])
    prevalence = sum(r["target"] for r in validation.predictions) / len(validation.predictions)
    axes[1].axhline(prevalence, linestyle="--", color="gray")
    if crack["recall"] is not None and crack["precision"] is not None:
        axes[1].scatter([crack["recall"]], [crack["precision"]], color="black", s=18)
    axes[1].set(
        xlabel="Crack recall",
        ylabel="Crack precision",
        title="Precision-recall",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    for label, name in ((0, "Source no-crack"), (1, "Source crack")):
        axes[2].hist(
            [r["probability_crack"] for r in validation.predictions if r["target"] == label],
            bins=np.linspace(0, 1, 26),
            alpha=0.5,
            density=True,
            label=name,
        )
    axes[2].axvline(
        validation.run.payload["threshold"]["threshold"], color="black", linestyle="--"
    )
    axes[2].set(
        xlabel="Uncalibrated crack score",
        ylabel="Within-class density",
        title="Scores and frozen threshold",
    )
    axes[2].legend(fontsize=8)
    figure.suptitle(
        f"{validation.run.root.name} / {validation.release.metadata['dataset']} validation"
    )
    figure.tight_layout()
    figure.savefig(output / "roc_pr_scores.png", dpi=120)
    plt.close(figure)
    if sensitivity is not None:
        figure, axis = plt.subplots(figsize=(7, 4))
        for key in ("crack_precision", "crack_recall", "crack_f1", "balanced_accuracy"):
            axis.plot(
                [r["threshold"] for r in sensitivity], [r[key] for r in sensitivity], label=key
            )
        axis.axvline(
            validation.run.payload["threshold"]["threshold"], color="black", linestyle="--"
        )
        axis.set(xlabel="Threshold (source validation only)", ylabel="Metric", ylim=(0, 1))
        axis.legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(output / "sdnet_threshold_sensitivity.png", dpi=120)
        plt.close(figure)
