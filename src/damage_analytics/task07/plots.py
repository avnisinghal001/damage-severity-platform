"""Task-native static figures without hardcoded RC1841 labels."""

from pathlib import Path

import numpy as np


def curves(folder: Path, history: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for key, label in (("train_loss", "Training"), ("validation_loss", "Validation")):
        axes[0].plot([r["epoch"] for r in history], [r[key] for r in history], label=label)
    axes[0].set(xlabel="Epoch", ylabel="Cross entropy")
    axes[0].legend()
    axes[1].plot([r["epoch"] for r in history], [r["validation_roc_auc"] for r in history])
    axes[1].set(xlabel="Epoch", ylabel="Validation ROC-AUC", ylim=(0, 1))
    figure.tight_layout()
    figure.savefig(folder / "learning_curves.png", dpi=120)
    plt.close(figure)


def confusion(path: Path, matrix: list, labels: list[str]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = np.asarray(matrix)
    figure, axis = plt.subplots(figsize=(6, 5))
    shown = axis.imshow(values, cmap="Blues")
    axis.set(
        xticks=range(len(labels)),
        yticks=range(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted",
        ylabel="Ground truth",
    )
    for (i, j), count in np.ndenumerate(values):
        axis.text(
            j,
            i,
            str(count),
            ha="center",
            va="center",
            color="white" if count > values.max() / 2 else "black",
        )
    figure.colorbar(shown, ax=axis)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)
