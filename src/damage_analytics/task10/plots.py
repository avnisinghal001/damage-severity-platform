"""Task-specific loss/accuracy/macro-F1 curves; no binary ROC or ordinal-grade assumptions."""

from pathlib import Path


def history_plot(output: Path, rows: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    epochs = [r["epoch"] for r in rows]
    for field, name in (("train_loss", "Training"), ("validation_loss", "Validation")):
        axes[0].plot(epochs, [r[field] for r in rows], label=name)
    for field in ("validation_accuracy", "validation_macro_f1"):
        axes[1].plot(epochs, [r[field] for r in rows], label=field)
    axes[0].set(xlabel="Epoch", ylabel="Weighted CE")
    axes[1].set(xlabel="Epoch", ylabel="Validation metric", ylim=(0, 1))
    for axis in axes:
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output / "learning_curves.png", dpi=120)
    plt.close(figure)
