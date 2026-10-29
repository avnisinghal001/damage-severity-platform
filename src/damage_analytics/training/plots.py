"""Static experiment artifacts with class legends and no safety/severity interpretation."""

from pathlib import Path

import numpy as np
from PIL import Image

from damage_analytics.labels import CLASS_NAMES, DISPLAY_COLORS


def plot_history(folder: Path, history: list[dict]) -> None:
    """Save training/validation loss and validation foreground mIoU curves."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    epochs = [row["epoch"] for row in history]
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="Training")
    axes[0].plot(epochs, [row["validation_loss"] for row in history], label="Validation")
    axes[0].set(xlabel="Epoch", ylabel="Loss (valid-pixel-weighted batch mean)")
    axes[0].legend()
    axes[1].plot(epochs, [row["validation_foreground_miou"] for row in history])
    axes[1].set(xlabel="Epoch", ylabel="Validation foreground mIoU", ylim=(0, 1))
    figure.tight_layout()
    figure.savefig(folder / "learning_curves.png", dpi=150)
    plt.close(figure)


def save_comparison(
    path: Path,
    image: Image.Image,
    target: np.ndarray | None,
    prediction: np.ndarray,
    sample_id: str,
) -> None:
    """Display native RGB, optional ground truth, and predicted IDs using project colors."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    colors = np.asarray(DISPLAY_COLORS, dtype=np.uint8)
    panels = [(image, "Original RGB")]
    if target is not None:
        panels.append((Image.fromarray(colors[target]), "Ground truth"))
    panels.append((Image.fromarray(colors[prediction]), "Prediction"))
    figure, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 5))
    for axis, (picture, title) in zip(axes, panels, strict=True):
        preview = picture.copy()
        preview.thumbnail((600, 600), Image.Resampling.NEAREST)
        axis.imshow(preview)
        axis.set_title(title)
        axis.axis("off")
    figure.legend(
        handles=[
            Patch(color=np.asarray(color) / 255, label=name)
            for name, color in zip(CLASS_NAMES, DISPLAY_COLORS, strict=True)
        ],
        loc="lower center",
        ncol=3,
        fontsize=8,
    )
    figure.suptitle(f"{sample_id} | segmentation categories, not structural safety")
    figure.tight_layout(rect=(0, 0.14, 1, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)


def plot_confusion(path: Path, matrix: np.ndarray) -> None:
    """Row-normalized confusion plot; exact integer counts are saved separately."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    normalized = np.divide(
        matrix,
        matrix.sum(axis=1, keepdims=True),
        out=np.zeros((9, 9), dtype=float),
        where=matrix.sum(axis=1, keepdims=True) > 0,
    )
    figure, axis = plt.subplots(figsize=(10, 8))
    shown = axis.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
    axis.set(
        xticks=range(9),
        yticks=range(9),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        xlabel="Prediction",
        ylabel="Ground truth",
    )
    plt.setp(axis.get_xticklabels(), rotation=50, ha="right")
    figure.colorbar(shown, ax=axis, label="Fraction of ground-truth class")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
