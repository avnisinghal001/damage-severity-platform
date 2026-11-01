"""Pillow failure panels using the exact project palette and original coordinate frame."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from damage_analytics.labels import CLASS_NAMES, DISPLAY_COLORS


def failure_panel(
    path: Path,
    image: Image.Image,
    target: np.ndarray,
    prediction: np.ndarray,
    sample_id: str,
    class_id: int,
) -> None:
    palette = np.asarray(DISPLAY_COLORS, dtype=np.uint8)
    error = np.full((*target.shape, 3), 235, dtype=np.uint8)
    error[(target == class_id) & (prediction == class_id)] = (35, 160, 75)
    error[(target != class_id) & (prediction == class_id)] = (220, 40, 40)
    error[(target == class_id) & (prediction != class_id)] = (40, 80, 220)
    panels = [
        (image, "Original RGB"),
        (Image.fromarray(palette[target]), "Ground truth IDs"),
        (Image.fromarray(palette[prediction]), "Predicted IDs"),
        (Image.fromarray(error), "Class: green TP / red FP / blue FN"),
    ]
    canvas = Image.new("RGB", (1120, 355), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (10, 8), f"{sample_id} | {CLASS_NAMES[class_id]} | native-grid evaluation", fill="black"
    )
    for index, (panel, title) in enumerate(panels):
        thumbnail = ImageOps.contain(panel, (270, 290), Image.Resampling.NEAREST)
        canvas.paste(thumbnail, (index * 280 + (280 - thumbnail.width) // 2, 35))
        draw.text((index * 280 + 6, 335), title, fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
