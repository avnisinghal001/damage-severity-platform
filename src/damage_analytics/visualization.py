"""Generate a deterministic gallery from validated, unchanged inputs."""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import TYPE_CHECKING

from damage_analytics.config import Config
from damage_analytics.io import sha256_file, write_json
from damage_analytics.labels import CLASS_NAMES, DISPLAY_COLORS

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage


def colorize_mask(mask: PILImage) -> PILImage:
    """Map class IDs to project RGB colors, rejecting any unknown mask IDs."""
    import numpy as np
    from PIL import Image

    values = np.asarray(mask)
    if values.ndim != 2 or values.min() < 0 or values.max() >= len(DISPLAY_COLORS):
        raise ValueError("Mask must contain only project class IDs 0..8")
    colors = np.asarray(DISPLAY_COLORS, dtype=np.uint8)[values]
    return Image.fromarray(colors)


def _contained_file(root: Path, relative: str) -> Path:
    """Validate manifest paths before opening them, including symlink resolution."""
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"Invalid manifest path: {relative}")
    return path


def create_gallery(config: Config) -> list[str]:
    """Save original/mask/overlay rows plus a legend; check selected-file hashes.

    Background is transparent in overlays. Stored image pixels are displayed
    without EXIF rotation to remain aligned with the polygon coordinate system.
    """
    import matplotlib
    import numpy as np
    from PIL import Image

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    with (config.output_dir / "manifest.csv").open(newline="", encoding="utf-8") as stream:
        records = list(csv.DictReader(stream))
    if not records:
        raise ValueError("No valid pairs available; inspect issues.csv and rerun analyze")
    selected = random.Random(config.random_seed).sample(
        records, min(config.gallery_size, len(records)))
    figure, axes = plt.subplots(len(selected), 3, figsize=(12, 3.3 * len(selected)), squeeze=False)
    try:
        for row_index, row in enumerate(selected):
            paths = {}
            inputs = (("image", config.extracted_dir), ("annotation", config.extracted_dir),
                      ("mask", config.output_dir))
            for field, root in inputs:
                path = _contained_file(root, row[field])
                if sha256_file(path) != row[f"{field}_sha256"]:
                    raise ValueError(f"{field} changed for {row['sample_id']}; rerun analyze")
                paths[field] = path
            with Image.open(paths["image"]) as source:
                original = source.convert("RGB")
            with Image.open(paths["mask"]) as source:
                mask = source.copy()
            if mask.size != original.size:
                raise ValueError("Generated mask/image dimension mismatch; rerun analyze")
            colored = colorize_mask(mask)
            blended = Image.blend(original, colored, config.overlay_alpha)
            foreground = Image.fromarray((np.asarray(mask) != 0).astype(np.uint8) * 255)
            overlay = Image.composite(blended, original, foreground)
            for axis, picture, title in zip(axes[row_index], (original, colored, overlay),
                                            ("Original", "Derived mask", "Overlay"), strict=True):
                axis.imshow(picture)
                axis.set_title(f"{row['sample_id']} | {title}", fontsize=9)
                axis.axis("off")
        legend = [Patch(color=np.array(color) / 255, label=name)
                  for name, color in zip(CLASS_NAMES, DISPLAY_COLORS, strict=True)]
        figure.legend(handles=legend, loc="lower center", ncol=3, fontsize=9)
        figure.suptitle("RC1841 | source polygons → derived masks | project display colors")
        # Reserve an absolute-height legend area, including for a one-row gallery.
        figure.tight_layout(rect=(0, 0.9 / (3.3 * len(selected)), 1, 0.975))
        figure.savefig(config.output_dir / "sample_gallery.png", dpi=140)
    finally:
        plt.close(figure)
    sample_ids = [row["sample_id"] for row in selected]
    write_json(config.output_dir / "gallery_samples.json",
               {"seed": config.random_seed, "sample_ids": sample_ids,
                "palette": "project display colors; not publisher RGB values"})
    return sample_ids
