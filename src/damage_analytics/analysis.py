"""Compute descriptive segmentation statistics and headless report plots."""

from collections import Counter

from damage_analytics.config import Config
from damage_analytics.io import write_csv, write_json
from damage_analytics.labels import CLASS_NAMES


def analyze_dataset(config: Config, records: list[dict], validation: dict) -> dict:
    """Aggregate valid-pair pixel counts, source polygon counts, and dimensions.

    Pixel fractions are pooled by pixels (large images weigh more). Image-level
    class presence uses final visible pixels; polygon counts use original JSON
    labels before overlap resolution. These measure different things.
    """
    import numpy as np

    total_pixels = sum(row["width"] * row["height"] for row in records)
    classes = []
    for index, name in enumerate(CLASS_NAMES):
        pixels = sum(row["pixel_counts"][index] for row in records)
        classes.append({
            "class_id": index, "source_label": name, "pixels": pixels,
            "pixel_fraction": pixels / total_pixels if total_pixels else 0,
            "images_with_pixels": sum(row["pixel_counts"][index] > 0 for row in records),
            "images_with_annotation": sum(name in row["polygon_counts"] for row in records),
            "polygons": sum(row["polygon_counts"].get(name, 0) for row in records),
        })
    dimension_counts = Counter((row["width"], row["height"]) for row in records)
    dimensions = [{"width": width, "height": height, "images": count}
                  for (width, height), count in sorted(dimension_counts.items())]
    dimension_stats = {}
    for key in ("width", "height"):
        values = [row[key] for row in records]
        dimension_stats[key] = ({"min": min(values), "max": max(values),
                                 "median": float(np.median(values)),
                                 "mean": float(np.mean(values))} if values else None)
    summary = {
        "validation_status": validation["status"], "valid_images": len(records),
        "total_pixels": total_pixels, "total_polygons": sum(r["polygon_count"] for r in records),
        "dimension_statistics": dimension_stats,
        "images_with_polygon_overlap": sum(row["overlap_pixels"] > 0 for row in records),
        "overlap_pixels": sum(row["overlap_pixels"] for row in records),
        "image_modes": dict(Counter(row["image_mode"] for row in records)),
        "classes": classes,
        "severity": "not available; no severity labels or scores created",
        "population": "valid pairs only; byte/pixel duplicates are retained and reported",
    }
    write_json(config.output_dir / "summary.json", summary)
    write_csv(config.output_dir / "class_distribution.csv", classes,
              ["class_id", "source_label", "pixels", "pixel_fraction", "images_with_pixels",
               "images_with_annotation", "polygons"])
    write_csv(config.output_dir / "dimensions.csv", dimensions, ["width", "height", "images"])
    per_image = [{"sample_id": row["sample_id"], "width": row["width"], "height": row["height"],
                  "polygons": row["polygon_count"], "overlap_pixels": row["overlap_pixels"],
                  **{f"pixels_{index}": value for index, value in enumerate(row["pixel_counts"])}}
                 for row in records]
    write_csv(config.output_dir / "per_image.csv", per_image,
              ["sample_id", "width", "height", "polygons", "overlap_pixels",
               *[f"pixels_{index}" for index in range(len(CLASS_NAMES))]])
    if records:
        _plot_statistics(config, records, classes)
    return summary


def _plot_statistics(config: Config, records: list[dict], classes: list[dict]) -> None:
    """Create three legible plots without requiring an interactive desktop."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder = config.output_dir / "plots"
    folder.mkdir(exist_ok=True)
    names = [row["source_label"] for row in classes]
    for filename, field, title, xlabel in (
        ("class_pixels.png", "pixel_fraction", "Pooled pixel distribution",
         "Fraction of all pixels"),
        ("category_frequency.png", "images_with_annotation", "Source category frequency",
         "Images containing source annotation (multiple categories allowed)"),
    ):
        selected = classes if field == "pixel_fraction" else classes[1:]
        labels = names if field == "pixel_fraction" else names[1:]
        figure, axis = plt.subplots(figsize=(10, 5))
        axis.barh(labels, [row[field] for row in selected])
        axis.set(title=title, xlabel=xlabel)
        axis.invert_yaxis()
        figure.tight_layout()
        figure.savefig(folder / filename, dpi=160)
        plt.close(figure)
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.scatter([row["width"] for row in records], [row["height"] for row in records],
                 alpha=0.3, s=15)
    axis.set(xlabel="Stored image width (pixels)", ylabel="Stored image height (pixels)",
             title="Image dimensions (valid pairs)")
    figure.tight_layout()
    figure.savefig(folder / "image_dimensions.png", dpi=160)
    plt.close(figure)
