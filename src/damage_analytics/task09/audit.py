"""Read-only RC1841 QA: sealed metadata, development masks and polygon decoding."""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from damage_analytics.annotations import read_annotation
from damage_analytics.curation.masks import checked_mask
from damage_analytics.io import write_csv, write_json
from damage_analytics.labels import CLASS_NAMES
from damage_analytics.task09.config import Config
from damage_analytics.training.manifests import Release, read_pair
from damage_analytics.training.transforms import frame_for, restore_prediction, transform_mask


def manifest_rows(release: Release) -> dict[str, dict]:
    with release.file("curated_manifest.csv").open() as stream:
        return {row["sample_id"]: row for row in csv.DictReader(stream)}


def selected_training(release: Release, count: int) -> tuple:
    """Cover present classes and spatial extremes deterministically, not model errors."""
    samples = release.samples("train")
    preferred = [max(samples, key=lambda s: s.pixel_counts[i]) for i in range(1, 9)]
    preferred += [
        max(samples, key=lambda s: s.width * s.height),
        max(samples, key=lambda s: max(s.width / s.height, s.height / s.width)),
    ]
    ordered = dict.fromkeys(
        s.sample_id for s in preferred + sorted(samples, key=lambda s: s.sample_id)
    )
    chosen = set(list(ordered)[:count])
    return tuple(s for s in samples if s.sample_id in chosen)


def audit(
    config: Config, release: Release, output: Path, *, full_development: bool = False
) -> dict:
    """Decode development samples only: all validation and bounded training by default."""
    output.mkdir(parents=True, exist_ok=False)
    rows = manifest_rows(release)
    quality = json.loads(release.file("quality_report.json").read_text())
    exact, groups = defaultdict(set), defaultdict(set)
    for row in rows.values():  # Membership/hash metadata only, including sealed test partition.
        for field in ("source_image_sha256", "image_sha256"):
            exact[(field, row[field])].add(row["split"])
        groups[row["group_id"]].add(row["split"])
    crossing = [key for key, splits in exact.items() if len(splits) > 1]
    if crossing or any(len(splits) > 1 for splits in groups.values()):
        raise ValueError("Known exact/source components cross frozen splits")
    stats = []
    for split in ("train", "validation"):
        samples = release.samples(split)
        counts = np.array([s.pixel_counts for s in samples], dtype=np.int64)
        for class_id, label in enumerate(CLASS_NAMES):
            stats.append(
                {
                    "split": split,
                    "class_id": class_id,
                    "label": label,
                    "images": int(np.count_nonzero(counts[:, class_id])),
                    "pixels": int(counts[:, class_id].sum()),
                    "pixel_percent": float(100 * counts[:, class_id].sum() / counts.sum()),
                }
            )
    training = (
        release.samples("train")
        if full_development
        else selected_training(release, config.audit_training_samples)
    )
    checked, issues, resize_records = [], [], []
    pooled_resize = np.zeros((9, 9), dtype=np.int64)
    for split, samples in (("train", training), ("validation", release.samples("validation"))):
        for sample in samples:
            try:
                image, mask = read_pair(release, sample)
                row = rows[sample.sample_id]
                polygons = read_annotation(
                    release.file(row["annotation"]), Path(sample.image).name, image.size
                )
                rendered, evidence = checked_mask(polygons, image.size)
                if evidence["conflict_pixels"] or not np.array_equal(
                    np.asarray(rendered), np.asarray(mask)
                ):
                    raise ValueError("Cleaned polygon raster differs from sealed class-ID mask")
                checked.append(
                    {
                        "sample_id": sample.sample_id,
                        "split": split,
                        "width": sample.width,
                        "height": sample.height,
                        "polygons": len(polygons),
                        "mask_ids": ";".join(map(str, np.unique(mask).tolist())),
                    }
                )
                if split == "validation":
                    frame = frame_for(image.size, config.data.image_size)
                    resized = transform_mask(mask, frame, config.data.image_size)
                    restored = restore_prediction(np.asarray(resized), frame)
                    target = np.asarray(mask)
                    pooled_resize += np.bincount(
                        9 * target.ravel().astype(np.int64) + restored.ravel(), minlength=81
                    ).reshape(9, 9)
                    for class_id in range(1, 9):
                        original = sample.pixel_counts[class_id]
                        if original:
                            resize_records.append(
                                {
                                    "sample_id": sample.sample_id,
                                    "class_id": class_id,
                                    "source_pixels": original,
                                    "model_grid_pixels": int(
                                        np.count_nonzero(np.asarray(resized) == class_id)
                                    ),
                                    "native_roundtrip_pixels": int(
                                        np.count_nonzero(restored == class_id)
                                    ),
                                }
                            )
            except (ValueError, OSError, KeyError) as exc:
                issues.append({"sample_id": sample.sample_id, "split": split, "reason": str(exc)})
    from damage_analytics.training.metrics import metrics_from_confusion

    report = {
        "dataset": release.identity(),
        "split_counts_metadata": {s: len(v) for s, v in release.rows.items()},
        "class_mapping": [{"class_id": i, "label": name} for i, name in enumerate(CLASS_NAMES)],
        "class_statistics": stats,
        "decoded_and_rerasterized": dict(Counter(r["split"] for r in checked)),
        "full_development_decode": full_development,
        "issues": issues,
        "known_exact_or_group_crossings": len(crossing),
        "dimensions_development": dict(
            Counter(
                f"{s.width}x{s.height}"
                for split in ("train", "validation")
                for s in release.samples(split)
            )
        ),
        "annotation_format": "Cleaned LabelMe polygons; L-mode PNG IDs 0..8; background is "
        "polygon complement",
        "resize_validation_roundtrip": metrics_from_confusion(pooled_resize),
        "entire_class_occurrences_lost": sum(r["model_grid_pixels"] == 0 for r in resize_records),
        "prior_release_exclusions": quality["excluded_samples"],
        "prior_exclusion_reasons": quality["exclusion_reasons_nonexclusive"],
        "limitations": quality["unresolved_limitations"],
        "all_positive_development_images": all(
            sum(s.pixel_counts[1:]) > 0
            for split in ("train", "validation")
            for s in release.samples(split)
        ),
        "test_pixels_annotations_or_metrics_read": False,
        "scope": "All split/hash metadata verified; fresh validation and selected training "
        "rasters checked. "
        "Historical full-release QA is referenced, not repeated on test. No expert label "
        "correction.",
    }
    write_json(output / "audit.json", report)
    write_csv(output / "class_statistics.csv", stats, list(stats[0]))
    write_csv(
        output / "checked_samples.csv",
        checked,
        ["sample_id", "split", "width", "height", "polygons", "mask_ids"],
    )
    write_csv(output / "issues.csv", issues, ["sample_id", "split", "reason"])
    write_csv(
        output / "resize_occurrences.csv",
        resize_records,
        ["sample_id", "class_id", "source_pixels", "model_grid_pixels", "native_roundtrip_pixels"],
    )
    if issues:
        raise ValueError(f"Dataset QA found {len(issues)} issues; see {output / 'issues.csv'}")
    return report
