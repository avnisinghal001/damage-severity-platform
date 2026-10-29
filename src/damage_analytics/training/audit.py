"""Measure destructive mask resizing on train/validation, without needing PyTorch."""

import numpy as np
from PIL import Image

from damage_analytics.io import write_csv, write_json
from damage_analytics.labels import CLASS_NAMES
from damage_analytics.training.config import IGNORE_INDEX, TrainingConfig
from damage_analytics.training.experiments import execution_provenance
from damage_analytics.training.manifests import Release
from damage_analytics.training.metrics import confusion_matrix, metrics_from_confusion
from damage_analytics.training.transforms import (
    choose_crop,
    frame_for,
    restore_prediction,
    transform_mask,
)
from damage_analytics.training.weights import training_weights


def audit_resizing(config: TrainingConfig, release: Release) -> dict:
    """Report native-grid mask roundtrip retention and disappearance, excluding test data."""
    rows, summaries = [], {}
    for split in ("train", "validation"):
        matrix = np.zeros((9, 9), dtype=np.int64)
        for index, sample in enumerate(release.samples(split)):
            with Image.open(release.file(sample.mask)) as source:
                source.load()
                mask = source.copy()
            original = np.asarray(mask)
            if (
                mask.mode != "L"
                or mask.size != (sample.width, sample.height)
                or original.max() > 8
            ):
                raise ValueError(f"Invalid mask: {sample.sample_id}")
            frame = frame_for(mask.size, config.data.image_size)
            resized = np.asarray(transform_mask(mask, frame, config.data.image_size))
            restored = restore_prediction(resized, frame)
            matrix += confusion_matrix(original, restored)
            counts = np.bincount(resized[resized != IGNORE_INDEX], minlength=9)
            crop_counts = None
            if split == "train":
                rng = np.random.default_rng(np.random.SeedSequence([config.seed, 0, index]))
                box = choose_crop(mask, config.data, rng)
                crop = mask.crop(box)
                crop_frame = frame_for(mask.size, config.data.image_size, box)
                crop_target = np.asarray(transform_mask(mask, crop_frame, config.data.image_size))
                crop_counts = (
                    np.bincount(np.asarray(crop).ravel(), minlength=9),
                    np.bincount(crop_target[crop_target != IGNORE_INDEX], minlength=9),
                )
            for class_id, label in enumerate(CLASS_NAMES[1:], 1):
                if not sample.pixel_counts[class_id]:
                    continue
                rows.append(
                    {
                        "split": split,
                        "sample_id": sample.sample_id,
                        "class_id": class_id,
                        "source_label": label,
                        "native_pixels": sample.pixel_counts[class_id],
                        "resized_pixels": int(counts[class_id]),
                        "entire_class_lost": bool(counts[class_id] == 0),
                        "crop_source_pixels": int(crop_counts[0][class_id]) if crop_counts else "",
                        "crop_resized_pixels": int(crop_counts[1][class_id])
                        if crop_counts
                        else "",
                    }
                )
        metrics = metrics_from_confusion(matrix)
        summaries[split] = {
            "samples": len(release.samples(split)),
            "roundtrip": metrics,
            "lost_class_occurrences": [
                {
                    "class_id": i,
                    "source_label": CLASS_NAMES[i],
                    "source_images": sum(
                        row["split"] == split and row["class_id"] == i for row in rows
                    ),
                    "entire_class_lost": sum(
                        row["split"] == split and row["class_id"] == i and row["entire_class_lost"]
                        for row in rows
                    ),
                }
                for i in range(1, 9)
            ],
        }
    result = {
        "dataset": release.identity(),
        "configuration": config.snapshot(),
        "execution": execution_provenance(),
        "target_hw": list(config.data.image_size),
        "splits": summaries,
        "test_accessed": False,
        "meaning": "Ground-truth resize/nearest-restore fidelity, NOT model performance",
        "crop_audit": "One seeded foreground-guided candidate per training sample; "
        "crop-out context "
        "is distinct from resize loss; not an estimate of full training distribution",
        "recommendation": "Inspect thin/rare-class losses. Training-only crops preserve "
        "more local "
        "resolution but introduce foreground-sampling bias; validation stays full-frame.",
    }
    folder = config.output_dir / "data_audit"
    _, class_weights = training_weights(release.samples("train"), config.optimization)
    result["training_class_weights"] = class_weights
    write_json(folder / "training_class_weights.json", class_weights)
    write_json(folder / "resize_audit.json", result)
    write_csv(
        folder / "resize_occurrences.csv",
        rows,
        [
            "split",
            "sample_id",
            "class_id",
            "source_label",
            "native_pixels",
            "resized_pixels",
            "entire_class_lost",
            "crop_source_pixels",
            "crop_resized_pixels",
        ],
    )
    return result
