"""Tiny sealed segmentation release fixture; never touches the real frozen dataset."""

import csv
from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from damage_analytics.curation.verification import write_checksums
from damage_analytics.io import sha256_file, write_json
from damage_analytics.labels import CLASS_NAMES
from damage_analytics.training.config import (
    DataConfig,
    ModelConfig,
    OptimizationConfig,
    SanityConfig,
    TrainingConfig,
)


@pytest.fixture
def training_release(tmp_path):
    """Mimic only the authenticated Task 02 schema needed by Task 03 unit tests."""
    root = tmp_path / "release"
    for folder in ("images", "masks", "splits"):
        (root / folder).mkdir(parents=True)
    mask = np.zeros((8, 12), dtype=np.uint8)
    mask[1:7, 1:5] = 1
    mask[2:6, 7:11] = 2
    counts = np.bincount(mask.ravel(), minlength=9).tolist()
    for split_index, split in enumerate(("train", "validation", "test")):
        image = np.stack([mask * 70, mask * 50, mask * 30], axis=-1)
        image[0, 0, 0] = split_index
        Image.fromarray(image).save(root / "images" / f"{split}.png")
        Image.fromarray(mask).save(root / "masks" / f"{split}.png")
        row = {
            "sample_id": split,
            "image": f"images/{split}.png",
            "mask": f"masks/{split}.png",
            "image_sha256": sha256_file(root / f"images/{split}.png"),
            "mask_sha256": sha256_file(root / f"masks/{split}.png"),
            "width": 12,
            "height": 8,
            "group_id": split,
            "split": split,
            **{f"pixels_{i}": count for i, count in enumerate(counts)},
        }
        with (root / "splits" / f"{split}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
    write_json(root / "release.json", {"dataset_version": "synthetic"})
    write_json(
        root / "quality_report.json",
        {
            "dataset_version": "synthetic",
            "valid_samples": 3,
            "verification": {"passed": True},
            "ml_readiness": {"technical_checks_passed": True},
            "label_mapping": [
                {"class_id": i, "source_label": name} for i, name in enumerate(CLASS_NAMES)
            ],
        },
    )
    write_checksums(root)
    config = TrainingConfig(
        root,
        "synthetic",
        sha256_file(root / "checksums.json"),
        tmp_path / "outputs",
        42,
        "cpu",
        True,
        0,
        DataConfig(image_size=(16, 16), crop_size=(8, 8)),
        ModelConfig(4, 2, 2),
        OptimizationConfig(epochs=2, batch_size=1, patience=2),
        SanityConfig(overfit_epochs=2),
    )
    return replace(config, data=replace(config.data, crop_probability=0))
