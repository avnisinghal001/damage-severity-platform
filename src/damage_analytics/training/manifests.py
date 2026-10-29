"""Authenticate the pinned Task 02 release and consume, never recreate, its splits."""

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from damage_analytics.curation.verification import verify_checksums
from damage_analytics.io import contained_file, sha256_file
from damage_analytics.labels import CLASS_NAMES
from damage_analytics.training.config import TrainingConfig


@dataclass(frozen=True)
class Sample:
    """One exact frozen-manifest row with release-relative image/mask locations."""

    sample_id: str
    image: str
    mask: str
    width: int
    height: int
    group_id: str
    pixel_counts: tuple[int, ...]


@dataclass(frozen=True)
class Release:
    """An authenticated input snapshot; test consumption always needs an explicit flag."""

    root: Path
    release_id: str
    checksums_sha256: str
    split_hashes: dict[str, str]
    rows: dict[str, tuple[Sample, ...]]
    checksums: dict[str, str]

    def samples(self, split: str, *, final_test: bool = False) -> tuple[Sample, ...]:
        """Forbid accidental test loading by routine training/validation commands."""
        if split == "test" and not final_test:
            raise ValueError("Test samples require the explicit final-test command")
        return self.rows[split]

    def file(self, relative: str) -> Path:
        """Check containment and content hash before selected sample data is opened."""
        path = contained_file(self.root, relative)
        if sha256_file(path) != self.checksums.get(relative):
            raise ValueError(f"Release file changed: {relative}")
        return path

    def identity(self) -> dict:
        """Portable experiment provenance independent of the mutable latest pointer."""
        return {
            "release_id": self.release_id,
            "checksums_sha256": self.checksums_sha256,
            "split_hashes": self.split_hashes,
        }


def open_release(config: TrainingConfig, *, full_integrity: bool = False) -> Release:
    """Authenticate seals, schemas, label mapping and partitions without decoding test pixels."""
    root = config.release_dir
    actual_hash = sha256_file(root / "checksums.json")
    if actual_hash != config.checksums_sha256:
        raise ValueError("Pinned release checksum manifest differs")
    checksums = json.loads((root / "checksums.json").read_text())

    def verified_json(relative: str) -> dict:
        path = contained_file(root, relative)
        if sha256_file(path) != checksums[relative]:
            raise ValueError(f"Sealed report changed: {relative}")
        return json.loads(path.read_text())

    snapshot = verified_json("release.json")
    quality = verified_json("quality_report.json")
    if (
        snapshot["dataset_version"] != config.release_id
        or quality["dataset_version"] != config.release_id
        or not quality["verification"]["passed"]
        or not quality["ml_readiness"]["technical_checks_passed"]
    ):
        raise ValueError("Release ID or Task 02 technical verification is invalid")
    expected_labels = [
        {"class_id": i, "source_label": label} for i, label in enumerate(CLASS_NAMES)
    ]
    if quality["label_mapping"] != expected_labels:
        raise ValueError("Frozen category mapping differs from IDs 0–8")
    rows, hashes, seen, groups = {}, {}, set(), {}
    for split in ("train", "validation", "test"):
        relative = f"splits/{split}.csv"
        path = contained_file(root, relative)
        hashes[split] = sha256_file(path)
        if hashes[split] != checksums[relative]:
            raise ValueError(f"Frozen {split} split changed")
        with path.open(newline="", encoding="utf-8") as stream:
            table = list(csv.DictReader(stream))
        parsed = []
        for row in table:
            sample_id, group = row["sample_id"], row["group_id"]
            counts = tuple(int(row[f"pixels_{i}"]) for i in range(len(CLASS_NAMES)))
            size = (int(row["width"]), int(row["height"]))
            if (
                sample_id in seen
                or row["split"] != split
                or min(size) < 1
                or min(counts) < 0
                or sum(counts) != size[0] * size[1]
                or groups.get(group, split) != split
            ):
                raise ValueError("Invalid/duplicate sample, mask counts, or cross-split group")
            for field in ("image", "mask"):
                if checksums.get(row[field]) != row[f"{field}_sha256"]:
                    raise ValueError("Manifest image/mask fingerprint differs from release seal")
                contained_file(root, row[field])
            seen.add(sample_id)
            groups[group] = split
            parsed.append(Sample(sample_id, row["image"], row["mask"], *size, group, counts))
        if not parsed:
            raise ValueError(f"Empty {split} split")
        rows[split] = tuple(parsed)
    if len(seen) != quality["valid_samples"]:
        raise ValueError("Split population differs from verified release")
    if full_integrity:
        # Integrity-only byte hashing includes sealed test artifacts, not test evaluation.
        verify_checksums(root)
    return Release(root, config.release_id, actual_hash, hashes, rows, checksums)


def read_pair(release: Release, sample: Sample) -> tuple[Image.Image, Image.Image]:
    """Reopen a selected RGB image and class-ID mask with exact alignment/ID checks."""
    import numpy as np

    with Image.open(release.file(sample.image)) as source:
        source.load()
        if source.mode != "RGB" or source.size != (sample.width, sample.height):
            raise ValueError(f"Image mode/dimensions changed: {sample.sample_id}")
        image = source.copy()
    with Image.open(release.file(sample.mask)) as source:
        source.load()
        values = np.asarray(source)
        if (
            source.mode != "L"
            or source.size != image.size
            or values.max() >= len(CLASS_NAMES)
            or tuple(np.bincount(values.ravel(), minlength=len(CLASS_NAMES)))
            != sample.pixel_counts
        ):
            raise ValueError(f"Mask IDs/dimensions/counts changed: {sample.sample_id}")
        mask = source.copy()
    return image, mask
