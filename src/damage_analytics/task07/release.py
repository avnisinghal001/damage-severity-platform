"""Authenticate immutable new-dataset releases; explicit test access only."""

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from damage_analytics.curation.verification import verify_checksums
from damage_analytics.io import contained_file, sha256_file
from damage_analytics.task07.splits import split_report


@dataclass(frozen=True)
class Release:
    root: Path
    metadata: dict
    quality: dict
    rows: tuple[dict, ...]
    seal: dict

    def identity(self) -> dict:
        return {
            "release_id": self.metadata["release_id"],
            "checksums_sha256": sha256_file(self.root / "checksums.json"),
            "splits": {s: self.seal[f"splits/{s}.csv"] for s in ("train", "validation", "test")},
        }

    def samples(
        self, split: str, task: str = "classification", *, final_test: bool = False
    ) -> tuple[dict, ...]:
        if split not in ("train", "validation", "test") or (split == "test" and not final_test):
            raise ValueError("Test samples require the explicit final-test command")
        return tuple(r for r in self.rows if r["split"] == split and r["task"] == task)

    def file(self, relative: str) -> Path:
        path = contained_file(self.root, relative)
        if sha256_file(path) != self.seal.get(relative):
            raise ValueError(f"Release file changed: {relative}")
        return path

    def image(self, row: dict) -> Image.Image:
        with Image.open(self.file(row["image"])) as source:
            source.load()
            result = ImageOps.exif_transpose(source).convert("RGB")
        if result.size != (row["width"], row["height"]):
            raise ValueError("Image dimensions changed")
        return result


def open_release(root: Path, *, full: bool = False) -> Release:
    """Verify seals and partition semantics without decoding test pixels for routine use."""
    root = root.resolve()
    seal = json.loads((root / "checksums.json").read_text())

    def checked(relative):
        path = contained_file(root, relative)
        if sha256_file(path) != seal.get(relative):
            raise ValueError(f"Sealed metadata changed: {relative}")
        return path

    metadata = json.loads(checked("release.json").read_text())
    quality = json.loads(checked("quality.json").read_text())
    rows = tuple(json.loads(line) for line in checked("manifest.jsonl").read_text().splitlines())
    near = json.loads(checked("near_candidates.json").read_text())
    grouping = json.loads(checked("grouping.json").read_text())
    if not rows or len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("Empty release or duplicate sample IDs")
    if any(r["dataset"] != metadata["dataset"] for r in rows):
        raise ValueError("Mixed dataset taxonomies")
    for row in rows:
        if row["task"] == "classification" and row["label"] not in (0, 1):
            raise ValueError("Invalid binary label")
        if row["task"] == "segmentation" and (
            len(row["pixel_counts"]) != 3
            or min(row["pixel_counts"]) < 0
            or sum(row["pixel_counts"]) != row["width"] * row["height"]
        ):
            raise ValueError("Invalid native mask counts")
        for field in ("image", "annotation", "yolo", "source_mask", "mask"):
            if row.get(field):
                contained_file(root, row[field])
                if seal.get(row[field]) != row[f"{field}_sha256"]:
                    raise ValueError("Manifest/file hash differs")
    seen = set()
    for split in ("train", "validation", "test"):
        with checked(f"splits/{split}.csv").open() as stream:
            table = list(csv.DictReader(stream))
        expected = {r["sample_id"]: r for r in rows if r["split"] == split}
        if len(table) != len(expected) or {r["sample_id"] for r in table} != set(expected):
            raise ValueError("Split population differs from manifest")
        for item in table:
            row = expected[item["sample_id"]]
            if (
                item["sample_id"] in seen
                or any(item[k] != str(row[k]) for k in ("group_id", "task", "image"))
                or item["label"] != ("" if row["label"] is None else str(row["label"]))
            ):
                raise ValueError("Split row changed or duplicated")
            seen.add(item["sample_id"])
    actual = split_report(list(rows), near, grouping["screening_truncated"])
    for key in (
        "known_component_checks_passed",
        "source_identity_verified",
        "cross_split_unresolved_near_pairs",
    ):
        if actual[key] != quality[key]:
            raise ValueError("Quality report does not match manifest evidence")
    if not actual["known_component_checks_passed"]:
        raise ValueError("Known duplicates/groups cross split boundaries")
    result = Release(root, metadata, quality, rows, seal)
    if full:
        verify_checksums(root)
        for row in rows:
            result.image(row)
            if row["mask"]:
                with Image.open(result.file(row["mask"])) as source:
                    values = np.asarray(source)
                    if (
                        source.mode != "L"
                        or source.size != (row["width"], row["height"])
                        or values.max() > 2
                        or np.bincount(values.ravel(), minlength=3).tolist() != row["pixel_counts"]
                    ):
                        raise ValueError("Mask contents differ from manifest")
    return result
