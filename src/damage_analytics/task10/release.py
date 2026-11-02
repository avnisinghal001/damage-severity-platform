"""Authenticate immutable Phi-Net tensor releases; official test is not a development loader."""

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from damage_analytics.curation.verification import verify_checksums
from damage_analytics.io import contained_file, sha256_file
from damage_analytics.task10.archives import checked_header
from damage_analytics.task10.config import CLASS_NAMES


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
            "seal_sha256": sha256_file(self.root / "checksums.json"),
            "manifest_hashes": {
                name: self.seal[f"manifests/{name}.csv"]
                for name in ("train", "validation", "official_test", "quarantine")
            },
        }

    def samples(self, split: str) -> tuple[dict, ...]:
        if split not in ("train", "validation"):
            raise ValueError(
                "Task 10 development cannot load the preserved official test partition"
            )
        return tuple(r for r in self.rows if r["split"] == split)

    def file(self, relative: str, *, verify: bool = True) -> Path:
        path = contained_file(self.root, relative)
        if verify and sha256_file(path) != self.seal.get(relative):
            raise ValueError(f"Release file changed: {relative}")
        return path


def open_release(root: Path, *, full: bool = False) -> Release:
    root = root.resolve()
    seal = json.loads((root / "checksums.json").read_text())

    def checked(relative):
        path = contained_file(root, relative)
        if sha256_file(path) != seal.get(relative):
            raise ValueError(f"Sealed metadata changed: {relative}")
        return path

    metadata, quality = (
        json.loads(checked("release.json").read_text()),
        json.loads(checked("quality.json").read_text()),
    )
    rows = tuple(json.loads(line) for line in checked("manifest.jsonl").read_text().splitlines())
    if (
        metadata["class_names"] != list(CLASS_NAMES)
        or not rows
        or len({r["sample_id"] for r in rows}) != len(rows)
    ):
        raise ValueError("Invalid class registration or manifest identities")
    components, hashes, seen = {}, {}, set()
    for split in ("train", "validation", "official_test", "quarantine"):
        with checked(f"manifests/{split}.csv").open() as stream:
            table = list(csv.DictReader(stream))
        expected = {r["sample_id"]: r for r in rows if r["split"] == split}
        if len(table) != len(expected) or {r["sample_id"] for r in table} != set(expected):
            raise ValueError("Saved split population differs")
        for item in table:
            row = expected[item["sample_id"]]
            if item["sample_id"] in seen or any(
                item[key] != ("" if row[key] is None else str(row[key])) for key in item
            ):
                raise ValueError("Saved split row differs")
            seen.add(item["sample_id"])
            if (row["official_partition"] == "test") != (split == "official_test"):
                raise ValueError("Official test membership moved")
            if split in ("train", "validation"):
                if (
                    not row["valid"]
                    or row["class_id"] not in range(4)
                    or row["disposition"] != "development_eligible"
                ):
                    raise ValueError("Invalid/quarantined row admitted to development")
                for mapping, key in (
                    (components, row["leakage_component"]),
                    (hashes, row["raw_sha256"]),
                ):
                    if key in mapping and mapping[key] != split:
                        raise ValueError(
                            "Known component or exact image crosses training/validation"
                        )
                    mapping[key] = split
    test_components = {r["leakage_component"] for r in rows if r["split"] == "official_test"}
    if test_components & set(components):
        raise ValueError("Official-test-related component admitted to development")
    for name, expected in metadata["official_array_headers"].items():
        path = contained_file(root, f"payload/task7/{name}")
        with path.open("rb") as stream:
            if checked_header(stream, path.stat().st_size) != expected:
                raise ValueError("Tensor file header/length changed")
    result = Release(root, metadata, quality, rows, seal)
    if full:
        verify_checksums(root)  # QA hashes official test bytes, never evaluates a model.
        for row in rows:
            x = np.load(
                result.file(row["image_array"], verify=False), mmap_mode="r", allow_pickle=False
            )
            if hashlib.sha256(x[row["array_index"]].tobytes()).hexdigest() != row["raw_sha256"]:
                raise ValueError("Tensor row hash differs")
    return result
