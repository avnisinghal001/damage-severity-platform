"""Reopen every release artifact and verify hashes, pixels, labels, partitions, and sources."""

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from damage_analytics.annotations import Polygon, rasterize, read_annotation
from damage_analytics.curation.cleaning import clean_annotation
from damage_analytics.curation.config import CleaningPolicy, CurationConfig
from damage_analytics.curation.images import pixel_hash, prepare_image
from damage_analytics.curation.masks import checked_mask
from damage_analytics.curation.similarity import connected_groups
from damage_analytics.curation.splitting import leakage_checks
from damage_analytics.io import contained_file, sha256_file, write_json
from damage_analytics.labels import CLASS_NAMES


def verify_dataset(folder: Path, config: CurationConfig) -> dict:
    """Validate actual saved artifacts, not just the records that created them."""
    with (folder / "curated_manifest.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    snapshot = json.loads((folder / "release.json").read_text())
    policy = CleaningPolicy(**snapshot["settings"]["cleaning"])
    failures = []
    for row in rows:
        try:
            _verify_sample(folder, config.paths.extracted_dir, row, policy)
        except (ValueError, OSError, SyntaxError) as exc:
            failures.append({"sample_id": row["sample_id"], "detail": str(exc)})
    partition_ids = []
    for split in ("train", "validation", "test"):
        with (folder / "splits" / f"{split}.csv").open(newline="", encoding="utf-8") as stream:
            split_rows = list(csv.DictReader(stream))
        expected = [row for row in rows if row["split"] == split]
        if split_rows != expected:
            failures.append(
                {"sample_id": "", "detail": f"{split} CSV differs from manifest subset"}
            )
        partition_ids.extend(row["sample_id"] for row in split_rows)
    if (
        len(partition_ids) != len(rows)
        or len(set(partition_ids)) != len(rows)
        or set(partition_ids) != {row["sample_id"] for row in rows}
    ):
        failures.append(
            {"sample_id": "", "detail": "Splits are not an exhaustive disjoint partition"}
        )
    with (folder / "similarity_edges.csv").open(newline="", encoding="utf-8") as stream:
        edges = list(csv.DictReader(stream))
    with (folder / "groups.csv").open(newline="", encoding="utf-8") as stream:
        groups = {row["sample_id"]: row["group_id"] for row in csv.DictReader(stream)}
    recomputed_groups = connected_groups(sorted(groups), edges)
    if groups != recomputed_groups or any(
        row["group_id"] != groups[row["sample_id"]] for row in rows
    ):
        failures.append(
            {"sample_id": "", "detail": "Stored groups differ from transitive evidence"}
        )
    leakage = leakage_checks(rows, edges, groups)
    if not leakage["passed"]:
        failures.append({"sample_id": "", "detail": "Known evidence crosses split boundaries"})
    actual_ids = {row["sample_id"] for row in rows}
    with (folder / "exclusions.csv").open(newline="", encoding="utf-8") as stream:
        excluded_ids = {row["sample_id"] for row in csv.DictReader(stream)}
    if actual_ids & excluded_ids or actual_ids | excluded_ids != set(groups):
        failures.append(
            {"sample_id": "", "detail": "Included/excluded IDs do not partition all IDs"}
        )
    expected_files = {row[field] for row in rows for field in ("image", "annotation", "mask")}
    actual_files = {
        path.relative_to(folder).as_posix()
        for name in ("images", "annotations", "masks")
        for path in (folder / name).rglob("*")
        if path.is_file()
    }
    if actual_files != expected_files:
        failures.append({"sample_id": "", "detail": "Missing or unreferenced sample artifacts"})
    return {
        "passed": not failures,
        "samples_verified": len(rows),
        "failures": failures,
        "leakage": leakage,
        "checks": [
            "source and artifact hashes",
            "RGB image pixel equality",
            "cleaned labels/vertices equal explicit policy output",
            "mask PNG roundtrip",
            "all mask pixels equal rerasterized JSON",
            "mask dimensions, dtype, IDs, counts",
            "background coverage complement",
            "disjoint exhaustive splits",
            "known grouping edges",
            "included/excluded partition",
            "no unreferenced sample artifacts",
        ],
    }


def _verify_sample(folder: Path, source_root: Path, row: dict, policy: CleaningPolicy) -> None:
    """Check one source-to-derivative transformation and its complete mask grid."""
    paths = {
        field: contained_file(folder, row[field]) for field in ("image", "annotation", "mask")
    }
    source_image = contained_file(source_root, row["source_image"])
    source_annotation = contained_file(source_root, row["source_annotation"])
    for path, expected in (
        (source_image, row["source_image_sha256"]),
        (source_annotation, row["source_annotation_sha256"]),
        *[(paths[field], row[f"{field}_sha256"]) for field in paths],
    ):
        if sha256_file(path) != expected:
            raise ValueError("Source/artifact hash mismatch")
    size = (int(row["width"]), int(row["height"]))
    with Image.open(paths["image"]) as image:
        image.load()
        if image.mode != "RGB" or image.size != size or image.getexif().get(274, 1) != 1:
            raise ValueError("Image mode/size/orientation mismatch")
        saved_image_hash = pixel_hash(image)
    document = json.loads(source_annotation.read_text())
    prepared, _, image_decisions = prepare_image(source_image, document, row["sample_id"], policy)
    if any(item["action"] == "exclude" for item in image_decisions):
        raise ValueError("Source image is not acceptable under release policy")
    if saved_image_hash != pixel_hash(prepared):
        raise ValueError("Exported RGB pixels differ from explicit image policy")
    polygons = read_annotation(paths["annotation"], paths["image"].name, size)
    cleaned, findings = clean_annotation(
        document, row["sample_id"], source_image.name, size, policy
    )
    if any(item["action"] == "exclude" for item in findings) or polygons != cleaned:
        raise ValueError("Saved annotation differs from explicit source-cleaning policy")
    expected_mask, info = checked_mask(polygons, size)
    if not any(item["code"] == "tolerance_clamp" for item in findings):
        original_polygons = [
            Polygon(shape["label"], tuple(map(tuple, shape["points"])))
            for shape in document["shapes"]
        ]
        original_mask, _ = rasterize(original_polygons, size)
        if not np.array_equal(original_mask, expected_mask):
            raise ValueError("Boundary-preserving cleaning changed source mask pixels")
    if policy.cross_class_overlap == "exclude" and info["conflict_pixels"]:
        raise ValueError("Disallowed cross-class mask conflict")
    with Image.open(paths["mask"]) as mask:
        mask.load()
        if mask.mode != "L" or mask.size != size or not np.array_equal(mask, expected_mask):
            raise ValueError("Mask is not the exact class-ID rasterization of cleaned JSON")
    counts = [int(row[f"pixels_{index}"]) for index in range(len(CLASS_NAMES))]
    if counts != info["pixel_counts"] or int(row["polygon_count"]) != len(polygons):
        raise ValueError("Manifest class counts/polygon counts differ from actual artifacts")
    if (
        int(row["overlap_pixels"]) != info["overlap_pixels"]
        or int(row["conflict_pixels"]) != info["conflict_pixels"]
    ):
        raise ValueError("Manifest overlap counts differ from actual polygons")
    if not info["foreground_pixels"] and not policy.allow_empty_annotations:
        raise ValueError("Disallowed empty mask")


def write_checksums(folder: Path) -> None:
    """Seal every release artifact except the checksum manifest itself."""
    files = {
        path.relative_to(folder).as_posix(): sha256_file(path)
        for path in sorted(folder.rglob("*"))
        if path.is_file() and path.name != "checksums.json"
    }
    write_json(folder / "checksums.json", files)


def verify_checksums(folder: Path) -> None:
    """Refuse tampering, missing/extra artifacts, and links in an immutable release."""
    expected = json.loads((folder / "checksums.json").read_text())
    paths = list(folder.rglob("*"))
    if any(path.is_symlink() for path in paths):
        raise ValueError("Symlinks are not allowed inside a release")
    actual = {
        path.relative_to(folder).as_posix()
        for path in paths
        if path.is_file() and path.name != "checksums.json"
    }
    if actual != set(expected):
        raise ValueError("Release artifact inventory changed")
    for relative, digest in expected.items():
        if sha256_file(contained_file(folder, relative)) != digest:
            raise ValueError(f"Release checksum mismatch: {relative}")
