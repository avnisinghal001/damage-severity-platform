"""Dependency-free validation of the versioned feature JSON and conservation invariants."""

import json
import math
import re

from damage_analytics.labels import CLASS_NAMES

SCHEMA_VERSION = "damage-features-v1"


def _scores(values: dict) -> None:
    if type(values["available"]) is not bool:
        raise ValueError("Score availability must be explicit boolean")
    for key in (
        "mean_maximum_softmax",
        "mean_normalized_entropy",
        "low_score_fraction",
        "high_entropy_fraction",
    ):
        value = values[key]
        if value is not None and (type(value) not in (int, float) or not 0 <= value <= 1):
            raise ValueError("Score summaries must be null or within [0,1]")
        if not values["available"] and value is not None:
            raise ValueError("Unavailable scores must stay unknown")


def validate_report(report: dict) -> None:
    """Convert malformed structure/type errors to a consistent public validation error."""
    try:
        _validate(report)
    except (KeyError, TypeError, IndexError, OverflowError) as exc:
        raise ValueError("Malformed damage-feature JSON structure") from exc


def _validate(report: dict) -> None:
    """Reject malformed labels/counts/coordinates; serialize only finite JSON primitives."""
    json.dumps(report, allow_nan=False)
    if report["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unknown damage feature schema")
    image, coverage = report["image"], report["coverage"]
    width, height = image["width"], image["height"]
    if any(type(v) is not int or v < 1 for v in (width, height)):
        raise ValueError("Image dimensions must be positive integers")
    total = width * height
    counts = coverage["raw_class_pixels"]
    if (
        len(counts) != 9
        or any(type(v) is not int or v < 0 for v in counts)
        or sum(counts) != total
    ):
        raise ValueError("Raw class pixels do not conserve image area")
    if report["class_names"] != list(CLASS_NAMES) or len(report["classes"]) != 8:
        raise ValueError("Feature class IDs/names differ from RC1841")
    foreground, retained = sum(counts[1:]), coverage["retained_foreground_pixels"]
    if any(
        type(coverage[key]) is not int or coverage[key] < 0
        for key in ("foreground_pixels", "retained_foreground_pixels", "filtered_pixels")
    ):
        raise ValueError("Coverage counts must be nonnegative integers")
    if (
        coverage["foreground_pixels"] != foreground
        or not 0 <= retained <= foreground
        or coverage["filtered_pixels"] != foreground - retained
        or not math.isclose(
            coverage["foreground_image_percent"], 100 * foreground / total, abs_tol=1e-9
        )
    ):
        raise ValueError("Foreground/filter coverage is inconsistent")
    indexed = {}
    for expected_id, row in enumerate(report["classes"], 1):
        if (
            type(row["class_id"]) is not int
            or row["class_id"] != expected_id
            or row["label"] != CLASS_NAMES[expected_id]
        ):
            raise ValueError("Class label/ID ordering differs")
        count_fields = (
            "raw_pixels",
            "retained_pixels",
            "filtered_pixels",
            "raw_region_count",
            "retained_region_count",
            "filtered_region_count",
            "reported_region_count",
            "unreported_retained_regions",
            "unreported_retained_pixels",
        )
        if any(type(row[key]) is not int or row[key] < 0 for key in count_fields):
            raise ValueError("Class counts must be nonnegative integers")
        if (
            row["raw_pixels"] != counts[expected_id]
            or row["retained_pixels"] + row["filtered_pixels"] != row["raw_pixels"]
            or min(row["retained_pixels"], row["filtered_pixels"]) < 0
            or not math.isclose(
                row["image_percent"], 100 * row["raw_pixels"] / total, abs_tol=1e-9
            )
            or row["retained_region_count"] + row["filtered_region_count"]
            != row["raw_region_count"]
        ):
            raise ValueError("Class area/region conservation failed")
        indexed[expected_id] = row
        _scores(row["score_statistics"])
    if sum(r["retained_pixels"] for r in report["classes"]) != retained:
        raise ValueError("Retained class sum differs")
    ids, reported = set(), {i: [0, 0] for i in range(1, 9)}
    for region in report["regions"]:
        class_id, identifier = region["class_id"], region["region_id"]
        if (
            type(class_id) is not int
            or class_id not in indexed
            or region["label"] != CLASS_NAMES[class_id]
            or identifier in ids
        ):
            raise ValueError("Invalid/duplicate region identity")
        ids.add(identifier)
        x1, y1, x2, y2 = region["bbox_xyxy"]
        if any(type(v) is not int for v in (x1, y1, x2, y2)) or not (
            0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height
        ):
            raise ValueError("Region bounding box escapes original image")
        area, (cx, cy) = region["area_pixels"], region["centroid_xy"]
        if (
            type(area) is not int
            or not 0 < area <= (x2 - x1) * (y2 - y1)
            or not x1 <= cx < x2
            or not y1 <= cy < y2
            or region["perimeter_pixel_edges"] <= 0
            or not math.isclose(region["image_percent"], 100 * area / total, abs_tol=1e-9)
        ):
            raise ValueError("Region geometry/coverage is inconsistent")
        reported[class_id][0] += 1
        reported[class_id][1] += area
        if (
            region["bbox_width_pixels"] != x2 - x1
            or region["bbox_height_pixels"] != y2 - y1
            or not math.isclose(
                region["bbox_fill_fraction"], area / ((x2 - x1) * (y2 - y1)), abs_tol=1e-12
            )
        ):
            raise ValueError("Bounding dimensions/fill disagree with region geometry")
        _scores(region["score_statistics"])
    for class_id, row in indexed.items():
        if (
            reported[class_id][0] != row["reported_region_count"]
            or reported[class_id][0] + row["unreported_retained_regions"]
            != row["retained_region_count"]
            or reported[class_id][1] + row["unreported_retained_pixels"] != row["retained_pixels"]
        ):
            raise ValueError("Reported/unreported region accounting differs")
    model = report["model"]
    if not isinstance(model["version"], str) or not model["version"]:
        raise ValueError("Model version is required, including explicit unknown status")
    if model["checkpoint_sha256"] is not None and not re.fullmatch(
        r"[0-9a-f]{64}", model["checkpoint_sha256"]
    ):
        raise ValueError("Invalid checkpoint provenance hash")
    if report["uncertainty"]["calibrated"] is not False:
        raise ValueError("This pipeline does not supply calibrated confidence")
    _scores(report["uncertainty"])
    if report["units"]["area"] != "image_pixels" or report["units"]["physical_scale"] is not None:
        raise ValueError(
            "This schema supplies image-pixel measurements without physical calibration"
        )
