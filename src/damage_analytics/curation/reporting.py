"""Human-readable audit explanations and machine-readable class/quality statistics."""

import csv
import json
from collections import Counter
from itertools import combinations
from pathlib import Path

from damage_analytics.io import write_csv, write_json
from damage_analytics.labels import CLASS_NAMES

DECISION_FIELDS = ["sample_id", "shape_index", "code", "action", "detail"]
EDGE_FIELDS = ["left", "right", "kind", "evidence"]
MANIFEST_FIELDS = [
    "sample_id",
    "source_image",
    "source_annotation",
    "image",
    "annotation",
    "mask",
    "source_image_sha256",
    "source_annotation_sha256",
    "image_sha256",
    "mask_sha256",
    "annotation_sha256",
    "width",
    "height",
    "source_mode",
    "source_format",
    "exif_orientation",
    "group_id",
    "split",
    "polygon_count",
    "overlap_pixels",
    "conflict_pixels",
    *[f"pixels_{index}" for index in range(len(CLASS_NAMES))],
]


def manifest_rows(records: list[dict]) -> list[dict]:
    """Flatten per-class counts into portable numeric CSV columns."""
    return [
        {
            **{key: row[key] for key in MANIFEST_FIELDS if not key.startswith("pixels_")},
            **{f"pixels_{index}": count for index, count in enumerate(row["pixel_counts"])},
        }
        for row in sorted(records, key=lambda item: item["sample_id"])
    ]


def statistics(records: list[dict], rare_threshold: int) -> dict:
    """Report foreground imbalance, multi-label occurrences, dimensions and modes."""
    population = {
        "all": records,
        **{
            split: [row for row in records if row["split"] == split]
            for split in ("train", "validation", "test")
        },
    }
    classes = []
    for split, subset in population.items():
        total = sum(row["width"] * row["height"] for row in subset)
        for index, name in enumerate(CLASS_NAMES):
            pixels = sum(row["pixel_counts"][index] for row in subset)
            images = sum(row["pixel_counts"][index] > 0 for row in subset)
            classes.append(
                {
                    "split": split,
                    "class_id": index,
                    "source_label": name,
                    "images": images,
                    "pixels": pixels,
                    "pixel_fraction": pixels / total if total else 0,
                    "annotation_images": sum(name in row["polygon_counts"] for row in subset),
                    "polygons": sum(row["polygon_counts"].get(name, 0) for row in subset),
                }
            )
    foreground = [row for row in classes if row["split"] == "all" and row["class_id"]]
    positive = [row["pixels"] for row in foreground if row["pixels"]]
    cooccurrences = Counter()
    combinations_count = Counter()
    for row in records:
        labels = tuple(
            index for index, pixels in enumerate(row["pixel_counts"]) if index and pixels
        )
        combinations_count[labels] += 1
        cooccurrences.update(combinations(labels, 2))
    dimensions = Counter((row["width"], row["height"]) for row in records)
    return {
        "population": "Curated samples after all exclusions and exact-pair deduplication",
        "samples": len(records),
        "total_pixels": sum(sum(row["pixel_counts"]) for row in records),
        "classes": classes,
        "rare_classes": [
            row["source_label"] for row in foreground if row["images"] < rare_threshold
        ],
        "rarity_threshold_images": rare_threshold,
        "foreground_max_min_pixel_ratio": max(positive) / min(positive) if positive else None,
        "multi_label_images": sum(
            count for labels, count in combinations_count.items() if len(labels) > 1
        ),
        "label_combinations": [
            {"class_ids": list(labels), "images": count}
            for labels, count in sorted(combinations_count.items())
        ],
        "cooccurrences": [
            {"left": left, "right": right, "images": count}
            for (left, right), count in sorted(cooccurrences.items())
        ],
        "dimensions": [
            {"width": width, "height": height, "images": count}
            for (width, height), count in sorted(dimensions.items())
        ],
        "source_modes": dict(Counter(row["source_mode"] for row in records)),
        "model_input_mode": "RGB",
        "severity_targets": "None; source labels retained verbatim",
    }


def audit_task01(folder: Path, decisions: list[dict], kept: set[str]) -> dict:
    """Explain every historical issue and link it to current sample decisions."""
    with (folder / "issues.csv").open(newline="", encoding="utf-8") as stream:
        previous = list(csv.DictReader(stream))
    explanations = {
        "polygon_overlap": "Task 01 used last-shape precedence; "
        "Task 02 separates same/cross-label overlap",
        "exif_orientation": "Nonidentity EXIF tag; compare oriented dimensions with JSON frame",
        "duplicate": "Same bytes/decoded pixels; linked transitively before any split assignment",
    }
    audit = []
    for row in previous:
        sample_ids = [row["sample_id"]] if row["sample_id"] else []
        if row["code"] == "duplicate":
            sample_ids = [
                Path(value.strip()).stem for value in row["detail"].split(": ", 1)[1].split(",")
            ]
        cause = explanations.get(
            row["code"], "Prior validation error; current audit records disposition"
        )
        if row["code"] == "invalid_pair":
            cause = (
                "JSON uses EXIF-oriented frame while Task 01 uses stored pixels"
                if "does not match decoded image" in row["detail"]
                else "Source polygon coordinate outside declared image bounds"
            )
        affected = [
            {
                "sample_id": sample_id,
                "disposition": "retained" if sample_id in kept else "excluded",
                "decisions": sorted(
                    {item["code"] for item in decisions if item["sample_id"] == sample_id}
                ),
            }
            for sample_id in sample_ids
        ]
        audit.append({**row, "root_cause": cause, "affected": affected})
    evidence = {}
    for name in ("validation.json", "summary.json", "duplicates.json"):
        path = folder / name
        if path.exists():
            evidence[name] = json.loads(path.read_text())
    for name in ("class_distribution.csv", "dimensions.csv"):
        path = folder / name
        if path.exists():
            with path.open(newline="", encoding="utf-8") as stream:
                evidence[name] = list(csv.DictReader(stream))
    return {
        "previous_issue_rows": len(previous),
        "issue_counts": dict(Counter(row["code"] for row in previous)),
        "previous_report_contents": evidence,
        "issues": audit,
    }


def write_reports(
    folder: Path,
    records: list[dict],
    all_records: list[dict],
    decisions: list[dict],
    edges: list[dict],
    stats: dict,
    split_report: dict,
    task01: dict,
) -> None:
    """Publish tables with stable schemas, including zero-row error/exclusion logs."""
    flattened = manifest_rows(records)
    audit_fields = [
        "sample_id",
        "included",
        "source_image",
        "source_annotation",
        "image_readable",
        "annotation_readable",
        "source_format",
        "source_mode",
        "exif_orientation",
        "camera_make",
        "camera_model",
        "camera_serial",
        "capture_time",
        "capture_time_basis",
        "stored_width",
        "stored_height",
        "declared_width",
        "declared_height",
        "width",
        "height",
        "mask_checks_performed",
        "source_polygon_counts",
    ]
    write_csv(
        folder / "sample_audit.csv",
        [
            {
                **{key: row.get(key, "") for key in audit_fields},
                "mask_checks_performed": "pixel_counts" in row,
                "source_polygon_counts": json.dumps(
                    row.get("source_polygon_counts", {}), sort_keys=True
                ),
            }
            for row in all_records
        ],
        audit_fields,
    )
    write_csv(folder / "curated_manifest.csv", flattened, MANIFEST_FIELDS)
    for split in ("train", "validation", "test"):
        write_csv(
            folder / "splits" / f"{split}.csv",
            [row for row in flattened if row["split"] == split],
            MANIFEST_FIELDS,
        )
    write_csv(folder / "decisions.csv", decisions, DECISION_FIELDS)
    write_csv(
        folder / "errors.csv",
        [row for row in decisions if row["action"] == "exclude"],
        DECISION_FIELDS,
    )
    excluded = [
        {
            "sample_id": row["sample_id"],
            "reasons": json.dumps(
                sorted(
                    {
                        item["code"]
                        for item in decisions
                        if item["sample_id"] == row["sample_id"] and item["action"] == "exclude"
                    }
                )
            ),
        }
        for row in all_records
        if not row["included"]
    ]
    write_csv(folder / "exclusions.csv", excluded, ["sample_id", "reasons"])
    write_csv(folder / "similarity_edges.csv", edges, EDGE_FIELDS)
    write_csv(
        folder / "groups.csv",
        [
            {
                "sample_id": row["sample_id"],
                "group_id": row["group_id"],
                "included": row["included"],
            }
            for row in all_records
        ],
        ["sample_id", "group_id", "included"],
    )
    write_csv(
        folder / "class_statistics.csv",
        stats["classes"],
        [
            "split",
            "class_id",
            "source_label",
            "images",
            "pixels",
            "pixel_fraction",
            "annotation_images",
            "polygons",
        ],
    )
    write_csv(folder / "dimensions.csv", stats["dimensions"], ["width", "height", "images"])
    write_json(folder / "statistics.json", stats)
    write_json(folder / "split_statistics.json", split_report)
    write_json(folder / "task01_audit.json", task01)
