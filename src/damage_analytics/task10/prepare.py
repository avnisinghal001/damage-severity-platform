"""Immutable tensor preparation, provenance, conservative test quarantine and saved manifests."""

import csv
import hashlib
import json
import os
import platform
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import PIL

from damage_analytics.curation.verification import write_checksums
from damage_analytics.io import sha256_file, write_csv, write_json
from damage_analytics.task10.archives import extract, inspect
from damage_analytics.task10.config import CLASS_NAMES, Config
from damage_analytics.task10.encoding import recovered_channels
from damage_analytics.task10.leakage import investigate, visual_views
from damage_analytics.task10.splits import development_split


def scan_arrays(root: Path, config: Config) -> tuple[list[dict], dict, dict]:
    rows, views, stats = [], {}, {}
    for partition in ("train", "test"):
        x_path, y_path = (
            f"payload/task7/task7_X_{partition}.npy",
            f"payload/task7/task7_y_{partition}.npy",
        )
        x, y = (
            np.load(root / x_path, mmap_mode="r", allow_pickle=False),
            np.load(root / y_path, mmap_mode="r", allow_pickle=False),
        )
        if x.shape[1:] != config.expected_image_shape or y.shape != (len(x), 4):
            raise ValueError("Extracted image/label tensor shapes differ")
        for index in range(len(x)):
            values, label = x[index], y[index]
            label_ok = bool(
                np.isfinite(label).all()
                and np.all((label == 0) | (label == 1))
                and label.sum() == 1
            )
            class_id = int(label.argmax()) if label_ok else None
            identifier = f"phi7-{partition}-{index:06d}"
            row = {
                "sample_id": identifier,
                "official_partition": partition,
                "array_index": index,
                "image_array": x_path,
                "label_array": y_path,
                "class_id": class_id,
                "label": CLASS_NAMES[class_id] if label_ok else None,
                "raw_sha256": hashlib.sha256(values.tobytes(order="C")).hexdigest(),
                "label_sha256": hashlib.sha256(label.tobytes(order="C")).hexdigest(),
                "source_group": "",
                "source_evidence": "",
                "recovered_sha256": "",
                "valid": False,
                "label_valid": label_ok,
                "image_valid": False,
                "issues": [] if label_ok else ["invalid_nonfinite_or_non_one_hot_label"],
            }
            try:
                channels, error = recovered_channels(values, config.roundtrip_tolerance)
                row.update(
                    image_valid=True,
                    maximum_inverse_integer_error=error,
                    recovered_sha256=hashlib.sha256(channels.tobytes()).hexdigest(),
                    stored_minimum=float(values.min()),
                    stored_maximum=float(values.max()),
                )
                views[identifier] = visual_views(channels, config.crop_fraction)
            except ValueError as exc:
                row["issues"].append(str(exc))
            row["valid"] = row["image_valid"] and label_ok
            row["disposition"] = (
                "official_test_preserved" if partition == "test" else "development_eligible"
            )
            if not row["valid"]:
                row["disposition"] = "invalid_official_row"
            rows.append(row)
            if (index + 1) % 1000 == 0:
                print(f"Validated {partition} tensor rows: {index + 1}/{len(x)}", flush=True)
        stats[partition] = {
            "rows": len(x),
            "class_counts": dict(
                Counter(
                    str(r["class_id"])
                    for r in rows
                    if r["official_partition"] == partition and r["label_valid"]
                )
            ),
            "invalid_rows": sum(
                not r["valid"] for r in rows if r["official_partition"] == partition
            ),
        }
        del x, y
    return rows, views, stats


def attach_source_evidence(rows: list[dict], path: Path | None) -> None:
    if path is None:
        return
    indexed, seen = {r["sample_id"]: r for r in rows}, set()
    with path.open() as stream:
        for item in csv.DictReader(stream):
            identifier = item["sample_id"]
            if (
                identifier not in indexed
                or identifier in seen
                or not item["source_group"].strip()
                or not item["evidence"].strip()
            ):
                raise ValueError(
                    "Source evidence needs unique known array IDs, supplied groups and evidence"
                )
            indexed[identifier].update(
                source_group=item["source_group"], source_evidence=item["evidence"]
            )
            seen.add(identifier)


def prepare(config: Config) -> Path:
    inspection = inspect(config)
    code = {
        f"task10/{p.name}": sha256_file(p)
        for p in Path(__file__).parent.glob("*.py")
        if p.stem
        in ("prepare", "archives", "config", "encoding", "leakage", "splits", "release", "reports")
    }
    package = Path(__file__).parent.parent
    for relative in (
        "archives.py",
        "io.py",
        "curation/similarity.py",
        "curation/verification.py",
        "task07/duplicates.py",
    ):
        code[relative] = sha256_file(package / relative)
    inputs = {
        "archive_sha256": inspection["archive_sha256"],
        "settings": config.preparation_settings(),
        "source_evidence_sha256": sha256_file(config.source_groups_csv)
        if config.source_groups_csv
        else None,
        "code_sha256": code,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "Pillow": PIL.__version__,
        },
    }
    version = (
        "phi7-" + hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()[:16]
    )
    destination = config.output_dir / "releases" / version
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        from damage_analytics.task10.release import open_release

        open_release(destination)
    else:
        with tempfile.TemporaryDirectory(prefix=".phi7-", dir=destination.parent) as temporary:
            stage = Path(temporary) / "release"
            stage.mkdir()
            extract(config, stage / "payload", inspection)
            rows, views, statistics = scan_arrays(stage, config)
            attach_source_evidence(rows, config.source_groups_csv)
            _, edges, near, screening = investigate(rows, views, config)
            blockers = []
            try:
                assignment = development_split(rows, config.validation_fraction, config.seed)
            except ValueError as exc:
                assignment = {}
                blockers.append(str(exc))
            for row in rows:
                row["split"] = (
                    "official_test"
                    if row["official_partition"] == "test"
                    else assignment.get(row["sample_id"], "quarantine")
                )
                if (
                    row["disposition"] == "development_eligible"
                    and row["sample_id"] not in assignment
                ):
                    row["disposition"] = "unassigned_infeasible_grouped_split"
            if screening["screening_truncated"]:
                blockers.append(
                    "Perceptual screening budget exhausted; no complete known-risk screen"
                )
            counts = {
                split: dict(
                    Counter(
                        str(r["class_id"])
                        for r in rows
                        if r["split"] == split and r["label_valid"]
                    )
                )
                for split in ("train", "validation", "official_test", "quarantine")
            }
            if any(
                set(counts[split]) != {"0", "1", "2", "3"} for split in ("train", "validation")
            ):
                blockers.append("Train/validation lack complete four-class coverage")
            quality = {
                "official_partitions": statistics,
                "prepared_class_counts": counts,
                "prepared_split_counts": dict(Counter(r["split"] for r in rows)),
                "dispositions": dict(Counter(r["disposition"] for r in rows)),
                "screening": screening,
                "blockers": blockers,
                "technical_training_ready": not blockers,
                "provided_source_evidence_complete": all(
                    r["source_group"] and r["source_evidence"] for r in rows
                ),
                "encoding_verified": config.encoding_verified,
                "encoding_evidence": config.encoding_evidence,
                "maximum_inverse_integer_error": max(
                    (r.get("maximum_inverse_integer_error", 0) for r in rows), default=0
                ),
                "official_test_preserved": len([r for r in rows if r["split"] == "official_test"])
                == statistics["test"]["rows"],
                "test_performance_inspected": False,
                "scientific_status": "Provisional until source identity and publisher color "
                "preprocessing are authenticated. "
                "Four class IDs are nominal, not numeric severity ranks. "
                "Visible pixel/object/component damage level only; no whole-building safety "
                "target.",
            }
            (stage / "manifest.jsonl").write_text(
                "".join(json.dumps(r, allow_nan=False) + "\n" for r in rows)
            )
            write_json(
                stage / "release.json",
                {
                    "release_id": version,
                    "inputs": inputs,
                    "class_names": list(CLASS_NAMES),
                    "official_array_headers": inspection["headers"],
                },
            )
            write_json(stage / "quality.json", quality)
            write_json(stage / "leakage_edges.json", edges)
            write_json(stage / "near_candidates.json", near)
            fields = [
                "sample_id",
                "official_partition",
                "array_index",
                "class_id",
                "label",
                "split",
                "leakage_component",
                "source_group",
                "image_array",
                "label_array",
                "valid",
                "disposition",
            ]
            for split in ("train", "validation", "official_test", "quarantine"):
                write_csv(
                    stage / f"manifests/{split}.csv",
                    [{k: r[k] for k in fields} for r in rows if r["split"] == split],
                    fields,
                )
            write_csv(
                stage / "issues.csv",
                [
                    {
                        "sample_id": r["sample_id"],
                        "official_partition": r["official_partition"],
                        "reason": reason,
                    }
                    for r in rows
                    for reason in r["issues"]
                ],
                ["sample_id", "official_partition", "reason"],
            )
            write_csv(
                stage / "quarantine.csv",
                [
                    {"sample_id": r["sample_id"], "reason": r["disposition"]}
                    for r in rows
                    if r["split"] == "quarantine"
                ],
                ["sample_id", "reason"],
            )
            write_csv(
                stage / "source_groups_template.csv",
                [{"sample_id": r["sample_id"], "source_group": "", "evidence": ""} for r in rows],
                ["sample_id", "source_group", "evidence"],
            )
            write_csv(
                stage / "near_review.csv",
                near,
                list(near[0]) if near else ["left", "right", "decision", "human_evidence"],
            )
            from damage_analytics.task10.reports import integrity_report

            integrity_report(stage / "INTEGRITY.md", inspection, quality)
            if sha256_file(config.archive) != inputs["archive_sha256"]:
                raise ValueError("Original archive changed during preparation")
            if (
                config.source_groups_csv
                and sha256_file(config.source_groups_csv) != inputs["source_evidence_sha256"]
            ):
                raise ValueError("Source evidence changed during preparation")
            write_checksums(stage)
            stage.rename(destination)
    write_json(
        config.output_dir / "latest.json",
        {
            "release_id": version,
            "release_dir": os.path.relpath(destination),
        },
    )
    return destination
