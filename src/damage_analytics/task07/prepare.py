"""Staged extraction, per-file QA, immutable manifests and split seals."""

import csv
import hashlib
import json
import os
import platform
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import PIL

from damage_analytics.curation.verification import write_checksums
from damage_analytics.io import sha256_file, write_csv, write_json
from damage_analytics.task07.adapters import discover
from damage_analytics.task07.archives import extract_payload
from damage_analytics.task07.config import Config
from damage_analytics.task07.duplicates import evidence_groups
from damage_analytics.task07.splits import assign_splits, split_report
from damage_analytics.task07.validation import validate_sample


def taxonomy(dataset: str, config: Config) -> dict:
    return {
        "classification": {
            "task": "binary_crack_presence",
            "labels": ["no_crack", "crack"],
            "meaning": "Source classification labels only; no-crack is NOT no structural damage",
        },
        "segmentation": None
        if dataset == "sdnet2018"
        else {
            "labels": ["background", *config.damsegment["category_names"]],
            "source_id_to_train_id": {"0": 1, "1": 2},
            "rgb_to_train_id": {"0,0,0": 0, "255,0,0": 1, "0,0,255": 2},
            **config.damsegment,
        },
        "detection": None
        if dataset == "sdnet2018"
        else {
            "source_ids": [0, 1],
            "category_names": config.damsegment["category_names"],
            "box_format": "native_xywh",
        },
    }


def prepare(config: Config, dataset: str) -> Path:
    """Heavy work occurs only on this explicit command. Same input/settings/code → same release."""
    archive = config.archive(dataset)
    code = {
        p.name: sha256_file(p)
        for p in Path(__file__).parent.glob("*.py")
        if p.stem
        in ("prepare", "archives", "adapters", "annotations", "splits", "duplicates", "validation")
    }
    package = Path(__file__).parent.parent
    for relative in (
        "archives.py",
        "config.py",
        "io.py",
        "curation/geometry.py",
        "curation/similarity.py",
        "curation/verification.py",
        "task07/config.py",
    ):
        code[relative] = sha256_file(package / relative)
    inputs = {
        "archive_sha256": sha256_file(archive),
        "dataset": dataset,
        "seed": config.seed,
        "fractions": config.fractions,
        "taxonomy": taxonomy(dataset, config),
        "code": code,
        "source_groups": sha256_file(config.source_groups_csv)
        if config.source_groups_csv
        else None,
        "near_reviews": sha256_file(config.near_reviews_csv) if config.near_reviews_csv else None,
        "phash_distance": config.phash_distance,
        "max_near_candidates": config.max_near_candidates,
        "max_image_pixels": config.max_image_pixels,
        "mask_min_polygon_iou": config.mask_min_polygon_iou,
        "environment": {
            "python": platform.python_version(),
            "Pillow": PIL.__version__,
            "numpy": np.__version__,
        },
    }
    digest = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()[:16]
    identifier = f"{dataset}-{digest}"
    destination = config.output_dir / "releases" / identifier
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        from damage_analytics.task07.release import open_release

        open_release(destination)
        write_json(
            config.output_dir / f"datasets/{dataset}/latest.json",
            {"release_dir": os.path.relpath(destination), "release_id": identifier},
        )
        return destination
    with tempfile.TemporaryDirectory(prefix=".prepare-", dir=destination.parent) as temporary:
        stage = Path(temporary) / "release"
        stage.mkdir()
        extract_payload(archive, dataset, stage / "payload")
        rows, exclusions = [], []
        discovered = discover(stage, dataset)
        for index, row in enumerate(discovered):
            try:
                rows.append(
                    validate_sample(
                        stage,
                        row,
                        max_pixels=config.max_image_pixels,
                        mask_min_iou=config.mask_min_polygon_iou,
                    )
                )
            except (ValueError, OSError, KeyError, TypeError, IndexError) as exc:
                exclusions.append(
                    {"sample_id": row["sample_id"], "image": row["image"], "reason": str(exc)}
                )
            if (index + 1) % 2000 == 0:
                print(f"Validated {index + 1}/{len(discovered)} records", flush=True)
        buckets = defaultdict(list)
        for row in rows:
            buckets[(row["task"], row["decoded_sha256"])].append(row)
        conflict = set()
        for (task, _), members in buckets.items():
            targets = {
                r["label"]
                if task == "classification"
                else r.get("mask_sha256", r.get("annotation_geometry_sha256"))
                for r in members
            }
            if len(targets) > 1:
                conflict.update(r["sample_id"] for r in members)
        for row in rows:
            if row["sample_id"] in conflict:
                exclusions.append(
                    {
                        "sample_id": row["sample_id"],
                        "image": row["image"],
                        "reason": "exact_image_conflicting_targets",
                    }
                )
        rows = [r for r in rows if r["sample_id"] not in conflict]
        if config.source_groups_csv:
            indexed = {r["sample_id"]: r for r in rows}
            seen = set()
            with config.source_groups_csv.open() as stream:
                for item in csv.DictReader(stream):
                    if (
                        item["sample_id"] not in indexed
                        or item["sample_id"] in seen
                        or not item["source_group"]
                        or not item["evidence"]
                    ):
                        raise ValueError(
                            "Source evidence requires unique included IDs, groups and evidence"
                        )
                    indexed[item["sample_id"]].update(
                        source_group=item["source_group"], source_evidence=item["evidence"]
                    )
                    seen.add(item["sample_id"])
        groups, near, grouping = evidence_groups(
            rows, config.phash_distance, config.max_near_candidates, config.near_reviews_csv
        )
        assignment = assign_splits(rows, groups, config.fractions, config.seed)
        for row in rows:
            row.update(group_id=groups[row["sample_id"]], split=assignment[row["sample_id"]])
        quality = split_report(rows, near, grouping["screening_truncated"])
        quality.update(
            discovered=len(discovered),
            included=len(rows),
            exclusions=len(exclusions),
            seed=config.seed,
            taxonomy=inputs["taxonomy"],
            class_counts={
                task: dict(Counter(str(r["label"]) for r in rows if r["task"] == task))
                for task in ("classification",)
            },
            mask_pixels=[sum(r.get("pixel_counts", [0, 0, 0])[i] for r in rows) for i in range(3)]
            if any(r["mask"] for r in rows)
            else None,
            dimensions=dict(Counter(f"{r['width']}x{r['height']}" for r in rows)),
            source_modes=dict(Counter(r["source_mode"] for r in rows)),
            source_labels=dict(Counter(r["source_label"] for r in rows)),
        )
        referenced = {
            r[key]
            for r in discovered
            for key in ("image", "annotation", "yolo", "source_mask")
            if r.get(key)
        }
        quality["orphan_label_files"] = [
            p.relative_to(stage).as_posix()
            for p in (stage / "payload").rglob("*")
            if p.is_file()
            and (p.suffix.lower() in (".json", ".txt") or "/Labels/Mask/" in p.as_posix())
            and p.relative_to(stage).as_posix() not in referenced
        ]
        if quality["orphan_label_files"]:
            quality["final_evaluation_eligible"] = False
        write_json(
            stage / "release.json",
            {"release_id": identifier, "dataset": dataset, "inputs": inputs},
        )
        (stage / "manifest.jsonl").write_text(
            "".join(json.dumps(r, allow_nan=False) + "\n" for r in rows)
        )
        write_json(stage / "quality.json", quality)
        (stage / "audit.md").write_text(
            f"# {dataset} prepared-dataset audit\n\n"
            f"Discovered {len(discovered)} records; included {len(rows)}; "
            f"excluded {len(exclusions)}.\n\n"
            f"Class counts: {quality['class_counts']}. "
            f"Native mask pixels: {quality['mask_pixels']}.\n\n"
            f"Verified source identity: {quality['source_identity_verified']}. "
            f"Cross-split unresolved pairs: {quality['cross_split_unresolved_near_pairs']}. "
            f"Screen truncated: {quality['screening_truncated']}.\n\n"
            f"Final-evaluation eligible: {quality['final_evaluation_eligible']}. "
            "This is not a guarantee of unknown-source independence.\n\n"
            "See exclusions.csv, quality.json, grouping.json and near_candidates.json. "
            "Raw files remain unchanged. Source PNGs are authoritative; polygon raster "
            "disagreement is recorded per sample, never silently repainted.\n"
        )
        write_json(stage / "grouping.json", grouping)
        write_json(stage / "near_candidates.json", near)
        write_csv(stage / "exclusions.csv", exclusions, ["sample_id", "image", "reason"])
        write_csv(
            stage / "source_groups_template.csv",
            [
                {
                    "sample_id": r["sample_id"],
                    "source_group": "",
                    "evidence": "",
                    "source_member": r["source_member"],
                    "candidate_source": r["candidate_source"],
                }
                for r in rows
            ],
            ["sample_id", "source_group", "evidence", "source_member", "candidate_source"],
        )
        for split in ("train", "validation", "test"):
            write_csv(
                stage / f"splits/{split}.csv",
                [
                    {
                        "sample_id": r["sample_id"],
                        "group_id": r["group_id"],
                        "task": r["task"],
                        "label": r["label"],
                        "image": r["image"],
                    }
                    for r in rows
                    if r["split"] == split
                ],
                ["sample_id", "group_id", "task", "label", "image"],
            )
        if sha256_file(archive) != inputs["archive_sha256"]:
            raise ValueError("Archive changed during preparation")
        for source, key in (
            (config.source_groups_csv, "source_groups"),
            (config.near_reviews_csv, "near_reviews"),
        ):
            if source and sha256_file(source) != inputs[key]:
                raise ValueError("Evidence CSV changed during preparation")
        write_checksums(stage)
        stage.rename(destination)
    write_json(
        config.output_dir / f"datasets/{dataset}/latest.json",
        {"release_dir": os.path.relpath(destination), "release_id": identifier},
    )
    return destination
