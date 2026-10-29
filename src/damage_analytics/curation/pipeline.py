"""Stage immutable, content-addressed Task 02 releases and verify before publication."""

import json
import tempfile
from collections import Counter
from pathlib import Path

from damage_analytics.curation.cleaning import issue
from damage_analytics.curation.config import CurationConfig
from damage_analytics.curation.metadata import capture_session_edges
from damage_analytics.curation.provenance import audit_inputs, release_identity, task01_evidence
from damage_analytics.curation.reporting import audit_task01, statistics, write_reports
from damage_analytics.curation.samples import curate_sample, resolve_exact_images
from damage_analytics.curation.similarity import connected_groups, similarity_edges, source_edges
from damage_analytics.curation.splitting import make_splits
from damage_analytics.curation.verification import (
    verify_checksums,
    verify_dataset,
    write_checksums,
)
from damage_analytics.discovery import discover_pairs
from damage_analytics.io import sha256_file, write_csv, write_json
from damage_analytics.labels import CLASS_NAMES, SOURCE_NOTE


def run_curation(config: CurationConfig) -> tuple[Path, dict]:
    """Reuse verified identical releases; publish new versions only after complete checks."""
    inventory, provenance = audit_inputs(config)
    previous_hashes = task01_evidence(config)
    version, snapshot = release_identity(config, inventory, provenance, previous_hashes)
    destination = config.output_dir / "versions" / version
    if destination.exists():
        verify_checksums(destination)
        verification = verify_dataset(destination, config)
        if not verification["passed"]:
            raise ValueError(
                f"Existing release verification failed: {verification['failures'][:3]}"
            )
        quality = json.loads((destination / "quality_report.json").read_text())
    else:
        config.output_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".task02-", dir=config.output_dir) as temporary:
            staging = Path(temporary) / "release"
            staging.mkdir()
            write_json(staging / "release.json", {"dataset_version": version, **snapshot})
            write_csv(staging / "source_inventory.csv", inventory, ["path", "sha256", "bytes"])
            quality = _build_release(config, staging, provenance, version, inventory)
            # Recheck sources/reports after all reads; no release mixes changing inputs.
            final_inventory, final_provenance = audit_inputs(config)
            if (
                inventory != final_inventory
                or provenance != final_provenance
                or previous_hashes != task01_evidence(config)
            ):
                raise ValueError("Inputs changed during curation; no release was published")
            write_json(staging / "quality_report.json", quality)
            write_checksums(staging)
            verify_checksums(staging)
            destination.parent.mkdir(parents=True, exist_ok=True)
            staging.rename(destination)
    _publish_latest(config, destination, quality)
    return destination, quality


def _build_release(
    config: CurationConfig, folder: Path, provenance: dict, version: str, inventory: list[dict]
) -> dict:
    """Audit samples, resolve duplicate ground truth, group, split, and verify saved data."""
    pairs, discovery_issues = discover_pairs(config.paths.extracted_dir)
    hashes = {row["path"]: row["sha256"] for row in inventory}
    records, decisions = [], []
    for number, pair in enumerate(pairs, 1):
        row, findings = curate_sample(
            pair, number, config.paths.extracted_dir, folder, hashes, config.cleaning
        )
        records.append(row)
        decisions.extend(findings)
    known = {row["sample_id"] for row in records}
    for finding in discovery_issues:
        sample_id = finding["sample_id"]
        action = "exclude" if finding["level"] == "error" else "accept"
        decisions.append(issue(sample_id, finding["code"], action, finding["detail"]))
        if sample_id and sample_id not in known:
            records.append({"sample_id": sample_id, "included": False})
            known.add(sample_id)
    decisions.extend(resolve_exact_images(records, config.cleaning))
    edges = similarity_edges(records, config.similarity)
    edges.extend(capture_session_edges(records, config.similarity))
    explicit_edges, evidence_ids = source_edges(config.source_groups_file, known)
    edges.extend(explicit_edges)
    edges.sort(key=lambda row: (row["left"], row["right"], row["kind"]))
    groups = connected_groups(sorted(known), edges)
    for row in records:
        row["group_id"] = groups[row["sample_id"]]
        if not row["included"]:
            for key in ("image", "annotation", "mask"):
                if row.get(key):
                    (folder / row[key]).unlink(missing_ok=True)
    curated = [row for row in records if row["included"]]
    group_count = len({row["group_id"] for row in curated})
    if group_count >= 3:
        assignments, split_report = make_splits(curated, config.splitting, config.random_seed)
    else:
        assignments = {row["sample_id"]: "unassigned" for row in curated}
        split_report = {
            "status": "blocked",
            "reason": "Fewer than three independent curated groups",
        }
    for row in curated:
        row["split"] = assignments[row["sample_id"]]
    stats = statistics(curated, config.splitting.rare_class_image_threshold)
    historical = audit_task01(config.task01_dir, decisions, {row["sample_id"] for row in curated})
    decisions.sort(
        key=lambda row: (row["sample_id"], row["shape_index"], row["code"], row["detail"])
    )
    write_reports(folder, curated, records, decisions, edges, stats, split_report, historical)
    verification = verify_dataset(folder, config)
    return _quality(
        config,
        records,
        curated,
        decisions,
        edges,
        evidence_ids,
        verification,
        stats,
        split_report,
        provenance,
        version,
    )


def _quality(
    config: CurationConfig,
    all_rows: list[dict],
    curated: list[dict],
    decisions: list[dict],
    edges: list[dict],
    evidence_ids: set[str],
    verification: dict,
    stats: dict,
    splits: dict,
    provenance: dict,
    version: str,
) -> dict:
    """Separate verified technical checks from unresolved scientific/source limitations."""
    issues = []
    kept = {row["sample_id"] for row in curated}
    for code in sorted({row["code"] for row in decisions}):
        subset = [row for row in decisions if row["code"] == code]
        issues.append(
            {
                "code": code,
                "events": len(subset),
                "samples": len({row["sample_id"] for row in subset}),
                "retained_samples": len({row["sample_id"] for row in subset} & kept),
                "actions": dict(Counter(row["action"] for row in subset)),
            }
        )
    known_count = len(all_rows)
    count_matches = known_count == config.paths.expected_pairs
    technical = verification["passed"] and count_matches and bool(curated)
    limitations = [
        "Physical structure/source IDs are unavailable unless supplied as explicit evidence; "
        "absence of unknown-source leakage is not established.",
        "Near-duplicate detection is heuristic; crops, rotations and exposure changes "
        "may be missed. "
        "Conservatively grouped candidates may be false positives.",
        "Microcrack vs non-structural crack and Delamination vs major spalling remain unresolved. "
        "Source strings are retained; no severity targets exist.",
        "EXIF dimension evidence determines orientation but is not expert confirmation "
        "of label alignment.",
        "Raster/geometry consistency does not prove that human annotations are complete "
        "or correct. "
        "Publisher RGB mask values are not applicable to polygon JSON inputs.",
        "RGB conversion is not calibrated color reproduction; "
        "pixel counts are not physical areas.",
        "Multi-label grouping search is heuristic; coverage and ratios are reported, "
        "not guaranteed.",
    ]
    return {
        "dataset_version": version,
        "source_integrity": provenance,
        "source_samples": known_count,
        "readable_images": sum(row.get("image_readable", False) for row in all_rows),
        "readable_annotations": sum(row.get("annotation_readable", False) for row in all_rows),
        "all_source_modes": dict(
            Counter(row.get("source_mode", "unreadable") for row in all_rows)
        ),
        "all_source_exif_orientations": dict(
            Counter(str(row.get("exif_orientation", "unreadable")) for row in all_rows)
        ),
        "expected_samples": config.paths.expected_pairs,
        "expected_count_matches": count_matches,
        "valid_samples": len(curated),
        "excluded_samples": known_count - len(curated),
        "issue_resolution": issues,
        "exclusion_reasons_nonexclusive": dict(
            Counter(row["code"] for row in decisions if row["action"] == "exclude")
        ),
        "excluded_samples_by_reason_nonexclusive": {
            code: len(
                {
                    row["sample_id"]
                    for row in decisions
                    if row["action"] == "exclude" and row["code"] == code
                }
            )
            for code in sorted({row["code"] for row in decisions if row["action"] == "exclude"})
        },
        "decision_actions": dict(Counter(row["action"] for row in decisions)),
        "similarity_evidence": dict(Counter(row["kind"] for row in edges)),
        "explicit_source_evidence_samples": len(evidence_ids),
        "samples_without_explicit_source_identity": known_count - len(evidence_ids),
        "grouping_uncertainty": "Unknown physical sources remain; "
        "filename prefixes are not source IDs",
        "label_mapping": [
            {"class_id": index, "source_label": name} for index, name in enumerate(CLASS_NAMES)
        ],
        "label_provenance": SOURCE_NOTE,
        "statistics": stats,
        "splitting": splits,
        "verification": verification,
        "ml_readiness": {
            "technical_checks_passed": technical,
            "verdict": ("CONDITIONAL_RESEARCH_USE" if technical else "NOT_READY"),
            "meaning": "Verified segmentation artifacts/splits only; no globally leakage-free or "
            "expert-clean ground-truth claim. Resolve listed limitations before benchmark claims.",
            "severity_prediction_ready": False,
        },
        "unresolved_limitations": limitations,
    }


def _publish_latest(config: CurationConfig, destination: Path, quality: dict) -> None:
    """Keep a small convenience pointer; immutable releases retain the frozen test CSV."""
    write_json(config.output_dir / "quality_report.json", quality)
    write_json(
        config.output_dir / "latest.json",
        {
            "dataset_version": quality["dataset_version"],
            "release_path": destination.relative_to(config.output_dir).as_posix(),
            "quality_report_sha256": sha256_file(destination / "quality_report.json"),
            "test_split_sha256": sha256_file(destination / "splits/test.csv"),
        },
    )


def latest_release(config: CurationConfig) -> Path:
    """Resolve a release pointer, refusing traversal outside the version directory."""
    pointer = json.loads((config.output_dir / "latest.json").read_text())
    path = (config.output_dir / pointer["release_path"]).resolve()
    versions = (config.output_dir / "versions").resolve()
    if path.parent != versions or not path.is_dir():
        raise ValueError("Invalid Task 02 release pointer")
    return path


def verify_latest(config: CurationConfig) -> dict:
    """Reverify sealed output artifacts plus every archived source file."""
    release = latest_release(config)
    verify_checksums(release)
    inventory, provenance = audit_inputs(config)
    snapshot = json.loads((release / "release.json").read_text())
    from damage_analytics.curation.provenance import content_hash

    if (
        content_hash(inventory) != snapshot["source_inventory_sha256"]
        or provenance != snapshot["provenance"]
    ):
        raise ValueError("Current source dataset differs from the frozen release")
    result = verify_dataset(release, config)
    result["source_integrity"] = provenance
    return result
