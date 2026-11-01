"""Sealed split/duplicate QA using metadata; contact sheets exclude test pixels."""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from damage_analytics.curation.similarity import connected_groups
from damage_analytics.io import sha256_file, write_csv, write_json
from damage_analytics.task07.crosscheck import overlap
from damage_analytics.task07.duplicates import HammingTree
from damage_analytics.task07.release import Release
from damage_analytics.task07.splits import split_report
from damage_analytics.task08.config import Config
from damage_analytics.task08.galleries import pair_sheets
from damage_analytics.task08.reports import integrity_report


def candidate_components(rows: tuple[dict, ...], near: list[dict]) -> tuple[dict, list[dict]]:
    """Trace possible indirect relationships, without converting them into source identities."""
    buckets = defaultdict(list)
    for row in rows:
        buckets[row["group_id"]].append(row["sample_id"])
    edges = [
        {"left": ids[0], "right": identifier} for ids in buckets.values() for identifier in ids[1:]
    ]
    edges.extend(
        {"left": p["left"], "right": p["right"]}
        for p in near
        if p["decision"] != "different_source"
    )
    components = connected_groups(sorted(r["sample_id"] for r in rows), edges)
    members = defaultdict(list)
    for row in rows:
        members[components[row["sample_id"]]].append(row)
    cross = [values for values in members.values() if len({r["split"] for r in values}) > 1]
    table = [
        {
            "candidate_component": components[r["sample_id"]],
            "sample_id": r["sample_id"],
            "task": r["task"],
            "split": r["split"],
            "source_member": r["source_member"],
        }
        for values in cross
        for r in values
    ]
    summary = {
        "potential_cross_split_components": len(cross),
        "potential_cross_split_records": len(table),
        "potential_classification_cross_split_components": sum(
            len({r["split"] for r in values if r["task"] == "classification"}) > 1
            for values in cross
        ),
        "scope": "Candidate graph only; neither confirmed source identities nor new splits",
    }
    return summary, table


def investigate_pairs(release: Release) -> tuple[list[dict], list[dict], dict]:
    """Enumerate actual affected views/samples without assigning identity or review decisions."""
    lookup = {r["sample_id"]: r for r in release.rows}
    near = json.loads(release.file("near_candidates.json").read_text())
    near.sort(
        key=lambda p: (
            lookup[p["left"]]["split"] == lookup[p["right"]]["split"],
            p["left"],
            p["right"],
        )
    )
    pairs, neighbors = [], defaultdict(list)
    for index, pair in enumerate(near, 1):
        left, right = lookup[pair["left"]], lookup[pair["right"]]
        cross = left["split"] != right["split"]
        item = {
            "pair_id": f"p{index:04d}",
            "left": left["sample_id"],
            "right": right["sample_id"],
            "decision": "",
            "evidence": "",
            "recorded_decision": pair["decision"],
            "cross_split": cross,
            "phash_distance": (left["phash"] ^ right["phash"]).bit_count(),
            "dhash_distance": (left["dhash"] ^ right["dhash"]).bit_count(),
        }
        for side, row in (("left", left), ("right", right)):
            for field in ("task", "split", "label", "source_label", "source_member", "group_id"):
                item[f"{side}_{field}"] = row[field]
            item[f"{side}_dimensions"] = f"{row.get('width', '?')}x{row.get('height', '?')}"
        item["same_raw_sha256"] = left["image_sha256"] == right["image_sha256"]
        item["same_decoded_sha256"] = left["decoded_sha256"] == right["decoded_sha256"]
        pairs.append(item)
        if cross and pair["decision"] == "unresolved":
            neighbors[left["sample_id"]].append(item["pair_id"])
            neighbors[right["sample_id"]].append(item["pair_id"])
    affected = [
        {
            "sample_id": identifier,
            "task": lookup[identifier]["task"],
            "split": lookup[identifier]["split"],
            "source_member": lookup[identifier]["source_member"],
            "source_label": lookup[identifier]["source_label"],
            "pair_ids": ";".join(ids),
        }
        for identifier, ids in sorted(neighbors.items())
    ]
    unresolved = [p for p in pairs if p["cross_split"] and p["recorded_decision"] == "unresolved"]
    summary = {
        "all_near_pairs": len(pairs),
        "unresolved_cross_split_pairs": len(unresolved),
        "cross_split_view_pairs": dict(
            Counter(" / ".join(sorted((p["left_task"], p["right_task"]))) for p in unresolved)
        ),
        "cross_split_partition_pairs": dict(
            Counter(" / ".join(sorted((p["left_split"], p["right_split"]))) for p in unresolved)
        ),
        "affected_samples": len(affected),
        "affected_by_view_partition": dict(Counter(f"{r['task']}/{r['split']}" for r in affected)),
        "affected_classification_samples": sum(r["task"] == "classification" for r in affected),
        "zero_phash_and_dhash_cross_split_pairs": sum(
            p["phash_distance"] == p["dhash_distance"] == 0 for p in unresolved
        ),
        "cross_split_dimension_pairs": dict(
            Counter(
                " / ".join(sorted((p["left_dimensions"], p["right_dimensions"])))
                for p in unresolved
            )
        ),
        "test_contact_images_redacted": True,
        "interpretation": "Near candidates are review evidence, not confirmed duplicates. "
        "View-level counts can repeat a common image's annotation views.",
    }
    return pairs, affected, summary


def audit_release(release: Release, output: Path, per_page: int) -> dict:
    """Audit all split metadata and exclusions while preserving every sealed release file."""
    near = json.loads(release.file("near_candidates.json").read_text())
    grouping = json.loads(release.file("grouping.json").read_text())
    checks = split_report(list(release.rows), near, grouping["screening_truncated"])
    with release.file("exclusions.csv").open() as stream:
        excluded = list(csv.DictReader(stream))
    if len(excluded) != release.quality["exclusions"]:
        raise ValueError("Excluded population differs from quality report")
    exact = defaultdict(list)
    for row in release.rows:
        exact[row["decoded_sha256"]].append(row)
    pairs, affected, pair_summary = investigate_pairs(release)
    potential, potential_rows = candidate_components(release.rows, near)
    pair_fields = list(pairs[0]) if pairs else ["pair_id", "left", "right", "decision", "evidence"]
    write_csv(output / "review_pairs.csv", pairs, pair_fields)
    write_csv(
        output / "affected_samples.csv",
        affected,
        ["sample_id", "task", "split", "source_member", "source_label", "pair_ids"],
    )
    write_csv(output / "excluded_records.csv", excluded, ["sample_id", "image", "reason"])
    write_csv(
        output / "potential_components.csv",
        potential_rows,
        ["candidate_component", "sample_id", "task", "split", "source_member"],
    )
    pair_sheets(release, pairs, output / "contact_sheets", per_page)
    report = {
        "dataset": release.identity(),
        "included": len(release.rows),
        "excluded": len(excluded),
        "included_by_view": dict(Counter(r["task"] for r in release.rows)),
        "exclusions_by_reason": dict(Counter(r["reason"] for r in excluded)),
        "exclusions_by_view": dict(Counter(Path(r["image"]).parts[1] for r in excluded)),
        "excluded_file_hashes": {r["sample_id"]: release.seal.get(r["image"]) for r in excluded},
        "exact_decoded_duplicate_components": sum(len(v) > 1 for v in exact.values()),
        "exact_cross_split_components": sum(
            len({r["split"] for r in v}) > 1 for v in exact.values()
        ),
        "exact_detection_segmentation_image_pairs": sum(
            {r["task"] for r in v} >= {"detection", "segmentation"} for v in exact.values()
        ),
        "split_checks": checks,
        "pair_investigation": pair_summary,
        "potential_candidate_graph": potential,
        "pixel_scope": "Only train/validation review thumbnails; test metadata used for split QA",
        "release_changed": False,
    }
    write_json(output / "quality.json", report)
    return report


def verify_search(left: Release, right: Release, radius: int, count: int) -> dict:
    """Compare an independent brute-force implementation with BK queries on validation anchors."""
    tree = HammingTree()
    for row in left.rows:
        tree.add(row["phash"], row["sample_id"])
    anchors = sorted(right.samples("validation"), key=lambda r: r["sample_id"])[:count]
    checks = []
    for row in anchors:
        distances = [(r["sample_id"], (r["phash"] ^ row["phash"]).bit_count()) for r in left.rows]
        expected = {identifier for identifier, distance in distances if distance <= radius}
        actual = set(tree.query(row["phash"], radius))
        if actual != expected:
            raise ValueError("BK radius search differs from brute force")
        checks.append(
            {
                "sample_id": row["sample_id"],
                "brute_force_records": len(left.rows),
                "minimum_phash_distance": min(d for _, d in distances),
                "radius_hits": len(actual),
                "matched": True,
            }
        )
    return {"validation_anchors": checks, "radius": radius, "all_matched": bool(checks)}


def audit(config: Config, releases: dict[str, Release], output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    reports = {
        name: audit_release(release, output / name, config.pairs_per_page)
        for name, release in releases.items()
    }
    left, right = releases["sdnet2018"], releases["damsegment"]
    screen = overlap(left.rows, right.rows, config.phash_distance, config.max_near_candidates)
    screen.update(
        left=left.identity(),
        right=right.identity(),
        scope="All-partition fingerprint metadata QA only; no test pixels or predictions",
    )
    write_json(output / "cross_dataset_screen.json", screen)
    verification = verify_search(left, right, config.phash_distance, config.brute_force_queries)
    write_json(output / "search_verification.json", verification)
    old_path = left.root.parent.parent / "cross_checks/external_overlap.json"
    legacy = json.loads(old_path.read_text()) if old_path.exists() else None
    result = {
        "datasets": reports,
        "overlap_search": {
            k: v for k, v in screen.items() if k not in ("exact_pairs", "near_candidates")
        },
        "legacy_overlap_report": {
            "path": str(old_path),
            "sha256": sha256_file(old_path),
            "screened_comparisons": legacy.get("screened_comparisons"),
        }
        if legacy
        else None,
        "search_verification": verification,
        "reporting_defect": "Legacy screened_comparisons counted radius candidates consumed, "
        "not search distances. Zero could mean a completed search with no hits. "
        "Schema 2 separates node distances, queries, radius hits and dHash checks.",
        "decisions": "No human identity decisions assigned; no records moved or releases changed. "
        "Unresolved source grouping and review candidates still block final eligibility.",
    }
    write_json(output / "integrity_report.json", result)
    integrity_report(output / "integrity_report.md", result)
    return result
