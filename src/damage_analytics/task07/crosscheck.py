"""Cross-dataset overlap evidence; no old RC1841 test pixels or predictions are read."""

import csv
from pathlib import Path

from damage_analytics.io import write_json
from damage_analytics.task07.duplicates import HammingTree
from damage_analytics.task07.release import Release


def overlap(
    left: tuple[dict, ...], right: tuple[dict, ...], radius: int = 4, limit: int = 100000
) -> dict:
    """Exact raw/decoded hashes plus perceptual review candidates, never identity assertions."""
    hashes = {}
    tree = HammingTree()
    indexed = {r["sample_id"]: r for r in left}
    for row in left:
        for field in ("image_sha256", "decoded_sha256"):
            hashes.setdefault(row[field], []).append(row["sample_id"])
        tree.add(row["phash"], row["sample_id"])
    exact, near, truncated, comparisons, exact_count = [], [], False, 0, 0
    search = {"queries": 0, "node_distance_evaluations": 0, "radius_hits": 0}
    dhash_comparisons = 0
    for row in right:
        hits = set(hashes.get(row["image_sha256"], []) + hashes.get(row["decoded_sha256"], []))
        exact_count += len(hits)
        exact.extend(
            {"left": identifier, "right": row["sample_id"]}
            for identifier in sorted(hits)[: max(0, limit - len(exact))]
        )
        for identifier in () if truncated else tree.query(row["phash"], radius, stats=search):
            comparisons += 1
            if comparisons > limit * 20:
                truncated = True
                break
            if identifier in hits:
                continue
            dhash_comparisons += 1
            if (row["dhash"] ^ indexed[identifier]["dhash"]).bit_count() <= 6:
                if len(near) < limit:
                    near.append(
                        {
                            "left": identifier,
                            "right": row["sample_id"],
                            "status": "review_required",
                        }
                    )
                else:
                    truncated = True
                    break
    return {
        "exact_pairs": exact,
        "exact_pair_count": exact_count,
        "exact_pair_listing_truncated": exact_count > len(exact),
        "near_candidates": near,
        "near_candidate_count": len(near),
        "screening_truncated": truncated,
        "screening_schema": 2,
        "screened_comparisons": search["node_distance_evaluations"],
        "screened_comparisons_definition": "BK-tree node-distance evaluations, not N*M pairs",
        "phash_candidate_hits_consumed": comparisons,
        "phash_search": search,
        "dhash_comparisons": dhash_comparisons,
        "indexed_records": len(left),
        "target_records": len(right),
        "queried_records": search["queries"],
        "possible_cartesian_pairs": len(left) * len(right),
        "phash_distance": radius,
        "dhash_distance": 6,
        "candidate_limit": limit,
        "limitations": "Crops, rotations and unknown physical origins can evade this screen. "
        "No near candidate is declared identical automatically.",
    }


def crosscheck(
    left: Release,
    right: Release,
    output: Path,
    rc_manifest: Path | None = None,
    *,
    radius: int = 4,
    limit: int = 100000,
) -> dict:
    report = {
        "left": left.identity(),
        "right": right.identity(),
        **overlap(left.rows, right.rows, radius, limit),
        "rc_metadata_only": None,
    }
    if rc_manifest:
        # Byte-hash evidence only. Do not decode old RC images, including its frozen test.
        with rc_manifest.open() as stream:
            rc = list(csv.DictReader(stream))
        by_hash = {}
        for row in rc:
            for field in ("image_sha256", "source_image_sha256"):
                by_hash.setdefault(row[field], []).append(
                    {"sample_id": row["sample_id"], "split": row["split"]}
                )
        report["rc_metadata_only"] = [
            {"external_sample_id": row["sample_id"], "rc_matches": by_hash[row["image_sha256"]]}
            for row in (*left.rows, *right.rows)
            if row["image_sha256"] in by_hash
        ]
        report["rc_limitation"] = (
            "No decoded/perceptual RC comparison; re-encoding/crops may be missed. "
            "Quarantine any matches to the previously used RC test from development."
        )
    write_json(output, report)
    return report
