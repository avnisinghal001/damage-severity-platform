"""Exact and full/crop visual evidence; graph components never imply invented photo identities."""

from collections import defaultdict

import numpy as np
from PIL import Image, ImageStat

from damage_analytics.curation.similarity import connected_groups, image_fingerprint
from damage_analytics.task07.duplicates import HammingTree
from damage_analytics.task10.config import Config


def visual_views(channels: np.ndarray, fraction: float) -> list[tuple[str, dict]]:
    """Channel-invariant gray fingerprints; screening crops never become labeled examples."""
    gray = Image.fromarray(np.rint(channels.mean(axis=2)).astype(np.uint8))
    width, height = gray.size
    w, h = max(2, round(width * fraction)), max(2, round(height * fraction))
    offsets = [
        ("center", (width - w) // 2, (height - h) // 2),
        ("top_left", 0, 0),
        ("top_right", width - w, 0),
        ("bottom_left", 0, height - h),
        ("bottom_right", width - w, height - h),
    ]
    images = [("full", gray), *[(name, gray.crop((x, y, x + w, y + h))) for name, x, y in offsets]]
    return [
        (name, {**image_fingerprint(image), "gray_std": ImageStat.Stat(image).stddev[0]})
        for name, image in images
    ]


def investigate(
    rows: list[dict], views: dict[str, list], config: Config
) -> tuple[dict, list, list, dict]:
    """Conservatively co-locate candidates; test-connected training components are quarantined."""
    edges, buckets = [], defaultdict(list)
    for row in rows:
        for field in ("raw_sha256", "recovered_sha256", "source_group"):
            if row.get(field):
                buckets[(field, row[field])].append(row["sample_id"])
    for (kind, _), ids in sorted(buckets.items()):
        edges.extend({"left": ids[0], "right": right, "kind": kind} for right in ids[1:])
    tree, indexed, pairs = HammingTree(), {}, {}
    stats = {"queries": 0, "node_distance_evaluations": 0, "radius_hits": 0}
    consumed, skipped_low_information, truncated = 0, 0, False
    by_id = {r["sample_id"]: r for r in rows}
    for row in sorted(rows, key=lambda r: r["sample_id"]):
        identifier = row["sample_id"]
        for view_name, evidence in views.get(identifier, []):
            if evidence["gray_std"] < 2:
                skipped_low_information += 1
                continue  # A uniform patch is not trustworthy visual source evidence.
            for key in (
                ()
                if truncated
                else tree.query(evidence["phash"], config.phash_distance, stats=stats)
            ):
                consumed += 1
                if consumed > config.maximum_near_pairs * 100:
                    truncated = True
                    break
                other_id, other_view, other = indexed[key]
                pair = tuple(sorted((identifier, other_id)))
                if (
                    identifier == other_id
                    or pair in pairs
                    or row["raw_sha256"] == by_id[other_id]["raw_sha256"]
                ):
                    continue
                dd = (evidence["dhash"] ^ other["dhash"]).bit_count()
                if dd > config.dhash_distance:
                    continue
                mae = float(
                    np.mean(
                        np.abs(
                            np.asarray(evidence["thumbnail"], dtype=float)
                            - np.asarray(other["thumbnail"], dtype=float)
                        )
                    )
                    / 255
                )
                if mae > config.thumbnail_mae:
                    continue
                if len(pairs) >= config.maximum_near_pairs:
                    truncated = True
                    break
                pairs[pair] = {
                    "left": other_id,
                    "right": identifier,
                    "left_view": other_view,
                    "right_view": view_name,
                    "phash_distance": (evidence["phash"] ^ other["phash"]).bit_count(),
                    "dhash_distance": dd,
                    "grayscale_thumbnail_mae": mae,
                    "decision": "",
                    "human_evidence": "",
                    "status": "unresolved_visual_candidate",
                }
            key = f"{identifier}|{view_name}"
            indexed[key] = (identifier, view_name, evidence)
            tree.add(evidence["phash"], key)
    near = list(pairs.values())
    edges.extend(
        {"left": p["left"], "right": p["right"], "kind": "conservative_visual_candidate"}
        for p in near
    )
    components = connected_groups(sorted(by_id), edges)
    test_components = {
        components[r["sample_id"]] for r in rows if r["official_partition"] == "test"
    }
    conflicts = set()
    for (kind, _), ids in buckets.items():
        if (
            kind in ("raw_sha256", "recovered_sha256")
            and len({by_id[i]["class_id"] for i in ids if by_id[i]["label_valid"]}) > 1
        ):
            conflicts.update(ids)
    for row in rows:
        row["leakage_component"] = components[row["sample_id"]]
        if row["official_partition"] == "train" and row["valid"]:
            if row["sample_id"] in conflicts:
                row["disposition"] = "quarantine_exact_image_label_conflict"
            elif row["leakage_component"] in test_components:
                row["disposition"] = "quarantine_official_test_related_component"
            else:
                row["disposition"] = "development_eligible"
    report = {
        "exact_or_supplied_source_edges": sum(
            e["kind"] != "conservative_visual_candidate" for e in edges
        ),
        "visual_candidate_pairs": len(near),
        "phash_search": stats,
        "low_information_views_not_used_for_near_identity": skipped_low_information,
        "screening_truncated": truncated,
        "consumed_radius_hits": consumed,
        "raw_or_recovered_label_conflict_samples": len(conflicts),
        "scope": "Conservative leakage components, not authenticated original-photo/source IDs. "
        "Unseen crops, rotations, repeat views and structures can evade screening. "
        "All original official test rows remain in that partition.",
    }
    return components, edges, near, report
