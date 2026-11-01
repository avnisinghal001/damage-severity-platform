"""Exact-image components and bounded Hamming-radius screening; near matches remain reviews."""

import csv
from collections import defaultdict
from pathlib import Path

from damage_analytics.curation.similarity import connected_groups


class HammingTree:
    """BK tree: exact radius search avoids enumerating all N² image pairs."""

    def __init__(self) -> None:
        self.nodes: dict[int, tuple[list[str], dict[int, int]]] = {}
        self.root: int | None = None

    def add(self, value: int, identifier: str) -> None:
        if value in self.nodes:
            self.nodes[value][0].append(identifier)
            return
        if self.root is None:
            self.root = value
        else:
            current = self.root
            while True:
                distance = (value ^ current).bit_count()
                children = self.nodes[current][1]
                if distance not in children:
                    children[distance] = value
                    break
                current = children[distance]
        self.nodes[value] = ([identifier], {})

    def query(self, value: int, radius: int, *, stats: dict | None = None):
        """Yield radius hits; optionally count search work even when no candidate is found."""
        if stats is not None:
            stats["queries"] = stats.get("queries", 0) + 1
        stack = [] if self.root is None else [self.root]
        while stack:
            current = stack.pop()
            distance = (value ^ current).bit_count()
            if stats is not None:
                stats["node_distance_evaluations"] = stats.get("node_distance_evaluations", 0) + 1
            if distance <= radius:
                if stats is not None:
                    stats["radius_hits"] = stats.get("radius_hits", 0) + len(
                        self.nodes[current][0]
                    )
                yield from self.nodes[current][0]
            stack.extend(
                child
                for edge, child in self.nodes[current][1].items()
                if distance - radius <= edge <= distance + radius
            )


def evidence_groups(
    records: list[dict], radius: int, limit: int, reviews: Path | None = None
) -> tuple[dict, list[dict], dict]:
    """Group exact/candidate-source evidence; do not deduplicate away supervised records."""
    edges, near, lookup = [], [], {r["sample_id"]: r for r in records}
    for field in ("image_sha256", "decoded_sha256", "candidate_source", "source_group"):
        buckets = defaultdict(list)
        for row in records:
            if field == "candidate_source" and row.get("source_group"):
                continue
            if row.get(field):
                buckets[row[field]].append(row["sample_id"])
        for members in buckets.values():
            edges.extend(
                {"left": members[0], "right": value, "kind": field} for value in members[1:]
            )
    tree, truncated, comparisons = HammingTree(), False, 0
    search = {"queries": 0, "node_distance_evaluations": 0, "radius_hits": 0}
    for row in sorted(records, key=lambda r: r["sample_id"]):
        for identifier in () if truncated else tree.query(row["phash"], radius, stats=search):
            comparisons += 1
            if comparisons > limit * 20:
                truncated = True
                break
            other = lookup[identifier]
            if row["decoded_sha256"] == other["decoded_sha256"]:
                continue
            if (row["dhash"] ^ other["dhash"]).bit_count() <= 6:
                if len(near) < limit:
                    near.append(
                        {
                            "left": identifier,
                            "right": row["sample_id"],
                            "kind": "perceptual_candidate",
                            "decision": "unresolved",
                        }
                    )
                else:
                    truncated = True
                    break
        tree.add(row["phash"], row["sample_id"])
    if reviews:
        known = {frozenset((x["left"], x["right"])): x for x in near}
        reviewed = set()
        with reviews.open() as stream:
            for review in csv.DictReader(stream):
                key = frozenset((review["left"], review["right"]))
                if (
                    key not in known
                    or key in reviewed
                    or review["decision"] not in ("same_source", "different_source")
                    or not review.get("evidence")
                ):
                    raise ValueError(
                        "Invalid/duplicate review: require unique pair, decision and evidence"
                    )
                reviewed.add(key)
                known[key].update(decision=review["decision"], evidence=review["evidence"])
                if review["decision"] == "same_source":
                    edges.append(
                        {
                            "left": review["left"],
                            "right": review["right"],
                            "kind": "reviewed_source",
                        }
                    )
    groups = connected_groups(sorted(lookup), edges)
    return (
        groups,
        near,
        {
            "edges": edges,
            "screening_truncated": truncated,
            "screened_comparisons": comparisons,
            "screened_comparisons_definition": "Legacy count of pHash radius hits consumed",
            "phash_search": search,
            "scope": "Leakage components, not invented physical-source identifiers",
        },
    )
