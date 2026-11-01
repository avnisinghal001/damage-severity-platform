"""Generic grouped multi-label splitting, independent of the RC1841 taxonomy."""

from collections import Counter, defaultdict

import numpy as np

NAMES = ("train", "validation", "test")


def assign_splits(
    records: list[dict], groups: dict[str, str], fractions: tuple, seed: int, restarts: int = 24
) -> dict:
    buckets = defaultdict(list)
    for row in records:
        buckets[groups[row["sample_id"]]].append(row)
    identifiers = sorted(buckets)
    if len(identifiers) < 3:
        raise ValueError("Three leakage components are required; no random-patch fallback")
    tokens = sorted({token for row in records for token in row["strata"]})
    features = np.array(
        [
            [sum(token in row["strata"] for row in buckets[g]) for token in tokens]
            for g in identifiers
        ],
        dtype=float,
    )
    sizes = np.array([len(buckets[g]) for g in identifiers], dtype=float)
    target_n = np.array(fractions) * sizes.sum()
    target_y = np.array(fractions)[:, None] * features.sum(0)
    required = (features > 0).sum(0) >= 3

    def cost(n, y):
        return float(
            ((n - target_n) ** 2 / np.maximum(target_n, 1)).sum()
            + ((y - target_y) ** 2 / np.maximum(target_y, 1)).sum()
            + 10000 * np.count_nonzero(y[:, required] == 0)
            + 10000 * np.count_nonzero(n == 0)
        )

    rng, best, assignment = np.random.default_rng(seed), float("inf"), None
    rarity = (features / np.maximum(features.sum(0), 1)).sum(1) + sizes / sizes.sum()
    for _ in range(restarts):
        n, y, chosen = np.zeros(3), np.zeros_like(target_y), {}
        order = np.argsort(-rarity * rng.uniform(0.8, 1.2, len(identifiers)), kind="stable")
        for index in order:
            scores = []
            for split in range(3):
                nn, yy = n.copy(), y.copy()
                nn[split] += sizes[index]
                yy[split] += features[index]
                scores.append(cost(nn, yy))
            split = int(np.argmin(scores))
            chosen[identifiers[index]] = NAMES[split]
            n[split] += sizes[index]
            y[split] += features[index]
        if cost(n, y) < best:
            best, assignment = cost(n, y), chosen
    if assignment is None or len(set(assignment.values())) != 3:
        raise ValueError("Nonempty grouped splits infeasible under this design")
    return {row["sample_id"]: assignment[groups[row["sample_id"]]] for row in records}


def split_report(records: list[dict], near: list[dict], truncated: bool) -> dict:
    components, sources, exact = defaultdict(set), defaultdict(set), defaultdict(set)
    for row in records:
        components[row["group_id"]].add(row["split"])
        if row.get("source_group"):
            sources[("verified", row["source_group"])].add(row["split"])
        elif row.get("candidate_source"):
            sources[("candidate", row["candidate_source"])].add(row["split"])
        exact[("decoded", row["decoded_sha256"])].add(row["split"])
        if row.get("image_sha256"):
            exact[("bytes", row["image_sha256"])].add(row["split"])
    indexed = {row["sample_id"]: row for row in records}
    cross_near = [
        x
        for x in near
        if x["decision"] == "unresolved"
        and indexed[x["left"]]["split"] != indexed[x["right"]]["split"]
    ]
    coverage = {}
    for split in NAMES:
        subset = [r for r in records if r["split"] == split]
        coverage[split] = {
            "samples_all_views": len(subset),
            "classification_labels": dict(
                Counter(str(r["label"]) for r in subset if r["task"] == "classification")
            ),
            "strata": dict(Counter(t for r in subset for t in r["strata"])),
            "components": len({r["group_id"] for r in subset}),
        }
    verified = all(r.get("source_group") and r.get("source_evidence") for r in records)
    passed = (
        len(indexed) == len(records)
        and all(len(v) == 1 for v in components.values())
        and all(len(v) == 1 for v in sources.values())
        and all(len(v) == 1 for v in exact.values())
    )
    binary_coverage = all(set(coverage[s]["classification_labels"]) == {"0", "1"} for s in NAMES)
    return {
        "known_component_checks_passed": passed,
        "source_identity_verified": verified,
        "cross_split_unresolved_near_pairs": len(cross_near),
        "screening_truncated": truncated,
        "classification_coverage_complete": binary_coverage,
        "final_evaluation_eligible": passed
        and verified
        and binary_coverage
        and not cross_near
        and not truncated,
        "coverage": coverage,
        "interpretation": "No claim of unknown-source leakage freedom. Candidate-prefix "
        "components preserve known evidence but are not authenticated physical-source IDs.",
    }
