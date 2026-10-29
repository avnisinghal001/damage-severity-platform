"""Deterministic grouped multi-label split search with explicit coverage diagnostics."""

from collections import defaultdict

import numpy as np

from damage_analytics.curation.config import SplitPolicy
from damage_analytics.labels import CLASS_NAMES

SPLIT_NAMES = ("train", "validation", "test")


def make_splits(
    records: list[dict], policy: SplitPolicy, seed: int
) -> tuple[dict[str, str], dict]:
    """Assign whole connected groups using size/class objectives and seeded restarts.

    This is a transparent heuristic, not a proof of globally optimal stratification.
    Coverage needs at least three independent groups per class; that necessary
    condition alone does not prove simultaneous multi-label coverage is feasible.
    Test labels are used once for dataset design, never to compare trained models.
    """
    grouped = defaultdict(list)
    for row in sorted(records, key=lambda item: item["sample_id"]):
        grouped[row["group_id"]].append(row)
    group_ids = sorted(grouped)
    if len(group_ids) < 3:
        raise ValueError("At least three independent curated groups are needed for three splits")
    sizes = np.array([len(grouped[key]) for key in group_ids], dtype=float)
    labels = np.array(
        [
            [
                sum(row["pixel_counts"][index] > 0 for row in grouped[key])
                for index in range(1, len(CLASS_NAMES))
            ]
            for key in group_ids
        ],
        dtype=float,
    )
    totals = labels.sum(axis=0)
    group_coverage = (labels > 0).sum(axis=0)
    fractions = np.asarray(policy.fractions)
    target_sizes = fractions * sizes.sum()
    target_labels = fractions[:, None] * totals
    required = np.zeros((3, len(CLASS_NAMES) - 1), dtype=bool)
    required[:, group_coverage >= 3] = True
    required[0, totals > 0] = True

    def objective(counts: np.ndarray, classes: np.ndarray) -> float:
        size_cost = np.sum((counts - target_sizes) ** 2 / np.maximum(target_sizes, 1))
        label_cost = np.sum((classes - target_labels) ** 2 / np.maximum(target_labels, 1))
        coverage_cost = 10000 * np.count_nonzero(required & (classes == 0))
        empty_cost = 10000 * np.count_nonzero(counts == 0)
        return float(size_cost + label_cost + coverage_cost + empty_cost)

    rng = np.random.default_rng(seed)
    rarity = (labels / np.maximum(totals, 1)).sum(axis=1) + sizes / sizes.sum()
    best_score, best_assignment = float("inf"), None
    for _ in range(policy.search_restarts):
        order = np.argsort(-(rarity * rng.uniform(0.8, 1.2, len(group_ids))), kind="stable")
        counts, classes = np.zeros(3), np.zeros_like(target_labels)
        assignment = np.full(len(group_ids), -1, dtype=int)
        for group in order:
            candidates = []
            for split in range(3):
                new_counts, new_classes = counts.copy(), classes.copy()
                new_counts[split] += sizes[group]
                new_classes[split] += labels[group]
                candidates.append(objective(new_counts, new_classes))
            choice = int(np.argmin(candidates))
            assignment[group] = choice
            counts[choice] += sizes[group]
            classes[choice] += labels[group]
        # Whole-group coordinate descent; never remove the last group from a split.
        for _pass in range(3):
            improved = False
            for group in rng.permutation(len(group_ids)):
                old = assignment[group]
                if np.count_nonzero(assignment == old) <= 1:
                    continue
                old_score = objective(counts, classes)
                for choice in rng.permutation(3):
                    if choice == old:
                        continue
                    new_counts, new_classes = counts.copy(), classes.copy()
                    new_counts[old] -= sizes[group]
                    new_counts[choice] += sizes[group]
                    new_classes[old] -= labels[group]
                    new_classes[choice] += labels[group]
                    score = objective(new_counts, new_classes)
                    if score < old_score - 1e-10:
                        counts, classes, old_score = new_counts, new_classes, score
                        assignment[group], old = choice, choice
                        improved = True
            if not improved:
                break
        score = objective(counts, classes)
        if score < best_score:
            best_score, best_assignment = score, assignment.copy()
    group_assignment = {
        key: SPLIT_NAMES[int(best_assignment[index])] for index, key in enumerate(group_ids)
    }
    result = {row["sample_id"]: group_assignment[row["group_id"]] for row in records}
    report = {
        "algorithm": "seeded whole-group greedy search + coordinate descent",
        "seed": seed,
        "restarts": policy.search_restarts,
        "objective": best_score,
        "target_fractions": list(policy.fractions),
        "independent_groups": len(group_ids),
        "largest_group": int(sizes.max()),
        "splits": {},
        "coverage_constraints": [],
        "test_policy": "Frozen split; no model feedback used",
    }
    for split in SPLIT_NAMES:
        subset = [row for row in records if result[row["sample_id"]] == split]
        report["splits"][split] = {
            "samples": len(subset),
            "fraction": len(subset) / len(records),
            "groups": len({row["group_id"] for row in subset}),
            "class_images": [
                sum(row["pixel_counts"][index] > 0 for row in subset)
                for index in range(len(CLASS_NAMES))
            ],
        }
    for index, name in enumerate(CLASS_NAMES[1:]):
        missing = [
            split
            for split in SPLIT_NAMES
            if report["splits"][split]["class_images"][index + 1] == 0
        ]
        report["coverage_constraints"].append(
            {
                "class_id": index + 1,
                "label": name,
                "groups": int(group_coverage[index]),
                "missing_splits": missing,
                "status": (
                    "absent"
                    if totals[index] == 0
                    else "infeasible_all_three"
                    if group_coverage[index] < 3
                    else "heuristic_unmet"
                    if missing
                    else "covered"
                ),
            }
        )
    return result, report


def leakage_checks(records: list[dict], edges: list[dict], groups: dict[str, str]) -> dict:
    """Verify partition coverage and evidence components, including excluded bridges."""
    assignments = {row["sample_id"]: row["split"] for row in records}
    grouped_splits = defaultdict(set)
    for sample_id, split in assignments.items():
        grouped_splits[groups[sample_id]].add(split)
    violations = [group for group, values in grouped_splits.items() if len(values) > 1]
    direct = [
        edge
        for edge in edges
        if edge["left"] in assignments
        and edge["right"] in assignments
        and assignments[edge["left"]] != assignments[edge["right"]]
    ]
    return {
        "checked_samples": len(records),
        "unique_sample_ids": len(assignments),
        "duplicate_manifest_ids": len(records) - len(assignments),
        "cross_split_components": violations,
        "cross_split_evidence_edges": direct,
        "passed": not violations and not direct and len(assignments) == len(records),
        "scope": "Known exact/similarity/source evidence only; unknown sources not verified",
    }
