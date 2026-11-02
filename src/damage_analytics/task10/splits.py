"""Deterministic stratified two-way splitting of official training components only."""

from collections import defaultdict

import numpy as np


def development_split(rows: list[dict], validation_fraction: float, seed: int) -> dict[str, str]:
    eligible = [r for r in rows if r["disposition"] == "development_eligible"]
    grouped = defaultdict(list)
    for row in eligible:
        grouped[row["leakage_component"]].append(row)
    identifiers = sorted(grouped)
    if len(identifiers) < 2:
        raise ValueError("Fewer than two independent leakage components remain for development")
    counts = np.array(
        [[sum(r["class_id"] == i for r in grouped[g]) for i in range(4)] for g in identifiers],
        dtype=float,
    )
    if np.any((counts > 0).sum(0) < 2):
        raise ValueError(
            "At least one class has fewer than two leakage components; stratification is "
            "infeasible"
        )
    fractions = np.array([1 - validation_fraction, validation_fraction])
    target = fractions[:, None] * counts.sum(0)
    target_size = fractions * counts.sum()

    def cost(y):
        return float(
            ((y - target) ** 2 / np.maximum(target, 1)).sum()
            + ((y.sum(1) - target_size) ** 2 / np.maximum(target_size, 1)).sum()
            + 10000 * np.count_nonzero(y == 0)
        )

    rng, best, selected = np.random.default_rng(seed), float("inf"), None
    priority = (counts / np.maximum(counts.sum(0), 1)).sum(1)
    for _ in range(24):
        assigned = np.zeros_like(target)
        result = {}
        for index in np.argsort(
            -priority * rng.uniform(0.8, 1.2, len(identifiers)), kind="stable"
        ):
            candidates = []
            for fold in range(2):
                updated = assigned.copy()
                updated[fold] += counts[index]
                candidates.append(cost(updated))
            fold = int(np.argmin(candidates))
            assigned[fold] += counts[index]
            result[identifiers[index]] = ("train", "validation")[fold]
        if cost(assigned) < best:
            best, selected = cost(assigned), result
    assignment = {r["sample_id"]: selected[r["leakage_component"]] for r in eligible}
    if any(
        {r["class_id"] for r in eligible if assignment[r["sample_id"]] == fold} != set(range(4))
        for fold in ("train", "validation")
    ):
        raise ValueError(
            "The grouped stratification search could not retain all four classes in both folds"
        )
    return assignment
