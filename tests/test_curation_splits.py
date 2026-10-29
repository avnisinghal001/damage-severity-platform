"""Whole-group split reproducibility, coverage, necessary constraints, and leakage checks."""

import copy

import pytest

from damage_analytics.curation.config import SplitPolicy
from damage_analytics.curation.splitting import leakage_checks, make_splits


def records(count=60):
    rows = []
    for index in range(count):
        counts = (
            [50] + [int((index % 3 == label) or (index % 5 == 0)) for label in range(3)] + [0] * 5
        )
        rows.append(
            {
                "sample_id": f"sample{index:03d}",
                "group_id": f"group{index // 2:03d}",
                "pixel_counts": counts,
            }
        )
    return rows


def test_seeded_splits_are_input_order_invariant_and_do_not_mutate_records():
    rows = records()
    original = copy.deepcopy(rows)
    first, report = make_splits(rows, SplitPolicy(search_restarts=3), 42)
    second, _ = make_splits(list(reversed(rows)), SplitPolicy(search_restarts=3), 42)
    assert first == second and original == rows
    assert set(first) == {row["sample_id"] for row in rows}
    for group in {row["group_id"] for row in rows}:
        assert len({first[row["sample_id"]] for row in rows if row["group_id"] == group}) == 1
    assert all(
        abs(report["splits"][split]["fraction"] - target) <= 0.05
        for split, target in zip(("train", "validation", "test"), (0.7, 0.15, 0.15), strict=True)
    )
    assert all(row["status"] == "covered" for row in report["coverage_constraints"][:3])


def test_rare_class_in_one_group_is_retained_and_coverage_infeasibility_reported():
    rows = records()
    rows[0]["pixel_counts"][8] = 1
    assignments, report = make_splits(rows, SplitPolicy(search_restarts=2), 4)
    rare = report["coverage_constraints"][-1]
    assert rare["groups"] == 1 and rare["status"] == "infeasible_all_three"
    assert assignments[rows[0]["sample_id"]] == "train"
    assert set(rare["missing_splits"]) == {"validation", "test"}


def test_fewer_than_three_groups_is_an_explicit_failure():
    with pytest.raises(ValueError, match="three independent"):
        make_splits(records(4), SplitPolicy(), 42)


def test_leakage_checks_reject_duplicate_identifiers():
    rows = [{"sample_id": "a", "split": "train"}, {"sample_id": "a", "split": "train"}]
    assert not leakage_checks(rows, [], {"a": "a"})["passed"]
