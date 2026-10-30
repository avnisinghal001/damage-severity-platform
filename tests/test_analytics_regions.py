"""Exact image-space components, geometry, semantic scope and uncertainty behavior."""

from collections import deque
from dataclasses import replace

import numpy as np
import pytest

from damage_analytics.analytics.components import connected_components
from damage_analytics.analytics.config import RegionPolicy, UncertaintyPolicy
from damage_analytics.analytics.regions import analyze_mask
from damage_analytics.analytics.severity import spalling_assessment


def reference_components(binary, connectivity):
    remaining = set(zip(*np.nonzero(binary), strict=True))
    groups = []
    offsets = [
        (dy, dx)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if (dy or dx) and (connectivity == 8 or abs(dy) + abs(dx) == 1)
    ]
    while remaining:
        first = remaining.pop()
        found, queue = {first}, deque([first])
        while queue:
            y, x = queue.popleft()
            for dy, dx in offsets:
                neighbor = (y + dy, x + dx)
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    found.add(neighbor)
                    queue.append(neighbor)
        groups.append(frozenset(found))
    return set(groups)


@pytest.mark.parametrize("connectivity", [4, 8])
def test_run_length_components_match_independent_breadth_first_reference(connectivity):
    rng = np.random.default_rng(15)
    for _ in range(30):
        values = rng.random((9, 13)) < 0.35
        parts = connected_components(values, connectivity)
        represented = {
            frozenset((y, x) for y, left, right in part.runs for x in range(left, right))
            for part in parts
        }
        assert represented == reference_components(values, connectivity)
        assert sum(part.area for part in parts) == int(values.sum())
        for part in parts:
            points = [(y, x) for y, left, right in part.runs for x in range(left, right)]
            assert part.centroid == pytest.approx(
                (np.mean([x for _, x in points]), np.mean([y for y, _ in points]))
            )


def test_diagonal_connectivity_and_transitive_bridge():
    diagonal = np.eye(3, dtype=bool)
    assert len(connected_components(diagonal, 4)) == 3
    assert len(connected_components(diagonal, 8)) == 1
    bridge = np.array([[1, 0, 1], [1, 1, 1]], dtype=bool)
    (part,) = connected_components(bridge, 4)
    assert part.area == 5 and part.bbox == (0, 0, 3, 2)
    assert connected_components(np.zeros((3, 4), dtype=bool)) == []


def example_mask():
    mask = np.zeros((5, 6), dtype=np.uint8)
    mask[1:3, 1:3] = 3
    mask[1:3, 3] = 4
    mask[4, 5] = 6
    return mask


def test_exact_areas_exclusive_boxes_and_mixed_complexes():
    result = analyze_mask(example_mask(), RegionPolicy(), UncertaintyPolicy())
    assert sum(row["area_pixels"] for row in result["classes"]) == 30
    minor = next(row for row in result["regions"] if row["class_id"] == 3)
    assert minor["area_pixels"] == 4
    assert minor["bbox_xyxy_exclusive"] == [1, 1, 3, 3]
    assert minor["centroid_xy_pixel_index"] == [1.5, 1.5]
    assert minor["image_area_percent"] == pytest.approx(100 * 4 / 30)
    assert not minor["touches_image_boundary"]
    assert minor["spalling_assessment"]["status"] == "uncertain_category_prediction"
    assert len(result["damage_complexes"]) == 2
    mixed = result["damage_complexes"][0]["spalling_assessment"]
    assert mixed["status"] == "mixed_minor_and_moderate" and mixed["label"] is None
    assert result["physical_measurements"]["available"] is False
    assert result["overall_structural_severity"]["score"] is None


def test_reporting_filter_never_changes_area_or_mask():
    mask = example_mask()
    original = mask.copy()
    result = analyze_mask(mask, RegionPolicy(minimum_reported_pixels=3), UncertaintyPolicy())
    assert [row["class_id"] for row in result["regions"]] == [3]
    assert result["classes"][4]["area_pixels"] == 2
    assert result["classes"][4]["connected_regions"] == 1
    assert result["classes"][4]["unreported_region_pixels"] == 2
    np.testing.assert_array_equal(mask, original)


def test_uncalibrated_scores_can_withhold_but_never_validate_severity():
    mask = np.full((3, 4), 3, dtype=np.uint8)
    result = analyze_mask(
        mask,
        RegionPolicy(),
        UncertaintyPolicy(),
        maximum_softmax=np.full(mask.shape, 0.9),
        normalized_entropy=np.full(mask.shape, 0.1),
    )
    region = result["regions"][0]
    assert region["spalling_assessment"]["label"] == "Minor spalling"
    assert region["spalling_assessment"]["status"] == "source_category_derived_unvalidated"
    assert region["uncertainty"]["calibrated"] is False
    uncertain = analyze_mask(
        mask,
        RegionPolicy(),
        UncertaintyPolicy(),
        maximum_softmax=np.full(mask.shape, 0.2),
        normalized_entropy=np.full(mask.shape, 0.9),
    )
    assert uncertain["regions"][0]["spalling_assessment"]["label"] is None
    assert uncertain["regions"][0]["uncertainty"]["review_flags"]


@pytest.mark.parametrize(
    "ids,status",
    [
        ([3], "source_category_derived_unvalidated"),
        ([4], "source_category_derived_unvalidated"),
        ([3, 4], "mixed_minor_and_moderate"),
        ([3, 6], "mixed_supported_and_unsupported"),
        ([5], "unsupported_category"),
        ([7], "unsupported_category"),
        ([], "not_applicable_no_foreground"),
    ],
)
def test_severity_is_only_the_supported_source_category(ids, status):
    assessment = spalling_assessment(ids)
    assert assessment["status"] == status
    assert assessment["structural_severity_or_safety"] == "not_assessed"
    if status != "source_category_derived_unvalidated":
        assert assessment["label"] is None


@pytest.mark.parametrize(
    "mask",
    [
        np.zeros((3, 4, 3), dtype=np.uint8),
        np.array([[255]]),
        np.array([[-1]]),
        np.array([[1.0]]),
        np.zeros((0, 4), dtype=np.uint8),
    ],
)
def test_invalid_masks_are_not_coerced(mask):
    with pytest.raises(ValueError):
        analyze_mask(mask, RegionPolicy(), UncertaintyPolicy())


def test_background_is_not_a_health_or_no_severity_claim():
    result = analyze_mask(np.zeros((2, 3), dtype=np.uint8), RegionPolicy(), UncertaintyPolicy())
    assert result["classes"][0]["area_pixels"] == 6
    assert not result["regions"] and not result["damage_complexes"]
    assert result["overall_structural_severity"]["status"] == "not_assessed"


def test_invalid_score_alignment_and_missing_map_are_rejected():
    mask = example_mask()
    with pytest.raises(ValueError, match="both"):
        analyze_mask(
            mask, RegionPolicy(), UncertaintyPolicy(), maximum_softmax=np.ones(mask.shape)
        )
    with pytest.raises(ValueError, match="align"):
        analyze_mask(
            mask,
            RegionPolicy(),
            UncertaintyPolicy(),
            maximum_softmax=np.ones((1, 1)),
            normalized_entropy=np.ones(mask.shape),
        )
    with pytest.raises(ValueError):
        analyze_mask(mask, replace(RegionPolicy(), connectivity=6), UncertaintyPolicy())
