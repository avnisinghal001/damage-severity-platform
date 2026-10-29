"""Cleaning policies, numeric edge cases, and hand-verifiable mask correctness."""

from dataclasses import replace

import numpy as np
import pytest

from damage_analytics.annotations import Polygon, rasterize
from damage_analytics.curation.cleaning import clean_annotation
from damage_analytics.curation.config import CleaningPolicy
from damage_analytics.curation.geometry import self_intersections, signed_area
from damage_analytics.curation.masks import annotation_similarity, checked_mask
from damage_analytics.labels import LABEL_TO_ID


def document(points=None, label="Microcrack"):
    """Return a minimal source annotation without embedded image data."""
    return {
        "imagePath": "sample.png",
        "imageWidth": 8,
        "imageHeight": 6,
        "shapes": [
            {
                "label": label,
                "shape_type": "polygon",
                "points": points or [[1, 1], [4, 1], [4, 4], [1, 4]],
            }
        ],
    }


def clean(data, policy=None):
    return clean_annotation(data, "sample", "sample.png", (8, 6), policy or CleaningPolicy())


@pytest.mark.parametrize(
    "points,intersecting",
    [
        ([(0, 0), (4, 0), (4, 4), (0, 4)], False),
        ([(0, 0), (4, 0), (2, 2), (4, 4), (0, 4)], False),
        ([(0, 0), (4, 4), (0, 4), (4, 0)], True),
        ([(0, 0), (4, 0), (1, 1), (3, 4), (0, 3), (4, 2)], True),
        ([(0, 0), (4, 0), (2, 0), (4, 4), (0, 4)], True),
        ([(0, 0), (4, 0), (4, 4), (2, 0), (0, 4)], True),
        ([(0, 0), (2, 0), (4, 0), (4, 4), (0, 4)], False),
    ],
)
def test_intersections_include_touches_backtracking_and_nonzero_area_crossings(
    points, intersecting
):
    assert bool(self_intersections(points, 1e-9)) == intersecting
    assert bool(self_intersections(list(reversed(points)), 1e-9)) == intersecting
    translated = [(x + 100, y - 20) for x, y in points]
    assert bool(self_intersections(translated, 1e-9)) == intersecting


def test_redundant_vertices_are_explicit_and_preserve_every_raster_pixel():
    data = document([[1, 1], [4, 1], [4, 1], [4, 4], [1, 4], [1, 1]])
    polygons, findings = clean(data)
    assert [(item["code"], item["action"]) for item in findings] == [("redundant_vertices", "fix")]
    before, _ = rasterize(
        [Polygon("Microcrack", tuple(map(tuple, data["shapes"][0]["points"])))], (8, 6)
    )
    after, _ = rasterize(polygons, (8, 6))
    np.testing.assert_array_equal(before, after)
    _, strict = clean(data, replace(CleaningPolicy(), remove_redundant_vertices=False))
    assert strict[0]["action"] == "exclude"


@pytest.mark.parametrize(
    "coordinate", [float("nan"), float("inf"), -float("inf"), True, "2", None]
)
def test_invalid_coordinates_are_never_coerced(coordinate):
    _, findings = clean(document([[coordinate, 1], [4, 1], [4, 4], [1, 4]]))
    assert any(item["action"] == "exclude" for item in findings)


def test_tolerance_clamp_is_off_by_default_and_records_exact_changes():
    data = document([[-0.2, 1], [4, 1], [4, 4], [1, 4]])
    _, default = clean(data)
    assert any(item["code"] == "out_of_bounds" for item in default)
    polygons, opted_in = clean(data, replace(CleaningPolicy(), out_of_bounds_tolerance=0.25))
    assert polygons[0].points[0] == (0, 1)
    fix = next(item for item in opted_in if item["code"] == "tolerance_clamp")
    assert "before" in fix["detail"] and "after" in fix["detail"]
    _, too_far = clean(data, replace(CleaningPolicy(), out_of_bounds_tolerance=0.1))
    assert any(item["action"] == "exclude" for item in too_far)


def test_boundary_coordinates_are_accepted_but_collapsed_pixel_rings_are_excluded():
    polygons, findings = clean(document([[0, 0], [8, 0], [8, 6], [0, 6]]))
    assert not any(item["action"] == "exclude" for item in findings)
    mask, info = checked_mask(polygons, (8, 6))
    assert info["pixel_counts"] == [0, 48, 0, 0, 0, 0, 0, 0, 0]
    assert mask.getpixel((7, 5)) == 1
    _, invalid = clean(document([[7, 1], [8, 1], [8, 4], [7, 4]]))
    assert any(item["code"] == "invalid_pixel_ring" for item in invalid)


@pytest.mark.parametrize("label,class_id", LABEL_TO_ID.items())
def test_all_eight_labels_keep_their_exact_ids(label, class_id):
    polygons, findings = clean(document(label=label))
    assert not findings
    mask, info = checked_mask(polygons, (8, 6))
    assert mask.getpixel((2, 2)) == class_id
    assert info["pixel_counts"][class_id] == 16
    assert info["pixel_counts"][0] == 32


@pytest.mark.parametrize("invalid", [[], None, {"shapes": []}, {"imagePath": "wrong.png"}])
def test_malformed_documents_are_reported(invalid):
    _, findings = clean(invalid)
    assert any(item["action"] == "exclude" for item in findings)


def test_every_shape_is_audited_and_bad_shape_never_disappears_silently():
    data = document()
    data["shapes"] += [
        document(label="severe")["shapes"][0],
        {"label": "Minor spalling", "shape_type": "rectangle", "points": []},
    ]
    polygons, findings = clean(data)
    assert len(polygons) == 1
    assert {item["code"] for item in findings} == {"unknown_label", "unsupported_shape"}
    assert {item["shape_index"] for item in findings} == {1, 2}


def test_zero_area_and_empty_annotations_are_explicit():
    _, findings = clean(document([[1, 1], [2, 2], [3, 3]]))
    assert any(item["code"] == "zero_area" for item in findings)
    data = document()
    data["shapes"] = []
    _, default = clean(data)
    _, opted_in = clean(data, replace(CleaningPolicy(), allow_empty_annotations=True))
    assert default[0]["action"] == "exclude"
    assert opted_in[0]["action"] == "accept"
    assert signed_area([(1, 1), (2, 2), (3, 3)]) == 0


def test_triangle_mask_matches_hand_computed_grid_and_background():
    polygon = Polygon("Structural crack", ((1, 1), (4, 1), (1, 4)))
    mask, info = checked_mask([polygon], (8, 6))
    expected = np.zeros((6, 8), dtype=np.uint8)
    for y, last_x in ((1, 4), (2, 3), (3, 2), (4, 1)):
        expected[y, 1 : last_x + 1] = 2
    np.testing.assert_array_equal(mask, expected)
    assert info["pixel_counts"][2] == 10
    assert sum(info["pixel_counts"]) == 48


def test_same_label_union_and_cross_label_conflict_are_distinguished():
    first = Polygon("Microcrack", ((1, 1), (4, 1), (4, 4), (1, 4)))
    _, same = checked_mask([first, first], (8, 6))
    assert same["overlap_pixels"] == 16 and same["conflict_pixels"] == 0
    second = Polygon("Structural crack", first.points)
    mask, cross = checked_mask([first, second], (8, 6))
    assert cross["conflict_pixels"] == 16
    assert mask.getpixel((2, 2)) == 2  # Explicit Task 01 rasterization, not policy approval.
    assert annotation_similarity([0, 0], [0, 0]) == 0
    assert annotation_similarity([0, 1], [0, 1]) == 1
    assert annotation_similarity([0, 1], [0, 2]) == 0
