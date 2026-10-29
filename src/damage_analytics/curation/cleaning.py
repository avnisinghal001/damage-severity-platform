"""Audit every polygon and record explicit fixes, acceptable cases, and exclusions."""

import math
from pathlib import PurePosixPath

from damage_analytics.annotations import Polygon
from damage_analytics.curation.config import CleaningPolicy
from damage_analytics.curation.geometry import redundant_vertices, self_intersections, signed_area
from damage_analytics.labels import LABEL_TO_ID


def issue(sample_id: str, code: str, action: str, detail: str, shape: int = -1) -> dict:
    """Use a common decision schema; shape indices are zero-based, -1 means sample-wide."""
    return {
        "sample_id": sample_id,
        "shape_index": shape,
        "code": code,
        "action": action,
        "detail": detail,
    }


def clean_annotation(
    document: dict, sample_id: str, image_name: str, size: tuple[int, int], policy: CleaningPolicy
) -> tuple[list[Polygon], list[dict]]:
    """Audit all shapes; any excluded shape excludes its sample, never just that shape."""
    decisions = []
    if not isinstance(document, dict):
        return [], [issue(sample_id, "annotation_schema", "exclude", "JSON must be an object")]
    if document.get("flags"):
        decisions.append(
            issue(
                sample_id,
                "unresolved_annotation_flags",
                "exclude",
                "Nonempty document flags require interpretation",
            )
        )
    reference = document.get("imagePath")
    if (
        not isinstance(reference, str)
        or PurePosixPath(reference.replace("\\", "/")).name != image_name
    ):
        decisions.append(
            issue(sample_id, "image_reference", "exclude", "imagePath basename mismatch")
        )
    if any(
        type(document.get(key)) is not int or document[key] != value
        for key, value in zip(("imageWidth", "imageHeight"), size, strict=True)
    ):
        decisions.append(
            issue(
                sample_id,
                "dimension_mismatch",
                "exclude",
                f"JSON dimensions do not match selected frame {size}",
            )
        )
    shapes = document.get("shapes")
    if not isinstance(shapes, list):
        return [], decisions + [
            issue(sample_id, "annotation_schema", "exclude", "shapes is not a list")
        ]
    if not shapes:
        action = "accept" if policy.allow_empty_annotations else "exclude"
        decisions.append(issue(sample_id, "empty_annotation", action, "No source damage polygons"))
    polygons = []
    for index, shape in enumerate(shapes):
        polygon, findings = _clean_shape(shape, sample_id, index, size, policy)
        decisions.extend(findings)
        if polygon:
            polygons.append(polygon)
    return polygons, decisions


def _clean_shape(
    shape: dict, sample_id: str, index: int, size: tuple[int, int], policy: CleaningPolicy
) -> tuple[Polygon | None, list[dict]]:
    """Keep source label strings, reject unknown semantics, and audit all usable geometry."""
    findings = []

    def record(code: str, action: str, detail: str) -> None:
        findings.append(issue(sample_id, code, action, detail, index))

    if not isinstance(shape, dict) or shape.get("shape_type") != "polygon":
        record("unsupported_shape", "exclude", "Only explicit polygon shapes are supported")
        return None, findings
    label = shape.get("label")
    if not isinstance(label, str) or label not in LABEL_TO_ID:
        record("unknown_label", "exclude", repr(label))
    if shape.get("flags") or shape.get("group_id") is not None:
        record("unresolved_shape_semantics", "exclude", "Nonempty flags/group_id require review")
    vertices = shape.get("points")
    if not isinstance(vertices, list) or len(vertices) < 3:
        record("malformed_polygon", "exclude", "At least three [x,y] vertices required")
        return None, findings
    points = []
    for vertex in vertices:
        if (
            not isinstance(vertex, list)
            or len(vertex) != 2
            or any(type(v) not in (int, float) for v in vertex)
        ):
            record("invalid_coordinate", "exclude", repr(vertex))
            continue
        try:
            finite = all(math.isfinite(value) for value in vertex)
        except OverflowError:
            finite = False
        if not finite:
            record("nonfinite_coordinate", "exclude", repr(vertex))
            continue
        points.append((float(vertex[0]), float(vertex[1])))
    if len(points) != len(vertices):
        return None, findings
    cleaned = redundant_vertices(points)
    if cleaned != points:
        action = "fix" if policy.remove_redundant_vertices else "exclude"
        record(
            "redundant_vertices", action, f"{len(points)} → {len(cleaned)} vertices; same boundary"
        )
        points = cleaned
    if len(set(points)) < 3:
        record("degenerate_polygon", "exclude", "Fewer than three distinct vertices")
        return None, findings
    changes = []
    for vertex_index, (x, y) in enumerate(points):
        clipped = (min(max(x, 0), size[0]), min(max(y, 0), size[1]))
        distance = max(abs(clipped[0] - x), abs(clipped[1] - y))
        if distance:
            if distance > policy.out_of_bounds_tolerance:
                record(
                    "out_of_bounds",
                    "exclude",
                    f"vertex {vertex_index}: {(x, y)}; delta={distance}",
                )
            else:
                changes.append({"vertex": vertex_index, "before": [x, y], "after": list(clipped)})
                points[vertex_index] = clipped
    if changes:
        record("tolerance_clamp", "fix", repr(changes))
    if any(x == size[0] or y == size[1] for x, y in points):
        record("boundary_edge_coordinate", "accept", "Edge coordinates use Task 01 pixel clipping")
    intersections = self_intersections(points, policy.geometry_epsilon)
    if intersections:
        record(
            "self_intersection", "exclude", f"{intersections} non-simple edge contacts/retracings"
        )
    if abs(signed_area(points)) <= policy.geometry_epsilon:
        record("zero_area", "exclude", "Absolute shoelace area is zero within epsilon")
    # Validate the actual clipped pixel-frame ring as well as source edge coordinates.
    pixel_points = redundant_vertices(
        [(min(x, size[0] - 1), min(y, size[1] - 1)) for x, y in points]
    )
    if (
        len(set(pixel_points)) < 3
        or abs(signed_area(pixel_points)) <= policy.geometry_epsilon
        or self_intersections(pixel_points, policy.geometry_epsilon)
    ):
        record("invalid_pixel_ring", "exclude", "Edge-to-pixel mapping degenerates/intersects")
    if any(item["action"] == "exclude" for item in findings):
        return None, findings
    return Polygon(label, tuple(points)), findings
