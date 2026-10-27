"""Read strict LabelMe-style polygons without trusting embedded image paths/data."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from damage_analytics.labels import LABEL_TO_ID

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage


@dataclass(frozen=True)
class Polygon:
    """A recognized source label and its original floating-point vertices."""

    label: str
    points: tuple[tuple[float, float], ...]


def read_annotation(path: Path, image_name: str, size: tuple[int, int]) -> list[Polygon]:
    """Validate labels, dimensions, filename reference, and polygon geometry.

    Coordinates equal to width/height are accepted as boundary-edge coordinates
    and are clipped to the last pixel during rasterization. Other out-of-bounds
    coordinates, non-finite values, zero-area polygons, and non-polygons fail.
    JSON imageData is never decoded and imagePath is never opened.
    """
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("Annotation must be a JSON object")
    width, height = size
    for field, expected in (("imageWidth", width), ("imageHeight", height)):
        value = document.get(field)
        if type(value) is not int or value != expected:
            raise ValueError(f"{field} does not match decoded image: {value!r} != {expected}")
    reference = document.get("imagePath")
    if (not isinstance(reference, str)
            or PurePosixPath(reference.replace("\\", "/")).name != image_name):
        raise ValueError("imagePath basename does not match paired image filename")
    shapes = document.get("shapes")
    if not isinstance(shapes, list):
        raise ValueError("shapes must be a list")
    polygons = []
    for shape in shapes:
        if not isinstance(shape, dict) or shape.get("shape_type") != "polygon":
            raise ValueError("Only explicitly declared polygon shapes are supported")
        label = shape.get("label")
        if not isinstance(label, str) or label not in LABEL_TO_ID:
            raise ValueError(f"Unexpected annotation label: {label!r}")
        vertices = shape.get("points")
        if not isinstance(vertices, list) or len(vertices) < 3:
            raise ValueError("A polygon needs at least three vertices")
        points = []
        for vertex in vertices:
            if not isinstance(vertex, list) or len(vertex) != 2:
                raise ValueError("Each vertex must be an [x, y] pair")
            x, y = vertex
            if any(type(v) not in (int, float) or not math.isfinite(v) for v in vertex):
                raise ValueError("Polygon coordinates must be finite numbers")
            if not 0 <= x <= width or not 0 <= y <= height:
                raise ValueError(f"Out-of-bounds polygon vertex: {vertex}")
            points.append((float(x), float(y)))
        area_twice = sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2)
                         in zip(points, points[1:] + points[:1], strict=True))
        if len(set(points)) < 3 or abs(area_twice) < 1e-8:
            raise ValueError("Polygon is degenerate or has zero signed area")
        polygons.append(Polygon(label, tuple(points)))
    return polygons


def rasterize(polygons: list[Polygon], size: tuple[int, int]) -> tuple[PILImage, int]:
    """Return uint8 IDs and overlap counts using Pillow's inclusive polygon fill.

    Later shapes overwrite earlier shapes, matching JSON order. Overlap counts
    count pixels covered by two or more polygons, including same-class overlap.
    """
    import numpy as np
    from PIL import Image, ImageDraw

    width, height = size
    mask = Image.new("L", size, 0)
    coverage = np.zeros((height, width), dtype=np.uint32)
    for polygon in polygons:
        points = [(min(x, width - 1), min(y, height - 1)) for x, y in polygon.points]
        layer = Image.new("1", size, 0)
        ImageDraw.Draw(layer).polygon(points, fill=1)
        coverage += np.asarray(layer, dtype=np.uint32)
        ImageDraw.Draw(mask).polygon(points, fill=LABEL_TO_ID[polygon.label])
    values = np.asarray(mask)
    if not set(np.unique(values)).issubset({0, *LABEL_TO_ID.values()}):
        raise ValueError("Generated mask contains unexpected class IDs")
    return mask, int(np.count_nonzero(coverage > 1))
