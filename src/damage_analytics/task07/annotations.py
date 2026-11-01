"""DamSegment's actual COCO-like JSON and YOLO schemas, not inferred from folder names."""

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from damage_analytics.curation.geometry import redundant_vertices, self_intersections, signed_area


def validate_json(value: dict, size: tuple[int, int], filename: str, *, polygons: bool) -> dict:
    """Validate XYWH boxes and optional simple polygon rings without repairing raw annotations."""
    if value.get("image") != {"file_name": filename, "width": size[0], "height": size[1]}:
        raise ValueError("JSON image filename/dimensions differ")
    annotations = value.get("annotations")
    if not isinstance(annotations, list):
        raise ValueError("Missing annotation list")
    counts = Counter()
    for annotation in annotations:
        if not isinstance(annotation, dict):
            raise ValueError("Annotation entries must be objects")
        category = annotation.get("category_id")
        if type(category) is not int or category not in (0, 1):
            raise ValueError("Unexpected source category ID")
        box = annotation.get("bbox", [])
        if (
            not isinstance(box, list)
            or len(box) != 4
            or not all(type(x) in (int, float) and math.isfinite(x) for x in box)
        ):
            raise ValueError("Malformed/nonfinite XYWH box")
        x, y, width, height = box
        if min(x, y) < 0 or min(width, height) <= 0 or x + width > size[0] or y + height > size[1]:
            raise ValueError("Out-of-bounds or empty box")
        if annotation.get("iscrowd", 0) != 0:
            raise ValueError("Crowd/RLE annotations are unsupported")
        if "area" in annotation and not math.isclose(
            annotation["area"], width * height, abs_tol=0.01
        ):
            raise ValueError("Detection area does not match XYWH")
        if polygons:
            rings = annotation.get("segmentation")
            if not isinstance(rings, list) or not rings:
                raise ValueError("Missing polygon segmentation")
            for ring in rings:
                if not isinstance(ring, list) or len(ring) < 6 or len(ring) % 2:
                    raise ValueError("Malformed polygon coordinates")
                if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in ring):
                    raise ValueError("Nonfinite polygon")
                points = redundant_vertices(list(zip(ring[::2], ring[1::2], strict=True)))
                if any(not 0 <= a <= size[0] or not 0 <= b <= size[1] for a, b in points):
                    raise ValueError("Polygon outside native image")
                if (
                    len(points) < 3
                    or abs(signed_area(points)) == 0
                    or self_intersections(points, 1e-8)
                ):
                    raise ValueError("Degenerate or self-intersecting polygon")
                if any(a < x or b < y or a > x + width or b > y + height for a, b in points):
                    raise ValueError("Polygon exceeds its bounding box")
        counts[category] += 1
    return {"objects": dict(counts), "empty": not annotations}


def validate_yolo(text: str, *, polygons: bool) -> Counter:
    """Validate normalized boxes/polygons; keep the two supplied formats distinct."""
    counts = Counter()
    for line in text.splitlines():
        if not line.strip():
            continue
        cells = line.split()
        category = int(cells[0])
        values = list(map(float, cells[1:]))
        if category not in (0, 1) or any(not math.isfinite(v) or not 0 <= v <= 1 for v in values):
            raise ValueError("Invalid YOLO class/normalized coordinate")
        if polygons:
            if len(values) < 6 or len(values) % 2:
                raise ValueError("Invalid YOLO polygon")
            points = redundant_vertices(list(zip(values[::2], values[1::2], strict=True)))
            if (
                len(points) < 3
                or abs(signed_area(points)) == 0
                or self_intersections(points, 1e-10)
            ):
                raise ValueError("Invalid YOLO polygon geometry")
        else:
            if len(values) != 4:
                raise ValueError("Expected YOLO center-x/center-y/width/height")
            x, y, width, height = values
            if (
                min(width, height) <= 0
                or min(x - width / 2, y - height / 2) < -1e-6
                or max(x + width / 2, y + height / 2) > 1 + 1e-6
            ):
                raise ValueError("YOLO box outside image")
        counts[category] += 1
    return counts


def decode_mask(image: Image.Image) -> np.ndarray:
    """Explicit local mapping: black→0 background, red→1 source 0, blue→2 source 1."""
    if image.mode not in ("RGB", "P") or "transparency" in image.info:
        raise ValueError("Expected opaque RGB/palette DamSegment mask")
    rgb = np.asarray(image.convert("RGB"))
    result = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    for color, identifier in (((0, 0, 0), 0), ((255, 0, 0), 1), ((0, 0, 255), 2)):
        result[np.all(rgb == color, axis=2)] = identifier
    if np.any(result == 255):
        raise ValueError("Unexpected mask color; no nearest-color guessing")
    return result


def polygon_mask_comparison(value: dict, mask: np.ndarray) -> dict:
    """Report native raster disagreement; source PNG remains authoritative, not repainted."""
    raster = Image.new("L", (mask.shape[1], mask.shape[0]))
    draw = ImageDraw.Draw(raster)
    for item in value["annotations"]:
        for ring in item["segmentation"]:
            draw.polygon(
                list(zip(ring[::2], ring[1::2], strict=True)), fill=item["category_id"] + 1
            )
    reference = np.asarray(raster)
    overlaps = []
    for identifier in (1, 2):
        union = np.count_nonzero((reference == identifier) | (mask == identifier))
        intersection = np.count_nonzero((reference == identifier) & (mask == identifier))
        overlaps.append(intersection / union if union else None)
    return {
        "disagreement_pixels": int(np.count_nonzero(reference != mask)),
        "class_iou": overlaps,
        "policy": "Supplied PNG authoritative; boundary/format disagreement is documented",
    }


def read_annotation(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError("Annotation JSON must be an object")
    return value
