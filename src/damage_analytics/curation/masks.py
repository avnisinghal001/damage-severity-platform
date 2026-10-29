"""Cross-label conflict detection and pixel-level invariants around Task 01 rasterization."""

import numpy as np
from PIL import Image, ImageDraw

from damage_analytics.annotations import Polygon, rasterize
from damage_analytics.labels import CLASS_NAMES, LABEL_TO_ID


def checked_mask(polygons: list[Polygon], size: tuple[int, int]) -> tuple[Image.Image, dict]:
    """Reuse Task 01's rasterizer, independently check label coverage using bit planes.

    Multiple same-label polygons form a union. Multi-label covered pixels are
    conflicts, including shared rasterized boundaries; no automatic precedence
    justification is inferred. The caller selects the explicit conflict policy.
    """
    mask, overlap = rasterize(polygons, size)
    values = np.asarray(mask)
    bits = np.zeros((size[1], size[0]), dtype=np.uint16)
    for polygon in polygons:
        layer = Image.new("1", size)
        points = [(min(x, size[0] - 1), min(y, size[1] - 1)) for x, y in polygon.points]
        ImageDraw.Draw(layer).polygon(points, fill=1)
        bits[np.asarray(layer)] |= 1 << LABEL_TO_ID[polygon.label]
    foreground = bits != 0
    conflicting = (bits & (bits - np.uint16(1))) != 0
    if values.shape != (size[1], size[0]) or values.dtype != np.uint8:
        raise ValueError("Mask shape/dtype invariant failed")
    if not np.array_equal(values != 0, foreground):
        raise ValueError("Background does not equal polygon-coverage complement")
    if values.max(initial=0) >= len(CLASS_NAMES):
        raise ValueError("Unknown mask ID")
    for class_id in LABEL_TO_ID.values():
        single_class = bits == (1 << class_id)
        if not np.all(values[single_class] == class_id):
            raise ValueError("Mask class does not match unambiguous polygon coverage")
    counts = np.bincount(values.ravel(), minlength=len(CLASS_NAMES)).tolist()
    if sum(counts) != size[0] * size[1]:
        raise ValueError("Pixel count conservation failed")
    return mask, {
        "overlap_pixels": overlap,
        "conflict_pixels": int(np.count_nonzero(conflicting)),
        "foreground_pixels": int(np.count_nonzero(foreground)),
        "pixel_counts": counts,
    }


def mask_thumbnail(mask: Image.Image) -> list[int]:
    """Small nearest-neighbor class-ID grid for annotation-similarity review only."""
    return np.asarray(mask.resize((32, 32), Image.Resampling.NEAREST)).ravel().tolist()


def annotation_similarity(left: list[int], right: list[int]) -> float:
    """Mean class IoU on nonempty foreground unions; empty/empty is not evidence."""
    a, b = np.asarray(left), np.asarray(right)
    labels = (set(a.tolist()) | set(b.tolist())) - {0}
    if not labels:
        return 0.0
    return float(
        np.mean(
            [
                np.count_nonzero((a == label) & (b == label))
                / np.count_nonzero((a == label) | (b == label))
                for label in labels
            ]
        )
    )
