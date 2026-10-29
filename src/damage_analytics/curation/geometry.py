"""Simple-ring geometry checks using explicit segment intersections, without repair."""

import numpy as np


def redundant_vertices(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Remove only exactly consecutive duplicate vertices and repeated closing points."""
    cleaned = []
    for point in points:
        if not cleaned or point != cleaned[-1]:
            cleaned.append(point)
    if len(cleaned) > 1 and cleaned[0] == cleaned[-1]:
        cleaned.pop()
    return cleaned


def signed_area(points: list[tuple[float, float]]) -> float:
    """Return signed shoelace area; winding direction is not a class signal."""
    array = np.asarray(points, dtype=np.float64)
    return float(
        np.sum(array[:, 0] * np.roll(array[:, 1], -1) - array[:, 1] * np.roll(array[:, 0], -1)) / 2
    )


def self_intersections(points: list[tuple[float, float]], epsilon: float) -> int:
    """Count intersections/touches of nonadjacent edges and adjacent backtracking.

    Adjacent edges may share their endpoint, but may not retrace an edge. Bounding
    boxes and orientation tests include collinear touches. Epsilon is expressed
    in pixel coordinates for bounds and pixel-squared units for cross products.
    This is a simple-ring validator, not a general polygon topology repair engine.
    """
    a = np.asarray(points, dtype=np.float64)
    b = np.roll(a, -1, axis=0)
    indices = np.arange(len(a))
    left, right = np.triu_indices(len(a), 1)
    adjacent = (right == left + 1) | ((left == 0) & (right == len(a) - 1))
    i, j = left[~adjacent], right[~adjacent]
    p, q, r, s = a[i], b[i], a[j], b[j]

    def cross(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        return u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]

    c1, c2 = cross(q - p, r - p), cross(q - p, s - p)
    c3, c4 = cross(s - r, p - r), cross(s - r, q - r)
    bounds = np.all(
        np.maximum(np.minimum(p, q), np.minimum(r, s))
        <= np.minimum(np.maximum(p, q), np.maximum(r, s)) + epsilon,
        axis=1,
    )
    straddle1 = ((c1 <= epsilon) & (c2 >= -epsilon)) | ((c2 <= epsilon) & (c1 >= -epsilon))
    straddle2 = ((c3 <= epsilon) & (c4 >= -epsilon)) | ((c4 <= epsilon) & (c3 >= -epsilon))
    incoming, outgoing = a - a[(indices - 1) % len(a)], b - a
    retraced = (np.abs(cross(incoming, outgoing)) <= epsilon) & (
        np.sum(incoming * outgoing, axis=1) < -epsilon
    )
    return int(np.count_nonzero(bounds & straddle1 & straddle2) + np.count_nonzero(retraced))
