"""Exact pixel-grid perimeter and population-moment descriptors from existing RLE regions."""

import math
from collections import defaultdict

import numpy as np

from damage_analytics.analytics.components import Component


def geometric_features(component: Component) -> dict:
    """Axes describe pixel-center dispersion, not physical crack length or thickness."""
    area, sx, sy, sxx, syy, sxy, horizontal = component.area, 0.0, 0.0, 0.0, 0.0, 0.0, 0
    rows = defaultdict(list)

    def square_sum(n: int) -> int:
        return n * (n + 1) * (2 * n + 1) // 6 if n >= 0 else 0

    for y, left, right in component.runs:
        n = right - left
        xx = (left + right - 1) * n / 2
        sx += xx
        sy += y * n
        sxx += square_sum(right - 1) - square_sum(left - 1)
        syy += y * y * n
        sxy += y * xx
        horizontal += n - 1
        rows[y].append((left, right))
    vertical = 0
    for y, current in rows.items():
        previous, pointer = rows.get(y - 1, []), 0
        for left, right in current:
            while pointer < len(previous) and previous[pointer][1] <= left:
                pointer += 1
            index = pointer
            while index < len(previous) and previous[index][0] < right:
                a, b = previous[index]
                vertical += max(0, min(right, b) - max(left, a))
                index += 1
    perimeter = 4 * area - 2 * (horizontal + vertical)
    covariance = np.array(
        [
            [sxx / area - (sx / area) ** 2, sxy / area - sx * sy / area**2],
            [sxy / area - sx * sy / area**2, syy / area - (sy / area) ** 2],
        ]
    )
    values, vectors = np.linalg.eigh(covariance)
    minor, major = np.maximum(values, 0).tolist()
    vector = vectors[:, 1]
    angle = (
        (math.degrees(math.atan2(vector[1], vector[0])) + 90) % 180 - 90
        if major - minor > 1e-10
        else None
    )
    x1, y1, x2, y2 = component.bbox
    return {
        "bbox_xyxy": [x1, y1, x2, y2],
        "bbox_width_pixels": x2 - x1,
        "bbox_height_pixels": y2 - y1,
        "centroid_xy": [sx / area + 0.5, sy / area + 0.5],
        "perimeter_pixel_edges": perimeter,
        "bbox_fill_fraction": area / ((x2 - x1) * (y2 - y1)),
        "equivalent_disk_diameter_pixels": math.sqrt(4 * area / math.pi),
        "moment_major_axis_pixels": 4 * math.sqrt(major),
        "moment_minor_axis_pixels": 4 * math.sqrt(minor),
        "moment_axis_ratio": math.sqrt(major / minor) if minor > 1e-12 else None,
        "orientation_degrees_image_xy": angle,
        "eccentricity": math.sqrt(max(0, 1 - minor / major)) if major > 0 else None,
        "compactness_pixel_grid": 4 * math.pi * area / perimeter**2,
    }
