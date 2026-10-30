"""Exact run-length connected components without SciPy/OpenCV dependencies."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Component:
    """Pixel runs, exact area and exclusive-maximum bounding box in image coordinates."""

    runs: tuple[tuple[int, int, int], ...]  # y, left inclusive, right exclusive
    area: int
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]  # mean pixel-index coordinates, x then y


def connected_components(binary: np.ndarray, connectivity: int = 8) -> list[Component]:
    """Label runs and merge adjacent-row overlaps; diagonal contact joins only for 8-connectivity.

    This counts mask-connected semantic regions, not physical damage instances.
    Work scales with row runs instead of Python objects for every foreground pixel.
    """
    if binary.ndim != 2 or connectivity not in (4, 8):
        raise ValueError("Expected a 2D mask and connectivity 4/8")
    parents, runs, previous = [], [], []

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for y, row in enumerate(binary.astype(bool, copy=False)):
        changes = np.diff(np.pad(row.astype(np.int8), (1, 1)))
        starts, ends = np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)
        current = []
        pointer = 0
        margin = int(connectivity == 8)
        for left, right in zip(starts, ends, strict=True):
            left, right = int(left), int(right)
            label = len(parents)
            parents.append(label)
            runs.append((y, left, right))
            while pointer < len(previous) and previous[pointer][1] <= left - margin:
                pointer += 1
            scan = pointer
            while scan < len(previous):
                old_left, old_right, old_label = previous[scan]
                if old_left >= right + margin:
                    break
                if old_right > left - margin:
                    a, b = find(label), find(old_label)
                    parents[max(a, b)] = min(a, b)
                scan += 1
            current.append((left, right, label))
        previous = current
    grouped: dict[int, list[tuple[int, int, int]]] = {}
    for label, run in enumerate(runs):
        grouped.setdefault(find(label), []).append(run)
    components = []
    for group in grouped.values():
        area = sum(right - left for _, left, right in group)
        sum_x = sum((left + right - 1) * (right - left) / 2 for _, left, right in group)
        sum_y = sum(y * (right - left) for y, left, right in group)
        bbox = (
            min(left for _, left, _ in group),
            min(y for y, _, _ in group),
            max(right for _, _, right in group),
            max(y for y, _, _ in group) + 1,
        )
        components.append(Component(tuple(group), area, bbox, (sum_x / area, sum_y / area)))
    return sorted(components, key=lambda value: (value.bbox[1], value.bbox[0]))


def component_values(values: np.ndarray, component: Component) -> np.ndarray:
    """Extract run values for small-label sets or per-component descriptive score statistics."""
    return np.concatenate([values[y, left:right] for y, left, right in component.runs])
