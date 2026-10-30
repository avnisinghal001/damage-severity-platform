"""Image-space area, class components, mixed damage complexes and uncalibrated score summaries."""

from dataclasses import asdict

import numpy as np

from damage_analytics.analytics.components import Component, component_values, connected_components
from damage_analytics.analytics.config import RegionPolicy, UncertaintyPolicy
from damage_analytics.analytics.severity import spalling_assessment
from damage_analytics.labels import CLASS_NAMES


def validate_mask(mask: np.ndarray) -> None:
    """Accept native 2D integer class IDs only; no RGB colors, padding or floating targets."""
    if (
        mask.ndim != 2
        or not mask.size
        or not np.issubdtype(mask.dtype, np.integer)
        or np.any((mask < 0) | (mask >= len(CLASS_NAMES)))
    ):
        raise ValueError("Analytics requires a nonempty 2D integer mask with IDs 0–8")


def analyze_mask(
    mask: np.ndarray,
    regions: RegionPolicy,
    uncertainty: UncertaintyPolicy,
    *,
    maximum_softmax: np.ndarray | None = None,
    normalized_entropy: np.ndarray | None = None,
) -> dict:
    """Measure native mask pixels without physical units or morphological alteration.

    Class components may touch other categories. Foreground complexes connect all
    damage pixels and expose mixed semantics rather than collapsing them to a grade.
    Minimum reported area filters tables only; totals and component counts remain exact.
    """
    validate_mask(mask)
    if regions.connectivity not in (4, 8) or regions.minimum_reported_pixels < 1:
        raise ValueError("Invalid region policy")
    if (maximum_softmax is None) != (normalized_entropy is None):
        raise ValueError("Provide both score maps or neither")
    for scores in (maximum_softmax, normalized_entropy):
        if scores is not None and (
            scores.shape != mask.shape
            or not np.isfinite(scores).all()
            or np.any((scores < 0) | (scores > 1))
        ):
            raise ValueError("Score maps must align with mask and be finite in [0,1]")
    height, width = mask.shape
    total = mask.size
    complexes = connected_components(mask != 0, regions.connectivity)
    complex_ids = np.zeros(mask.shape, dtype=np.int32)
    complex_rows = []
    for number, component in enumerate(complexes, 1):
        for y, left, right in component.runs:
            complex_ids[y, left:right] = number
        ids = np.unique(component_values(mask, component)).tolist()
        row = _region_row(component, mask.shape)
        scores = _score_summary(component, maximum_softmax, normalized_entropy, uncertainty)
        row.update(
            {
                "complex_id": number,
                "class_ids": ids,
                "uncertainty": scores,
                "spalling_assessment": spalling_assessment(
                    ids, uncertain_prediction=scores["status"] != "uncalibrated_scores_available"
                ),
            }
        )
        complex_rows.append(row)
    counts = np.bincount(mask.ravel().astype(np.int64), minlength=len(CLASS_NAMES))
    classes, component_rows = [], []
    for class_id, name in enumerate(CLASS_NAMES):
        parts = connected_components(mask == class_id, regions.connectivity) if class_id else []
        reported = [part for part in parts if part.area >= regions.minimum_reported_pixels]
        classes.append(
            {
                "class_id": class_id,
                "source_label": name,
                "area_pixels": int(counts[class_id]),
                "image_area_percent": 100 * int(counts[class_id]) / total,
                "connected_regions": len(parts) if class_id else None,
                "reported_regions": len(reported) if class_id else None,
                "unreported_region_pixels": sum(
                    part.area for part in parts if part.area < regions.minimum_reported_pixels
                ),
            }
        )
        for component in reported:
            row = _region_row(component, mask.shape)
            y, left, _ = component.runs[0]
            scores = _score_summary(component, maximum_softmax, normalized_entropy, uncertainty)
            row.update(
                {
                    "region_id": len(component_rows) + 1,
                    "class_id": class_id,
                    "source_label": name,
                    "complex_id": int(complex_ids[y, left]),
                    "uncertainty": scores,
                    "spalling_assessment": spalling_assessment(
                        [class_id],
                        uncertain_prediction=scores["status"] != "uncalibrated_scores_available",
                    ),
                }
            )
            component_rows.append(row)
    if sum(row["area_pixels"] for row in classes) != total:
        raise ValueError("Class-area conservation failed")
    return {
        "schema_version": 1,
        "measurement_space": "image_pixels",
        "image": {"width": width, "height": height, "area_pixels": total},
        "region_policy": asdict(regions),
        "classes": classes,
        "regions": component_rows,
        "damage_complexes": complex_rows,
        "spalling_summary": spalling_assessment(
            (np.flatnonzero(counts[1:]) + 1).tolist(),
            uncertain_prediction=maximum_softmax is None
            or any(
                row["spalling_assessment"]["status"] == "uncertain_category_prediction"
                for row in complex_rows
            ),
        ),
        "uncertainty": {
            "calibrated": False,
            "review_thresholds": asdict(uncertainty),
            "status": "unknown_no_model_scores"
            if maximum_softmax is None
            else "uncalibrated_softmax_and_entropy_only",
        },
        "physical_measurements": {
            "available": False,
            "reason": "No scale, geometry or calibration",
        },
        "overall_structural_severity": {"status": "not_assessed", "score": None},
        "limitations": [
            "Connected mask components are not verified physical instances.",
            "Image-area percentages are not physical damage fractions or severity scores.",
            "Source-category spalling labels are not an independently trained severity model.",
            "Background/unobserved damage is not evidence of structural safety.",
        ],
    }


def _region_row(component: Component, shape: tuple[int, int]) -> dict:
    """Use integer pixel area, exclusive bbox maxima and pixel-index centroid coordinates."""
    x0, y0, x1, y1 = component.bbox
    return {
        "area_pixels": component.area,
        "image_area_percent": 100 * component.area / (shape[0] * shape[1]),
        "bbox_xyxy_exclusive": list(component.bbox),
        "bbox_width_pixels": x1 - x0,
        "bbox_height_pixels": y1 - y0,
        "bbox_fill_fraction": component.area / ((x1 - x0) * (y1 - y0)),
        "centroid_xy_pixel_index": list(component.centroid),
        "touches_image_boundary": x0 == 0 or y0 == 0 or x1 == shape[1] or y1 == shape[0],
    }


def _score_summary(
    component: Component,
    maximum: np.ndarray | None,
    entropy: np.ndarray | None,
    policy: UncertaintyPolicy,
) -> dict:
    """Scores remain uncalibrated even if no heuristic review threshold is triggered."""
    if maximum is None:
        return {
            "status": "unknown_no_model_scores",
            "calibrated": False,
            "mean_maximum_softmax": None,
            "mean_normalized_entropy": None,
            "review_flags": [],
        }
    probability_mean = float(component_values(maximum, component).mean())
    entropy_mean = float(component_values(entropy, component).mean())
    flags = []
    if probability_mean < policy.maximum_softmax_below:
        flags.append("low_uncalibrated_maximum_softmax")
    if entropy_mean > policy.normalized_entropy_above:
        flags.append("high_uncalibrated_entropy")
    return {
        "status": "review_requested" if flags else "uncalibrated_scores_available",
        "calibrated": False,
        "mean_maximum_softmax": probability_mean,
        "mean_normalized_entropy": entropy_mean,
        "review_flags": flags,
    }
