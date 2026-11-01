"""Native-resolution, per-class damage descriptors; filtering never changes raw coverage."""

from dataclasses import asdict

import numpy as np

from damage_analytics.analytics.components import component_values, connected_components
from damage_analytics.analytics.regions import validate_mask
from damage_analytics.labels import CLASS_NAMES
from damage_analytics.task09.config import FeaturePolicy
from damage_analytics.task09.geometry import geometric_features
from damage_analytics.task09.schema import SCHEMA_VERSION, validate_report


def score_statistics(
    maximum: np.ndarray | None, entropy: np.ndarray | None, policy: FeaturePolicy
) -> dict:
    if maximum is None:
        return {
            "available": False,
            "mean_maximum_softmax": None,
            "mean_normalized_entropy": None,
            "low_score_fraction": None,
            "high_entropy_fraction": None,
        }
    return {
        "available": True,
        "mean_maximum_softmax": float(maximum.mean()) if maximum.size else None,
        "mean_normalized_entropy": float(entropy.mean()) if entropy.size else None,
        "low_score_fraction": float(np.mean(maximum < policy.maximum_softmax_below))
        if maximum.size
        else None,
        "high_entropy_fraction": float(np.mean(entropy > policy.normalized_entropy_above))
        if entropy.size
        else None,
    }


def extract_features(
    mask: np.ndarray,
    policy: FeaturePolicy,
    *,
    maximum_softmax: np.ndarray | None = None,
    normalized_entropy: np.ndarray | None = None,
    model: dict | None = None,
    coordinate_frame: str = "supplied_mask_grid",
) -> tuple[dict, np.ndarray]:
    """Return validated JSON and a separately filtered ID mask; source data stays unchanged."""
    validate_mask(mask)
    policy.validate()
    if (maximum_softmax is None) != (normalized_entropy is None):
        raise ValueError("Provide both uncertainty maps or neither")
    for values in (maximum_softmax, normalized_entropy):
        if values is not None and (
            values.shape != mask.shape
            or not np.isfinite(values).all()
            or np.any((values < 0) | (values > 1))
        ):
            raise ValueError("Uncertainty maps must match original mask coordinates and [0,1]")
    filtered = mask.astype(np.uint8, copy=True)
    counts = np.bincount(mask.ravel().astype(np.int64), minlength=9).tolist()
    classes, candidates, regions = [], [], []
    for class_id in range(1, 9):
        components = connected_components(mask == class_id, policy.connectivity)
        threshold = (policy.minimum_by_class or {}).get(
            str(class_id), policy.minimum_region_pixels
        )
        kept = []
        for component in components:
            if component.area < threshold:
                for y, left, right in component.runs:
                    filtered[y, left:right] = 0
            else:
                kept.append(component)
                candidates.append((class_id, component))
        retained = sum(c.area for c in kept)
        selected = mask == class_id
        scores = score_statistics(
            maximum_softmax[selected] if maximum_softmax is not None else None,
            normalized_entropy[selected] if normalized_entropy is not None else None,
            policy,
        )
        classes.append(
            {
                "class_id": class_id,
                "label": CLASS_NAMES[class_id],
                "raw_pixels": counts[class_id],
                "image_percent": 100 * counts[class_id] / mask.size,
                "retained_pixels": retained,
                "filtered_pixels": counts[class_id] - retained,
                "raw_region_count": len(components),
                "retained_region_count": len(kept),
                "filtered_region_count": len(components) - len(kept),
                "minimum_region_pixels": threshold,
                "reported_region_count": 0,
                "unreported_retained_regions": len(kept),
                "unreported_retained_pixels": retained,
                "score_statistics": scores,
            }
        )
    candidates.sort(key=lambda item: (-item[1].area, item[0], item[1].bbox[1], item[1].bbox[0]))
    for number, (class_id, component) in enumerate(
        candidates[: policy.maximum_reported_regions], 1
    ):
        scores = score_statistics(
            component_values(maximum_softmax, component) if maximum_softmax is not None else None,
            component_values(normalized_entropy, component)
            if normalized_entropy is not None
            else None,
            policy,
        )
        regions.append(
            {
                "region_id": f"r{number:05d}",
                "class_id": class_id,
                "label": CLASS_NAMES[class_id],
                "area_pixels": component.area,
                "image_percent": 100 * component.area / mask.size,
                **geometric_features(component),
                "score_statistics": scores,
            }
        )
        row = classes[class_id - 1]
        row["reported_region_count"] += 1
        row["unreported_retained_regions"] -= 1
        row["unreported_retained_pixels"] -= component.area
    foreground = mask != 0
    scores = score_statistics(
        maximum_softmax[foreground] if maximum_softmax is not None else None,
        normalized_entropy[foreground] if normalized_entropy is not None else None,
        policy,
    )
    retained_pixels = int(np.count_nonzero(filtered))
    report = {
        "schema_version": SCHEMA_VERSION,
        "class_names": list(CLASS_NAMES),
        "image": {
            "width": mask.shape[1],
            "height": mask.shape[0],
            "coordinate_frame": coordinate_frame,
        },
        "model": model
        or {"name": "unknown_for_mask_only", "version": "unknown", "checkpoint_sha256": None},
        "coverage": {
            "raw_class_pixels": counts,
            "foreground_pixels": int(foreground.sum()),
            "foreground_image_percent": 100 * int(foreground.sum()) / mask.size,
            "retained_foreground_pixels": retained_pixels,
            "filtered_pixels": int(foreground.sum()) - retained_pixels,
        },
        "classes": classes,
        "regions": regions,
        "filtering": {
            **asdict(policy),
            "raw_mask_unchanged": True,
            "region_table_truncated": len(regions) < len(candidates),
            "meaning": "Below-size regions are reporting candidates, not confirmed artifacts; "
            "raw totals and evaluation are unfiltered. Thin cracks may be small.",
        },
        "uncertainty": {
            **scores,
            "calibrated": False,
            "status": "uncalibrated_descriptive_scores"
            if maximum_softmax is not None
            else "unknown_without_scores",
            "domain_assessment": "unknown_no_evaluated_OOD_mechanism",
        },
        "units": {
            "area": "image_pixels",
            "length": "pixel_grid_units",
            "physical_scale": None,
            "bbox": "[x_min,y_min,x_max_exclusive,y_max_exclusive]",
            "centroid": "pixel-center coordinates: upper-left pixel center is (0.5,0.5)",
            "axes": "4*sqrt(population covariance eigenvalue); not physical crack length/width",
        },
        "interpretation": "Predicted semantic image coverage and connected regions, not "
        "physical damage "
        "instances, validated severity, structural condition or safety scores",
    }
    validate_report(report)
    return report, filtered
