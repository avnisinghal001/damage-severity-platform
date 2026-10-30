"""Assess existing spalling-category supervision using train/validation groups, never test."""

import json
from collections import Counter, defaultdict

import numpy as np
from PIL import Image

from damage_analytics.analytics.components import component_values, connected_components
from damage_analytics.analytics.config import AnalyticsConfig, runtime_training_config
from damage_analytics.analytics.severity import SUPPORTED_SPALLING, spalling_assessment
from damage_analytics.io import write_csv, write_json
from damage_analytics.training.manifests import open_release


def assess_supervision(config: AnalyticsConfig) -> dict:
    """Count pure/mixed/unsupported foreground components and independent-group support.

    Pure spalling regions provide source-category targets for a possible two-class
    crop experiment. Region fragments are not independent observations, and these
    targets share lineage with segmentation rather than independently rating severity.
    No classifier is created or trained, regardless of the planning-count gate.
    """
    release = open_release(runtime_training_config(config))
    candidates, complex_audit, summaries = [], [], {}
    for split in ("train", "validation"):
        statuses = Counter()
        groups, images = defaultdict(set), defaultdict(set)
        counts = Counter()
        for sample in release.samples(split):
            with Image.open(release.file(sample.mask)) as source:
                values = np.array(source)
            for index, component in enumerate(
                connected_components(values != 0, config.regions.connectivity)
            ):
                ids = np.unique(component_values(values, component)).tolist()
                assessment = spalling_assessment(ids)
                statuses[assessment["status"]] += 1
                complex_audit.append(
                    {
                        "split": split,
                        "sample_id": sample.sample_id,
                        "source_group": sample.group_id,
                        "component_index": index,
                        "class_ids": json.dumps(ids),
                        "assessment_status": assessment["status"],
                        "eligible_category_label": assessment["label"],
                        "area_pixels": component.area,
                        "bbox_xyxy_exclusive": json.dumps(component.bbox),
                    }
                )
                if assessment["status"] != "source_category_derived_unvalidated":
                    continue
                class_id = ids[0]
                counts[class_id] += 1
                groups[class_id].add(sample.group_id)
                images[class_id].add(sample.sample_id)
                candidates.append(
                    {
                        "split": split,
                        "sample_id": sample.sample_id,
                        "source_group": sample.group_id,
                        "component_index": index,
                        "source_class_id": class_id,
                        "source_label": SUPPORTED_SPALLING[class_id],
                        "area_pixels": component.area,
                        "bbox_x0": component.bbox[0],
                        "bbox_y0": component.bbox[1],
                        "bbox_x1_exclusive": component.bbox[2],
                        "bbox_y1_exclusive": component.bbox[3],
                        "label_lineage": "Existing segmentation pixel category, not independent "
                        "severity annotation",
                    }
                )
        summaries[split] = {
            "samples_audited": len(release.samples(split)),
            "complex_status_counts": dict(statuses),
            "eligible_pure_category_regions": [
                {
                    "class_id": class_id,
                    "source_label": label,
                    "regions": counts[class_id],
                    "images": len(images[class_id]),
                    "known_groups": len(groups[class_id]),
                }
                for class_id, label in SUPPORTED_SPALLING.items()
            ],
        }
    train_min = config.supervision.minimum_training_groups
    val_min = config.supervision.minimum_validation_groups
    gate = all(
        row["known_groups"] >= threshold
        for split, threshold in (("train", train_min), ("validation", val_min))
        for row in summaries[split]["eligible_pure_category_regions"]
    )
    result = {
        "dataset": release.identity(),
        "splits": summaries,
        "planning_group_gate": {
            "passed": gate,
            "minimum_training_groups": train_min,
            "minimum_validation_groups": val_min,
            "meaning": "Exploratory planning heuristic, not statistical power evidence",
        },
        "independent_region_severity_classifier": {
            "implemented_or_trained": False,
            "verdict": "not_justified_by_available_supervision",
            "reasons": [
                "No independent region-level severity judgments or "
                "clinical/engineering outcome targets.",
                "Minor/Moderate source categories can support category "
                "discrimination, not global severity.",
                "Multiple regions from the same image/group are "
                "correlated, not extra independent samples.",
                "Mixed/unsupported complexes are not defensible binary targets.",
                "Source identity remains incomplete; small "
                "validation-group support limits reliability.",
            ],
        },
        "possible_future_scope": "A separately approved RGB-region Minor-vs-Moderate "
        "category experiment "
        "with known groups, expert audit, independent evaluation and no mask-label leakage",
        "test_masks_decoded": False,
        "overall_structural_severity": "not_assessed",
    }
    folder = config.output_dir / "severity_supervision"
    write_json(folder / "assessment.json", result)
    fields = [
        "split",
        "sample_id",
        "source_group",
        "component_index",
        "source_class_id",
        "source_label",
        "area_pixels",
        "bbox_x0",
        "bbox_y0",
        "bbox_x1_exclusive",
        "bbox_y1_exclusive",
        "label_lineage",
    ]
    write_csv(folder / "category_region_candidates.csv", candidates, fields)
    write_csv(
        folder / "complex_audit.csv",
        complex_audit,
        [
            "split",
            "sample_id",
            "source_group",
            "component_index",
            "class_ids",
            "assessment_status",
            "eligible_category_label",
            "area_pixels",
            "bbox_xyxy_exclusive",
        ],
    )
    return result
