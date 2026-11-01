"""Concise scientific handover reports, populated only from executed audit/evaluation evidence."""

from pathlib import Path

import numpy as np

from damage_analytics.io import write_json
from damage_analytics.labels import CLASS_NAMES


def ratio(value: float | None) -> str:
    return "undefined" if value is None else f"{value:.2%}"


def summarize(output: Path, audit: dict, evaluations: dict[str, dict]) -> None:
    diagnostics = {}
    for name, result in evaluations.items():
        matrix = np.asarray(result["metrics"]["confusion_matrix"])
        rows = []
        for class_id in range(1, 9):
            missed, false_positive = matrix[class_id].copy(), matrix[:, class_id].copy()
            missed[class_id] = false_positive[class_id] = 0
            fn_id, fp_id = int(missed.argmax()), int(false_positive.argmax())
            rows.append(
                {
                    "class_id": class_id,
                    "label": CLASS_NAMES[class_id],
                    "dominant_missed_as": CLASS_NAMES[fn_id],
                    "missed_pixels": int(missed[fn_id]),
                    "dominant_false_positive_source": CLASS_NAMES[fp_id],
                    "false_positive_pixels_from_source": int(false_positive[fp_id]),
                }
            )
        diagnostics[name] = rows
    write_json(output / "poor_class_diagnostics.json", diagnostics)
    lines = [
        "# Task 09: RC1841 audit and segmentation evaluation",
        "",
        "All artifacts are isolated under Task 09. Dataset releases/checkpoints/splits are "
        "unchanged. "
        "Test membership/hash metadata is used only for integrity QA. No test pixels, test "
        "annotations "
        "or final-test results were read. Task 08 training code and device allocation were "
        "not changed.",
        "",
        "## Dataset verification",
        "",
        f"Split counts: {audit['split_counts_metadata']}. Fresh image/mask/polygon raster checks: "
        f"{audit['decoded_and_rerasterized']}. Fresh issues: {len(audit['issues'])}; known "
        f"exact/group "
        f"crossings: {audit['known_exact_or_group_crossings']}.",
        "",
        f"Historical release exclusions: {audit['prior_release_exclusions']}; reasons "
        f"{audit['prior_exclusion_reasons']}. Default training decoding is bounded; full training "
        "decoding is a separate explicit command. All development class counts come from "
        "sealed manifests. Exclusion reasons are nonexclusive and must not be summed as samples.",
        "",
        f"Validation class occurrences entirely lost on the model grid at the configured resize: "
        f"{audit['entire_class_occurrences_lost']}. Ground-truth resize roundtrip is a sampling "
        "diagnostic, not model performance; some boundary/thin-feature pixels can still be lost.",
        "",
        "Background is the polygon complement. All development images contain foreground "
        "annotation; "
        "background pixels do not establish intact-concrete, non-concrete or unrelated-image "
        "recognition.",
        "",
        "## Recomputed validation evaluation",
        "",
        "The complete saved native masks were authenticated against checkpoint SHA, "
        "dataset/split IDs, "
        "prediction hashes and exact validation membership. Metrics below were recomputed "
        "against native "
        "ground truth, independently checked, and matched historical confusion counts. This "
        "is cached "
        "prediction replay, not a newly executed full model forward pass. A separate fresh "
        "command is provided.",
        "",
        "| Model | Foreground mIoU | Background pixels predicted as damage |",
        "|---|---:|---:|",
    ]
    for name, result in evaluations.items():
        metrics = result["metrics"]
        lines.append(
            f"| {name} | {metrics['foreground_miou']:.4%} | "
            f"{sum(metrics['confusion_matrix'][0][1:]):,} |"
        )
    for name, result in evaluations.items():
        lines += [
            "",
            f"### {name}",
            "",
            "| Class | IoU | Dice | Precision | Recall | Validation images |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        val_images = {
            row["class_id"]: row["images"]
            for row in audit["class_statistics"]
            if row["split"] == "validation"
        }
        for row in result["metrics"]["classes"][1:]:
            lines.append(
                f"| {row['source_label']} | {ratio(row['iou'])} | {ratio(row['dice'])} | "
                f"{ratio(row['precision'])} | {ratio(row['recall'])} | "
                f"{val_images[row['class_id']]} |"
            )
        worst = sorted(
            result["metrics"]["classes"][1:],
            key=lambda row: row["iou"] if row["iou"] is not None else 1,
        )[:3]
        lines += ["", "Lowest-IoU foreground categories:", ""]
        for row in worst:
            diagnostic = diagnostics[name][row["class_id"] - 1]
            lines.append(
                f"* {row['source_label']}: IoU {ratio(row['iou'])}; dominant missed-as "
                f"{diagnostic['dominant_missed_as']} ({diagnostic['missed_pixels']:,} pixels); "
                f"dominant FP source {diagnostic['dominant_false_positive_source']} "
                f"({diagnostic['false_positive_pixels_from_source']:,} pixels)."
            )
    lines += [
        "",
        "## Interpretation and experiments requiring later execution",
        "",
        "Poor precision/recall is demonstrated relative to supplied annotations; false "
        "positives and "
        "false negatives are shown separately for each class. Rare-category validation "
        "counts are small "
        "and physical source independence is unverified. Large image/pixel counts are not "
        "independent "
        "damage-instance counts. The failure gallery selects the largest absolute FP/FN "
        "pixel errors, "
        "so it emphasizes large regions/images and is not a random sample.",
        "",
        "The existing paired preprocessing already uses nearest-neighbor masks, ignored padding, "
        "synchronized augmentation and native restoration. No mismatch or wrong historical "
        "loss formula "
        "was established. New Task 09 inference adds fail-closed finite-logit/normalization "
        "checks. "
        "No raw labels, masks, frozen split membership or existing model mathematics were "
        "changed.",
        "",
        "The planned cohort retrains the original U-Net and a TorchVision LR-ASPP "
        "MobileNetV3-Large, "
        "with common ImageNet normalization, 256-pixel letterboxing, training-only class "
        "weights, CE+Dice, "
        "AdamW, matched 25-epoch ceiling and early stopping on pooled native validation "
        "foreground mIoU. "
        "LR-ASPP initializes from COCO/VOC weights only when real training is explicitly "
        "invoked, replaces "
        "both output heads with nine RC channels and optionally freezes BN running "
        "statistics for small "
        "batches. No source taxonomy is merged into RC labels. Pretraining/model family are "
        "a combined "
        "intervention; coarse features could worsen thin cracks. No improvement or speedup "
        "is claimed.",
        "",
        "Historical U-Net runs used a different normalization/LR/50-epoch budget and "
        "different backends. "
        "They are contextual baselines, not matched causal controls for the new cohort. "
        "Train both new "
        "models only after Task 08 finishes. The already-used final test remains excluded "
        "from development.",
        "",
        "Region filtering is a configurable reporting heuristic, default off (minimum one "
        "pixel). Raw "
        "mask coverage and segmentation metrics are always unfiltered. Truncation is counted "
        "explicitly. "
        "Bounding boxes, centroids, perimeter and moment axes use original-image pixel "
        "coordinates; "
        "moment axes do not estimate physical crack length or width. No physical scale, "
        "overall severity "
        "or structural safety target is available. Scores/entropy are uncalibrated and are "
        "not OOD evidence.",
        "",
        "## Files",
        "",
        "audit/: class statistics, checked samples, issues and resize diagnostics. "
        "evaluations/{model}/: pooled/per-image metrics, confusion plots and per-class "
        "failure panels. "
        "inference/: validated feature JSON, raw/reporting-filtered ID masks and uncertainty "
        "maps. "
        "See docs/task09.md for exact manual commands and module responsibilities.",
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    write_json(
        output / "summary.json",
        {
            "audit": audit,
            "evaluations": evaluations,
            "real_training_executed": False,
            "test_evaluated": False,
        },
    )
