"""Only annotation-category-derived spalling labels; no invented overall severity."""

SUPPORTED_SPALLING = {3: "Minor spalling", 4: "Moderate spalling"}


def spalling_assessment(class_ids: list[int], *, uncertain_prediction: bool = False) -> dict:
    """Describe supported labels, mixed categories and unsupported cases explicitly.

    These are names inherited from the segmentation taxonomy, not predictions from
    independent region-severity supervision. Region area never determines the label.
    A score heuristic can withhold a label but cannot validate or calibrate one.
    """
    foreground = sorted(set(class_ids) - {0})
    supported = [SUPPORTED_SPALLING[value] for value in foreground if value in SUPPORTED_SPALLING]
    unsupported = [value for value in foreground if value not in SUPPORTED_SPALLING]
    if not foreground:
        status = "not_applicable_no_foreground"
    elif len(supported) > 1:
        status = "mixed_minor_and_moderate"
    elif supported and unsupported:
        status = "mixed_supported_and_unsupported"
    elif not supported:
        status = "unsupported_category"
    elif uncertain_prediction:
        status = "uncertain_category_prediction"
    else:
        status = "source_category_derived_unvalidated"
    return {
        "status": status,
        "label": supported[0] if status == "source_category_derived_unvalidated" else None,
        "candidate_source_labels": supported,
        "unsupported_class_ids": unsupported,
        "supervision": "Same pixel-category annotations as segmentation; no "
        "independent severity target",
        "structural_severity_or_safety": "not_assessed",
    }
