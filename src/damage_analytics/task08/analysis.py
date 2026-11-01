"""Reproduce saved validation results and render selected, traceable error examples."""

from itertools import combinations
from pathlib import Path

from damage_analytics.io import sha256_file, write_csv, write_json
from damage_analytics.task07.plots import confusion
from damage_analytics.task07.release import Release
from damage_analytics.task08.config import Config
from damage_analytics.task08.diagnostics import (
    curve_points,
    figures,
    flat_metrics,
    material_breakdown,
    score_summary,
    threshold_sensitivity,
)
from damage_analytics.task08.galleries import sample_gallery
from damage_analytics.task08.inputs import Validation, load_run, load_validation
from damage_analytics.task08.reports import failure_report
from damage_analytics.training.experiments import execution_provenance


def error_galleries(validation: Validation, output: Path, count: int) -> None:
    frozen = validation.run.payload["threshold"]["threshold"]
    records = list(validation.predictions)
    categories = {
        "false_positives": sorted(
            [r for r in records if r["target"] == 0 and r["prediction"] == 1],
            key=lambda r: (-r["probability_crack"], r["sample_id"]),
        ),
        "false_negatives": sorted(
            [r for r in records if r["target"] == 1 and r["prediction"] == 0],
            key=lambda r: (r["probability_crack"], r["sample_id"]),
        ),
        "borderline": sorted(
            records, key=lambda r: (abs(r["probability_crack"] - frozen), r["sample_id"])
        ),
    }
    for category, selected in categories.items():
        items = [
            {
                "sample_id": r["sample_id"],
                "row": r["row"],
                "score": r["probability_crack"],
                "target": r["target"],
                "prediction": r["prediction"],
                "lines": [
                    r["sample_id"],
                    f"source target={r['target']} prediction={r['prediction']}",
                    f"score={r['probability_crack']:.4f} threshold={frozen:.4f}",
                    f"material/tag={r['row']['domain']}",
                    Path(r["row"]["source_member"]).name,
                ],
            }
            for r in selected
        ]
        sample_gallery(validation.release, items, output / f"{category}.png", count)


def analyze_validation(validation: Validation, output: Path, count: int) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    records = validation.predictions
    points = curve_points(
        [r["target"] for r in records], [r["probability_crack"] for r in records]
    )
    write_csv(output / "roc_pr_points.csv", points, list(points[0]))
    sensitivity = None
    if validation.release.metadata["dataset"] == "sdnet2018":
        sensitivity = threshold_sensitivity(validation)
        write_csv(output / "sdnet_threshold_sensitivity.csv", sensitivity, list(sensitivity[0]))
        materials = material_breakdown(validation)
        write_csv(output / "material_metrics.csv", materials, list(materials[0]))
    else:
        materials = None
    error_galleries(validation, output / "galleries", count)
    figures(output, validation, points, sensitivity)
    confusion(
        output / "confusion_matrix.png",
        validation.metrics["confusion_matrix"],
        ["No crack", "Crack"],
    )
    write_csv(
        output / "class_metrics.csv",
        validation.metrics["classes"],
        list(validation.metrics["classes"][0]),
    )
    error_index = [
        {
            "sample_id": r["sample_id"],
            "target": r["target"],
            "prediction": r["prediction"],
            "probability_crack": r["probability_crack"],
            "absolute_threshold_distance": abs(
                r["probability_crack"] - validation.metrics["threshold"]
            ),
            "error": "false_positive"
            if r["prediction"] > r["target"]
            else "false_negative"
            if r["prediction"] < r["target"]
            else "correct",
            "material_tag": r["row"]["domain"],
            "group_id": r["row"]["group_id"],
        }
        for r in records
    ]
    write_csv(output / "prediction_index.csv", error_index, list(error_index[0]))
    summary = {
        "run_id": validation.run.root.name,
        "backbone": validation.run.payload["backbone"],
        "dataset_name": validation.release.metadata["dataset"],
        "dataset": validation.release.identity(),
        "samples": len(records),
        "crack_prevalence": sum(r["target"] for r in records) / len(records),
        "threshold": validation.metrics["threshold"],
        "metrics": validation.metrics,
        "scores": score_summary(records),
        "material_metrics": materials,
        "checkpoint_sha256": validation.run.checkpoint_sha256,
        "selected_epoch": validation.run.payload["epoch"],
        "epochs_trained": len(validation.run.history),
        "validation_metrics_reproduced": True,
        "saved_report": str(validation.report_path),
        "saved_report_sha256": sha256_file(validation.report_path),
        "saved_predictions_sha256": sha256_file(validation.report_path.parent / "predictions.csv"),
        "model_runtime": validation.report.get("runtime"),
        "runtime_note": "Historical runtime cannot be inferred from checkpoint timestamps; "
        "new evaluation commands record synchronized end-to-end pass timing.",
        "threshold_tuned_here": False,
        "inference_rerun": False,
        "test_evaluated": False,
    }
    write_json(output / "metrics.json", summary)
    return summary


def disagreements(a: Validation, b: Validation, output: Path, count: int) -> dict:
    if a.release.identity() != b.release.identity():
        raise ValueError("Disagreements require identical evaluation populations")
    lookup = {r["sample_id"]: r for r in b.predictions}
    if set(lookup) != {r["sample_id"] for r in a.predictions}:
        raise ValueError("Disagreement population differs")
    rows, both_wrong, only_a_wrong, only_b_wrong = [], 0, 0, 0
    for left in a.predictions:
        right = lookup[left["sample_id"]]
        wrong_a, wrong_b = (
            left["prediction"] != left["target"],
            right["prediction"] != right["target"],
        )
        both_wrong += wrong_a and wrong_b
        only_a_wrong += wrong_a and not wrong_b
        only_b_wrong += wrong_b and not wrong_a
        if left["prediction"] != right["prediction"]:
            rows.append(
                {
                    "sample_id": left["sample_id"],
                    "target": left["target"],
                    "a_prediction": left["prediction"],
                    "b_prediction": right["prediction"],
                    "a_score": left["probability_crack"],
                    "b_score": right["probability_crack"],
                    "row": left["row"],
                }
            )
    rows.sort(key=lambda r: (-abs(r["a_score"] - r["b_score"]), r["sample_id"]))
    write_csv(
        output / "disagreements.csv",
        [{k: v for k, v in r.items() if k != "row"} for r in rows],
        ["sample_id", "target", "a_prediction", "b_prediction", "a_score", "b_score"],
    )
    items = [
        {
            **r,
            "lines": [
                r["sample_id"],
                f"source target={r['target']}",
                f"A: pred={r['a_prediction']} score={r['a_score']:.4f}",
                f"B: pred={r['b_prediction']} score={r['b_score']:.4f}",
                "Each model uses its frozen source threshold",
            ],
        }
        for r in rows
    ]
    sample_gallery(a.release, items, output / "disagreements.png", count)
    result = {
        "a": a.run.root.name,
        "b": b.run.root.name,
        "dataset": a.release.metadata["dataset"],
        "samples": len(a.predictions),
        "disagreements": len(rows),
        "both_wrong": both_wrong,
        "only_a_wrong": only_a_wrong,
        "only_b_wrong": only_b_wrong,
        "thresholds": [a.metrics["threshold"], b.metrics["threshold"]],
        "scope": "Distinct frozen thresholds; selected disagreements are not calibration evidence",
    }
    write_json(output / "summary.json", result)
    return result


def analyze(
    config: Config,
    releases: dict[str, Release],
    output: Path,
    run_paths: tuple[Path, ...] | None = None,
    *,
    include_external: bool = False,
) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    paths = run_paths or config.baseline_runs
    if len({p.name for p in paths}) != len(paths):
        raise ValueError("Analysis run IDs must be unique")
    source = releases["sdnet2018"]
    runs = [load_run(path, source) for path in paths]
    datasets = [source, releases["damsegment"]] if include_external else [source]
    summaries, compared = [], []
    for release in datasets:
        validations = [load_validation(run, source, release) for run in runs]
        for validation in validations:
            folder = output / validation.run.root.name / release.metadata["dataset"]
            summaries.append(analyze_validation(validation, folder, config.gallery_samples))
        for a, b in combinations(validations, 2):
            folder = (
                output
                / "disagreements"
                / release.metadata["dataset"]
                / f"{a.run.root.name}__{b.run.root.name}"
            )
            compared.append(disagreements(a, b, folder, config.gallery_samples))
    table = [
        {
            "run_id": r["run_id"],
            "dataset": r["dataset_name"],
            "threshold": r["threshold"],
            "samples": r["samples"],
            **flat_metrics(r["metrics"]),
            "runtime_seconds": r["model_runtime"]["seconds"] if r["model_runtime"] else None,
        }
        for r in summaries
    ]
    write_csv(output / "comparison.csv", table, list(table[0]))
    result = {
        "results": summaries,
        "disagreements": compared,
        "execution": execution_provenance(),
        "external_included": include_external,
        "inference_rerun": False,
        "test_pixels_or_predictions_read": False,
        "external_threshold_tuning": False,
        "limits": "Validation-only exploratory analysis. Source group independence is unverified; "
        "external validation has now been inspected and is not an unbiased final test.",
    }
    write_json(output / "analysis_report.json", result)
    failure_report(output / "failure_analysis.md", result, runs)
    return result
