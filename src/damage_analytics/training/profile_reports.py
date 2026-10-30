"""Readable CSV summaries of immutable hardware benchmark reports."""

import json
from pathlib import Path

import numpy as np

from damage_analytics.io import write_csv, write_json
from damage_analytics.training.profiling import profile_destination


def summarize_profiles(output: Path, identifiers: list[str], report_id: str) -> Path:
    """Summarize measured passes; never extrapolate full training or claim model improvements."""
    rows = []
    for identifier in identifiers:
        source = profile_destination(output, identifier) / "profile.json"
        report = json.loads(source.read_text())
        if (
            not report["cases"]
            or report.get("status", "completed") != "completed"
            or (source.parent / "failure.json").exists()
        ):
            raise ValueError(f"No completed cases in {identifier}; inspect its failure report")
        for case in report["cases"]:
            base = {
                "profile": identifier,
                "suite": report["suite"],
                "device": report["device"],
                "deterministic": report["config"]["deterministic"],
                "cpu_threads": report["cpu_threads"],
                "workers": case["workers"],
                "persistent_workers": case["persistent_workers"],
                "cache_mb_per_process": case["cache_mb_per_process"],
                "channels_last": report.get("channels_last", False),
                "batch_size": case["batch_size"],
            }
            for split in ("loader",) if report["suite"] == "loaders" else ("train", "validation"):
                passes = (
                    case["passes"] if split == "loader" else [p[split] for p in case["passes"]]
                )
                times = [item["seconds"] for item in passes]
                memory = case["memory"]["sampled_process_tree_peak_bytes"]
                rows.append(
                    {
                        **base,
                        "split": split,
                        "images": passes[0]["images"],
                        "first_seconds": times[0],
                        "last_seconds": times[-1],
                        "median_seconds": float(np.median(times)),
                        "p95_seconds": float(np.quantile(times, 0.95)),
                        "images_per_second_at_median": passes[0]["images"]
                        / float(np.median(times)),
                        "sampled_tree_peak_mib": memory / 1024**2 if memory else None,
                    }
                )
    destination = profile_destination(output, report_id)
    destination.mkdir(parents=True, exist_ok=False)
    write_csv(destination / "timings.csv", rows, list(rows[0]))
    write_json(
        destination / "summary.json",
        {
            "profiles": identifiers,
            "rows": rows,
            "interpretation": "Short hardware measurements, not full-run forecasts or learned "
            "accuracy. Worker/cache first vs later passes differ; compare identical samples, "
            "batch sizes and settings. MPS nondeterminism is explicit. RSS is approximate.",
        },
    )
    return destination
