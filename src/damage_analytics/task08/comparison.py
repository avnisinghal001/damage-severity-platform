"""Matched source-validation comparison; external evidence cannot drive this command."""

import json
from pathlib import Path

from damage_analytics.io import sha256_file
from damage_analytics.task07.evaluation import compare
from damage_analytics.task07.release import Release


def compare_runs(runs: list[Path], source: Release, output: Path) -> dict:
    if len(runs) < 2 or len(set(runs)) != len(runs):
        raise ValueError("Supply at least two unique runs")
    configs, paths = [], []
    for run in runs:
        metadata = json.loads((run / "run.json").read_text())
        status = json.loads((run / "status.json").read_text())
        if metadata["dataset"] != source.identity() or status["status"] != "completed":
            raise ValueError("Comparison requires completed runs on the pinned source release")
        configs.append(
            {
                "settings": metadata["settings"],
                "seed": metadata["seed"],
                "normalization": metadata["normalization"],
                "selection": metadata["selection"],
                "class_weights": metadata["class_weights"],
            }
        )
        paths.append(
            run.parent.parent
            / "evaluations"
            / sha256_file(run / "best.pt")[:16]
            / source.metadata["release_id"]
            / "validation/metrics.json"
        )
    if any(config != configs[0] for config in configs[1:]):
        raise ValueError("Matched comparison requires identical source settings/budget/seed/loss")
    return compare(paths, output)
