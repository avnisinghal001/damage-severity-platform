"""Evaluate supplied native DamSegment predictions, never RC1841's nine-class output."""

from pathlib import Path

import numpy as np
from PIL import Image

from damage_analytics.io import sha256_file, write_csv, write_json
from damage_analytics.task07.annotations import decode_mask
from damage_analytics.task07.metrics import from_confusion
from damage_analytics.task07.plots import confusion
from damage_analytics.task07.release import Release


def evaluate_masks(
    release: Release,
    predictions: Path,
    output: Path,
    *,
    rgb: bool = False,
    final_test: bool = False,
) -> dict:
    """Ground truth uses supplied PNGs; predictions must already align, with explicit encoding."""
    taxonomy = release.metadata["inputs"]["taxonomy"]["segmentation"]
    if taxonomy is None:
        raise ValueError("This dataset has no segmentation targets")
    if final_test and (
        not release.quality["final_evaluation_eligible"] or not taxonomy["semantic_names_verified"]
    ):
        raise ValueError(
            "Final segmentation test requires verified groups and documented category semantics"
        )
    rows = release.samples(
        "test" if final_test else "validation", "segmentation", final_test=final_test
    )
    if not rows:
        raise ValueError("No segmentation records in split")
    if output.exists():
        raise FileExistsError("Evaluation output already exists")
    expected = {f"{r['sample_id']}.png" for r in rows}
    if {p.name for p in predictions.glob("*.png")} != expected:
        raise ValueError(
            "Prediction directory must contain exactly one sample-ID PNG per evaluated sample"
        )
    output.mkdir(parents=True)
    matrix = np.zeros((3, 3), dtype=np.int64)
    hashes = []
    try:
        for row in rows:
            with Image.open(release.file(row["mask"])) as source:
                truth = np.array(source)
            path = predictions / f"{row['sample_id']}.png"
            with Image.open(path) as source:
                source.load()
                if rgb:
                    predicted = decode_mask(source)
                else:
                    if source.mode != "L":
                        raise ValueError("ID predictions must be L-mode PNGs")
                    predicted = np.asarray(source)
            if predicted.shape != truth.shape or predicted.max() > 2:
                raise ValueError(
                    "Prediction shape/IDs disagree; no implicit resize or RC label remapping"
                )
            matrix += np.bincount(
                3 * truth.ravel().astype(int) + predicted.ravel(), minlength=9
            ).reshape(3, 3)
            hashes.append({"sample_id": row["sample_id"], "prediction_sha256": sha256_file(path)})
        result = {
            "metrics": from_confusion(matrix),
            "taxonomy": taxonomy,
            "dataset": release.identity(),
            "split": "test" if final_test else "validation",
            "test_evaluated": final_test,
            "semantic_names_verified": taxonomy["semantic_names_verified"],
            "scope": "Native source-category masks; no RC label merge or severity inference",
        }
        write_json(output / "metrics.json", result)
        write_csv(output / "predictions.csv", hashes, list(hashes[0]))
        confusion(output / "confusion_matrix.png", matrix.tolist(), taxonomy["labels"])
        return result
    except Exception as exc:
        write_json(output / "failure.json", {"error": str(exc), "test_attempted": final_test})
        raise
