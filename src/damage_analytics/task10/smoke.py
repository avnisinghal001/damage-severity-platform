"""Small synthetic end-to-end plumbing check; never measures real dataset performance."""

import io
from dataclasses import replace
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from damage_analytics.io import write_json
from damage_analytics.task10.config import (
    SOURCE_CODES,
    SOURCE_LABELS,
    Config,
    TrainConfig,
    safe_id,
)
from damage_analytics.task10.encoding import CAFFE_OFFSETS


def synthetic_archive(
    path: Path, *, size: int = 32, train_rows: int = 40, test_rows: int = 8, seed: int = 123
) -> None:
    """Write a documented, known-color fixture with the actual six-file archive schema."""
    rng = np.random.default_rng(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(path, "x", compression=ZIP_DEFLATED) as archive:
        registration = "\n".join(
            f" {i}: {label} ({code})"
            for i, (label, code) in enumerate(zip(SOURCE_LABELS, SOURCE_CODES, strict=True))
        )
        archive.writestr("task7/README.txt", "SYNTHETIC FIXTURE ONLY\n" + registration + "\n")
        archive.writestr(
            "task7/license.txt", "Synthetic random pixels generated for software tests."
        )
        for split, count in (("train", train_rows), ("test", test_rows)):
            rgb = rng.integers(0, 256, (count, size, size, 3), dtype=np.uint8)
            arrays = {
                "X": np.ascontiguousarray(rgb[..., ::-1].astype(np.float32) - CAFFE_OFFSETS),
                "y": np.eye(4, dtype=np.float32)[np.arange(count) % 4],
            }
            for kind, values in arrays.items():
                stream = io.BytesIO()
                np.save(stream, values, allow_pickle=False)
                archive.writestr(f"task7/task7_{kind}_{split}.npy", stream.getvalue())


def run_smoke(config: Config, run_id: str) -> Path:
    """Prepare random tensors, train a tiny CNN for two CPU epochs, reload and validate."""
    from damage_analytics.task10.engine import evaluate, train
    from damage_analytics.task10.prepare import prepare
    from damage_analytics.task10.release import open_release

    safe_id(run_id)
    output = config.output_dir / "smoke" / run_id
    output.mkdir(parents=True, exist_ok=False)
    synthetic_archive(output / "synthetic.zip")
    fixture = replace(
        config,
        archive=output / "synthetic.zip",
        output_dir=output,
        expected_image_shape=(32, 32, 3),
        source_groups_csv=None,
        encoding_verified=True,
        encoding_evidence="Synthetic fixture explicitly generated with Caffe/BGR encoding",
        training=TrainConfig(epochs=2, image_size=32, batch_size=8, patience=2),
    )
    release = open_release(prepare(fixture), full=True)
    experiment = train(fixture, release, "tiny_cpu", allow_provisional=True, synthetic=True)
    evaluation = evaluate(fixture, release, experiment / "best.pt")
    write_json(
        output / "smoke_result.json",
        {
            "status": "passed",
            "purpose": "Synthetic plumbing only; not an ML accuracy result",
            "experiment": str(experiment),
            "evaluation": str(evaluation),
            "test_evaluated": False,
            "real_model_trained": False,
        },
    )
    return output
