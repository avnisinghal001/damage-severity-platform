"""Small synthetic release and actual two-epoch loop; no torchvision/weights/network required."""

import io
import tempfile
from dataclasses import replace
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image

from damage_analytics.io import write_json
from damage_analytics.task07.config import Config
from damage_analytics.task07.prepare import prepare


def synthetic_archive(path: Path, groups: int = 9) -> None:
    rng = np.random.default_rng(17)
    with ZipFile(path, "w") as archive:
        for group in range(groups):
            for label, folder in ((0, "UD"), (1, "CD")):
                for patch in range(2):
                    values = rng.integers(0, 25, (16, 16, 3), dtype=np.uint8) + (
                        25 if label == 0 else 190
                    )
                    stream = io.BytesIO()
                    Image.fromarray(values).save(stream, format="JPEG")
                    archive.writestr(
                        f"D/{folder}/{group:03d}-{label}-{patch}.jpg", stream.getvalue()
                    )


def smoke(config: Config, output: Path) -> dict:
    import torch

    from damage_analytics.task07.engine import train
    from damage_analytics.task07.evaluation import evaluate
    from damage_analytics.task07.release import open_release

    output.mkdir(parents=True, exist_ok=False)
    threads, deterministic = torch.get_num_threads(), torch.are_deterministic_algorithms_enabled()
    try:
        with tempfile.TemporaryDirectory(prefix="task07-smoke-") as temporary:
            archive = Path(temporary) / "synthetic.zip"
            synthetic_archive(archive)
            temporary_config = replace(
                config,
                sdnet2018_archive=archive,
                output_dir=Path(temporary) / "outputs",
                source_groups_csv=None,
                near_reviews_csv=None,
            )
            release = open_release(prepare(temporary_config, "sdnet2018"), full=True)
            settings = replace(
                config.training,
                epochs=2,
                batch_size=4,
                image_size=16,
                device="cpu",
                cpu_threads=1,
                num_workers=0,
                deterministic=True,
                learning_rate=0.02,
                class_weighting="none",
            )
            folder = train(
                release,
                settings,
                output,
                "tiny_synthetic",
                "tiny",
                config.seed,
                allow_provisional=True,
                synthetic=True,
            )
            result = evaluate(folder / "best.pt", release, output)
            # Evaluation is completed before temporary source data is removed.
            report = {
                "purpose": "synthetic_plumbing_only",
                "completed": True,
                "epochs": 2,
                "validation_result": str(result),
                "pretrained_downloaded": False,
                "real_dataset_training": False,
                "test_evaluated": False,
            }
            write_json(output / "smoke.json", report)
            return report
    finally:
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(deterministic)
