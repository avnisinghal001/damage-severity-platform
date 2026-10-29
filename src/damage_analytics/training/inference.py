"""Single RGB-image inference with checkpoint-bound preprocessing and native-size output."""

import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from damage_analytics.io import sha256_file, write_json
from damage_analytics.training.checkpoints import load_checkpoint
from damage_analytics.training.config import TrainingConfig
from damage_analytics.training.evaluation import effective_config
from damage_analytics.training.experiments import execution_provenance
from damage_analytics.training.manifests import open_release
from damage_analytics.training.plots import save_comparison
from damage_analytics.training.runtime import configure_runtime
from damage_analytics.training.transforms import prepare_pair, restore_prediction


def predict_image(
    config: TrainingConfig, checkpoint: Path, image_path: Path, prediction_id: str
) -> Path:
    """Save class IDs 0..8 and a display comparison; refuse known test-release images."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", prediction_id):
        raise ValueError("Provide a safe prediction ID")
    release = open_release(config)
    image_hash = sha256_file(image_path)
    test_hashes = {release.checksums[row.image] for row in release.rows["test"]}
    if image_hash in test_hashes or any(
        image_path.resolve() == (release.root / row.image).resolve()
        for row in release.rows["test"]
    ):
        raise ValueError("Known test images are reserved for explicit final-test evaluation")
    runtime_seed = config.seed
    device = configure_runtime(runtime_seed, config.device, config.deterministic)
    model, payload = load_checkpoint(checkpoint, device, expected_dataset=release.identity())
    config = effective_config(config, payload)
    with Image.open(image_path) as source:
        source.load()
        if source.mode != "RGB":
            raise ValueError(
                "Inference expects an RGB image; no unrecorded color/alpha conversion"
            )
        image = ImageOps.exif_transpose(source).copy()
    rgb, _, frame = prepare_pair(
        image,
        Image.new("L", image.size),
        config.data,
        training=False,
        rng=np.random.default_rng(config.seed),
    )
    model.eval()
    with torch.inference_mode():
        prediction = (
            model(torch.from_numpy(rgb).unsqueeze(0).to(device)).argmax(1)[0].cpu().numpy()
        )
    native = restore_prediction(prediction.astype(np.uint8), frame)
    folder = config.output_dir / "inference" / prediction_id
    folder.mkdir(parents=True, exist_ok=False)
    Image.fromarray(native).save(folder / "prediction_ids.png")
    save_comparison(folder / "comparison.png", image, None, native, prediction_id)
    write_json(
        folder / "prediction.json",
        {
            "dataset": release.identity(),
            "execution": execution_provenance(),
            "checkpoint_sha256": sha256_file(checkpoint),
            "image_sha256": image_hash,
            "native_size": list(image.size),
            "device": str(device),
            "runtime_seed": runtime_seed,
            "deterministic_requested": config.deterministic,
            "pixel_counts": np.bincount(native.ravel(), minlength=9).tolist(),
            "data_config": payload["metadata"]["config"]["data"],
            "meaning": "Semantic category prediction, not severity or structural safety",
            "test_evaluated": False,
        },
    )
    return folder
