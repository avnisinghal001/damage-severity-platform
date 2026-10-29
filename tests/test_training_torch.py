"""Executable PyTorch checks; each test is explicitly skipped when torch is absent."""

import importlib.util
from dataclasses import replace

import numpy as np
import pytest

HAS_TORCH = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(
    not HAS_TORCH, reason="PyTorch is absent; installation is prohibited"
)
if HAS_TORCH:
    import torch


@pytest.fixture(autouse=True)
def tiny_cpu_threads():
    """Avoid thread-pool overhead in tiny synthetic CPU tests; restore caller settings."""
    if not HAS_TORCH:
        yield
        return
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_model_forward_backward_preserves_spatial_shape():
    from damage_analytics.training.config import ModelConfig
    from damage_analytics.training.model import LightweightUNet

    model = LightweightUNet(ModelConfig(4, 2, 2))
    image = torch.randn(2, 3, 17, 19, requires_grad=True)
    logits = model(image)
    assert tuple(logits.shape) == (2, 9, 17, 19)
    logits.square().mean().backward()
    assert image.grad is not None and bool(torch.isfinite(image.grad).all())
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_real_dataset_adapter_loader_shapes_and_repeatable_epochs(training_release):
    from damage_analytics.training.dataset import SegmentationDataset, make_loader
    from damage_analytics.training.manifests import open_release

    release = open_release(training_release)
    dataset = SegmentationDataset(
        release, release.samples("train"), training_release.data, training=True, seed=42
    )
    dataset.set_epoch(2)
    first, second = dataset[0], dataset[0]
    assert torch.equal(first["image"], second["image"]) and torch.equal(
        first["mask"], second["mask"]
    )
    batch = next(iter(make_loader(dataset, 1, 0, 42, shuffle=True)))
    assert tuple(batch["images"].shape) == (1, 3, 16, 16)
    assert tuple(batch["masks"].shape) == (1, 16, 16)
    assert batch["images"].dtype == torch.float32 and batch["masks"].dtype == torch.int64
    assert set(batch["masks"].unique().tolist()).issubset({0, 1, 2, 255})
    assert batch["native_masks"] == [None]


def test_loss_ignores_padding_and_backpropagates_finite_gradients():
    from damage_analytics.training.losses import SegmentationLoss

    logits = torch.randn(1, 9, 4, 4, requires_grad=True)
    target = torch.ones(1, 4, 4, dtype=torch.long)
    target[0, 0, 0] = 255
    criterion = SegmentationLoss(torch.ones(9), 0.5)
    loss = criterion(logits, target)
    assert bool(torch.isfinite(loss))
    loss.backward()
    assert bool(torch.isfinite(logits.grad).all())
    assert bool((logits.grad[0, :, 0, 0] == 0).all())
    with pytest.raises(ValueError, match="no supervised"):
        criterion(logits, torch.full_like(target, 255))


def test_checkpoint_roundtrip_and_dataset_binding(tmp_path):
    from damage_analytics.training.checkpoints import load_checkpoint, save_checkpoint
    from damage_analytics.training.config import ModelConfig
    from damage_analytics.training.model import LightweightUNet

    architecture = ModelConfig(4, 2, 2)
    model = LightweightUNet(architecture)
    optimizer = torch.optim.AdamW(model.parameters())
    path = tmp_path / "model.pt"
    save_checkpoint(
        path,
        model,
        optimizer,
        epoch=1,
        score=0.2,
        metadata={"dataset": {"release_id": "synthetic"}},
        architecture=architecture,
    )
    restored, payload = load_checkpoint(
        path, torch.device("cpu"), expected_dataset={"release_id": "synthetic"}
    )
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)
    assert payload["epoch"] == 1
    with pytest.raises(ValueError, match="different dataset"):
        load_checkpoint(path, torch.device("cpu"), expected_dataset={"release_id": "wrong"})


def test_synthetic_training_smoke(tmp_path):
    from damage_analytics.training.smoke import synthetic_smoke

    result = synthetic_smoke(tmp_path / "smoke")
    assert result["steps"] == 3 and result["loss_decreased"] and result["checkpoint_roundtrip"]


def test_end_to_end_training_validation_inference_and_explicit_synthetic_test(training_release):
    from damage_analytics.training.engine import train_model
    from damage_analytics.training.evaluation import evaluate_checkpoint
    from damage_analytics.training.inference import predict_image

    config = training_release
    folder = train_model(config, "unit_run")
    assert (folder / "best.pt").is_file() and (folder / "history.csv").is_file()
    validation = evaluate_checkpoint(config, folder / "best.pt")
    assert (validation / "confusion_matrix.npy").is_file()
    prediction = predict_image(
        config, folder / "best.pt", config.release_dir / "images/validation.png", "unit"
    )
    from PIL import Image

    with Image.open(prediction / "prediction_ids.png") as mask:
        assert mask.size == (12, 8) and np.asarray(mask).max() <= 8
    with pytest.raises(ValueError, match="test images"):
        predict_image(
            config, folder / "best.pt", config.release_dir / "images/test.png", "forbidden"
        )
    final = evaluate_checkpoint(config, folder / "best.pt", final_test=True)
    assert (final / "metrics.json").is_file()
    with pytest.raises(FileExistsError):
        evaluate_checkpoint(config, folder / "best.pt", final_test=True)


def test_overfit_mode_is_diagnostic_and_refuses_final_test(training_release):
    import json

    from damage_analytics.training.engine import train_model
    from damage_analytics.training.evaluation import evaluate_checkpoint

    config = replace(training_release, sanity=replace(training_release.sanity, overfit_samples=1))
    folder = train_model(config, "overfit_unit", overfit=True)
    status = json.loads((folder / "status.json").read_text())
    assert status["diagnostic_only"] and not status["test_evaluated"]
    with pytest.raises(ValueError, match="validation-selected"):
        evaluate_checkpoint(config, folder / "best.pt", final_test=True)


def test_unavailable_mps_is_not_silently_selected(monkeypatch):
    from damage_analytics.training.runtime import configure_runtime

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert str(configure_runtime(42, "auto", True)) == "cpu"
    with pytest.raises(ValueError, match="unavailable"):
        configure_runtime(42, "mps", True)
