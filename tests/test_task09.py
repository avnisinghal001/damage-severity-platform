"""Synthetic, single-thread CPU tests; never touch real test images or pretrained downloads."""

import copy
import csv
import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from damage_analytics.analytics.components import connected_components
from damage_analytics.curation.verification import write_checksums
from damage_analytics.io import sha256_file
from damage_analytics.task09.audit import audit, selected_training
from damage_analytics.task09.config import Config, FeaturePolicy, load_config
from damage_analytics.task09.features import extract_features
from damage_analytics.task09.geometry import geometric_features
from damage_analytics.task09.schema import validate_report
from damage_analytics.training.manifests import open_release


@pytest.fixture(autouse=True, scope="module")
def single_thread_cpu():
    torch = pytest.importorskip("torch")
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(autouse=True)
def no_downloads(monkeypatch):
    import torch

    def forbidden(*args, **kwargs):
        raise AssertionError("Tests may never download weights")

    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", forbidden)
    if importlib.util.find_spec("torchvision"):
        import torchvision.models._api as api

        monkeypatch.setattr(api, "load_state_dict_from_url", forbidden)


@pytest.fixture
def context(training_release, tmp_path, monkeypatch):
    base = training_release
    root = base.release_dir
    (root / "annotations").mkdir()
    rows = []
    for split in ("train", "validation", "test"):
        with (root / f"splits/{split}.csv").open() as stream:
            row = next(csv.DictReader(stream))
        annotation = {
            "imagePath": f"{split}.png",
            "imageWidth": 12,
            "imageHeight": 8,
            "shapes": [
                {
                    "label": "Microcrack",
                    "shape_type": "polygon",
                    "points": [[1, 1], [4, 1], [4, 6], [1, 6]],
                },
                {
                    "label": "Structural crack",
                    "shape_type": "polygon",
                    "points": [[7, 2], [10, 2], [10, 5], [7, 5]],
                },
            ],
        }
        relative = f"annotations/{split}.json"
        (root / relative).write_text(json.dumps(annotation))
        rows.append(
            {
                **row,
                "annotation": relative,
                "source_image_sha256": row["image_sha256"],
                "source_annotation_sha256": sha256_file(root / relative),
            }
        )
    with (root / "curated_manifest.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    quality = json.loads((root / "quality_report.json").read_text())
    quality.update(
        excluded_samples=0, exclusion_reasons_nonexclusive={}, unresolved_limitations=["synthetic"]
    )
    (root / "quality_report.json").write_text(json.dumps(quality))
    write_checksums(root)
    base = replace(base, checksums_sha256=sha256_file(root / "checksums.json"))
    config = replace(
        load_config(Path("config/task09.toml")),
        output_dir=tmp_path / "task09",
        data=base.data,
        optimization=replace(base.optimization, epochs=2, batch_size=1),
        baselines=(),
        audit_training_samples=1,
        gallery_samples_per_class=1,
    )

    def training(self, *, device=None):
        return replace(
            base,
            output_dir=self.output_dir,
            data=self.data,
            optimization=self.optimization,
            device=device or "cpu",
            deterministic=False,
            num_workers=0,
        )

    monkeypatch.setattr(Config, "training", training)
    return config, open_release(config.training())


def test_area_geometry_filtering_and_truncation_conserve_pixels():
    mask = np.zeros((8, 9), dtype=np.uint8)
    mask[:2, :2] = 1
    mask[3:5, 3:5] = 3
    mask[7, 8] = 3
    original = mask.copy()
    policy = FeaturePolicy(
        minimum_region_pixels=4, minimum_by_class={"1": 1}, maximum_reported_regions=1
    )
    report, filtered = extract_features(mask, policy)
    assert np.array_equal(mask, original) and np.count_nonzero(filtered) == 8
    assert report["coverage"]["foreground_pixels"] == 9
    assert report["coverage"]["filtered_pixels"] == 1
    assert report["filtering"]["region_table_truncated"]
    region = report["regions"][0]
    assert region["bbox_xyxy"] == [0, 0, 2, 2] and region["centroid_xy"] == [1.0, 1.0]
    assert region["perimeter_pixel_edges"] == 8 and region["area_pixels"] == 4
    assert report["classes"][2]["unreported_retained_pixels"] == 4
    validate_report(json.loads(json.dumps(report)))


@pytest.mark.parametrize("connectivity", [4, 8])
def test_rle_geometry_matches_independent_pixel_edge_and_moment_calculations(connectivity):
    rng = np.random.default_rng(5)
    for _ in range(4):
        binary = rng.random((9, 11)) < 0.35
        for component in connected_components(binary, connectivity):
            region = np.zeros(binary.shape, dtype=bool)
            for y, left, right in component.runs:
                region[y, left:right] = True
            y, x = np.nonzero(region)
            padded = np.pad(region, 1)
            perimeter = sum(
                int(np.count_nonzero(region & ~neighbor))
                for neighbor in (
                    padded[:-2, 1:-1],
                    padded[2:, 1:-1],
                    padded[1:-1, :-2],
                    padded[1:-1, 2:],
                )
            )
            features = geometric_features(component)
            assert features["perimeter_pixel_edges"] == perimeter
            np.testing.assert_allclose(features["centroid_xy"], [x.mean() + 0.5, y.mean() + 0.5])
            variance = np.array(
                [
                    [np.mean((x - x.mean()) ** 2), np.mean((x - x.mean()) * (y - y.mean()))],
                    [np.mean((x - x.mean()) * (y - y.mean())), np.mean((y - y.mean()) ** 2)],
                ]
            )
            eigenvalues = np.maximum(np.linalg.eigvalsh(variance), 0)
            assert features["moment_major_axis_pixels"] == pytest.approx(
                4 * np.sqrt(eigenvalues[1])
            )


def test_holes_diagonals_and_degenerate_regions_have_defined_geometry():
    ring = np.ones((3, 3), dtype=bool)
    ring[1, 1] = False
    assert geometric_features(connected_components(ring)[0])["perimeter_pixel_edges"] == 16
    diagonal = np.eye(3, dtype=np.uint8)
    four, _ = extract_features(diagonal, FeaturePolicy(connectivity=4))
    eight, _ = extract_features(diagonal, FeaturePolicy(connectivity=8))
    assert four["classes"][0]["raw_region_count"] == 3
    assert eight["classes"][0]["raw_region_count"] == 1
    assert four["regions"][0]["moment_axis_ratio"] is None
    assert four["regions"][0]["orientation_degrees_image_xy"] is None
    background, _ = extract_features(np.zeros((3, 4), np.uint8), FeaturePolicy())
    assert background["coverage"]["foreground_pixels"] == 0 and not background["regions"]
    assert background["uncertainty"]["mean_maximum_softmax"] is None


@pytest.mark.parametrize(
    "mask",
    [
        np.zeros((2, 3, 3), np.uint8),
        np.array([[9]], np.uint8),
        np.array([[-1]], np.int32),
        np.zeros((2, 2), np.float32),
    ],
)
def test_invalid_class_id_masks_are_rejected(mask):
    with pytest.raises(ValueError):
        extract_features(mask, FeaturePolicy())


def test_uncertainty_maps_are_aligned_finite_and_explicitly_uncalibrated():
    mask = np.ones((3, 4), np.uint8)
    maximum, entropy = np.full(mask.shape, 0.55), np.full(mask.shape, 0.8)
    report, _ = extract_features(
        mask, FeaturePolicy(), maximum_softmax=maximum, normalized_entropy=entropy
    )
    assert (
        not report["uncertainty"]["calibrated"]
        and report["uncertainty"]["low_score_fraction"] == 1
    )
    assert report["regions"][0]["score_statistics"]["high_entropy_fraction"] == 1
    with pytest.raises(ValueError):
        extract_features(mask, FeaturePolicy(), maximum_softmax=maximum)
    for wrong in (np.zeros((2, 4)), np.full(mask.shape, np.nan), np.full(mask.shape, 1.1)):
        with pytest.raises(ValueError):
            extract_features(
                mask, FeaturePolicy(), maximum_softmax=wrong, normalized_entropy=entropy
            )


@pytest.mark.parametrize(
    "issue",
    [
        "bbox",
        "coverage",
        "class",
        "nan",
        "calibration",
        "float_count",
        "score_range",
        "bbox_dimension",
    ],
)
def test_json_contract_rejects_inconsistent_or_misleading_payloads(issue):
    report, _ = extract_features(np.ones((3, 4), np.uint8), FeaturePolicy())
    bad = copy.deepcopy(report)
    if issue == "bbox":
        bad["regions"][0]["bbox_xyxy"][0] = -1
    if issue == "coverage":
        bad["coverage"]["foreground_pixels"] = 0
    if issue == "class":
        bad["classes"][0]["label"] = "invented severity"
    if issue == "nan":
        bad["regions"][0]["centroid_xy"][0] = float("nan")
    if issue == "calibration":
        bad["uncertainty"]["calibrated"] = True
    if issue == "float_count":
        bad["classes"][0]["raw_pixels"] = 12.0
    if issue == "score_range":
        bad["uncertainty"]["available"] = True
        bad["uncertainty"]["mean_maximum_softmax"] = 1.1
    if issue == "bbox_dimension":
        bad["regions"][0]["bbox_width_pixels"] = 999
    with pytest.raises(ValueError):
        validate_report(bad)


def test_audit_verifies_rasterization_without_decoding_test(context, tmp_path, monkeypatch):
    from damage_analytics.task09 import audit as module

    config, release = context
    original = module.read_pair
    seen = []

    def guarded(selected_release, sample):
        assert sample.sample_id != "test"
        seen.append(sample.sample_id)
        return original(selected_release, sample)

    monkeypatch.setattr(module, "read_pair", guarded)
    result = audit(config, release, tmp_path / "audit")
    assert result["decoded_and_rerasterized"] == {"train": 1, "validation": 1}
    assert set(seen) == {"train", "validation"} and not result["issues"]
    assert len(selected_training(release, 1)) == 1


def test_validation_metric_population_and_native_pixel_correctness(context, tmp_path):
    from damage_analytics.task09.evaluation import evaluate_predictions

    _, release = context
    sample = release.samples("validation")[0]
    with Image.open(release.file(sample.mask)) as source:
        values = np.array(source)
    result = evaluate_predictions(release, [(sample, values)], tmp_path / "perfect", {})
    assert result["metrics"]["foreground_miou"] == 1
    assert result["metrics"]["evaluated_pixels"] == sample.width * sample.height
    with pytest.raises(ValueError, match="outside frozen validation"):
        evaluate_predictions(
            release, [(release.rows["test"][0], values)], tmp_path / "blocked", {}
        )
    with pytest.raises(ValueError, match="Incomplete"):
        evaluate_predictions(release, [], tmp_path / "missing", {})


def test_known_test_copy_is_rejected_by_metadata_without_decoding(context, tmp_path):
    from damage_analytics.task09.pipeline import forbid_test_input

    _, release = context
    path = release.root / release.rows["test"][0].image
    copied = tmp_path / "unrelated_name.png"
    copied.write_bytes(path.read_bytes())  # Synthetic fixture only.
    with pytest.raises(ValueError, match="Known frozen test"):
        forbid_test_input(release, copied)


def test_pretrained_architecture_forward_backward_freezes_bn_without_downloads():
    import torch

    from damage_analytics.task09.models import build_model
    from damage_analytics.training.config import ModelConfig

    pytest.importorskip("torchvision")
    model = build_model("lraspp_mobilenet_v3_large", ModelConfig(), pretrained=False).train()
    normalizers = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    before = [m.running_mean.clone() for m in normalizers]
    logits = model(torch.randn(2, 3, 32, 32))
    assert logits.shape == (2, 9, 32, 32)
    logits.square().mean().backward()
    assert all(not m.training and m.weight.requires_grad for m in normalizers)
    assert all(
        torch.equal(old, layer.running_mean)
        for old, layer in zip(before, normalizers, strict=True)
    )


def test_inference_native_alignment_and_nonfinite_logit_rejection(tmp_path):
    import torch

    from damage_analytics.task09.pipeline import Segmenter
    from damage_analytics.training.config import DataConfig

    class Model(torch.nn.Module):
        invalid = False

        def forward(self, images):
            output = torch.zeros((images.shape[0], 9, *images.shape[-2:]))
            output[:, 0] = 1
            output[:, 2, :, images.shape[-1] // 2 :] = 8
            if self.invalid:
                output[0, 0, 0, 0] = float("nan")
            return output

    runner = Segmenter.__new__(Segmenter)
    runner.device, runner.model = torch.device("cpu"), Model()
    runner.data = DataConfig(image_size=(16, 16))
    runner.maximum_image_pixels = 1000
    runner.identity = {"name": "fixture", "version": "fixture", "checkpoint_sha256": None}
    runner.checkpoint = tmp_path / "fixture.pt"
    runner.checkpoint.write_bytes(b"local fixture fingerprint, not a model")
    runner.digest = sha256_file(runner.checkpoint)
    report, arrays, oriented = runner.predict(Image.new("RGB", (21, 11)), FeaturePolicy())
    assert arrays["raw"].shape == (11, 21) and oriented.size == (21, 11)
    assert arrays["raw"][0, 0] == 0 and arrays["raw"][-1, -1] == 2
    assert report["regions"][0]["bbox_xyxy"][2:] == [21, 11]
    assert sum(report["coverage"]["raw_class_pixels"]) == 231
    runner.model.invalid = True
    with pytest.raises(ValueError, match="nonfinite"):
        runner.predict(Image.new("RGB", (21, 11)), FeaturePolicy())


def test_synthetic_checkpoint_native_features_for_both_models(tmp_path):
    pytest.importorskip("torchvision")
    from damage_analytics.task09.smoke import smoke

    result = smoke(tmp_path / "smoke")
    assert len(result["models"]) == 2 and not result["pretrained_downloaded"]
    assert all(m["checkpoint_reload"] and m["native_features_validated"] for m in result["models"])


def test_tiny_training_fresh_evaluation_and_structured_inference(context, tmp_path, monkeypatch):
    from damage_analytics.task09.evaluation import fresh_evaluation
    from damage_analytics.task09.experiments import train
    from damage_analytics.task09.pipeline import predict_file

    config, release = context
    folder = train(config, release, "unet", "synthetic", synthetic=True)
    assert (folder / "best.pt").exists()
    result = fresh_evaluation(config, release, folder / "best.pt", tmp_path / "evaluation")
    assert result["samples"] == 1 and result["fresh_model_forward"]
    image = release.root / release.samples("validation")[0].image
    report = predict_file(config, release, folder / "best.pt", image, tmp_path / "inference")
    assert report["model"]["name"] == "unet" and report["uncertainty"]["available"]
    assert report["image"]["width"] == 12 and report["image"]["height"] == 8
    validate_report(json.loads((tmp_path / "inference/damage_features.json").read_text()))
    with monkeypatch.context() as patch:
        patch.setattr(
            Image.Image, "load", lambda *args, **kwargs: pytest.fail("Oversized pixels decoded")
        )
        with pytest.raises(ValueError, match="bounded JPEG/PNG"):
            predict_file(
                replace(config, maximum_image_pixels=1),
                release,
                folder / "best.pt",
                image,
                tmp_path / "oversized",
            )


def test_comparison_rejects_unmatched_settings_before_metric_reads(context, monkeypatch):
    from damage_analytics.task09 import experiments

    config, release = context
    for index in range(2):
        folder = config.output_dir / "experiments" / f"run{index}"
        folder.mkdir(parents=True)
        metadata = {
            "dataset": release.identity(),
            "data": {},
            "optimization": {"epochs": 20 + index},
            "seed": 42,
            "device": "cpu",
        }
        (folder / "run.json").write_text(json.dumps(metadata))
        (folder / "status.json").write_text(json.dumps({"status": "completed"}))
    monkeypatch.setattr(
        experiments, "read_validation", lambda p: pytest.fail("Must reject before loading metrics")
    )
    with pytest.raises(ValueError, match="settings differ"):
        experiments.compare_runs(
            config, release, ["run0", "run1"], config.output_dir / "comparison.json"
        )
