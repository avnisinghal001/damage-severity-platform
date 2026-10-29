"""Torch-independent seals, aligned transforms, native restoration and resizing audit checks."""

from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from damage_analytics.training.audit import audit_resizing
from damage_analytics.training.config import DataConfig, load_training_config, validate_config
from damage_analytics.training.manifests import open_release, read_pair
from damage_analytics.training.transforms import (
    choose_crop,
    frame_for,
    prepare_pair,
    restore_prediction,
    transform_mask,
)
from damage_analytics.training.weights import training_weights


def test_pinned_splits_and_test_access_guard(training_release):
    release = open_release(training_release, full_integrity=True)
    assert len(release.samples("train")) == len(release.samples("validation")) == 1
    with pytest.raises(ValueError, match="explicit final-test"):
        release.samples("test")
    assert len(release.samples("test", final_test=True)) == 1
    image, mask = read_pair(release, release.samples("train")[0])
    assert image.size == mask.size == (12, 8)
    with pytest.raises(ValueError, match="Pinned release"):
        open_release(replace(training_release, checksums_sha256="0" * 64))


def test_tampered_mask_and_split_are_rejected(training_release):
    release = open_release(training_release)
    path = release.root / release.samples("train")[0].mask
    path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="file changed"):
        read_pair(release, release.samples("train")[0])
    (release.root / "splits/train.csv").write_text("tampered")
    with pytest.raises(ValueError, match="split changed"):
        open_release(training_release)


@pytest.mark.parametrize("horizontal,vertical", [(0, 0), (1, 0), (0, 1), (1, 1)])
def test_paired_flips_padding_and_normalization(horizontal, vertical):
    labels = np.zeros((4, 8), dtype=np.uint8)
    labels[1:3, 1:4] = 1
    rgb = np.stack([labels * 255] * 3, axis=-1)
    config = DataConfig(
        image_size=(8, 8),
        mean=(0, 0, 0),
        std=(1, 1, 1),
        crop_probability=0,
        horizontal_flip=horizontal,
        vertical_flip=vertical,
    )
    image, mask, _ = prepare_pair(
        Image.fromarray(rgb),
        Image.fromarray(labels),
        config,
        training=True,
        rng=np.random.default_rng(42),
    )
    assert image.shape == (3, 8, 8) and image.dtype == np.float32 and mask.dtype == np.int64
    valid = mask != 255
    np.testing.assert_array_equal(image[0, valid], mask[valid])
    assert np.all(image[:, ~valid] == 0)
    assert set(np.unique(mask)) == {0, 1, 255}


def test_foreground_crop_includes_guided_rare_pixel_and_is_repeatable():
    values = np.zeros((40, 60), dtype=np.uint8)
    values[39, 59] = 8
    mask = Image.fromarray(values)
    config = DataConfig(crop_size=(8, 8), foreground_crop_probability=1)
    for seed in range(10):
        box = choose_crop(mask, config, np.random.default_rng(seed))
        assert 8 in np.asarray(mask.crop(box))
        assert box == choose_crop(mask, config, np.random.default_rng(seed))


def test_validation_ignores_all_augmentation_and_keeps_nearest_ids():
    values = np.zeros((11, 17), dtype=np.uint8)
    values[2:9, 2:9] = 7
    image, mask = Image.new("RGB", (17, 11)), Image.fromarray(values)
    config = DataConfig(
        image_size=(16, 16), crop_probability=1, horizontal_flip=1, vertical_flip=1
    )
    first = prepare_pair(image, mask, config, training=False, rng=np.random.default_rng(1))
    second = prepare_pair(image, mask, config, training=False, rng=np.random.default_rng(99))
    np.testing.assert_array_equal(first[0], second[0])
    assert first[2].crop_box == (0, 0, 17, 11)
    assert set(np.unique(first[1])).issubset({0, 7, 255})
    restored = restore_prediction(first[1], first[2])
    assert restored.shape == values.shape and set(np.unique(restored)).issubset({0, 7})
    with pytest.raises(ValueError, match="crop"):
        restore_prediction(first[1], frame_for((17, 11), (16, 16), (0, 0, 8, 8)))


def test_thin_foreground_destruction_can_be_detected():
    values = np.zeros((101, 101), dtype=np.uint8)
    values[1, 1] = 1
    mask = Image.fromarray(values)
    frame = frame_for(mask.size, (8, 8))
    resized = np.asarray(transform_mask(mask, frame, (8, 8)))
    assert not np.any(resized == 1)
    assert not np.any(restore_prediction(resized, frame) == 1)


def test_resize_audit_serializes_and_never_requests_test(training_release, monkeypatch):
    release = open_release(training_release)
    original = type(release).samples

    def guard(self, split, **kwargs):
        assert split != "test"
        return original(self, split, **kwargs)

    monkeypatch.setattr(type(release), "samples", guard)
    report = audit_resizing(training_release, release)
    assert report["test_accessed"] is False
    assert set(report["splits"]) == {"train", "validation"}
    assert (training_release.output_dir / "data_audit/resize_audit.json").is_file()


def test_class_weights_only_use_supplied_training_samples(training_release):
    release = open_release(training_release)
    weights, report = training_weights(release.samples("train"), training_release.optimization)
    expected = release.samples("train")[0].pixel_counts
    assert report["native_training_pixel_counts"] == list(expected)
    assert weights[2] > weights[0] and np.all(weights[3:] == 0)
    assert weights.max() <= training_release.optimization.max_class_weight
    plain, _ = training_weights(
        release.samples("train"), replace(training_release.optimization, class_weighting="none")
    )
    assert np.all(plain == 1)


@pytest.mark.parametrize(
    "change",
    [
        {"device": "cuda"},
        {"seed": -1},
        {"checksums_sha256": "bad"},
        {"num_workers": -1},
    ],
)
def test_invalid_configuration_rejected(training_release, change):
    with pytest.raises(ValueError):
        validate_config(replace(training_release, **change))


def test_configuration_loads_actual_pin_and_rejects_fractional_model_size():
    from pathlib import Path

    config = load_training_config(Path("config/task03.toml"))
    assert config.release_id == "rc1841-curated-v1-d34771dbb662406a"
    with pytest.raises(ValueError, match="divisible"):
        validate_config(replace(config, data=replace(config.data, image_size=(255, 256))))
