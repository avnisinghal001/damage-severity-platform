"""Image format, orientation, color, and source-preservation checks."""

from dataclasses import replace

import numpy as np
import pytest
from PIL import Image, ImageOps

from damage_analytics.curation.config import CleaningPolicy
from damage_analytics.curation.images import prepare_image
from damage_analytics.io import sha256_file


def test_exif_alignment_requires_unique_dimension_evidence_and_preserves_source(tmp_path):
    path = tmp_path / "sample.jpg"
    array = np.random.default_rng(5).integers(0, 255, (6, 8, 3), dtype=np.uint8)
    exif = Image.Exif()
    exif[274] = 6
    Image.fromarray(array).save(path, exif=exif)
    before = sha256_file(path)
    result, metadata, decisions = prepare_image(
        path, {"imageWidth": 6, "imageHeight": 8}, "sample", CleaningPolicy()
    )
    with Image.open(path) as original:
        expected = ImageOps.exif_transpose(original).convert("RGB")
        np.testing.assert_array_equal(result, expected)
    assert result.size == (6, 8) and metadata["exif_orientation"] == 6
    assert decisions[0]["code"] == "exif_frame_alignment"
    assert before == sha256_file(path)
    assert not result.info
    _, _, strict = prepare_image(
        path,
        {"imageWidth": 6, "imageHeight": 8},
        "sample",
        replace(CleaningPolicy(), exif_policy="exclude"),
    )
    assert any(row["action"] == "exclude" for row in strict)


def test_same_dimension_exif_rotation_is_ambiguous(tmp_path):
    path = tmp_path / "sample.jpg"
    exif = Image.Exif()
    exif[274] = 3
    Image.new("RGB", (8, 6)).save(path, exif=exif)
    _, _, decisions = prepare_image(
        path, {"imageWidth": 8, "imageHeight": 6}, "sample", CleaningPolicy()
    )
    assert decisions[0]["code"] == "ambiguous_exif_frame"


@pytest.mark.parametrize("mode,suffix", [("CMYK", ".jpg"), ("L", ".png"), ("P", ".png")])
def test_rgb_conversion_is_explicit_and_configurable(tmp_path, mode, suffix):
    path = tmp_path / f"sample{suffix}"
    Image.new(mode, (8, 6)).save(path)
    result, _, decisions = prepare_image(path, {}, "sample", CleaningPolicy())
    assert result.mode == "RGB"
    assert any(row["code"] == "rgb_conversion" and row["action"] == "fix" for row in decisions)
    _, _, strict = prepare_image(
        path, {}, "sample", replace(CleaningPolicy(), allow_rgb_conversion=False)
    )
    assert any(row["action"] == "exclude" for row in strict)


def test_transparency_and_mislabeled_format_are_not_silently_accepted(tmp_path):
    path = tmp_path / "sample.jpg"
    Image.new("RGBA", (8, 6), (20, 20, 20, 0)).save(path, format="PNG")
    _, _, findings = prepare_image(path, {}, "sample", CleaningPolicy())
    assert {row["code"] for row in findings} >= {"transparent_pixels", "format_extension_mismatch"}


def test_multiframe_image_is_excluded(tmp_path):
    path = tmp_path / "sample.tif"
    Image.new("RGB", (8, 6)).save(path, save_all=True, append_images=[Image.new("RGB", (8, 6))])
    _, _, findings = prepare_image(path, {}, "sample", CleaningPolicy())
    assert any(row["code"] == "multiframe_image" for row in findings)
