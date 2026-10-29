"""Synthetic end-to-end releases, provenance, idempotence, and tamper detection."""

import json
import shutil
from dataclasses import replace
from zipfile import ZipFile

import numpy as np
import pytest
from PIL import Image

from damage_analytics.archives import extract_archives
from damage_analytics.config import ArchiveLimits, Config
from damage_analytics.curation.config import (
    CleaningPolicy,
    CurationConfig,
    SimilarityPolicy,
    SplitPolicy,
    load_curation_config,
)
from damage_analytics.curation.pipeline import run_curation, verify_latest
from damage_analytics.curation.provenance import audit_inputs
from damage_analytics.curation.verification import verify_checksums
from damage_analytics.io import sha256_file, write_json
from damage_analytics.validation import validate_dataset


@pytest.fixture
def synthetic_config(tmp_path):
    source = tmp_path / "source"
    for role in ("images", "annotations"):
        (source / role).mkdir(parents=True)
    rng = np.random.default_rng(1337)

    def sample(name, index, label="Microcrack", points=None):
        Image.fromarray(rng.integers(0, 255, (24, 32, 3), dtype=np.uint8)).save(
            source / "images" / f"{name}.png"
        )
        x, y = 1 + index % 7, 1 + index // 7
        write_json(
            source / "annotations" / f"{name}.json",
            {
                "imagePath": f"{name}.png",
                "imageWidth": 32,
                "imageHeight": 24,
                "shapes": [
                    {
                        "label": label,
                        "shape_type": "polygon",
                        "points": points or [[x, y], [x + 4, y], [x + 4, y + 4], [x, y + 4]],
                    }
                ],
            },
        )

    for index in range(20):
        sample(f"base{index:02d}", index, "Microcrack" if index % 2 == 0 else "Structural crack")
    for name, original, change_label in (
        ("zz_duplicate", "base00", False),
        ("conflict", "base01", True),
    ):
        shutil.copyfile(source / "images" / f"{original}.png", source / "images" / f"{name}.png")
        data = json.loads((source / "annotations" / f"{original}.json").read_text())
        data["imagePath"] = f"{name}.png"
        if change_label:
            data["shapes"][0]["label"] = "Microcrack"
        write_json(source / "annotations" / f"{name}.json", data)
    sample("crossing", 21, points=[[1, 1], [5, 5], [1, 5], [5, 1]])
    sample("outside", 22, points=[[-1, 1], [5, 1], [5, 5], [1, 5]])
    sample("unknown", 23, label="invented severity")
    sample("empty", 24)
    data = json.loads((source / "annotations/empty.json").read_text())
    data["shapes"] = []
    write_json(source / "annotations/empty.json", data)
    sample("corrupt", 25)
    (source / "images/corrupt.png").write_bytes(b"unreadable image")
    exif = Image.Exif()
    exif[274] = 6
    Image.fromarray(rng.integers(0, 255, (6, 8, 3), dtype=np.uint8)).save(
        source / "images/oriented.jpg", exif=exif
    )
    write_json(
        source / "annotations/oriented.json",
        {
            "imagePath": "oriented.jpg",
            "imageWidth": 6,
            "imageHeight": 8,
            "shapes": [
                {
                    "label": "Microcrack",
                    "shape_type": "polygon",
                    "points": [[1, 1], [4, 1], [4, 5], [1, 5]],
                }
            ],
        },
    )
    raw, interim, task01, task02 = (
        tmp_path / name for name in ("raw", "interim", "task01", "task02")
    )
    raw.mkdir()
    for role, filename in (("images", "imagedata.zip"), ("annotations", "labeldata.zip")):
        with ZipFile(raw / filename, "w") as archive:
            for path in sorted((source / role).iterdir()):
                archive.write(path, path.name)
    base = Config(raw, interim, task01, ArchiveLimits(), expected_pairs=28)
    extract_archives(base)
    validate_dataset(base)
    return CurationConfig(
        replace(base, output_dir=task02),
        task01,
        task02,
        "synthetic",
        42,
        CleaningPolicy(),
        SimilarityPolicy(),
        SplitPolicy(search_restarts=2),
    )


def test_end_to_end_counts_integrity_and_idempotence(synthetic_config):
    config = synthetic_config
    before = {path.name: sha256_file(path) for path in config.paths.raw_dir.iterdir()}
    folder, quality = run_curation(config)
    assert quality["source_samples"] == 28
    assert quality["valid_samples"] == 20
    assert quality["excluded_samples"] == 8
    assert quality["verification"]["passed"]
    assert quality["verification"]["leakage"]["passed"]
    assert quality["ml_readiness"]["verdict"] == "CONDITIONAL_RESEARCH_USE"
    assert quality["explicit_source_evidence_samples"] == 0
    assert quality["statistics"]["total_pixels"] == 19 * 32 * 24 + 6 * 8
    assert (
        quality["excluded_samples_by_reason_nonexclusive"]["exact_image_annotation_conflict"] == 2
    )
    assert (folder / "task01_audit.json").is_file()
    original_checksums = (folder / "checksums.json").read_bytes()
    original_mtime = (folder / "splits/test.csv").stat().st_mtime_ns
    again, report = run_curation(config)
    assert again == folder and report == quality
    assert original_checksums == (folder / "checksums.json").read_bytes()
    assert original_mtime == (folder / "splits/test.csv").stat().st_mtime_ns
    assert verify_latest(config)["passed"]
    assert before == {path.name: sha256_file(path) for path in config.paths.raw_dir.iterdir()}


def test_tampering_with_output_or_source_is_rejected(synthetic_config):
    folder, _ = run_curation(synthetic_config)
    (folder / "splits/test.csv").write_text("tampered")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_checksums(folder)
    path = next((synthetic_config.paths.extracted_dir / "annotations").iterdir())
    path.write_text("{}")
    with pytest.raises(ValueError, match="Source integrity audit failed"):
        audit_inputs(synthetic_config)


@pytest.mark.parametrize(
    "setting",
    [
        '[cleaning]\nexif_policy="guess"\n',
        "[cleaning]\nout_of_bounds_tolerance=-1\n",
        "[splitting]\nfractions=[0.7,0.2,0.2]\n",
        "[similarity]\nphash_distance=99\n",
    ],
)
def test_invalid_policies_are_rejected(tmp_path, setting):
    config = tmp_path / "config"
    config.mkdir()
    (config / "base.toml").write_text(
        'project_root=".."\nraw_dir="raw"\nextracted_dir="interim"\noutput_dir="task01"\n'
    )
    path = config / "task02.toml"
    path.write_text(
        'base_config="base.toml"\noutput_dir="../task02"\n'
        'dataset_name="test"\nrandom_seed=42\n' + setting
    )
    with pytest.raises(ValueError):
        load_curation_config(path)


def test_output_cannot_overlap_input_or_project_source(tmp_path):
    config = tmp_path / "config"
    config.mkdir()
    (config / "base.toml").write_text(
        'project_root=".."\nraw_dir="raw"\nextracted_dir="interim"\noutput_dir="task01"\n'
    )
    for output in ("../raw/nested", "../task01", "../src", ".."):
        path = config / "task02.toml"
        path.write_text(
            f'base_config="base.toml"\noutput_dir="{output}"\n'
            'dataset_name="test"\nrandom_seed=42\n'
        )
        with pytest.raises(ValueError):
            load_curation_config(path)


@pytest.mark.parametrize("name,seed", [("../escape", 42), ("unsafe/name", 42), ("safe", -1)])
def test_release_name_and_seed_cannot_bypass_release_safety(tmp_path, name, seed):
    config = tmp_path / "config"
    config.mkdir()
    (config / "base.toml").write_text(
        'project_root=".."\nraw_dir="raw"\nextracted_dir="interim"\noutput_dir="task01"\n'
    )
    path = config / "task02.toml"
    path.write_text(
        f'base_config="base.toml"\noutput_dir="../task02"\n'
        f'dataset_name="{name}"\nrandom_seed={seed}\n'
    )
    with pytest.raises(ValueError):
        load_curation_config(path)
