"""Synthetic archive, annotation, leakage and immutable-release checks."""

import io
import json
from dataclasses import replace
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pytest
from PIL import Image, ImageDraw

from damage_analytics.task07.annotations import decode_mask, validate_json, validate_yolo
from damage_analytics.task07.archives import dataset_archives
from damage_analytics.task07.config import Config, TrainConfig
from damage_analytics.task07.crosscheck import overlap
from damage_analytics.task07.duplicates import HammingTree, evidence_groups
from damage_analytics.task07.prepare import prepare
from damage_analytics.task07.release import open_release
from damage_analytics.task07.segmentation import evaluate_masks
from damage_analytics.task07.smoke import synthetic_archive
from damage_analytics.task07.splits import assign_splits, split_report


@pytest.fixture
def task07_config(tmp_path):
    archive = tmp_path / "sdnet.zip"
    synthetic_archive(archive)
    return Config(
        archive,
        tmp_path / "dam.zip",
        tmp_path / "output",
        42,
        (0.7, 0.15, 0.15),
        None,
        None,
        0,
        1000,
        {
            "category_names": ["source_category_0", "source_category_1"],
            "semantic_names_verified": False,
            "semantic_evidence": "",
        },
        TrainConfig(),
    )


def encoded(image, format="PNG"):
    stream = io.BytesIO()
    image.save(stream, format=format)
    return stream.getvalue()


def dam_fixture(path):
    rng = np.random.default_rng(3)
    nested = {}
    for role in ("classification", "detection", "segmentation"):
        stream = io.BytesIO()
        nested[role] = stream
        with ZipFile(stream, "w", ZIP_DEFLATED) as archive:
            for index in range(6):
                if role == "classification":
                    folder = "Crack" if index % 2 else "Non-Crack"
                    image = Image.fromarray(rng.integers(0, 255, (16, 16, 3), dtype=np.uint8))
                    archive.writestr(
                        f"Damage Classification/{folder}/{index}.jpg", encoded(image, "JPEG")
                    )
                    if index == 0:
                        archive.writestr(
                            f"Damage Classification/{folder}/{index}.png", encoded(image)
                        )
                    continue
                name = f"E ({index}).jpg"
                # The same actual image bytes occur in both views.
                image = Image.new("RGB", (16, 16), (20 + index, 40 + index, 60 + index))
                base = "Damage Detection" if role == "detection" else "Damage Segmentaion/Easy"
                archive.writestr(f"{base}/Images/{name}", encoded(image, "JPEG"))
                annotations = []
                mask = Image.new("RGB", (16, 16))
                draw = ImageDraw.Draw(mask)
                lines = []
                for category, (x, y) in enumerate(((1, 1), (9, 9))):
                    ring = [x, y, x + 3, y, x + 3, y + 3, x, y + 3]
                    item = {"category_id": category, "bbox": [x, y, 3, 3], "iscrowd": 0}
                    if role == "segmentation":
                        item["segmentation"] = [ring]
                        lines.append(str(category) + " " + " ".join(str(v / 16) for v in ring))
                    else:
                        item["area"] = 9
                        lines.append(
                            f"{category} {(x + 1.5) / 16} {(y + 1.5) / 16} {3 / 16} {3 / 16}"
                        )
                    annotations.append(item)
                    draw.rectangle(
                        (x, y, x + 3, y + 3), fill=(255, 0, 0) if category == 0 else (0, 0, 255)
                    )
                value = {
                    "image": {"file_name": name, "width": 16, "height": 16},
                    "annotations": annotations,
                }
                stem = Path(name).stem
                archive.writestr(f"{base}/Labels/Pascal VOC/{stem}.json", json.dumps(value))
                archive.writestr(f"{base}/Labels/Yolo/{stem}.txt", "\n".join(lines))
                if role == "segmentation":
                    archive.writestr(f"{base}/Labels/Mask/{stem}_mask.png", encoded(mask))
    names = {
        "classification": "Damage Classification.zip",
        "detection": "Damage Detection.zip",
        "segmentation": "Damage Segmentaion.zip",
    }
    with ZipFile(path, "w") as outer:
        for role, stream in nested.items():
            outer.writestr(f"DamSegment/{names[role]}", stream.getvalue())


def test_prepare_is_idempotent_and_splits_are_whole_prefix_groups(task07_config):
    path = prepare(task07_config, "sdnet2018")
    before = (path / "checksums.json").read_bytes()
    assert prepare(task07_config, "sdnet2018") == path
    assert before == (path / "checksums.json").read_bytes()
    release = open_release(path, full=True)
    assert len(release.rows) == 36
    assert release.quality["known_component_checks_passed"]
    assert not release.quality["source_identity_verified"]
    assert not release.quality["final_evaluation_eligible"]
    for prefix in {r["candidate_source"] for r in release.rows}:
        assert len({r["split"] for r in release.rows if r["candidate_source"] == prefix}) == 1
    with pytest.raises(ValueError, match="explicit"):
        release.samples("test")
    changed = replace(task07_config, seed=99)
    assert prepare(changed, "sdnet2018") != path


def test_full_seal_and_manifest_tampering_detected(task07_config):
    release = open_release(prepare(task07_config, "sdnet2018"))
    image = release.root / release.rows[0]["image"]
    image.write_bytes(image.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="changed"):
        release.file(release.rows[0]["image"])
    (release.root / "splits/train.csv").write_text("tampered")
    with pytest.raises(ValueError, match="metadata"):
        open_release(release.root)


def test_dam_views_same_split_stem_collision_and_native_mask_evaluation(task07_config, tmp_path):
    dam_fixture(task07_config.damsegment_archive)
    release = open_release(prepare(task07_config, "damsegment"), full=True)
    assert sum(r["task"] == "classification" for r in release.rows) == 7
    assert sum(r["task"] == "segmentation" for r in release.rows) == 6
    for row in release.rows:
        assert (
            len({r["split"] for r in release.rows if r["decoded_sha256"] == row["decoded_sha256"]})
            == 1
        )
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    rows = release.samples("validation", "segmentation")
    assert rows
    for row in rows:
        (predictions / f"{row['sample_id']}.png").write_bytes(
            release.file(row["mask"]).read_bytes()
        )
    report = evaluate_masks(release, predictions, tmp_path / "mask_eval")
    assert report["metrics"]["foreground_miou"] == 1
    assert report["taxonomy"]["labels"] == ["background", "source_category_0", "source_category_1"]
    with pytest.raises(ValueError, match="verified"):
        evaluate_masks(release, predictions, tmp_path / "blocked", final_test=True)
    from damage_analytics.task07.visualization import detection_gallery

    gallery = detection_gallery(release, tmp_path / "gallery", count=2)
    assert list(gallery.glob("*.png")) and (gallery / "gallery.json").exists()


@pytest.mark.parametrize("name", ["../escape.jpg", "/absolute.jpg", "a\\b.jpg"])
def test_nested_zip_traversal_rejected(tmp_path, name):
    nested = io.BytesIO()
    with ZipFile(nested, "w") as archive:
        archive.writestr(name, b"image")
    path = tmp_path / "bad.zip"
    with ZipFile(path, "w") as archive:
        archive.writestr("DamSegment/Damage Detection.zip", nested.getvalue())
    with pytest.raises(ValueError, match="Unsafe"):
        with dataset_archives(path, "damsegment"):
            pass


def test_native_mask_mapping_rejects_unknown_colors():
    mask = Image.new("RGB", (3, 1))
    mask.putdata([(0, 0, 0), (255, 0, 0), (0, 0, 255)])
    assert decode_mask(mask).tolist() == [[0, 1, 2]]
    mask.putpixel((0, 0), (1, 0, 0))
    with pytest.raises(ValueError, match="color"):
        decode_mask(mask)


@pytest.mark.parametrize(
    "box", [[0, 0, 0, 1], [-1, 0, 2, 2], [0, 0, 99, 2], [float("nan"), 0, 1, 2]]
)
def test_invalid_boxes_rejected(box):
    value = {
        "image": {"file_name": "x.jpg", "width": 16, "height": 16},
        "annotations": [{"category_id": 0, "bbox": box}],
    }
    with pytest.raises(ValueError):
        validate_json(value, (16, 16), "x.jpg", polygons=False)


def test_bad_polygon_and_yolo_rejected():
    for text, polygons in (
        ("0 .1 .1 .8 .8", False),
        ("0 0 0 1 1 0 1 1 0", True),
        ("2 .5 .5 .2 .2", False),
    ):
        with pytest.raises(ValueError):
            validate_yolo(text, polygons=polygons)


def test_bk_tree_matches_brute_force_and_near_matches_are_not_identity():
    tree, values = HammingTree(), [0, 1, 3, 7, 15, 16, 32]
    for i, value in enumerate(values):
        tree.add(value, str(i))
    for query in values:
        assert set(tree.query(query, 2)) == {
            str(i) for i, value in enumerate(values) if (query ^ value).bit_count() <= 2
        }
    rows = [
        {
            "sample_id": str(i),
            "image_sha256": str(i),
            "decoded_sha256": str(i),
            "phash": i,
            "dhash": i,
            "candidate_source": "",
            "source_group": "",
        }
        for i in range(2)
    ]
    groups, near, _ = evidence_groups(rows, 2, 10)
    assert len(near) == 1 and groups["0"] != groups["1"]


def test_duplicate_reviews_require_unique_evidence_and_group_reviewed_sources(tmp_path):
    rows = [
        {
            "sample_id": str(i),
            "image_sha256": str(i),
            "decoded_sha256": str(i),
            "phash": i,
            "dhash": i,
            "candidate_source": "",
            "source_group": "",
        }
        for i in range(2)
    ]
    _, near, _ = evidence_groups(rows, 2, 10)
    pair = near[0]
    reviews = tmp_path / "reviews.csv"
    header = "left,right,decision,evidence\n"
    line = f"{pair['left']},{pair['right']},same_source,Known synthetic parent\n"
    reviews.write_text(header + line)
    groups, reviewed, _ = evidence_groups(rows, 2, 10, reviews)
    assert groups["0"] == groups["1"] and reviewed[0]["decision"] == "same_source"
    reviews.write_text(header + line + line)
    with pytest.raises(ValueError, match="duplicate review"):
        evidence_groups(rows, 2, 10, reviews)


def test_corrupt_sample_is_logged_excluded_and_preserved(task07_config):
    with ZipFile(task07_config.sdnet2018_archive, "a") as archive:
        archive.writestr("D/UD/999-corrupt.jpg", b"not an image")
    release = open_release(prepare(task07_config, "sdnet2018"), full=True)
    assert release.quality["discovered"] == 37 and len(release.rows) == 36
    assert release.quality["exclusions"] == 1
    assert "999-corrupt" in (release.root / "exclusions.csv").read_text()
    assert (
        release.root / "payload/classification/D/UD/999-corrupt.jpg"
    ).read_bytes() == b"not an image"


def test_split_reproducibility_and_leakage_failure():
    rows = [
        {
            "sample_id": str(i),
            "strata": [f"class:{i % 2}"],
            "task": "classification",
            "label": i % 2,
            "decoded_sha256": str(i),
            "source_group": "",
            "source_evidence": "",
        }
        for i in range(12)
    ]
    groups = {r["sample_id"]: r["sample_id"] for r in rows}
    a = assign_splits(rows, groups, (0.7, 0.15, 0.15), 42)
    assert a == assign_splits(list(reversed(rows)), groups, (0.7, 0.15, 0.15), 42)
    for row in rows:
        row.update(split=a[row["sample_id"]], group_id=groups[row["sample_id"]])
    assert split_report(rows, [], False)["known_component_checks_passed"]
    first = rows[0]
    other = next(r for r in rows if r["split"] != first["split"])
    other["decoded_sha256"] = first["decoded_sha256"]
    assert not split_report(rows, [], False)["known_component_checks_passed"]


def test_verified_synthetic_source_evidence_and_exact_cross_dataset_match(task07_config, tmp_path):
    import csv

    original = open_release(prepare(task07_config, "sdnet2018"))
    evidence = tmp_path / "sources.csv"
    with evidence.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["sample_id", "source_group", "evidence"])
        writer.writeheader()
        for row in original.rows:
            writer.writerow(
                {
                    "sample_id": row["sample_id"],
                    "source_group": row["candidate_source"],
                    "evidence": "Known synthetic generator parent identity",
                }
            )
    verified = open_release(
        prepare(replace(task07_config, source_groups_csv=evidence), "sdnet2018")
    )
    assert verified.quality["source_identity_verified"]
    assert verified.quality["classification_coverage_complete"]
    assert verified.quality["final_evaluation_eligible"]
    target = dict(verified.rows[0], sample_id="external-image")
    report = overlap(verified.rows, (target,), radius=0)
    assert report["exact_pairs"] and report["exact_pair_count"] > 0


def test_perceptual_budget_is_reported_not_claimed_complete():
    rows = [
        {
            "sample_id": str(i),
            "image_sha256": str(i),
            "decoded_sha256": str(i),
            "phash": 0,
            "dhash": 0,
            "candidate_source": "",
            "source_group": "",
        }
        for i in range(8)
    ]
    _, near, evidence = evidence_groups(rows, 0, 2)
    assert len(near) == 2 and evidence["screening_truncated"]


def test_invalid_polygon_and_unknown_source_id():
    base = {
        "image": {"file_name": "x.jpg", "width": 16, "height": 16},
        "annotations": [
            {
                "category_id": 0,
                "bbox": [0, 0, 16, 16],
                "segmentation": [[0, 0, 10, 10, 0, 10, 10, 0]],
            }
        ],
    }
    with pytest.raises(ValueError, match="Degenerate|self-intersecting"):
        validate_json(base, (16, 16), "x.jpg", polygons=True)
    base["annotations"][0]["category_id"] = 7
    with pytest.raises(ValueError, match="category"):
        validate_json(base, (16, 16), "x.jpg", polygons=True)
    base["annotations"] = [None]
    with pytest.raises(ValueError, match="objects"):
        validate_json(base, (16, 16), "x.jpg", polygons=False)


def test_known_source_crossing_is_detected_independently_of_component_names():
    rows = [
        {
            "sample_id": str(i),
            "group_id": str(i),
            "decoded_sha256": str(i),
            "source_group": "known-parent",
            "source_evidence": "fixture",
            "task": "classification",
            "label": i % 2,
            "strata": [],
            "split": split,
        }
        for i, split in enumerate(("train", "validation"))
    ]
    assert not split_report(rows, [], False)["known_component_checks_passed"]


def test_final_test_refuses_unknown_sources_before_loading_a_backbone(task07_config, tmp_path):
    torch = pytest.importorskip("torch")
    from damage_analytics.task07.evaluation import evaluate

    release = open_release(prepare(task07_config, "sdnet2018"))
    folder = tmp_path / "experiment"
    folder.mkdir()
    checkpoint = folder / "best.pt"
    torch.save(
        {
            "task07_format": 1,
            "task": "binary_crack_presence",
            "purpose": "training",
            "dataset": release.identity(),
            "release_dir": str(release.root),
            "epoch": 1,
            "threshold": {
                "selected_on": "validation",
                "threshold": 0.7,
                "split_sha256": release.identity()["splits"]["validation"],
            },
        },
        checkpoint,
    )
    (folder / "status.json").write_text(json.dumps({"status": "completed", "best_epoch": 1}))
    with pytest.raises(ValueError, match="Final evaluation"):
        evaluate(checkpoint, release, tmp_path / "evaluations", final_test=True)
    assert not (tmp_path / "evaluations").exists()
