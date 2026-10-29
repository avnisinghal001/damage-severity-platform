"""Duplicate evidence, annotation identity, source-group evidence and transitive leakage."""

from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from damage_analytics.annotations import Polygon
from damage_analytics.curation.config import CleaningPolicy, SimilarityPolicy
from damage_analytics.curation.images import pixel_hash
from damage_analytics.curation.samples import resolve_exact_images
from damage_analytics.curation.similarity import (
    annotation_hash,
    connected_groups,
    image_fingerprint,
    similarity_edges,
    source_edges,
)
from damage_analytics.curation.splitting import leakage_checks


def fingerprint_row(name, image):
    return {
        "sample_id": name,
        "decoded_sha256": pixel_hash(image),
        "width": image.width,
        "height": image.height,
        **image_fingerprint(image),
    }


def test_exact_duplicate_and_near_resized_images_have_explicit_evidence():
    image = Image.fromarray(
        np.random.default_rng(19).integers(0, 255, (64, 64, 3), dtype=np.uint8)
    )
    duplicate = fingerprint_row("b", image.copy())
    resized = image.resize((128, 128), Image.Resampling.LANCZOS)
    rows = [fingerprint_row("a", image), duplicate, fingerprint_row("c", resized)]
    edges = similarity_edges(rows, SimilarityPolicy())
    assert any(row["kind"] == "decoded_sha256" for row in edges)
    assert any(row["kind"] == "near_image_candidate" and row["right"] == "c" for row in edges)
    groups = connected_groups(["a", "b", "c"], edges)
    assert len(set(groups.values())) == 1


def test_similar_labels_alone_do_not_create_near_image_evidence():
    red, blue = Image.new("RGB", (16, 16), "red"), Image.new("RGB", (16, 16), "blue")
    edges = similarity_edges(
        [fingerprint_row("red", red), fingerprint_row("blue", blue)], SimilarityPolicy()
    )
    assert not edges


def test_annotation_hash_is_ring_start_and_winding_invariant_but_label_sensitive():
    points = ((1, 1), (4, 1), (4, 4), (1, 4))
    original = annotation_hash([Polygon("Microcrack", points)], (8, 6))
    rotated = points[2:] + points[:2]
    assert original == annotation_hash([Polygon("Microcrack", rotated)], (8, 6))
    assert original == annotation_hash([Polygon("Microcrack", tuple(reversed(points)))], (8, 6))
    assert original != annotation_hash([Polygon("Structural crack", points)], (8, 6))
    assert annotation_hash([], (8, 6)) == ""


def test_duplicate_ground_truth_conflicts_exclude_every_member():
    records = [
        {"sample_id": name, "decoded_sha256": "same", "included": True, "mask_pixel_sha256": mask}
        for name, mask in (("a", "one"), ("b", "two"))
    ]
    findings = resolve_exact_images(records, CleaningPolicy())
    assert not any(row["included"] for row in records)
    assert {row["code"] for row in findings} == {"exact_image_annotation_conflict"}


def test_identical_pair_deduplication_is_deterministic_and_configurable():
    def records():
        return [
            {
                "sample_id": name,
                "decoded_sha256": "same",
                "included": True,
                "mask_pixel_sha256": "same",
            }
            for name in ("b", "a")
        ]

    rows = records()
    resolve_exact_images(rows, CleaningPolicy())
    assert [row["sample_id"] for row in rows if row["included"]] == ["a"]
    rows = records()
    assert not resolve_exact_images(
        rows, replace(CleaningPolicy(), deduplicate_identical_pairs=False)
    )
    assert all(row["included"] for row in rows)


def test_unusable_annotation_on_identical_image_is_not_arbitrarily_resolved():
    records = [
        {"sample_id": "a", "decoded_sha256": "same", "included": True, "mask_pixel_sha256": "one"},
        {"sample_id": "b", "decoded_sha256": "same", "included": False},
    ]
    resolve_exact_images(records, CleaningPolicy())
    assert not any(row["included"] for row in records)


def test_excluded_intermediates_still_prevent_transitive_leakage():
    edges = [
        {"left": "a", "right": "excluded", "kind": "near_image_candidate"},
        {"left": "excluded", "right": "c", "kind": "source_group"},
    ]
    groups = connected_groups(["a", "excluded", "c"], edges)
    result = leakage_checks(
        [{"sample_id": "a", "split": "train"}, {"sample_id": "c", "split": "test"}], edges, groups
    )
    assert not result["passed"]
    assert result["cross_split_components"] == ["a"]


def test_explicit_source_evidence_is_required_and_validated(tmp_path):
    path = tmp_path / "sources.csv"
    path.write_text(
        "sample_id,source_group,evidence\na,structure_1,inspection log\n"
        "b,structure_1,inspection log\n"
    )
    edges, assigned = source_edges(path, {"a", "b"})
    assert assigned == {"a", "b"} and edges[0]["kind"] == "source_group"
    with pytest.raises(ValueError):
        source_edges(path, {"a"})
    path.write_text("sample_id,source_group,evidence\na,group,\n")
    with pytest.raises(ValueError):
        source_edges(path, {"a"})
    path.write_text("wrong,columns\n")
    with pytest.raises(ValueError):
        source_edges(path, {"a"})
