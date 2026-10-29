"""Temporal grouping uses actual camera/time evidence and never filename families."""

from dataclasses import replace

import pytest
from PIL import Image

from damage_analytics.curation.config import SimilarityPolicy
from damage_analytics.curation.metadata import camera_metadata, capture_session_edges
from damage_analytics.curation.similarity import connected_groups


def test_camera_metadata_is_read_and_time_basis_is_explicit(tmp_path):
    path = tmp_path / "sample.jpg"
    exif = Image.Exif()
    exif[271], exif[272], exif[306] = "Camera Maker", "Camera Model", "2020:01:01 12:00:00"
    Image.new("RGB", (8, 6)).save(path, exif=exif)
    with Image.open(path) as image:
        result = camera_metadata(image)
    assert result["camera_make"] == "Camera Maker"
    assert result["capture_time"] == "2020:01:01 12:00:00"
    assert result["capture_time_basis"] == "DateTime"
    assert result["camera_serial"] == ""


def test_capture_sessions_are_transitive_and_do_not_bridge_long_gaps():
    rows = [
        {
            "sample_id": name,
            "camera_make": "Camera",
            "camera_model": "Model",
            "capture_time": timestamp,
        }
        for name, timestamp in (
            ("a", "2020:01:01 12:00:00"),
            ("excluded_b", "2020:01:01 12:05:00"),
            ("c", "2020:01:01 12:10:00"),
            ("d", "2020:01:02 12:00:00"),
        )
    ]
    edges = capture_session_edges(rows, SimilarityPolicy(capture_session_gap_seconds=300))
    assert len(edges) == 2
    groups = connected_groups([row["sample_id"] for row in rows], edges)
    assert groups["a"] == groups["c"] and groups["a"] != groups["d"]
    assert all("not proven physical" in edge["evidence"] for edge in edges)
    assert not capture_session_edges(
        rows, replace(SimilarityPolicy(), group_capture_sessions=False)
    )


@pytest.mark.parametrize("value", ["", "invalid", "2020:99:99 12:00:00"])
def test_invalid_metadata_does_not_invent_capture_groups(value):
    rows = [
        {
            "sample_id": name,
            "camera_make": "Camera",
            "camera_model": "Model",
            "capture_time": value,
        }
        for name in ("a", "b")
    ]
    assert not capture_session_edges(rows, SimilarityPolicy())


def test_same_time_but_different_device_evidence_is_not_grouped():
    rows = [
        {
            "sample_id": name,
            "camera_make": "Camera",
            "camera_model": model,
            "capture_time": "2020:01:01 12:00:00",
            "camera_serial": serial,
        }
        for name, model, serial in (("a", "M1", "one"), ("b", "M1", "two"), ("c", "M2", "one"))
    ]
    assert not capture_session_edges(rows, SimilarityPolicy())
