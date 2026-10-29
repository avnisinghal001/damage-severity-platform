"""Conservative temporal source candidates from actual camera/time metadata."""

from collections import defaultdict
from datetime import datetime

from PIL import Image

from damage_analytics.curation.config import SimilarityPolicy


def camera_metadata(image: Image.Image) -> dict:
    """Read available EXIF identity/time fields without inventing structure IDs.

    DateTimeOriginal is preferred; DateTime is a weaker fallback (it can be an edit
    time). Missing serial numbers mean camera make/model is not device-unique.
    Candidates are grouped conservatively and never called proven physical sources.
    """
    exif = image.getexif()
    detailed = exif.get_ifd(34665) if 34665 in exif else {}
    timestamp = detailed.get(36867) or exif.get(306, "")
    fields = {
        "camera_make": exif.get(271, ""),
        "camera_model": exif.get(272, ""),
        "camera_serial": detailed.get(42033, ""),
        "capture_time": timestamp,
        "capture_time_basis": "DateTimeOriginal" if detailed.get(36867) else "DateTime",
    }
    return {
        key: str(value).strip().strip("\x00") if value is not None else ""
        for key, value in fields.items()
    }


def capture_session_edges(records: list[dict], policy: SimilarityPolicy) -> list[dict]:
    """Link adjacent same-camera/time candidates, retaining excluded intermediates.

    Same make/model without a serial is weak evidence, not a unique camera ID.
    A configurable short timestamp gap defines candidate sessions. All session
    nodes remain in one split; false-positive grouping is safer than known overlap.
    Invalid/missing timestamps produce no temporal evidence and remain uncertain.
    """
    if not policy.group_capture_sessions:
        return []
    cameras = defaultdict(list)
    for row in records:
        if not row.get("camera_make") or not row.get("camera_model"):
            continue
        try:
            timestamp = datetime.strptime(row.get("capture_time", ""), "%Y:%m:%d %H:%M:%S")
        except (ValueError, TypeError):
            continue
        key = (row["camera_make"], row["camera_model"], row.get("camera_serial", ""))
        cameras[key].append(
            (timestamp, row["sample_id"], row.get("capture_time_basis", "unknown"))
        )
    edges = []
    for camera, members in sorted(cameras.items()):
        ordered = sorted(members)
        for left, right in zip(ordered[:-1], ordered[1:], strict=True):
            seconds = (right[0] - left[0]).total_seconds()
            if seconds <= policy.capture_session_gap_seconds:
                edges.append(
                    {
                        "left": left[1],
                        "right": right[1],
                        "kind": "capture_session_candidate",
                        "evidence": f"{camera[0]} {camera[1]}; serial={camera[2] or 'unknown'}; "
                        f"gap_seconds={seconds}; time_fields={left[2]}/{right[2]}; "
                        "temporal candidate, not proven physical structure identity",
                    }
                )
    return edges
