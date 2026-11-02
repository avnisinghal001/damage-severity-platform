"""Verify a real upload, both HTTP services, persisted retrieval, history and native artifacts."""

import argparse
import hashlib
import io
import json
import tomllib
from pathlib import Path

import httpx
import numpy as np
from PIL import Image, ImageOps

from damage_analytics.api.schemas import HistoryPage, InspectionResponse
from damage_analytics.io import write_json
from damage_analytics.task09.schema import validate_report


def verify(base_url: str, image: Path, config: Path, output: Path) -> dict:
    """A serving check, not labeled model evaluation. It creates one saved inspection."""
    if output.exists():
        raise FileExistsError("Verification output already exists; use a new path")
    pins = tomllib.loads(config.read_text())
    content = image.read_bytes()
    with Image.open(io.BytesIO(content)) as original:
        media_type = {"JPEG": "image/jpeg", "PNG": "image/png"}[original.format]
        rgb = ImageOps.exif_transpose(original).convert("RGB")
    with httpx.Client(base_url=base_url.rstrip("/"), timeout=120.0, trust_env=False) as client:
        response = client.get("/ready")
        response.raise_for_status()
        response = client.post("/inspections", files={"file": (image.name, content, media_type)})
        response.raise_for_status()
        document = response.json()
        inspection = InspectionResponse.model_validate(document)
        if inspection.width != rgb.width or inspection.height != rgb.height:
            raise ValueError("Inspection dimensions differ from EXIF-oriented upload")
        if (
            inspection.model.checkpoint_sha256 != pins["segmentation"]["sha256"]
            or not inspection.classification
            or inspection.classification.model.checkpoint_sha256 != pins["crack"]["sha256"]
            or inspection.severity is None
            or inspection.severity.status != "available"
            or inspection.severity.model.checkpoint_sha256 != pins["severity"]["sha256"]
            or inspection.severity.input_rgb_sha256 != hashlib.sha256(rgb.tobytes()).hexdigest()
        ):
            raise ValueError("The complete pinned model/HTTP flow did not succeed")
        if not inspection.features:
            raise ValueError("Task 09 features missing")
        validate_report(inspection.features)
        response = client.get(f"/inspections/{inspection.id}")
        response.raise_for_status()
        if response.json() != document:
            raise ValueError("Persisted retrieval differs from the completed prediction")
        response = client.get("/inspections", params={"limit": 100})
        response.raise_for_status()
        page = HistoryPage.model_validate(response.json())
        if inspection.id not in {row.id for row in page.items}:
            raise ValueError("New inspection was not present in saved history")
        for kind, reference in (("image", inspection.image), ("mask", inspection.mask)):
            # API-root references remain within the supplied /api proxy prefix.
            expected = f"/inspections/{inspection.id}/artifacts/{kind}"
            if reference.url != expected:
                raise ValueError("Unexpected artifact URL")
            artifact = client.get(expected)
            artifact.raise_for_status()
            if hashlib.sha256(artifact.content).hexdigest() != reference.sha256:
                raise ValueError("Artifact content hash mismatch")
            with Image.open(io.BytesIO(artifact.content)) as decoded:
                if decoded.size != rgb.size:
                    raise ValueError("Artifact native resolution mismatch")
                if kind == "mask":
                    values = np.asarray(decoded)
                    if values.ndim != 2 or not np.isin(values, range(9)).all():
                        raise ValueError("Mask is not a native-resolution 0..8 class-ID image")
                    counts = np.bincount(values.ravel(), minlength=9).tolist()
                    if counts != inspection.features["coverage"]["raw_class_pixels"]:
                        raise ValueError("Mask pixels disagree with persisted class coverage")
                elif decoded.mode != "RGB" or decoded.tobytes() != rgb.tobytes():
                    raise ValueError("Saved image coordinates/pixels differ from oriented upload")
        invalid = client.post("/inspections", files={"file": ("bad.png", b"invalid", "image/png")})
        if invalid.status_code != 422:
            raise ValueError("Malformed upload was not rejected")
    report = {
        "status": "passed",
        "purpose": "Serving integration, not accuracy evaluation",
        "base_url": base_url,
        "inspection_id": str(inspection.id),
        "width": rgb.width,
        "height": rgb.height,
        "models": {role: pins[role]["sha256"] for role in ("segmentation", "crack", "severity")},
        "checks": [
            "real_image_upload",
            "native_segmentation_mask",
            "task09_features",
            "frozen_crack_decision",
            "severity_http_image_hash_and_pin",
            "persisted_retrieval",
            "inspection_history",
            "artifact_hashes_and_pixel_counts",
            "malformed_upload_rejection",
        ],
        "test_performance_evaluated": False,
    }
    write_json(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/api")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config/task12.toml"))
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/task12/verification/upload.json")
    )
    args = parser.parse_args()
    print(json.dumps(verify(args.base_url, args.image, args.config, args.output), indent=2))


if __name__ == "__main__":
    main()
