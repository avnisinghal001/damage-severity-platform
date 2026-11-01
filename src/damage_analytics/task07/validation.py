"""Per-sample validation for external classification, detection and mask records."""

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from damage_analytics.curation.similarity import image_fingerprint
from damage_analytics.io import contained_file, sha256_file
from damage_analytics.task07.annotations import (
    decode_mask,
    polygon_mask_comparison,
    read_annotation,
    validate_json,
    validate_yolo,
)


def validate_sample(
    root: Path, row: dict, *, max_pixels: int = 32_000_000, mask_min_iou: float = 0.5
) -> dict:
    """Validate bytes, RGB derivation, pairing and native labels; return reproducible evidence."""
    image_path = contained_file(root, row["image"])
    with Image.open(image_path) as source:
        expected_format = "PNG" if image_path.suffix.lower() == ".png" else "JPEG"
        if source.format != expected_format:
            raise ValueError("Image encoding disagrees with its JPEG/PNG extension")
        if source.width * source.height > max_pixels:
            raise ValueError(f"Native image exceeds the configured {max_pixels}-pixel budget")
        source.load()
        if source.mode not in ("RGB", "L", "CMYK") or getattr(source, "n_frames", 1) != 1:
            raise ValueError("Unsupported image mode/frame count")
        orientation = source.getexif().get(274, 1)
        if type(orientation) is not int or orientation not in range(1, 9):
            raise ValueError("Invalid EXIF orientation")
        if row["task"] != "classification" and orientation != 1:
            raise ValueError("Annotated image has unresolved EXIF coordinate frame")
        image = ImageOps.exif_transpose(source).convert("RGB")
        tags = source.getexif()
        row.update(
            source_mode=source.mode,
            source_format=source.format,
            exif_orientation=orientation,
            source_exif={
                name: str(tags[key])
                for key, name in (
                    (42016, "image_unique_id"),
                    (36867, "capture_time"),
                    (42033, "camera_serial"),
                )
                if key in tags
            },
        )
    fingerprint = image_fingerprint(image)
    row.update(
        width=image.width,
        height=image.height,
        image_sha256=sha256_file(image_path),
        decoded_sha256=hashlib.sha256(
            f"RGB:{image.width}:{image.height}:".encode() + image.tobytes()
        ).hexdigest(),
        phash=fingerprint["phash"],
        dhash=fingerprint["dhash"],
    )
    row["mask"] = ""
    row["strata"] = [f"classification:{row['label']}"] if row["task"] == "classification" else []
    if row["task"] == "classification":
        row["strata"].append(f"classification_domain:{row['domain']}:{row['label']}")
    if row["annotation"]:
        annotation_path = contained_file(root, row["annotation"])
        value = read_annotation(annotation_path)
        evidence = validate_json(
            value, image.size, image_path.name, polygons=row["task"] == "segmentation"
        )
        yolo = contained_file(root, row["yolo"])
        if (
            dict(validate_yolo(yolo.read_text(), polygons=row["task"] == "segmentation"))
            != evidence["objects"]
        ):
            raise ValueError("JSON/YOLO category/object counts differ")
        row.update(
            annotation_sha256=sha256_file(annotation_path),
            yolo_sha256=sha256_file(yolo),
            annotation_geometry_sha256=hashlib.sha256(
                json.dumps(value["annotations"], sort_keys=True).encode()
            ).hexdigest(),
            annotation_evidence=evidence,
        )
        row["strata"] = [f"{row['task']}:{key}" for key in evidence["objects"]] or [
            f"{row['task']}:empty"
        ]
    if row["source_mask"]:
        source_mask = contained_file(root, row["source_mask"])
        with Image.open(source_mask) as source:
            source.load()
            if source.size != image.size:
                raise ValueError("Image/mask size mismatch")
            mask = decode_mask(source)
        row["mask_evidence"] = polygon_mask_comparison(value, mask)
        # A category absent from polygons must not appear in its encoded mask, or vice versa.
        if set(np.unique(mask)) - {0} != {int(c) + 1 for c in evidence["objects"]}:
            raise ValueError("Mask/JSON category presence disagrees")
        if any(
            iou is not None and iou < mask_min_iou for iou in row["mask_evidence"]["class_iou"]
        ):
            raise ValueError("Mask/source-ID correspondence is unsupported by polygons")
        row["mask"] = f"masks/{row['sample_id']}.png"
        target = root / row["mask"]
        target.parent.mkdir(exist_ok=True)
        Image.fromarray(mask).save(target)
        row.update(
            mask_sha256=sha256_file(target),
            source_mask_sha256=sha256_file(source_mask),
            pixel_counts=np.bincount(mask.ravel(), minlength=3).tolist(),
        )
    return row
