"""Validate paired data, preserve provenance, and generate lossless class-ID masks."""

import hashlib
import warnings
from collections import Counter, defaultdict

from damage_analytics.annotations import rasterize, read_annotation
from damage_analytics.config import Config
from damage_analytics.discovery import discover_pairs
from damage_analytics.io import sha256_file, write_csv, write_json
from damage_analytics.labels import CLASS_NAMES, DATASET_URL, DISPLAY_COLORS, SOURCE_NOTE

MANIFEST_FIELDS = [
    "sample_id", "image", "annotation", "mask", "width", "height", "image_mode",
    "image_sha256", "annotation_sha256", "mask_sha256", "overlap_pixels", "polygon_count",
]
ISSUE_FIELDS = ["level", "sample_id", "code", "detail"]


def validate_dataset(config: Config) -> tuple[list[dict], dict]:
    """Write reports for all unambiguous pairs; exclude invalid pairs from statistics.

    Missing/ambiguous/unsupported files are discovery errors. Byte duplicates in
    all extracted regular files and decoded-pixel duplicates in paired images
    are warnings; they remain in statistics and must be addressed before splits.
    """
    import numpy as np
    from PIL import Image

    pairs, issues = discover_pairs(config.extracted_dir)
    if len(pairs) != config.expected_pairs:
        issues.append({"level": "error", "sample_id": "", "code": "pair_count",
                       "detail": (f"Expected {config.expected_pairs}; "
                                  f"found {len(pairs)} unique pairs")})
    config.output_dir.mkdir(parents=True, exist_ok=True)
    # Invalidate visual products from earlier runs, including a zero-valid-pair run.
    for relative in ("sample_gallery.png", "gallery_samples.json", "plots/class_pixels.png",
                     "plots/category_frequency.png", "plots/image_dimensions.png"):
        (config.output_dir / relative).unlink(missing_ok=True)
    (config.output_dir / "masks").mkdir(exist_ok=True)
    duplicate_groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    file_hashes = {}
    for role in ("images", "annotations"):
        for path in sorted((config.extracted_dir / role).rglob("*")):
            if path.is_file():
                digest = sha256_file(path)
                file_hashes[path] = digest
                duplicate_groups[(role + "_bytes", digest)].append(
                    path.relative_to(config.extracted_dir).as_posix())
    records = []
    for number, pair in enumerate(pairs, 1):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(pair.image) as image:
                    image.verify()
                with Image.open(pair.image) as image:
                    image.load()
                    size, mode = image.size, image.mode
                    rgb = image.convert("RGB")
                    digest = hashlib.sha256(str(size).encode() + rgb.tobytes()).hexdigest()
                    duplicate_groups[("decoded_image_pixels", digest)].append(pair.sample_id)
                    orientation = image.getexif().get(274, 1)
            if orientation != 1:
                issues.append({"level": "warning", "sample_id": pair.sample_id,
                               "code": "exif_orientation",
                               "detail": "Stored pixels used; EXIF rotation was not applied"})
            polygons = read_annotation(pair.annotation, pair.image.name, size)
            mask, overlap = rasterize(polygons, size)
            mask_relative = f"masks/{number:06d}.png"
            mask_path = config.output_dir / mask_relative
            mask.save(mask_path)
            if not polygons:
                issues.append({"level": "warning", "sample_id": pair.sample_id,
                               "code": "empty_annotation",
                               "detail": "All pixels become background"})
            if overlap:
                issues.append({"level": "warning", "sample_id": pair.sample_id,
                               "code": "polygon_overlap",
                               "detail": f"{overlap} pixels; later JSON shapes take precedence"})
            counts = np.bincount(np.asarray(mask).ravel(), minlength=len(CLASS_NAMES)).tolist()
            records.append({
                "sample_id": pair.sample_id,
                "image": pair.image.relative_to(config.extracted_dir).as_posix(),
                "annotation": pair.annotation.relative_to(config.extracted_dir).as_posix(),
                "mask": mask_relative, "width": size[0], "height": size[1], "image_mode": mode,
                "image_sha256": file_hashes[pair.image],
                "annotation_sha256": file_hashes[pair.annotation],
                "mask_sha256": sha256_file(mask_path), "overlap_pixels": overlap,
                "polygon_count": len(polygons), "pixel_counts": counts,
                "polygon_counts": dict(Counter(polygon.label for polygon in polygons)),
            })
        except (ValueError, TypeError, KeyError, OverflowError, OSError, SyntaxError,
                Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
            issues.append({"level": "error", "sample_id": pair.sample_id,
                           "code": "invalid_pair", "detail": str(exc)})
    duplicates = []
    for (kind, digest), members in sorted(duplicate_groups.items()):
        if len(members) > 1:
            duplicates.append({"kind": kind, "sha256": digest, "members": members})
            issues.append({"level": "warning", "sample_id": "", "code": "duplicate",
                           "detail": f"{kind}: {', '.join(members)}"})
    error_count = sum(issue["level"] == "error" for issue in issues)
    summary = {
        "status": "failed" if error_count else "passed",
        "expected_pairs": config.expected_pairs, "discovered_pairs": len(pairs),
        "valid_pairs": len(records), "excluded_pairs": len(pairs) - len(records),
        "errors": error_count, "warnings": len(issues) - error_count,
        "duplicate_groups": len(duplicates),
        "source_mask_color_check": "not applicable: source annotations are JSON polygons",
        "unexpected_annotation_labels": "rejected; see invalid_pair issues",
        "generated_mask_check": "IDs limited to 0..8; RGB display palette is project-defined",
        "source": DATASET_URL, "label_provenance": SOURCE_NOTE,
        "classes": [{"id": index, "source_label": name, "display_rgb": DISPLAY_COLORS[index]}
                    for index, name in enumerate(CLASS_NAMES)],
        "overlap_rule": "last polygon in JSON order wins",
        "coordinate_rule": "Pillow polygon fill; clip edge coordinates to width-1/height-1",
        "statistics_population": "valid pairs only; duplicates included",
    }
    write_csv(config.output_dir / "manifest.csv",
              [{key: row[key] for key in MANIFEST_FIELDS} for row in records], MANIFEST_FIELDS)
    write_csv(config.output_dir / "issues.csv", issues, ISSUE_FIELDS)
    write_json(config.output_dir / "duplicates.json", duplicates)
    write_json(config.output_dir / "validation.json", summary)
    return records, summary
