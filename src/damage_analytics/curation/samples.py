"""Per-sample audit/export and conservative handling of exact duplicate ground truth."""

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from damage_analytics.curation.cleaning import clean_annotation, issue
from damage_analytics.curation.config import CleaningPolicy
from damage_analytics.curation.images import prepare_image
from damage_analytics.curation.masks import checked_mask, mask_thumbnail
from damage_analytics.curation.similarity import annotation_hash, image_fingerprint
from damage_analytics.discovery import Pair
from damage_analytics.io import sha256_file, write_json


def curate_sample(
    pair: Pair,
    number: int,
    source_root: Path,
    output: Path,
    hashes: dict[str, str],
    policy: CleaningPolicy,
) -> tuple[dict, list[dict]]:
    """Audit an image and all its polygons; export only a completely accepted pair."""
    image_path = pair.image.relative_to(source_root).as_posix()
    annotation_path = pair.annotation.relative_to(source_root).as_posix()
    row = {
        "sample_id": pair.sample_id,
        "source_image": image_path,
        "source_annotation": annotation_path,
        "source_image_sha256": hashes[image_path],
        "source_annotation_sha256": hashes[annotation_path],
        "included": False,
        "image_readable": False,
        "annotation_readable": False,
    }
    decisions, document = [], {}
    try:
        document = json.loads(pair.annotation.read_text(encoding="utf-8"))
        row["annotation_readable"] = True
    except (ValueError, OSError) as exc:
        decisions.append(issue(pair.sample_id, "unreadable_annotation", "exclude", str(exc)))
    if isinstance(document, dict):
        row["declared_width"] = document.get("imageWidth")
        row["declared_height"] = document.get("imageHeight")
        row["source_polygon_counts"] = (
            dict(
                Counter(
                    shape["label"]
                    for shape in document.get("shapes", [])
                    if isinstance(shape, dict) and isinstance(shape.get("label"), str)
                )
            )
            if isinstance(document.get("shapes"), list)
            else {}
        )
    try:
        image, metadata, image_decisions = prepare_image(
            pair.image, document, pair.sample_id, policy
        )
        row.update(metadata)
        row["image_readable"] = True
        row.update(image_fingerprint(image))
        decisions.extend(image_decisions)
    except (
        ValueError,
        OSError,
        SyntaxError,
        Image.DecompressionBombWarning,
        Image.DecompressionBombError,
    ) as exc:
        decisions.append(issue(pair.sample_id, "unreadable_image", "exclude", str(exc)))
        # Use declared sizes solely for geometry audit; no pixels/masks can be accepted.
        declared = (
            (document.get("imageWidth"), document.get("imageHeight"))
            if isinstance(document, dict)
            else (None, None)
        )
        if not all(type(value) is int and value > 0 for value in declared):
            return row, decisions
        metadata = {"width": declared[0], "height": declared[1]}
    polygons, findings = clean_annotation(
        document, pair.sample_id, pair.image.name, (metadata["width"], metadata["height"]), policy
    )
    decisions.extend(findings)
    if any(item["action"] == "exclude" for item in decisions):
        return row, decisions
    size = image.size
    row["annotation_geometry_sha256"] = annotation_hash(polygons, size)
    mask, mask_info = checked_mask(polygons, size)
    row.update(mask_info)
    row["mask_thumbnail"] = mask_thumbnail(mask)
    if mask_info["conflict_pixels"]:
        action = "exclude" if policy.cross_class_overlap == "exclude" else "accept"
        decisions.append(
            issue(
                pair.sample_id,
                "cross_class_overlap",
                action,
                f"{mask_info['conflict_pixels']} pixels; policy={policy.cross_class_overlap}",
            )
        )
    elif mask_info["overlap_pixels"]:
        decisions.append(
            issue(
                pair.sample_id,
                "same_class_overlap",
                "accept",
                f"{mask_info['overlap_pixels']} multiply covered pixels; class union unchanged",
            )
        )
    if not mask_info["foreground_pixels"] and not policy.allow_empty_annotations:
        decisions.append(issue(pair.sample_id, "empty_mask", "exclude", "No foreground pixels"))
    if any(item["action"] == "exclude" for item in decisions):
        return row, decisions
    row.update(
        {
            "image": f"images/{number:06d}.png",
            "annotation": f"annotations/{number:06d}.json",
            "mask": f"masks/{number:06d}.png",
            "polygon_count": len(polygons),
            "polygon_counts": dict(Counter(polygon.label for polygon in polygons)),
            "mask_pixel_sha256": hashlib.sha256(str(size).encode() + mask.tobytes()).hexdigest(),
        }
    )
    for name in ("images", "annotations", "masks"):
        (output / name).mkdir(exist_ok=True)
    image.save(output / row["image"])
    mask.save(output / row["mask"])
    write_json(
        output / row["annotation"],
        {
            "imagePath": Path(row["image"]).name,
            "imageWidth": size[0],
            "imageHeight": size[1],
            "source_sample_id": pair.sample_id,
            "shapes": [
                {"label": polygon.label, "shape_type": "polygon", "points": polygon.points}
                for polygon in polygons
            ],
        },
    )
    for field in ("image", "annotation", "mask"):
        row[f"{field}_sha256"] = sha256_file(output / row[field])
    row["included"] = True
    return row, decisions


def resolve_exact_images(records: list[dict], policy: CleaningPolicy) -> list[dict]:
    """Exclude conflicting/unknown annotations; optionally keep one identical pair.

    No majority vote, union, or arbitrary choice resolves competing ground truth.
    All members of an exact decoded-image group are excluded when one is unusable
    or masks disagree. Identical image+mask duplicates keep the smallest source ID.
    """
    groups = defaultdict(list)
    for row in records:
        if row.get("decoded_sha256"):
            groups[row["decoded_sha256"]].append(row)
    decisions = []
    for members in groups.values():
        if len(members) < 2:
            continue
        unusable = any(not row["included"] for row in members)
        conflicting = len({row.get("mask_pixel_sha256") for row in members}) > 1
        if unusable or conflicting:
            code = (
                "exact_image_unresolved_annotation"
                if unusable
                else "exact_image_annotation_conflict"
            )
            for row in members:
                row["included"] = False
                decisions.append(
                    issue(
                        row["sample_id"],
                        code,
                        "exclude",
                        "Exact decoded image has unusable or conflicting ground truth: "
                        + ", ".join(member["sample_id"] for member in members),
                    )
                )
        elif policy.deduplicate_identical_pairs:
            ordered = sorted(members, key=lambda row: row["sample_id"])
            for row in ordered[1:]:
                row["included"] = False
                decisions.append(
                    issue(
                        row["sample_id"],
                        "redundant_exact_pair",
                        "exclude",
                        "Identical decoded image and mask; "
                        f"representative={ordered[0]['sample_id']}",
                    )
                )
    return decisions
