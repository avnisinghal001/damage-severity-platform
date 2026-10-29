"""Deterministic image/annotation similarity evidence and transitive grouping."""

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from damage_analytics.annotations import Polygon
from damage_analytics.curation.config import SimilarityPolicy
from damage_analytics.curation.masks import annotation_similarity


def image_fingerprint(image: Image.Image) -> dict:
    """Compute a 63-AC-bit perceptual hash, 64-bit difference hash and RGB thumbnail."""
    gray = np.asarray(image.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=float)
    positions = np.arange(32) + 0.5
    basis = np.cos(np.pi / 32 * np.arange(8)[:, None] * positions)
    coefficients = (basis @ gray @ basis.T).ravel()[1:]
    differences = np.asarray(image.convert("L").resize((9, 8), Image.Resampling.LANCZOS))

    def bits_integer(bits: np.ndarray) -> int:
        return int("".join("1" if bit else "0" for bit in bits.ravel()), 2)

    return {
        "phash": bits_integer(coefficients > np.median(coefficients)),
        "dhash": bits_integer(differences[:, 1:] > differences[:, :-1]),
        "thumbnail": np.asarray(image.resize((16, 16), Image.Resampling.LANCZOS), dtype=np.uint8)
        .ravel()
        .tolist(),
    }


def annotation_hash(polygons: list[Polygon], size: tuple[int, int]) -> str:
    """Hash ordered geometry/labels independently of JSON metadata and imagePath.

    Exact ring equality is winding/start invariant. Polygon order is preserved
    because it defines precedence under the optional last-wins conflict policy.
    Empty annotations do not generate grouping evidence.
    """
    shapes = []
    for polygon in polygons:
        ring = list(polygon.points)
        # Choose canonical rotation in both windings; no coordinate rounding.
        variants = [
            sequence[index:] + sequence[:index]
            for sequence in (ring, list(reversed(ring)))
            for index in range(len(ring))
        ]
        shapes.append((polygon.label, min(variants)))
    payload = json.dumps([size, shapes], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest() if shapes else ""


def similarity_edges(records: list[dict], policy: SimilarityPolicy) -> list[dict]:
    """Compare all readable samples, including excluded intermediates; record evidence.

    Near-image evidence requires both hash distances, aspect agreement, and RGB
    thumbnail MAE. Near annotations are assessed only within these visual candidates;
    similar segmentation shapes alone do not establish common physical source.
    """
    edges = []
    for field in (
        "source_image_sha256",
        "decoded_sha256",
        "stored_pixel_sha256",
        "source_annotation_sha256",
        "annotation_geometry_sha256",
    ):
        groups: dict[str, list[str]] = defaultdict(list)
        for row in records:
            if row.get(field):
                groups[row[field]].append(row["sample_id"])
        for members in groups.values():
            for member in members[1:]:
                edges.append(
                    {
                        "left": members[0],
                        "right": member,
                        "kind": field,
                        "evidence": "Exact matching fingerprint",
                    }
                )
    readable = [row for row in records if "phash" in row]
    for index, left in enumerate(readable):
        for right in readable[index + 1 :]:
            if left["decoded_sha256"] == right["decoded_sha256"]:
                continue
            phash = (left["phash"] ^ right["phash"]).bit_count()
            if phash > policy.phash_distance:
                continue
            dhash = (left["dhash"] ^ right["dhash"]).bit_count()
            if dhash > policy.dhash_distance:
                continue
            aspect = abs(
                math.log((left["width"] / left["height"]) / (right["width"] / right["height"]))
            )
            if aspect > policy.aspect_log_tolerance:
                continue
            mae = float(
                np.mean(
                    np.abs(
                        np.array(left["thumbnail"], dtype=float)
                        - np.array(right["thumbnail"], dtype=float)
                    )
                )
                / 255
            )
            if mae > policy.thumbnail_mae:
                continue
            evidence = {
                "phash_distance": phash,
                "dhash_distance": dhash,
                "thumbnail_mae": mae,
                "aspect_log_difference": aspect,
            }
            edges.append(
                {
                    "left": left["sample_id"],
                    "right": right["sample_id"],
                    "kind": "near_image_candidate",
                    "evidence": json.dumps(evidence),
                }
            )
            if left.get("mask_thumbnail") and right.get("mask_thumbnail"):
                iou = annotation_similarity(left["mask_thumbnail"], right["mask_thumbnail"])
                if iou >= policy.annotation_iou:
                    edges.append(
                        {
                            "left": left["sample_id"],
                            "right": right["sample_id"],
                            "kind": "near_annotation_candidate",
                            "evidence": f"Visual candidate; thumbnail mean class IoU={iou}",
                        }
                    )
    return sorted(edges, key=lambda row: (row["left"], row["right"], row["kind"]))


def source_edges(path: Path | None, sample_ids: set[str]) -> tuple[list[dict], set[str]]:
    """Read optional explicit source evidence; reject unknown/duplicated sample IDs."""
    if path is None:
        return [], set()
    groups, assigned = defaultdict(list), set()
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not {"sample_id", "source_group", "evidence"}.issubset(reader.fieldnames or []):
            raise ValueError(
                "Source evidence CSV requires sample_id,source_group,evidence headers"
            )
        for row in reader:
            sample_id = row["sample_id"]
            if (
                sample_id not in sample_ids
                or sample_id in assigned
                or not (row["source_group"] or "").strip()
                or not (row["evidence"] or "").strip()
            ):
                raise ValueError(
                    "Source group CSV requires known unique IDs, groups, and evidence"
                )
            assigned.add(sample_id)
            groups[row["source_group"]].append(row)
    edges = []
    for name, members in groups.items():
        for member in members[1:]:
            edges.append(
                {
                    "left": members[0]["sample_id"],
                    "right": member["sample_id"],
                    "kind": "source_group",
                    "evidence": f"{name}: {member['evidence']}",
                }
            )
    return edges, assigned


def connected_groups(sample_ids: list[str], edges: list[dict]) -> dict[str, str]:
    """Union all evidence, retaining excluded intermediates; choose stable minimum IDs."""
    parent = {sample_id: sample_id for sample_id in sample_ids}

    def find(sample_id: str) -> str:
        while parent[sample_id] != sample_id:
            parent[sample_id] = parent[parent[sample_id]]
            sample_id = parent[sample_id]
        return sample_id

    for edge in edges:
        left, right = find(edge["left"]), find(edge["right"])
        parent[max(left, right)] = min(left, right)
    return {sample_id: find(sample_id) for sample_id in sorted(sample_ids)}
