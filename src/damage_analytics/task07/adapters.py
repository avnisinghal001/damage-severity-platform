"""Discover the inspected layouts without ambiguous stem-only identifiers."""

import hashlib
import re
from pathlib import Path


def discover(root: Path, dataset: str) -> list[dict]:
    """Preserve source taxonomy and all view identities, including duplicate JPG/PNG stems."""
    rows = []
    for image in sorted((root / "payload").rglob("*")):
        if image.suffix.lower() not in (".jpg", ".jpeg", ".png") or not image.is_file():
            continue
        relative = image.relative_to(root).as_posix()
        parts = image.relative_to(root / "payload").parts
        task = parts[0]
        if task not in ("classification", "detection", "segmentation"):
            raise ValueError("Unrecognized view")
        if task != "classification" and image.parent.name != "Images":
            continue
        row = {
            "sample_id": hashlib.sha256(f"{dataset}:{relative}".encode()).hexdigest()[:24],
            "dataset": dataset,
            "task": task,
            "image": relative,
            "source_member": "/".join(parts[1:]),
            "label": None,
            "annotation": "",
            "yolo": "",
            "source_mask": "",
            "candidate_source": "",
            "candidate_basis": "",
            "source_group": "",
            "source_evidence": "",
        }
        if dataset == "sdnet2018":
            folder = "/".join(parts[1:3])
            mapping = {"D/CD": 1, "D/UD": 0, "P/CP": 1, "P/UP": 0, "W/CW": 1, "W/UW": 0}
            if folder not in mapping:
                raise ValueError(f"Unexpected SDNET folder: {folder}")
            prefix = re.fullmatch(r"(\d+)-.+", image.stem)
            if prefix is None:
                raise ValueError("SDNET patch lacks an interpretable prefix")
            row.update(
                label=mapping[folder],
                source_label=parts[2],
                domain=parts[1],
                candidate_source=f"sdnet-prefix:{parts[1]}:{prefix[1]}",
                candidate_basis="Filename prefix only; original-photo identity unverified",
            )
        elif task == "classification":
            if image.parent.name not in ("Crack", "Non-Crack"):
                raise ValueError("Unexpected DamSegment classification folder")
            row.update(
                label=int(image.parent.name == "Crack"),
                source_label=image.parent.name,
                domain="classification",
                candidate_source=f"dam-stem:{image.parent.name}:{image.stem}",
                candidate_basis="Conservative same-class stem bucket, not a physical source ID",
            )
        else:
            base = image.parent.parent
            row.update(
                source_label="source_category_ids_0_1",
                domain=base.name,
                annotation=(base / "Labels/Pascal VOC" / f"{image.stem}.json")
                .relative_to(root)
                .as_posix(),
                yolo=(base / "Labels/Yolo" / f"{image.stem}.txt").relative_to(root).as_posix(),
                candidate_source=f"dam-view-stem:{image.stem}",
                candidate_basis="Cross-view name match; full hashes also checked",
            )
            if task == "segmentation":
                row["source_mask"] = (
                    (base / "Labels/Mask" / f"{image.stem}_mask.png").relative_to(root).as_posix()
                )
        rows.append(row)
    if not rows or len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("No records or duplicate identifiers")
    return rows
