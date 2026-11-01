"""Lightweight archive metadata and representative pixels, not full extraction."""

import csv
import hashlib
import io
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from damage_analytics.io import write_json
from damage_analytics.task07.archives import central_identity, dataset_archives, useful
from damage_analytics.task07.config import Config
from damage_analytics.task07.metrics import from_confusion


def audit_archives(config: Config) -> dict:
    """Read central directories, small JSON annotations and a bounded number of image samples."""
    report = {
        "scope": "Archive inventories and sampled decoding; full integrity/preparation NOT run",
        "datasets": {},
        "old_rc_test_accessed": False,
    }
    evidence = {}
    for dataset in ("sdnet2018", "damsegment"):
        path = config.archive(dataset)
        views, indexes = {}, {}
        with dataset_archives(path, dataset) as archives:
            for task, archive in archives.items():
                files = [x for x in archive.infolist() if not x.is_dir() and useful(x.filename)]
                images = [
                    x
                    for x in files
                    if Path(x.filename).suffix.lower() in (".jpg", ".png", ".jpeg")
                    and (task == "classification" or "/Images/" in x.filename)
                ]
                counts = Counter(str(Path(x.filename).parent) for x in images)
                indexes[task] = {(x.CRC, x.file_size) for x in images}
                image_categories, objects, empty = Counter(), Counter(), 0
                for member in files:
                    if not member.filename.endswith(".json"):
                        continue
                    value = json.loads(archive.read(member))
                    identifiers = [a["category_id"] for a in value["annotations"]]
                    objects.update(identifiers)
                    image_categories.update(set(identifiers))
                    empty += not identifiers
                sampled = []
                directories = sorted(
                    {
                        str(Path(x.filename).parent)
                        for x in files
                        if Path(x.filename).suffix.lower() in (".jpg", ".png")
                    }
                )
                for directory in directories[:12]:
                    member = next(x for x in files if str(Path(x.filename).parent) == directory)
                    with Image.open(io.BytesIO(archive.read(member))) as image:
                        image.load()
                        sampled.append(
                            {
                                "member": member.filename,
                                "size": list(image.size),
                                "mode": image.mode,
                                "colors_if_small_palette": image.getcolors(8),
                            }
                        )
                for member in images[:3]:
                    evidence[(task, Path(member.filename).name)] = hashlib.sha256(
                        archive.read(member)
                    ).hexdigest()
                views[task] = {
                    "files": len(files),
                    "images": len(images),
                    "image_directories": dict(counts),
                    "directory_fingerprint": central_identity(archive),
                    "annotations": {
                        "objects_by_source_id": dict(objects),
                        "images_by_source_id": dict(image_categories),
                        "empty": empty,
                    },
                    "same_stem_collisions": len(images)
                    - len({Path(x.filename).stem for x in images}),
                    "samples": sampled,
                    "extensions": dict(Counter(Path(x.filename).suffix.lower() for x in files)),
                }
                if task != "classification":
                    names = {x.filename for x in files}
                    expected = []
                    for member in images:
                        base = Path(member.filename).parent.parent
                        stem = Path(member.filename).stem
                        expected.extend(
                            [
                                f"{base}/Labels/Pascal VOC/{stem}.json",
                                f"{base}/Labels/Yolo/{stem}.txt",
                            ]
                        )
                        if task == "segmentation":
                            expected.append(f"{base}/Labels/Mask/{stem}_mask.png")
                    views[task]["missing_expected_pairs"] = [
                        name for name in expected if name not in names
                    ]
                if dataset == "sdnet2018":
                    prefixes = {
                        f"{Path(x.filename).parts[0]}/{Path(x.filename).stem.split('-')[0]}"
                        for x in images
                    }
                    views[task]["candidate_source_prefixes"] = len(prefixes)
        item = {
            "archive": str(path),
            "archive_bytes": path.stat().st_size,
            "views": views,
            "full_archive_sha256": "Deferred to explicit prepare command",
        }
        if dataset == "damsegment":
            item["cross_view_crc_size_candidates"] = len(
                indexes["detection"] & indexes["segmentation"]
            )
            item["classification_segmentation_crc_size_candidates"] = len(
                indexes["classification"] & indexes["segmentation"]
            )
            item["sha256_verified_sample_pairs"] = [
                {"filename": name, "sha256": value}
                for (task, name), value in evidence.items()
                if task == "detection" and evidence.get(("segmentation", name)) == value
            ]
        report["datasets"][dataset] = item
    report["limitations"] = [
        "SDNET filename prefixes are candidate source groups, not supplied verified photo IDs.",
        "DamSegment classification C/N stems do not establish original-photo identity.",
        "Folders named Pascal VOC contain COCO-like JSON, not XML.",
        "Source annotation IDs 0/1 have no semantic category-name map in these archives.",
        "No-crack is NOT no structural damage; no general safety/severity labels exist.",
        "CRC/size matches are screening evidence, not cryptographic identity for all files.",
        "Full hashes, geometry, mask distributions and exclusions await preparation.",
    ]
    write_json(config.output_dir / "archive_audit.json", report)
    return report


def audit_rc(output: Path) -> dict:
    """Reuse existing reports and train/validation manifests, never open RC test data/results."""
    root = Path("outputs/task02/versions/rc1841-curated-v1-d34771dbb662406a")
    quality = json.loads((root / "quality_report.json").read_text())
    stats = quality["statistics"]
    empty = sum(x["images"] for x in stats["label_combinations"] if not x["class_ids"])
    subsets = {}
    for split in ("train", "validation"):
        with (root / f"splits/{split}.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        subsets[split] = {
            "samples": len(rows),
            "zero_foreground": sum(
                sum(int(r[f"pixels_{i}"]) for i in range(1, 9)) == 0 for r in rows
            ),
        }
    results = []
    for task in ("task03", "task04"):
        for path in sorted(Path(f"outputs/{task}/evaluations").glob("*/validation/metrics.json")):
            saved = json.loads(path.read_text())
            if saved.get("split") != "validation" or saved.get("test_evaluated"):
                raise ValueError("Unexpected evaluation scope")
            matrix = saved["metrics"]["confusion_matrix"]
            derived = from_confusion(np.asarray(matrix, dtype=np.int64))
            if not np.isclose(
                derived["foreground_miou"], saved["metrics"]["foreground_miou"], atol=1e-12, rtol=0
            ):
                raise ValueError("Saved validation mIoU does not match its confusion matrix")
            confusion_pairs = sorted(
                (
                    {"ground_truth": i, "predicted": j, "pixels": matrix[i][j]}
                    for i in range(1, 9)
                    for j in range(1, 9)
                    if i != j
                ),
                key=lambda row: -row["pixels"],
            )
            results.append(
                {
                    "path": str(path),
                    "run": saved.get("run_id", "unet_001"),
                    "foreground_miou": saved["metrics"]["foreground_miou"],
                    "background_to_damage_pixels": sum(matrix[0][1:]),
                    "background_pixel_false_damage_fraction": sum(matrix[0][1:]) / sum(matrix[0]),
                    "per_class": saved["metrics"]["classes"],
                    "top_foreground_confusions": confusion_pairs[:6],
                    "metrics_recomputed_from_saved_confusion": True,
                    "negative_image_false_positive_rate": None,
                }
            )
    report = {
        "samples": stats["samples"],
        "annotated_damage_images": stats["samples"] - empty,
        "zero_foreground_images": empty,
        "verified_undamaged_or_nonconcrete_negatives": 0,
        "classes": [x for x in stats["classes"] if x["split"] == "all"],
        "subsets": subsets,
        "validation": results,
        "source_limitations": quality["unresolved_limitations"],
        "interpretation": "Background is polygon complement, not an intact-concrete label. "
        "Segmentation is conditional research; no material classifier or image-level "
        "minor/moderate/severe supervision exists. Unrelated-photo failures are user-reported; "
        "no independent OOD-negative benchmark was found or rerun.",
        "test_data_or_test_results_accessed": False,
    }
    write_json(output / "rc1841_audit.json", report)
    return report
