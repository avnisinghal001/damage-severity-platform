"""Integrity inventories and train-only grayscale near-pair contact sheets."""

import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from damage_analytics.io import write_csv, write_json
from damage_analytics.task10.config import safe_id
from damage_analytics.task10.encoding import recovered_channels
from damage_analytics.task10.release import Release


def build_review(release: Release, output_dir: Path, report_id: str, limit: int = 32) -> Path:
    """Never render test rows; exports are review aids, not duplicate adjudications."""
    if type(limit) is not int or not 0 <= limit <= 200:
        raise ValueError("Contact sheet limit must be 0..200")
    output = output_dir / "reviews" / safe_id(report_id)
    output.mkdir(parents=True, exist_ok=False)
    pairs = json.loads(release.file("near_candidates.json").read_text())
    by_id = {row["sample_id"]: row for row in release.rows}
    inventory = []
    for pair in pairs:
        item = dict(pair)
        for side in ("left", "right"):
            row = by_id[pair[side]]
            for field in ("official_partition", "split", "class_id", "label", "array_index"):
                item[f"{side}_{field}"] = row[field]
        inventory.append(item)
    write_csv(
        output / "near_pair_inventory.csv",
        inventory,
        list(inventory[0]) if inventory else ["left", "right", "decision", "human_evidence"],
    )
    counts = release.quality["prepared_class_counts"]
    classes = [
        {
            "split": split,
            "class_id": i,
            "label": name,
            "rows": values.get(str(i), 0),
            "percentage": 100
            * values.get(str(i), 0)
            / release.quality["prepared_split_counts"][split],
        }
        for split, values in counts.items()
        if release.quality["prepared_split_counts"].get(split, 0)
        for i, name in enumerate(release.metadata["class_names"])
    ]
    write_csv(output / "class_distribution.csv", classes, list(classes[0]))
    train_pairs = [
        p
        for p in pairs
        if all(by_id[p[s]]["official_partition"] == "train" for s in ("left", "right"))
    ]
    chosen = train_pairs[:limit]
    if chosen:
        relative = by_id[chosen[0]["left"]]["image_array"]
        values = np.load(release.file(relative), mmap_mode="r", allow_pickle=False)
        for offset in range(0, len(chosen), 8):
            batch = chosen[offset : offset + 8]
            sheet = Image.new("RGB", (640, len(batch) * 220), "white")
            draw = ImageDraw.Draw(sheet)
            for number, pair in enumerate(batch):
                for column, side in enumerate(("left", "right")):
                    row = by_id[pair[side]]
                    channels, _ = recovered_channels(values[row["array_index"]])
                    gray = Image.fromarray(np.rint(channels.mean(axis=2)).astype(np.uint8))
                    thumb = ImageOps.contain(gray, (200, 170))
                    x, y = column * 320, number * 220
                    sheet.paste(thumb, (x + 60, y + 45))
                    draw.text((x + 5, y + 4), row["sample_id"], fill="black")
                    draw.text(
                        (x + 5, y + 19), f"{row['label']} | {pair[f'{side}_view']}", fill="black"
                    )
            sheet.save(output / f"train_candidates_{offset // 8 + 1:02d}.png")
        del values
    summary = {
        "dataset": release.identity(),
        "candidate_pairs": len(pairs),
        "official_partition_pairs": dict(
            Counter(
                "/".join(sorted(by_id[p[s]]["official_partition"] for s in ("left", "right")))
                for p in pairs
            )
        ),
        "prepared_partition_pairs": dict(
            Counter(
                "/".join(sorted(by_id[p[s]]["split"] for s in ("left", "right"))) for p in pairs
            )
        ),
        "different_registered_labels": sum(
            by_id[p["left"]]["class_id"] != by_id[p["right"]]["class_id"] for p in pairs
        ),
        "distinct_candidate_rows": len({p[s] for p in pairs for s in ("left", "right")}),
        "multirow_leakage_components": sum(
            n > 1 for n in Counter(r["leakage_component"] for r in release.rows).values()
        ),
        "train_only_sheets_pairs": len(chosen),
        "official_test_images_rendered": 0,
        "human_decisions_made": 0,
        "test_performance_inspected": False,
        "scope": "Grayscale full images; view names identify the matching screening crops. "
        "Source group IDs and review decisions remain blank unless supplied by a human. "
        "Different labels on a near pair do not establish a label error.",
    }
    write_json(output / "summary.json", summary)
    return output
