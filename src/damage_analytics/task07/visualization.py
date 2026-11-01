"""Small detection ground-truth sanity gallery, with original numeric category IDs."""

from pathlib import Path

from PIL import ImageDraw

from damage_analytics.io import write_json
from damage_analytics.task07.annotations import read_annotation, validate_json
from damage_analytics.task07.release import Release


def detection_gallery(release: Release, output: Path, count: int = 6) -> Path:
    if not 1 <= count <= 20:
        raise ValueError("Choose 1..20 gallery samples")
    rows = release.samples("validation", "detection")[:count]
    if not rows:
        raise ValueError("No detection annotations available")
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for row in rows:
        image = release.image(row)
        value = read_annotation(release.file(row["annotation"]))
        validate_json(value, image.size, Path(row["image"]).name, polygons=False)
        draw = ImageDraw.Draw(image)
        for item in value["annotations"]:
            x, y, width, height = item["bbox"]
            category = item["category_id"]
            color = (255, 0, 0) if category == 0 else (0, 80, 255)
            draw.rectangle((x, y, x + width, y + height), outline=color, width=2)
            draw.text((x, y), f"source ID {category}", fill=color)
        image.save(output / f"{row['sample_id']}.png")
        records.append({"sample_id": row["sample_id"], "annotation": row["annotation"]})
    write_json(
        output / "gallery.json",
        {
            "samples": records,
            "split": "validation",
            "meaning": "Ground-truth boxes only; not detector predictions or severity targets",
        },
    )
    return output
