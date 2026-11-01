"""Small, paginated Pillow contact sheets; test pixels are never opened."""

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from damage_analytics.io import write_json
from damage_analytics.task07.release import Release

WIDTH, HEIGHT = 300, 280


def card(release: Release, row: dict, lines: list[str]) -> Image.Image:
    """Protect test pixels even if an integrity-review caller accidentally supplies them."""
    result = Image.new("RGB", (WIDTH, HEIGHT), "#f3f5f7")
    draw = ImageDraw.Draw(result)
    if row["split"] == "test":
        draw.text((12, 70), "TEST IMAGE REDACTED\nMetadata review only", fill="#922b21")
    else:
        image = ImageOps.contain(release.image(row), (WIDTH - 16, 194))
        result.paste(image, ((WIDTH - image.width) // 2, 4))
    for index, line in enumerate(lines[:5]):
        draw.text((8, 202 + 14 * index), str(line), fill="#17202a")
    return result


def write_cards(path: Path, cards: list[Image.Image], columns: int = 3) -> None:
    """Write a bounded page; empty selections get an explicit empty-state image."""
    rows = max(1, math.ceil(len(cards) / columns))
    canvas = Image.new("RGB", (WIDTH * columns, HEIGHT * rows), "white")
    if not cards:
        ImageDraw.Draw(canvas).text((20, 25), "No samples meet this definition", fill="black")
    for index, image in enumerate(cards):
        canvas.paste(image, ((index % columns) * WIDTH, (index // columns) * HEIGHT))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def sample_gallery(release: Release, items: list[dict], output: Path, limit: int) -> None:
    """Validation error galleries carry the exact sample/score definitions in their index."""
    if any(item["row"]["split"] != "validation" for item in items):
        raise ValueError("Error galleries require validation samples only")
    chosen = items[:limit]
    write_cards(output, [card(release, item["row"], item["lines"]) for item in chosen])
    write_json(
        output.with_suffix(".json"),
        {
            "eligible_samples": len(items),
            "shown": len(chosen),
            "samples": [{k: v for k, v in item.items() if k != "row"} for item in chosen],
            "scope": "Selected validation examples, source labels, uncalibrated model scores",
        },
    )


def pair_sheets(release: Release, pairs: list[dict], output: Path, per_page: int) -> None:
    """Show both views; every test endpoint is a placeholder, never a thumbnail."""
    lookup = {r["sample_id"]: r for r in release.rows}
    for offset in range(0, len(pairs), per_page):
        cards = []
        for pair in pairs[offset : offset + per_page]:
            for side in ("left", "right"):
                row = lookup[pair[side]]
                cards.append(
                    card(
                        release,
                        row,
                        [
                            f"Pair {pair['pair_id']} / {side} / {row['task']}",
                            row["sample_id"],
                            f"{row['split']} | source={row['source_label']}",
                            Path(row["source_member"]).name,
                            f"pHash={pair['phash_distance']} dHash={pair['dhash_distance']}",
                        ],
                    )
                )
        write_cards(output / f"pairs_{offset // per_page + 1:03d}.png", cards, columns=2)
