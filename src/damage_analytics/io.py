"""Small, shared helpers for portable reports and provenance."""

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    """Hash file contents in bounded chunks, without loading images into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def contained_file(root: Path, relative: str) -> Path:
    """Resolve a relative report path without allowing escapes or missing files."""
    path = (root / relative).resolve()
    if (
        Path(relative).is_absolute()
        or not path.is_relative_to(root.resolve())
        or not path.is_file()
    ):
        raise ValueError(f"Invalid manifest path: {relative}")
    return path


def write_json(path: Path, value: Any) -> None:
    """Write human-readable UTF-8 JSON, disallowing NaN and infinity."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    """Write stable headers even when no records were found."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
