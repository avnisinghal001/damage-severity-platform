"""Load and validate the single Task 01 configuration file."""

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ArchiveLimits:
    """Upper bounds checked before extraction and enforced during reads."""

    max_files: int = 20000
    max_total_bytes: int = 5_000_000_000
    max_member_bytes: int = 100_000_000
    max_compression_ratio: int = 1000


@dataclass(frozen=True)
class Config:
    """Resolved paths and reproducible processing settings."""

    raw_dir: Path
    extracted_dir: Path
    output_dir: Path
    limits: ArchiveLimits
    expected_pairs: int = 1841
    gallery_size: int = 8
    random_seed: int = 42
    overlay_alpha: float = 0.45


def load_config(path: Path) -> Config:
    """Resolve paths relative to the config's project root; reject unsafe overlap."""
    with path.open("rb") as stream:
        settings = tomllib.load(stream)
    root = (path.resolve().parent / settings.pop("project_root", "..")).resolve()
    limits = ArchiveLimits(**settings.pop("archive_limits", {}))
    if any(value <= 0 for value in vars(limits).values()):
        raise ValueError("Archive limits must be positive")
    paths = {key: (root / settings.pop(key)).resolve()
             for key in ("raw_dir", "extracted_dir", "output_dir")}
    values = list(paths.values())
    for index, left in enumerate(values):
        for right in values[index + 1:]:
            if left == right or left in right.parents or right in left.parents:
                raise ValueError("Raw, extracted, and output directories must not overlap")
    if any(root == value or value in root.parents for value in values):
        raise ValueError("Data/output paths must not be the project root or its ancestors")
    config = Config(**paths, limits=limits, **settings)
    if config.expected_pairs < 1 or not 1 <= config.gallery_size <= 50:
        raise ValueError("Expected count must be positive; gallery_size must be 1..50")
    if not 0 < config.overlay_alpha <= 1:
        raise ValueError("overlay_alpha must be in (0, 1]")
    return config
