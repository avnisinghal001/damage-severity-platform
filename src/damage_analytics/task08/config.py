"""Pinned research inputs and small, explicit Task 08 settings."""

import tomllib
from dataclasses import dataclass
from pathlib import Path

from damage_analytics.io import sha256_file
from damage_analytics.task07.config import TrainConfig, validate_training
from damage_analytics.task07.release import Release, open_release


@dataclass(frozen=True)
class Config:
    output_dir: Path
    sdnet_release: Path
    sdnet_seal_sha256: str
    damsegment_release: Path
    damsegment_seal_sha256: str
    baseline_runs: tuple[Path, ...]
    seed: int
    phash_distance: int
    max_near_candidates: int
    brute_force_queries: int
    gallery_samples: int
    pairs_per_page: int
    training: TrainConfig


def load_config(path: Path) -> Config:
    values = tomllib.loads(path.read_text())
    for key in ("output_dir", "sdnet_release", "damsegment_release"):
        values[key] = (path.parent / values[key]).resolve()
    values["baseline_runs"] = tuple((path.parent / p).resolve() for p in values["baseline_runs"])
    values["training"] = TrainConfig(**values["training"])
    result = Config(**values)
    allowed = Path(__file__).resolve().parents[3] / "outputs/task08"
    if result.output_dir != allowed and allowed not in result.output_dir.parents:
        raise ValueError("Task 08 outputs must remain under outputs/task08")
    if (
        not 0 <= result.phash_distance <= 8
        or result.max_near_candidates < 1
        or not 1 <= result.brute_force_queries <= 100
        or not 1 <= result.gallery_samples <= 40
        or not 1 <= result.pairs_per_page <= 12
        or type(result.seed) is not int
        or result.seed < 0
    ):
        raise ValueError("Invalid Task 08 analysis budgets")
    validate_training(result.training)
    return result


def load_releases(config: Config) -> dict[str, Release]:
    """Check pinned metadata/seals without decoding any test pixels."""
    return {name: load_release(config, name) for name in ("sdnet2018", "damsegment")}


def load_release(config: Config, name: str) -> Release:
    """Training opens source metadata only; external data is an explicit separate operation."""
    if name not in ("sdnet2018", "damsegment"):
        raise ValueError("Unknown Task 08 dataset")
    prefix = "sdnet" if name == "sdnet2018" else "damsegment"
    root, expected = getattr(config, f"{prefix}_release"), getattr(config, f"{prefix}_seal_sha256")
    if sha256_file(root / "checksums.json") != expected:
        raise ValueError(f"Pinned {name} release seal changed")
    release = open_release(root)
    if release.metadata["dataset"] != name:
        raise ValueError("Configured dataset identity differs")
    return release
