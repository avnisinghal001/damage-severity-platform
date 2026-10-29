"""Typed, validated Task 02 policy configuration, reusing Task 01 paths."""

import math
import re
import tomllib
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from damage_analytics.config import Config, load_config


@dataclass(frozen=True)
class CleaningPolicy:
    """Conservative defaults: no topology repair, no conflicting-label precedence."""

    remove_redundant_vertices: bool = True
    out_of_bounds_tolerance: float = 0.0
    geometry_epsilon: float = 1e-9
    exif_policy: str = "match_annotation_dimensions"
    allow_rgb_conversion: bool = True
    cross_class_overlap: str = "exclude"
    allow_empty_annotations: bool = False
    deduplicate_identical_pairs: bool = True


@dataclass(frozen=True)
class SimilarityPolicy:
    """Conservative similarity candidates are grouped, not declared identical."""

    phash_distance: int = 6
    dhash_distance: int = 8
    thumbnail_mae: float = 0.12
    aspect_log_tolerance: float = 0.15
    annotation_iou: float = 0.98
    group_capture_sessions: bool = True
    capture_session_gap_seconds: int = 600


@dataclass(frozen=True)
class SplitPolicy:
    """Fixed fractions and finite deterministic search; no model feedback."""

    fractions: tuple[float, ...] = (0.7, 0.15, 0.15)
    search_restarts: int = 24
    rare_class_image_threshold: int = 60


@dataclass(frozen=True)
class CurationConfig:
    """Task 01 data locations plus Task 02 release policies and evidence paths."""

    paths: Config
    task01_dir: Path
    output_dir: Path
    dataset_name: str
    random_seed: int
    cleaning: CleaningPolicy
    similarity: SimilarityPolicy
    splitting: SplitPolicy
    source_groups_file: Path | None = None

    def snapshot(self) -> dict:
        """Return path-independent scientific settings for release identity."""
        return {
            "dataset_name": self.dataset_name,
            "random_seed": self.random_seed,
            "cleaning": asdict(self.cleaning),
            "similarity": asdict(self.similarity),
            "splitting": asdict(self.splitting),
        }


def load_curation_config(path: Path) -> CurationConfig:
    """Load policies without changing Task 01 defaults or requiring another package."""
    with path.open("rb") as stream:
        settings = tomllib.load(stream)
    base = load_config(path.parent / settings.pop("base_config"))
    output = (path.parent / settings.pop("output_dir")).resolve()
    root = path.resolve().parent.parent
    for other in (
        base.raw_dir,
        base.extracted_dir,
        base.output_dir,
        root / "src",
        root / "tests",
        root / "config",
        root / "docs",
    ):
        if output == other or output in other.parents or other in output.parents:
            raise ValueError("Task 02 outputs must be separate from inputs and project source")
    if output == root or output in root.parents:
        raise ValueError("Task 02 output cannot be the project root or an ancestor")
    source_file = settings.pop("source_groups_file", None)
    config = CurationConfig(
        paths=replace(base, output_dir=output),
        task01_dir=base.output_dir,
        output_dir=output,
        cleaning=CleaningPolicy(**settings.pop("cleaning", {})),
        similarity=SimilarityPolicy(**settings.pop("similarity", {})),
        splitting=SplitPolicy(**settings.pop("splitting", {})),
        source_groups_file=(path.parent / source_file).resolve() if source_file else None,
        **settings,
    )
    _validate(config)
    return config


def _validate(config: CurationConfig) -> None:
    """Reject unsupported policies rather than accepting misspelled settings."""
    policy, similarity, split = config.cleaning, config.similarity, config.splitting
    if (
        not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", config.dataset_name)
        or type(config.random_seed) is not int
        or config.random_seed < 0
    ):
        raise ValueError(
            "dataset_name must be a safe slug (1..80 characters); seed must be nonnegative"
        )
    if policy.exif_policy not in ("exclude", "match_annotation_dimensions"):
        raise ValueError("Unsupported exif_policy")
    if policy.cross_class_overlap not in ("exclude", "last_wins"):
        raise ValueError("Unsupported cross_class_overlap policy")
    for key, value in asdict(policy).items():
        if (
            key
            not in (
                "out_of_bounds_tolerance",
                "geometry_epsilon",
                "exif_policy",
                "cross_class_overlap",
            )
            and type(value) is not bool
        ):
            raise ValueError(f"{key} must be a boolean")
    if (
        any(
            type(value) not in (int, float)
            for value in (policy.out_of_bounds_tolerance, policy.geometry_epsilon)
        )
        or not math.isfinite(policy.out_of_bounds_tolerance)
        or policy.out_of_bounds_tolerance < 0
        or not math.isfinite(policy.geometry_epsilon)
        or policy.geometry_epsilon <= 0
    ):
        raise ValueError("Geometry tolerances must be finite and nonnegative (epsilon > 0)")
    for distance in (similarity.phash_distance, similarity.dhash_distance):
        if type(distance) is not int or not 0 <= distance <= 64:
            raise ValueError("Hash distances must be integers in 0..64")
    if (
        type(similarity.group_capture_sessions) is not bool
        or type(similarity.capture_session_gap_seconds) is not int
        or similarity.capture_session_gap_seconds < 1
    ):
        raise ValueError(
            "Capture-session grouping requires a boolean and positive integer seconds"
        )
    if (
        not 0 <= similarity.thumbnail_mae <= 1
        or not math.isfinite(similarity.aspect_log_tolerance)
        or similarity.aspect_log_tolerance < 0
        or not 0 <= similarity.annotation_iou <= 1
    ):
        raise ValueError("Invalid similarity thresholds")
    if (
        len(split.fractions) != 3
        or any(not 0 < value < 1 for value in split.fractions)
        or not math.isclose(sum(split.fractions), 1, abs_tol=1e-9)
    ):
        raise ValueError("Exactly three positive train/validation/test fractions must sum to 1")
    if (
        type(split.search_restarts) is not int
        or not 1 <= split.search_restarts <= 1000
        or type(split.rare_class_image_threshold) is not int
        or split.rare_class_image_threshold < 1
    ):
        raise ValueError("Invalid split search/rarity settings")
