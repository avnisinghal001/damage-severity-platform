"""Central research configuration without importing PyTorch or torchvision."""

import math
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    patience: int = 6
    min_delta: float = 1e-4
    scheduler: str = "cosine"
    class_weighting: str = "balanced"
    threshold_objective: str = "balanced_accuracy"
    image_size: int = 224
    device: str = "auto"
    deterministic: bool = False
    cpu_threads: int = 4
    num_workers: int = 0


@dataclass(frozen=True)
class Config:
    sdnet2018_archive: Path
    damsegment_archive: Path
    output_dir: Path
    seed: int
    fractions: tuple[float, ...]
    source_groups_csv: Path | None
    near_reviews_csv: Path | None
    phash_distance: int
    max_near_candidates: int
    damsegment: dict
    training: TrainConfig
    max_image_pixels: int = 32_000_000
    mask_min_polygon_iou: float = 0.5

    def archive(self, dataset: str) -> Path:
        if dataset not in ("sdnet2018", "damsegment"):
            raise ValueError("Unknown dataset")
        return getattr(self, f"{dataset}_archive")

    def snapshot(self) -> dict:
        value = asdict(self)
        for key in (
            "sdnet2018_archive",
            "damsegment_archive",
            "output_dir",
            "source_groups_csv",
            "near_reviews_csv",
        ):
            value[key] = str(value[key]) if value[key] is not None else None
        return value


def load_config(path: Path) -> Config:
    values = tomllib.loads(path.read_text())
    for name in (
        "sdnet2018_archive",
        "damsegment_archive",
        "output_dir",
        "source_groups_csv",
        "near_reviews_csv",
    ):
        value = values.get(name)
        values[name] = (path.parent / value).resolve() if value else None
    values["fractions"] = tuple(values["fractions"])
    result = Config(**{**values, "training": TrainConfig(**values["training"])})
    root = Path(__file__).resolve().parents[3]
    allowed = root / "outputs/task07"
    if result.output_dir != allowed and allowed not in result.output_dir.parents:
        raise ValueError("Task 07 writes must remain under outputs/task07")
    if (
        len(result.fractions) != 3
        or min(result.fractions) <= 0
        or not math.isclose(sum(result.fractions), 1.0)
        or type(result.seed) is not int
        or result.seed < 0
        or not 0 <= result.phash_distance <= 8
        or result.max_near_candidates < 1
    ):
        raise ValueError("Invalid split/duplicate settings")
    if len(result.damsegment["category_names"]) != 2:
        raise ValueError("The inspected DamSegment schema has two source categories")
    if type(result.damsegment["semantic_names_verified"]) is not bool:
        raise ValueError("Taxonomy verification must be explicit boolean")
    if not result.damsegment["semantic_names_verified"] and result.damsegment[
        "category_names"
    ] != ["source_category_0", "source_category_1"]:
        raise ValueError("Unverified numeric categories must retain neutral source-ID names")
    if result.damsegment["semantic_names_verified"] and not result.damsegment["semantic_evidence"]:
        raise ValueError("Verified semantic names require explicit documentary evidence")
    validate_training(result.training)
    if result.max_image_pixels < 1 or not 0 <= result.mask_min_polygon_iou <= 1:
        raise ValueError("Invalid pixel/mask validation budgets")
    return result


def validate_training(config: TrainConfig) -> None:
    for value in (
        config.epochs,
        config.batch_size,
        config.patience,
        config.image_size,
        config.cpu_threads,
    ):
        if type(value) is not int or value < 1:
            raise ValueError("Training counts must be positive integers")
    if config.num_workers < 0 or config.device not in ("cpu", "mps", "auto"):
        raise ValueError("Invalid device/workers")
    for value in (config.learning_rate, config.weight_decay, config.min_delta):
        if not math.isfinite(value) or value < 0:
            raise ValueError("Invalid optimizer values")
    if config.learning_rate == 0 or config.scheduler not in ("cosine", "none"):
        raise ValueError("Invalid learning rate/scheduler")
    if config.class_weighting not in ("balanced", "none") or config.threshold_objective not in (
        "balanced_accuracy",
        "macro_f1",
    ):
        raise ValueError("Invalid weighting/threshold objective")
