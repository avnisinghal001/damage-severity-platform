"""Validated Task 10 configuration and immutable source label registration."""

import math
import re
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path

CLASS_NAMES = ("Heavy", "Minor", "Moderate", "Undamaged")
SOURCE_LABELS = ("Heavy damage", "Minor damage", "Moderate damage", "Undamaged state")
SOURCE_CODES = ("HvD", "MiD", "MoD", "UD")


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 25
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    patience: int = 6
    min_delta: float = 1e-4
    class_weighting: str = "balanced"
    scheduler: str = "cosine"
    image_size: int = 224
    num_workers: int = 0
    cpu_threads: int = 1
    deterministic: bool = False

    def validate(self) -> None:
        for value in (
            self.epochs,
            self.batch_size,
            self.patience,
            self.image_size,
            self.cpu_threads,
        ):
            if type(value) is not int or value < 1:
                raise ValueError("Training counts must be positive integers")
        if (
            self.num_workers < 0
            or self.class_weighting not in ("balanced", "none")
            or self.scheduler not in ("cosine", "none")
        ):
            raise ValueError("Invalid training workers/weighting/scheduler")
        if (
            any(
                not math.isfinite(v) or v < 0
                for v in (self.learning_rate, self.weight_decay, self.min_delta)
            )
            or not self.learning_rate
        ):
            raise ValueError("Invalid optimizer settings")
        if type(self.deterministic) is not bool:
            raise ValueError("Determinism policy must be explicit boolean")


@dataclass(frozen=True)
class Config:
    archive: Path
    output_dir: Path
    seed: int
    validation_fraction: float
    expected_image_shape: tuple[int, int, int]
    source_groups_csv: Path | None
    encoding: str
    encoding_verified: bool
    encoding_evidence: str
    roundtrip_tolerance: float
    phash_distance: int
    dhash_distance: int
    thumbnail_mae: float
    crop_fraction: float
    maximum_near_pairs: int
    training: TrainConfig

    def preparation_settings(self) -> dict:
        values = asdict(self)
        for key in ("archive", "output_dir", "training", "source_groups_csv"):
            values.pop(key)
        return values


def safe_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", value):
        raise ValueError("Use a safe new Task 10 run/report ID")
    return value


def load_config(path: Path) -> Config:
    values = tomllib.loads(path.read_text())
    for key in ("archive", "output_dir", "source_groups_csv"):
        value = values[key]
        values[key] = (path.parent / value).resolve() if value else None
    values["expected_image_shape"] = tuple(values["expected_image_shape"])
    values["training"] = TrainConfig(**values["training"])
    config = Config(**values)
    allowed = Path(__file__).resolve().parents[3] / "outputs/task10"
    if config.output_dir != allowed and allowed not in config.output_dir.parents:
        raise ValueError("Task 10 writes must remain under outputs/task10")
    if (
        type(config.seed) is not int
        or config.seed < 0
        or not 0 < config.validation_fraction < 1
        or len(config.expected_image_shape) != 3
        or config.expected_image_shape[2] != 3
        or any(type(x) is not int or x < 1 for x in config.expected_image_shape)
        or config.encoding != "caffe_bgr"
        or type(config.encoding_verified) is not bool
        or not 0 < config.roundtrip_tolerance < 0.1
        or not 0 <= config.phash_distance <= 8
        or not 0 <= config.dhash_distance <= 16
        or not 0 < config.thumbnail_mae < 0.5
        or not 0.5 <= config.crop_fraction < 1
        or config.maximum_near_pairs < 1
    ):
        raise ValueError("Invalid tensor/preparation policy")
    if config.encoding_verified and not config.encoding_evidence.strip():
        raise ValueError("Verified encoding requires documentary evidence, not inference alone")
    config.training.validate()
    return config
