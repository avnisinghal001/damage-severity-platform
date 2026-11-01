"""Task 09 settings reuse validated Task 03 configuration without editing shared modules."""

import re
import tomllib
from dataclasses import dataclass, replace
from pathlib import Path

from damage_analytics.training.config import (
    DataConfig,
    OptimizationConfig,
    TrainingConfig,
    load_training_config,
    validate_config,
)


@dataclass(frozen=True)
class FeaturePolicy:
    connectivity: int = 8
    minimum_region_pixels: int = 1
    maximum_reported_regions: int = 1000
    minimum_by_class: dict[str, int] | None = None
    maximum_softmax_below: float = 0.6
    normalized_entropy_above: float = 0.65

    def validate(self) -> None:
        if self.connectivity not in (4, 8) or any(
            type(v) is not int or v < 1
            for v in (self.minimum_region_pixels, self.maximum_reported_regions)
        ):
            raise ValueError("Invalid feature region policy")
        for key, value in (self.minimum_by_class or {}).items():
            if key not in {str(i) for i in range(1, 9)} or type(value) is not int or value < 1:
                raise ValueError("Per-class minimum requires IDs 1..8 and positive integer pixels")
        if not 0 <= self.maximum_softmax_below <= 1 or not 0 <= self.normalized_entropy_above <= 1:
            raise ValueError("Uncertainty review thresholds must be in [0,1]")


@dataclass(frozen=True)
class Baseline:
    name: str
    checkpoint: Path
    sha256: str
    validation: Path


@dataclass(frozen=True)
class Config:
    source_config: Path
    output_dir: Path
    seed: int
    device: str
    cpu_threads: int
    audit_training_samples: int
    gallery_samples_per_class: int
    maximum_image_pixels: int
    freeze_pretrained_batch_norm: bool
    data: DataConfig
    optimization: OptimizationConfig
    features: FeaturePolicy
    baselines: tuple[Baseline, ...]

    def training(self, *, device: str | None = None) -> TrainingConfig:
        base = load_training_config(self.source_config)
        config = replace(
            base,
            output_dir=self.output_dir,
            seed=self.seed,
            device=device or self.device,
            deterministic=False,
            num_workers=0,
            data=self.data,
            optimization=self.optimization,
        )
        validate_config(config)
        return config


def safe_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", value):
        raise ValueError("Use a safe new Task 09 output ID")
    return value


def load_config(path: Path) -> Config:
    values = tomllib.loads(path.read_text())
    for name in ("source_config", "output_dir"):
        values[name] = (path.parent / values[name]).resolve()
    values["data"] = DataConfig(**values["data"])
    values["optimization"] = OptimizationConfig(**values["optimization"])
    values["features"] = FeaturePolicy(**values["features"])
    values["baselines"] = tuple(
        Baseline(
            **{
                **item,
                "checkpoint": (path.parent / item["checkpoint"]).resolve(),
                "validation": (path.parent / item["validation"]).resolve(),
            }
        )
        for item in values["baselines"]
    )
    config = Config(**values)
    allowed = Path(__file__).resolve().parents[3] / "outputs/task09"
    if config.output_dir != allowed and allowed not in config.output_dir.parents:
        raise ValueError("Task 09 writes must remain under outputs/task09")
    config.features.validate()
    config.training()
    if (
        config.device not in ("cpu", "mps")
        or not 1 <= config.cpu_threads <= 4
        or not 1 <= config.audit_training_samples <= 1132
        or not 1 <= config.gallery_samples_per_class <= 6
        or config.maximum_image_pixels < 1
        or type(config.freeze_pretrained_batch_norm) is not bool
    ):
        raise ValueError("Invalid Task 09 runtime/audit settings")
    for baseline in config.baselines:
        safe_id(baseline.name)
        if not re.fullmatch(r"[0-9a-f]{64}", baseline.sha256):
            raise ValueError("Baseline checkpoint hash must be pinned")
    return config
