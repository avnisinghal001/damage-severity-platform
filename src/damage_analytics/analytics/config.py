"""Central Task 04 settings, preserving the earlier training configuration and outputs."""

import math
import re
import tomllib
from dataclasses import dataclass, replace
from pathlib import Path

from damage_analytics.training.config import (
    PerformanceConfig,
    TrainingConfig,
    load_training_config,
)


@dataclass(frozen=True)
class RegionPolicy:
    """Component connectivity and non-destructive reporting filter."""

    connectivity: int = 8
    minimum_reported_pixels: int = 1


@dataclass(frozen=True)
class UncertaintyPolicy:
    """Uncalibrated review triggers; never convert these to severity thresholds."""

    maximum_softmax_below: float = 0.6
    normalized_entropy_above: float = 0.65


@dataclass(frozen=True)
class VerificationPolicy:
    """Fixed numerical comparison tolerances for cross-backend baseline verification."""

    metric_absolute_tolerance: float = 1e-5
    confusion_l1_half_fraction_tolerance: float = 1e-5


@dataclass(frozen=True)
class SupervisionPolicy:
    """Exploratory independent-group count gates, not a power analysis."""

    minimum_training_groups: int = 30
    minimum_validation_groups: int = 10


@dataclass(frozen=True)
class ComparisonPolicy:
    """Paired validation component bootstrap, conditional on fixed trained models."""

    bootstrap_repeats: int = 200
    bootstrap_seed: int = 123


@dataclass(frozen=True)
class Experiment:
    """One registered single-factor variant, or the unchanged matched control."""

    name: str
    hypothesis: str
    image_size: tuple[int, int] | None = None
    class_weighting: str | None = None
    max_class_weight: float | None = None


@dataclass(frozen=True)
class SchedulePolicy:
    """Shared training budget and validation cadence for a comparison cohort."""

    epochs: int | None = None
    validation_every: int = 1
    batch_size: int | None = None


@dataclass(frozen=True)
class AnalyticsConfig:
    """Resolved read-only baseline inputs and isolated Task 04 output root."""

    training: TrainingConfig
    baseline_checkpoint: Path
    baseline_sha256: str
    baseline_validation: Path
    output_dir: Path
    regions: RegionPolicy
    uncertainty: UncertaintyPolicy
    verification: VerificationPolicy
    supervision: SupervisionPolicy
    comparison: ComparisonPolicy
    experiments: tuple[Experiment, ...]
    device: str = "cpu"
    deterministic: bool = True
    cpu_threads: int = 4
    comparison_samples: int = 4
    performance: PerformanceConfig = PerformanceConfig()
    num_workers: int = 0
    schedule: SchedulePolicy = SchedulePolicy()


def load_analytics_config(path: Path) -> AnalyticsConfig:
    """Load new settings without editing the original Task 03 config/checkpoint."""
    with path.open("rb") as stream:
        values = tomllib.load(stream)
    training = load_training_config(path.parent / values.pop("task03_config"))
    for key in ("baseline_checkpoint", "baseline_validation", "output_dir"):
        values[key] = (path.parent / values[key]).resolve()
    config = AnalyticsConfig(
        training=training,
        regions=RegionPolicy(**values.pop("regions", {})),
        uncertainty=UncertaintyPolicy(**values.pop("uncertainty", {})),
        verification=VerificationPolicy(**values.pop("verification", {})),
        supervision=SupervisionPolicy(**values.pop("supervision", {})),
        comparison=ComparisonPolicy(**values.pop("comparison", {})),
        experiments=tuple(Experiment(**row) for row in values.pop("experiments", [])),
        performance=PerformanceConfig(**values.pop("performance", {})),
        schedule=SchedulePolicy(**values.pop("schedule", {})),
        **values,
    )
    root = Path(__file__).resolve().parents[3]
    for protected in (
        root / "data",
        root / "src",
        root / "config",
        root / "docs",
        root / "tests",
        root / "outputs/task01",
        root / "outputs/task02",
        root / "outputs/task03",
        training.release_dir,
        training.output_dir,
    ):
        if (
            config.output_dir == protected
            or protected in config.output_dir.parents
            or config.output_dir in protected.parents
        ):
            raise ValueError(
                "Task 04 outputs must be separate from Tasks 01–03 and source data/code"
            )
    validate_analytics_config(config)
    return config


def validate_analytics_config(config: AnalyticsConfig) -> None:
    """Reject unsupported policies or accidental test-report inputs before writing outputs."""
    if config.device not in ("auto", "cpu", "mps") or type(config.deterministic) is not bool:
        raise ValueError("Invalid device/determinism setting")
    from damage_analytics.training.config import validate_config

    validate_config(runtime_training_config(config))
    if any(
        type(value) is not int or value < 1
        for value in (
            config.schedule.epochs,
            config.schedule.validation_every,
            config.schedule.batch_size,
        )
        if value is not None
    ):
        raise ValueError("Schedule counts must be positive integers")
    if not re.fullmatch(r"[0-9a-f]{64}", config.baseline_sha256):
        raise ValueError("Pin the full baseline checkpoint SHA-256")
    if "test_final" in config.baseline_validation.parts:
        raise ValueError("Task 04 must not consume final-test reports")
    if config.regions.connectivity not in (4, 8):
        raise ValueError("Connectivity must be 4 or 8")
    for number in (
        config.regions.minimum_reported_pixels,
        config.cpu_threads,
        config.comparison_samples,
        config.supervision.minimum_training_groups,
        config.supervision.minimum_validation_groups,
    ):
        if type(number) is not int or number < 1:
            raise ValueError("Counts/threads must be positive integers")
    for value in (
        config.uncertainty.maximum_softmax_below,
        config.uncertainty.normalized_entropy_above,
        config.verification.metric_absolute_tolerance,
        config.verification.confusion_l1_half_fraction_tolerance,
    ):
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Review thresholds/tolerances must be finite in [0,1]")
    if (
        type(config.comparison.bootstrap_repeats) is not int
        or not 0 <= config.comparison.bootstrap_repeats <= 5000
        or type(config.comparison.bootstrap_seed) is not int
        or config.comparison.bootstrap_seed < 0
    ):
        raise ValueError("Invalid bootstrap settings")
    names = [item.name for item in config.experiments]
    if len(names) != len(set(names)) or not 1 <= len(names) <= 3:
        raise ValueError("Register one to three unique experiments")
    for item in config.experiments:
        if (
            not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,59}", item.name)
            or not item.hypothesis.strip()
        ):
            raise ValueError("Experiments need safe names and explicit hypotheses")
        if (
            sum(
                value is not None
                for value in (item.image_size, item.class_weighting, item.max_class_weight)
            )
            > 1
        ):
            raise ValueError("Change one factor at a time")
        if item.name == "control" and any(
            value is not None
            for value in (item.image_size, item.class_weighting, item.max_class_weight)
        ):
            raise ValueError("The matched control must keep the baseline scientific settings")
        if item.image_size is not None and (
            len(item.image_size) != 2 or any(type(v) is not int or v < 1 for v in item.image_size)
        ):
            raise ValueError("Resolution needs two positive integers")
        if item.class_weighting not in (None, "none", "inverse_sqrt"):
            raise ValueError("Unsupported class weighting")
        if item.max_class_weight is not None and (
            not math.isfinite(item.max_class_weight) or item.max_class_weight <= 0
        ):
            raise ValueError("Class-weight cap must be finite and positive")


def runtime_training_config(config: AnalyticsConfig) -> TrainingConfig:
    """Isolate all reused training helpers' writes under Task 04."""
    return replace(
        config.training,
        output_dir=config.output_dir,
        device=config.device,
        deterministic=config.deterministic,
        comparison_samples=config.comparison_samples,
        num_workers=config.num_workers,
        performance=config.performance,
    )
