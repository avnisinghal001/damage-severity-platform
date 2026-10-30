"""Centralized, validated training settings with no dependency on PyTorch."""

import math
import re
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path

IGNORE_INDEX = 255


@dataclass(frozen=True)
class DataConfig:
    """Paired preprocessing settings; spatial tuples are always height, width."""

    image_size: tuple[int, int] = (256, 256)
    mean: tuple[float, ...] = (0.5, 0.5, 0.5)
    std: tuple[float, ...] = (0.5, 0.5, 0.5)
    horizontal_flip: float = 0.5
    vertical_flip: float = 0.0
    crop_probability: float = 0.5
    crop_size: tuple[int, int] = (512, 512)
    foreground_crop_probability: float = 0.8


@dataclass(frozen=True)
class ModelConfig:
    """Small GroupNorm U-Net, always with nine semantic output channels."""

    base_channels: int = 16
    depth: int = 3
    normalization_groups: int = 8


@dataclass(frozen=True)
class OptimizationConfig:
    """Training and validation-only model selection parameters."""

    epochs: int = 50
    batch_size: int = 4
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    dice_weight: float = 0.5
    class_weighting: str = "inverse_sqrt"
    max_class_weight: float = 10.0
    patience: int = 8
    min_delta: float = 1e-4
    gradient_clip: float = 1.0
    validation_every: int = 1


@dataclass(frozen=True)
class PerformanceConfig:
    """Execution policies; defaults preserve historical data and tensor behavior."""

    persistent_workers: bool = False
    cache_mb: int = 0
    prefetch_factor: int = 2
    channels_last: bool = False
    plot_every: int = 1


@dataclass(frozen=True)
class SanityConfig:
    """Tiny training-only overfit experiment, never a held-out performance claim."""

    overfit_samples: int = 2
    overfit_epochs: int = 100
    overfit_target_miou: float = 0.9


@dataclass(frozen=True)
class TrainingConfig:
    """Pinned input and isolated experiment output roots."""

    release_dir: Path
    release_id: str
    checksums_sha256: str
    output_dir: Path
    seed: int
    device: str
    deterministic: bool
    num_workers: int
    data: DataConfig
    model: ModelConfig
    optimization: OptimizationConfig
    sanity: SanityConfig
    comparison_samples: int = 4
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)

    def snapshot(self) -> dict:
        """Serialize effective settings with portable path strings."""
        values = asdict(self)
        for name in ("release_dir", "output_dir"):
            values[name] = str(values[name])
        return values


def load_training_config(path: Path) -> TrainingConfig:
    """Resolve config-relative paths and validate before any output is created."""
    with path.open("rb") as stream:
        settings = tomllib.load(stream)
    for name in ("release_dir", "output_dir"):
        settings[name] = (path.parent / settings[name]).resolve()
    config = TrainingConfig(
        data=DataConfig(**settings.pop("data", {})),
        model=ModelConfig(**settings.pop("model", {})),
        optimization=OptimizationConfig(**settings.pop("optimization", {})),
        sanity=SanityConfig(**settings.pop("sanity", {})),
        performance=PerformanceConfig(**settings.pop("performance", {})),
        **settings.pop("reporting", {}),
        **settings,
    )
    validate_config(config)
    project_root = Path(__file__).resolve().parents[3]
    for protected in (
        project_root / "data",
        project_root / "src",
        project_root / "config",
        project_root / "docs",
        project_root / "tests",
        project_root / "outputs/task01",
        project_root / "outputs/task02",
        config.release_dir,
    ):
        if (
            config.output_dir == protected
            or protected in config.output_dir.parents
            or config.output_dir in protected.parents
        ):
            raise ValueError(
                "Training output must be separate from source data/code and Tasks 01/02"
            )
    return config


def validate_config(config: TrainingConfig) -> None:
    """Reject unsafe values, malformed probabilities, and incompatible tensor sizes."""
    if config.device not in ("auto", "cpu", "mps") or type(config.deterministic) is not bool:
        raise ValueError("device must be auto/cpu/mps; deterministic must be boolean")
    if not re.fullmatch(r"[0-9a-f]{64}", config.checksums_sha256):
        raise ValueError("Pin the SHA-256 of Task 02 checksums.json")
    for value in (config.seed, config.num_workers):
        if type(value) is not int or value < 0:
            raise ValueError("seed and num_workers must be nonnegative integers")
    data, model, opt = config.data, config.model, config.optimization
    for size in (data.image_size, data.crop_size):
        if len(size) != 2 or any(type(v) is not int or v < 1 for v in size):
            raise ValueError("Spatial sizes need two positive integers (height, width)")
    if (
        type(model.depth) is not int
        or not 1 <= model.depth <= 5
        or type(model.base_channels) is not int
        or not 2 <= model.base_channels <= 64
        or type(model.normalization_groups) is not int
        or model.normalization_groups < 1
    ):
        raise ValueError("Invalid U-Net architecture")
    if any(v % 2**model.depth for v in data.image_size):
        raise ValueError("image_size must be divisible by 2**depth")
    if (
        len(data.mean) != 3
        or len(data.std) != 3
        or any(not math.isfinite(v) or not 0 <= v <= 1 for v in data.mean)
        or any(not math.isfinite(v) or v <= 0 for v in data.std)
    ):
        raise ValueError("RGB mean/std require three finite values; std must be positive")
    for probability in (
        data.horizontal_flip,
        data.vertical_flip,
        data.crop_probability,
        data.foreground_crop_probability,
        config.sanity.overfit_target_miou,
    ):
        if not 0 <= probability <= 1:
            raise ValueError("Probabilities/overfit target must be in [0,1]")
    for value in (
        opt.epochs,
        opt.batch_size,
        opt.patience,
        config.comparison_samples,
        config.sanity.overfit_samples,
        config.sanity.overfit_epochs,
        opt.validation_every,
        config.performance.prefetch_factor,
        config.performance.plot_every,
    ):
        if type(value) is not int or value < 1:
            raise ValueError("Counts must be positive integers")
    for value in (opt.learning_rate, opt.max_class_weight, opt.gradient_clip):
        if not math.isfinite(value) or value <= 0:
            raise ValueError("Learning rate, weight bound, and clipping must be positive")
    for value in (opt.weight_decay, opt.dice_weight, opt.min_delta):
        if not math.isfinite(value) or value < 0:
            raise ValueError("Loss/optimizer settings must be finite and nonnegative")
    if opt.class_weighting not in ("none", "inverse_sqrt"):
        raise ValueError("Unknown class weighting policy")
    performance = config.performance
    if (
        type(performance.cache_mb) is not int
        or performance.cache_mb < 0
        or type(performance.persistent_workers) is not bool
        or type(performance.channels_last) is not bool
    ):
        raise ValueError("Cache size must be nonnegative; execution switches must be boolean")
    if performance.persistent_workers and config.num_workers == 0:
        raise ValueError("Persistent workers require num_workers > 0")
