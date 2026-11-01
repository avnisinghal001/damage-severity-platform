"""Environment-based API settings, with no database credentials stored in the repository."""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """One worker's immutable serving settings; paths resolve from the working directory."""

    database_url: str = field(repr=False)
    checkpoint: Path
    checkpoint_sha256: str
    storage_dir: Path = Path("outputs/task05/inspections")
    device: str = "cpu"
    cpu_threads: int = 2
    max_upload_bytes: int = 10 * 1024**2
    max_pixels: int = 4_000_000
    max_dimension: int = 8192
    database_timeout_seconds: int = 3
    upload_timeout_seconds: int = 30

    def __post_init__(self) -> None:
        if not self.database_url.startswith(("postgresql://", "postgres://")):
            raise ValueError("DAMAGE_API_DATABASE_URL must be a PostgreSQL connection URL")
        if not re.fullmatch(r"[0-9a-f]{64}", self.checkpoint_sha256):
            raise ValueError("DAMAGE_API_CHECKPOINT_SHA256 must pin the full SHA-256")
        if self.device not in ("cpu", "mps"):
            raise ValueError("DAMAGE_API_DEVICE must be cpu or mps")
        for value in (
            self.cpu_threads,
            self.max_upload_bytes,
            self.max_pixels,
            self.max_dimension,
            self.database_timeout_seconds,
            self.upload_timeout_seconds,
        ):
            if type(value) is not int or value < 1:
                raise ValueError("API resource limits must be positive integers")
        root = Path(__file__).resolve().parents[3]
        storage = self.storage_dir.resolve()
        protected = [
            root / name
            for name in (
                "src",
                "config",
                "docs",
                "tests",
                "data",
                ".git",
                "outputs/task01",
                "outputs/task02",
                "outputs/task03",
                "outputs/task04",
            )
        ] + [self.checkpoint.resolve()]
        if any(
            storage == path or storage in path.parents or path in storage.parents
            for path in protected
        ):
            raise ValueError(
                "API storage must not overlap source, earlier tasks or the checkpoint"
            )

    @classmethod
    def from_env(cls) -> "Settings":
        """Fail clearly on missing pins/credentials; never connect or run migrations here."""
        required = ("DAMAGE_API_DATABASE_URL", "DAMAGE_API_CHECKPOINT_SHA256")
        if any(not os.environ.get(name) for name in required):
            raise ValueError("Set DAMAGE_API_DATABASE_URL and DAMAGE_API_CHECKPOINT_SHA256")
        values = {
            "database_url": os.environ[required[0]],
            "checkpoint_sha256": os.environ[required[1]],
            "checkpoint": Path(
                os.getenv(
                    "DAMAGE_API_CHECKPOINT", "outputs/task04/experiments/control_s42/best.pt"
                )
            ),
            "storage_dir": Path(os.getenv("DAMAGE_API_STORAGE_DIR", "outputs/task05/inspections")),
            "device": os.getenv("DAMAGE_API_DEVICE", "cpu"),
        }
        for name, default in (
            ("cpu_threads", 2),
            ("max_upload_bytes", 10 * 1024**2),
            ("max_pixels", 4_000_000),
            ("max_dimension", 8192),
            ("database_timeout_seconds", 3),
            ("upload_timeout_seconds", 30),
        ):
            values[name] = int(os.getenv(f"DAMAGE_API_{name.upper()}", str(default)))
        return cls(**values)
