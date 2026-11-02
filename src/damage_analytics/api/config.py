"""Environment-based API settings, with no database credentials stored in the repository."""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


class UploadLimits(Protocol):
    max_upload_bytes: int
    max_pixels: int
    max_dimension: int
    upload_timeout_seconds: int


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
    crack_checkpoint: Path | None = None
    crack_checkpoint_sha256: str | None = None
    severity_url: str | None = None
    severity_checkpoint_sha256: str | None = None
    severity_timeout_seconds: int = 60

    def __post_init__(self) -> None:
        if not self.database_url.startswith(("postgresql://", "postgres://")):
            raise ValueError("DAMAGE_API_DATABASE_URL must be a PostgreSQL connection URL")
        if not re.fullmatch(r"[0-9a-f]{64}", self.checkpoint_sha256):
            raise ValueError("DAMAGE_API_CHECKPOINT_SHA256 must pin the full SHA-256")
        if self.device not in ("cpu", "mps"):
            raise ValueError("DAMAGE_API_DEVICE must be cpu or mps")
        integration = (
            self.crack_checkpoint,
            self.crack_checkpoint_sha256,
            self.severity_url,
            self.severity_checkpoint_sha256,
        )
        if any(v is not None for v in integration):
            from urllib.parse import urlsplit

            if not all(integration):
                raise ValueError(
                    "Integrated serving needs crack checkpoint/pin and severity URL/pin"
                )
            for digest in (self.crack_checkpoint_sha256, self.severity_checkpoint_sha256):
                if not re.fullmatch(r"[0-9a-f]{64}", digest):
                    raise ValueError("Every deployed model needs a SHA-256 pin")
            url = urlsplit(self.severity_url)
            if (
                url.scheme not in ("http", "https")
                or not url.netloc
                or url.username
                or url.password
                or url.path not in ("", "/")
                or url.query
                or url.fragment
            ):
                raise ValueError("Severity URL must be a trusted HTTP service origin")
        for value in (
            self.cpu_threads,
            self.max_upload_bytes,
            self.max_pixels,
            self.max_dimension,
            self.database_timeout_seconds,
            self.upload_timeout_seconds,
            self.severity_timeout_seconds,
        ):
            if type(value) is not int or value < 1:
                raise ValueError("API resource limits must be positive integers")
        root = Path(__file__).resolve().parents[3]
        storage = self.storage_dir.resolve()
        protected = (
            [
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
                    "outputs/task06",
                    "outputs/task07",
                    "outputs/task08",
                    "outputs/task09",
                    "outputs/task10",
                    "outputs/task11",
                )
            ]
            + [self.checkpoint.resolve()]
            + ([self.crack_checkpoint.resolve()] if self.crack_checkpoint else [])
        )
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
            "crack_checkpoint": Path(os.environ["DAMAGE_API_CRACK_CHECKPOINT"])
            if os.environ.get("DAMAGE_API_CRACK_CHECKPOINT")
            else None,
            "crack_checkpoint_sha256": os.getenv("DAMAGE_API_CRACK_CHECKPOINT_SHA256"),
            "severity_url": os.getenv("DAMAGE_API_SEVERITY_URL"),
            "severity_checkpoint_sha256": os.getenv("DAMAGE_API_SEVERITY_CHECKPOINT_SHA256"),
        }
        for name, default in (
            ("cpu_threads", 2),
            ("max_upload_bytes", 10 * 1024**2),
            ("max_pixels", 4_000_000),
            ("max_dimension", 8192),
            ("database_timeout_seconds", 3),
            ("upload_timeout_seconds", 30),
            ("severity_timeout_seconds", 60),
        ):
            values[name] = int(os.getenv(f"DAMAGE_API_{name.upper()}", str(default)))
        return cls(**values)
