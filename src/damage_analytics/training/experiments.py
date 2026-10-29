"""Create new, auditable experiment directories without overwriting earlier runs."""

import importlib.metadata
import platform
import re
import subprocess
from pathlib import Path

from damage_analytics.io import sha256_file, write_json
from damage_analytics.training.config import TrainingConfig
from damage_analytics.training.manifests import Release


def create_experiment(
    config: TrainingConfig,
    release: Release,
    run_id: str,
    device: str,
    *,
    purpose: str = "training",
) -> tuple[Path, dict]:
    """Record actual code/config/device/runtime before training; refuse existing run IDs."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", run_id):
        raise ValueError("Use a safe run ID of 1..80 letters/numbers/dots/underscores/hyphens")
    folder = config.output_dir / "experiments" / run_id
    folder.mkdir(parents=True, exist_ok=False)
    metadata = {
        "run_id": run_id,
        "purpose": purpose,
        "dataset": release.identity(),
        "config": config.snapshot(),
        "seed": config.seed,
        "device": device,
        **execution_provenance(),
        "selection": "Best native-resolution validation foreground mIoU; no test model selection",
        "reproducibility": "Seeded workers/epoch transforms; deterministic ops requested. "
        "Bitwise equivalence across devices/versions is not guaranteed.",
        "test_evaluated": False,
        "scientific_scope": "Semantic categories only; Task 02 conditional-research limitations "
        "and unknown physical source identities remain unresolved.",
    }
    write_json(folder / "run.json", metadata)
    return folder, metadata


def execution_provenance() -> dict:
    """Capture source hashes and available versions, including torch=None before installation."""
    package = Path(__file__).resolve().parents[1]
    code = {
        path.relative_to(package).as_posix(): sha256_file(path)
        for path in sorted(package.rglob("*.py"))
    }
    try:
        commit = (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=package,
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip()
            or None
        )
    except OSError:
        commit = None
    versions = {}
    for name in ("torch", "Pillow", "numpy", "matplotlib"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "code_sha256": code,
        "git_commit": commit,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            **versions,
        },
    }
