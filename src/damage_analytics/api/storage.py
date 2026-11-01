"""Publish complete PNG artifacts under server-generated UUIDs, outside PostgreSQL."""

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import UUID

import numpy as np
from PIL import Image

from damage_analytics.analytics.regions import validate_mask
from damage_analytics.io import sha256_file


@dataclass(frozen=True)
class Artifacts:
    """Relative keys and content hashes; filesystem paths never come from upload filenames."""

    image_key: str
    image_sha256: str
    mask_key: str
    mask_sha256: str


class ArtifactStore:
    """One private storage root; only known database-backed artifact keys are served."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)

    def publish(self, identifier: UUID, image: Image.Image, mask: np.ndarray) -> Artifacts:
        """Stage both files, then rename the complete directory before database insertion."""
        validate_mask(mask)
        if mask.shape != (image.height, image.width):
            raise ValueError("Model mask does not match the oriented image")
        final = self.root / str(identifier)
        if final.exists():
            raise FileExistsError("Inspection artifacts already exist")
        with TemporaryDirectory(prefix=".pending-", dir=self.root) as temporary:
            stage = Path(temporary) / "artifacts"
            stage.mkdir(mode=0o700)
            image.save(stage / "image.png", format="PNG")
            Image.fromarray(mask.astype(np.uint8)).save(stage / "prediction_ids.png", format="PNG")
            image_hash = sha256_file(stage / "image.png")
            mask_hash = sha256_file(stage / "prediction_ids.png")
            stage.rename(final)
        return Artifacts(
            f"{identifier}/image.png", image_hash, f"{identifier}/prediction_ids.png", mask_hash
        )

    def file(self, identifier: UUID, key: str, kind: str, expected_sha256: str) -> Path:
        """Reject traversal, unrelated keys, symlink escapes, and changed artifact bytes."""
        name = {"image": "image.png", "mask": "prediction_ids.png"}[kind]
        if key != f"{identifier}/{name}":
            raise ValueError("Artifact key does not match its inspection")
        path = (self.root / key).resolve()
        if self.root not in path.parents or not path.is_file():
            raise FileNotFoundError("Artifact is unavailable")
        if sha256_file(path) != expected_sha256:
            raise ValueError("Artifact content hash changed")
        return path
