"""Bounded nested ZIP access and staged extraction using the established ZIP validator."""

import hashlib
import json
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from zipfile import ZipFile

from damage_analytics.archives import checked_members
from damage_analytics.config import ArchiveLimits

LIMITS = ArchiveLimits(
    max_files=100000,
    max_total_bytes=10_000_000_000,
    max_member_bytes=1_000_000_000,
    max_compression_ratio=2000,
)


def useful(name: str) -> bool:
    return "__MACOSX" not in Path(name).parts and not Path(name).name.startswith(".")


def central_identity(archive: ZipFile) -> str:
    """Fast directory fingerprint only; full archive SHA-256 is recorded at preparation."""
    rows = [
        (x.filename, x.CRC, x.file_size, x.compress_size) for x in checked_members(archive, LIMITS)
    ]
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()


@contextmanager
def dataset_archives(path: Path, dataset: str):
    """Inspect nested ZIPs in bounded temporary storage, never extract image trees here."""
    with ZipFile(path) as outer:
        members = checked_members(outer, LIMITS)
        if dataset == "sdnet2018":
            yield {"classification": outer}
            return
        if dataset != "damsegment":
            raise ValueError("Unknown dataset")
        with tempfile.TemporaryDirectory(prefix="task07-nested-") as temporary:
            archives = {}
            try:
                for member in members:
                    if member.is_dir() or not useful(member.filename):
                        continue
                    role = {
                        "Damage Classification.zip": "classification",
                        "Damage Detection.zip": "detection",
                        "Damage Segmentaion.zip": "segmentation",
                    }.get(Path(member.filename).name)
                    if role is None or role in archives:
                        raise ValueError(f"Unexpected/duplicate nested archive: {member.filename}")
                    target = Path(temporary) / f"{role}.zip"
                    with outer.open(member) as source, target.open("xb") as stream:
                        shutil.copyfileobj(source, stream, 1024 * 1024)
                    archives[role] = ZipFile(target)
                    checked_members(archives[role], LIMITS)
                if set(archives) != {"classification", "detection", "segmentation"}:
                    raise ValueError("DamSegment requires its three inspected nested archives")
                yield archives
            finally:
                for archive in archives.values():
                    archive.close()


def extract_payload(path: Path, dataset: str, destination: Path) -> None:
    """Extract validated regular entries with exclusive writes and CRC checks through EOF."""
    with dataset_archives(path, dataset) as archives:
        total = 0
        for role, archive in archives.items():
            for member in checked_members(archive, LIMITS):
                if not useful(member.filename):
                    continue
                total += member.file_size
                if total > LIMITS.max_total_bytes:
                    raise ValueError("Combined nested extraction exceeds byte budget")
                target = destination / role / member.filename
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(member) as source, target.open("xb") as output:
                        shutil.copyfileobj(source, output, 1024 * 1024)
