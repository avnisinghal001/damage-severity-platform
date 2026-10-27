"""Inspect ZIPs and extract regular files into a new, isolated directory."""

import shutil
import stat
import tempfile
import unicodedata
from pathlib import Path, PurePosixPath
from zipfile import ZipFile, ZipInfo

from damage_analytics.config import ArchiveLimits, Config
from damage_analytics.io import sha256_file, write_csv, write_json

ARCHIVES = {"images": "imagedata.zip", "annotations": "labeldata.zip"}


def checked_members(archive: ZipFile, limits: ArchiveLimits) -> list[ZipInfo]:
    """Reject traversal, links, collisions, encryption, and excessive ZIP sizes."""
    members = archive.infolist()
    if len(members) > limits.max_files:
        raise ValueError("ZIP exceeds max_files")
    total = 0
    destinations: dict[str, bool] = {}
    for member in members:
        name = member.filename
        path = PurePosixPath(name)
        if (not name or "\\" in name or ":" in name or "\x00" in name
                or path.is_absolute() or ".." in path.parts or path == PurePosixPath(".")):
            raise ValueError(f"Unsafe ZIP path: {name!r}")
        mode = member.external_attr >> 16
        file_type = stat.S_IFMT(mode)
        if file_type not in (0, stat.S_IFREG, stat.S_IFDIR):
            raise ValueError(f"Non-regular ZIP entry: {name!r}")
        if member.flag_bits & 1:
            raise ValueError(f"Encrypted ZIP entry: {name!r}")
        key = unicodedata.normalize("NFC", path.as_posix()).casefold()
        if key in destinations:
            raise ValueError(f"Duplicate/case-colliding ZIP path: {name!r}")
        destinations[key] = member.is_dir()
        total += member.file_size
        if member.file_size > limits.max_member_bytes or total > limits.max_total_bytes:
            raise ValueError("ZIP exceeds uncompressed byte limits")
        if member.file_size / max(member.compress_size, 1) > limits.max_compression_ratio:
            raise ValueError(f"ZIP exceeds compression-ratio limit: {name!r}")
    for name in destinations:
        for parent in PurePosixPath(name).parents:
            if destinations.get(parent.as_posix()) is False:
                raise ValueError(f"ZIP file/directory collision: {name!r}")
    return members


def inspect_archives(config: Config) -> dict:
    """Save a member inventory and archive hashes without extracting anything."""
    rows, summary = [], {}
    for role, filename in ARCHIVES.items():
        path = config.raw_dir / filename
        with ZipFile(path) as archive:
            members = checked_members(archive, config.limits)
            summary[role] = {
                "archive": filename, "sha256": sha256_file(path),
                "files": sum(not item.is_dir() for item in members),
                "uncompressed_bytes": sum(item.file_size for item in members),
            }
            rows.extend({"role": role, "member": item.filename,
                         "bytes": item.file_size, "compressed_bytes": item.compress_size,
                         "is_directory": item.is_dir()} for item in members)
    write_csv(config.output_dir / "archive_inventory.csv", rows,
              ["role", "member", "bytes", "compressed_bytes", "is_directory"])
    write_json(config.output_dir / "archives.json", summary)
    return summary


def extract_archives(config: Config) -> None:
    """Preflight both archives; publish only a fully extracted, CRC-checked tree.

    Existing destinations are never merged or overwritten. Temporary files stay
    beside the destination so the final rename remains on the same filesystem.
    """
    if config.extracted_dir.exists():
        raise FileExistsError(f"Extraction destination already exists: {config.extracted_dir}")
    summary = inspect_archives(config)
    config.extracted_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".rc1841-", dir=config.extracted_dir.parent) as temp:
        staging = Path(temp) / "dataset"
        staging.mkdir()
        for role, filename in ARCHIVES.items():
            archive_path = config.raw_dir / filename
            if sha256_file(archive_path) != summary[role]["sha256"]:
                raise ValueError("Archive changed after inspection")
            with ZipFile(archive_path) as archive:
                for member in checked_members(archive, config.limits):
                    target = staging / role / member.filename
                    if member.is_dir():
                        target.mkdir(parents=True, exist_ok=True)
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(member) as source, target.open("xb") as output:
                        # ZipExtFile checks CRC when consumed through EOF.
                        shutil.copyfileobj(source, output, length=1024 * 1024)
        write_json(staging / "provenance.json", summary)
        staging.rename(config.extracted_dir)
