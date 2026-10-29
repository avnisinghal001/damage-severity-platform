"""Audit original archives against extracted bytes and identify reproducible releases."""

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
from zipfile import ZipFile

from damage_analytics.archives import ARCHIVES, checked_members
from damage_analytics.curation.config import CurationConfig
from damage_analytics.io import sha256_file


def content_hash(value: object) -> str:
    """Hash canonical JSON; reject nonfinite configuration/report values."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def audit_inputs(config: CurationConfig) -> tuple[list[dict], dict]:
    """Check archive provenance, CRCs, member bytes, missing files, and unexpected files.

    Integrity failures abort publication: repairing an extracted copy silently would
    hide which source was curated. Raw archives are only read, never rewritten.
    """
    root = config.paths.extracted_dir
    recorded = json.loads((root / "provenance.json").read_text())
    inventory, archives, failures = [], {}, []
    for role, filename in ARCHIVES.items():
        path = config.paths.raw_dir / filename
        digest = sha256_file(path)
        archives[role] = {"archive": filename, "sha256": digest}
        if digest != recorded[role]["sha256"]:
            failures.append(f"{filename}: hash differs from extraction provenance")
        expected = set()
        with ZipFile(path) as archive:
            for member in checked_members(archive, config.paths.limits):
                if member.is_dir():
                    continue
                relative = f"{role}/{member.filename}"
                expected.add(relative)
                target = root / relative
                with archive.open(member) as stream:
                    archived_hash = hashlib.file_digest(stream, "sha256").hexdigest()
                actual_hash = sha256_file(target) if target.is_file() else ""
                if target.is_symlink() or actual_hash != archived_hash:
                    failures.append(f"{relative}: missing, symlinked, or different from archive")
                inventory.append(
                    {"path": relative, "sha256": actual_hash, "bytes": member.file_size}
                )
        actual = {
            path.relative_to(root).as_posix()
            for path in (root / role).rglob("*")
            if path.is_file() and path.name != ".DS_Store" and "__MACOSX" not in path.parts
        }
        failures.extend(f"Unexpected extracted file: {path}" for path in sorted(actual - expected))
    if failures:
        raise ValueError("Source integrity audit failed: " + "; ".join(failures))
    return sorted(inventory, key=lambda row: row["path"]), {
        "archives": archives,
        "extracted_files_verified": len(inventory),
        "archive_and_extracted_integrity": "passed",
        "failures": failures,
    }


def task01_evidence(config: CurationConfig) -> dict:
    """Fingerprint every available top-level Task 01 report used as audit evidence."""
    reports = {
        path.name: sha256_file(path)
        for path in sorted(config.task01_dir.iterdir())
        if path.suffix in (".json", ".csv") and path.is_file()
    }
    if not {"issues.csv", "validation.json", "manifest.csv"}.issubset(reports):
        raise ValueError("Task 01 reports are missing; run Task 01 analyze before Task 02")
    return reports


def release_identity(
    config: CurationConfig, inventory: list[dict], provenance: dict, task01_hashes: dict
) -> tuple[str, dict]:
    """Include scientific config, all source bytes, code, environment, and audit evidence."""
    package_root = Path(__file__).resolve().parents[1]
    code = {
        path.relative_to(package_root).as_posix(): sha256_file(path)
        for path in sorted(package_root.rglob("*.py"))
    }
    environment = {
        "python": platform.python_version(),
        **{name: importlib.metadata.version(name) for name in ("Pillow", "numpy", "matplotlib")},
    }
    evidence_hash = sha256_file(config.source_groups_file) if config.source_groups_file else None
    snapshot = {
        "settings": config.snapshot(),
        "expected_pairs": config.paths.expected_pairs,
        "source_inventory_sha256": content_hash(inventory),
        "provenance": provenance,
        "task01_reports": task01_hashes,
        "code": code,
        "environment": environment,
        "source_groups_sha256": evidence_hash,
    }
    version = f"{config.dataset_name}-{content_hash(snapshot)[:16]}"
    return version, snapshot
