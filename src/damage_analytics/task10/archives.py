"""Strict six-file NumPy ZIP schema; bounded streaming extraction and safe tensor headers."""

import re
import shutil
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from numpy.lib import format as npformat

from damage_analytics.archives import checked_members
from damage_analytics.config import ArchiveLimits
from damage_analytics.io import sha256_file, write_json
from damage_analytics.task10.config import SOURCE_CODES, SOURCE_LABELS, Config

FILES = (
    "task7_X_train.npy",
    "task7_y_train.npy",
    "task7_X_test.npy",
    "task7_y_test.npy",
    "README.txt",
    "license.txt",
)
LIMITS = ArchiveLimits(
    max_files=30,
    max_total_bytes=4_000_000_000,
    max_member_bytes=3_000_000_000,
    max_compression_ratio=2000,
)


def checked_header(stream, file_size: int) -> dict:
    version = npformat.read_magic(stream)
    if version == (1, 0):
        shape, fortran, dtype = npformat.read_array_header_1_0(stream)
    elif version == (2, 0):
        shape, fortran, dtype = npformat.read_array_header_2_0(stream)
    else:
        raise ValueError("Only inspected NPY versions 1/2 are supported")
    if (
        dtype.hasobject
        or dtype != np.dtype("float32")
        or fortran
        or not shape
        or any(type(x) is not int or x < 1 for x in shape)
    ):
        raise ValueError("Expected non-object, C-order float32 tensors with positive dimensions")
    expected = stream.tell() + int(np.prod(shape, dtype=object)) * dtype.itemsize
    if expected != file_size:
        raise ValueError("Tensor header/body byte length differs; possibly truncated/corrupt")
    return {"shape": list(shape), "dtype": dtype.str, "fortran_order": fortran, "bytes": file_size}


def inspect(config: Config) -> dict:
    with ZipFile(config.archive) as archive:
        members = [m for m in checked_members(archive, LIMITS) if not m.is_dir()]
        if {m.filename for m in members} != {f"task7/{name}" for name in FILES}:
            raise ValueError("Archive differs from the inspected six-file task7 schema")
        readme = archive.read("task7/README.txt").decode("utf-8")
        found = re.findall(r"(?m)^\s*([0-3]):\s*(.*?)\s*\((\w+)\)\s*$", readme)
        expected = [
            (str(i), label, code)
            for i, (label, code) in enumerate(zip(SOURCE_LABELS, SOURCE_CODES, strict=True))
        ]
        if found != expected:
            raise ValueError("README label registration differs; do not infer/reorder class IDs")
        headers = {}
        for member in members:
            if member.filename.endswith(".npy"):
                with archive.open(member) as stream:
                    headers[Path(member.filename).name] = checked_header(stream, member.file_size)
        for split in ("train", "test"):
            x, y = (
                headers[f"task7_X_{split}.npy"]["shape"],
                headers[f"task7_y_{split}.npy"]["shape"],
            )
            if x[1:] != list(config.expected_image_shape) or y != [x[0], 4]:
                raise ValueError("Image/one-hot label shapes or row alignment differ")
        return {
            "archive_sha256": sha256_file(config.archive),
            "archive_bytes": config.archive.stat().st_size,
            "headers": headers,
            "readme": readme,
            "class_mapping": [
                {
                    "class_id": i,
                    "label": label,
                    "source_label": SOURCE_LABELS[i],
                    "source_code": SOURCE_CODES[i],
                }
                for i, label in enumerate(("Heavy", "Minor", "Moderate", "Undamaged"))
            ],
            "provided_source_id_files": [],
            "official_validation_present": False,
            "published_task7_sample_count_reference": None,
            "count_reference_note": "Local README has no counts; no per-task count was "
            "verified on official pages. "
            "The published 36,413 is across multiple tasks, not this archive.",
            "label_alignment_scope": "X[i] and one-hot y[i] share their official "
            "partition/index; no original filenames supplied",
        }


def extract(config: Config, destination: Path, inspection: dict) -> dict:
    """All entries are consumed through EOF/CRC and copied without changing their bytes."""
    destination.mkdir(parents=True)
    hashes = {}
    with ZipFile(config.archive) as archive:
        for member in checked_members(archive, LIMITS):
            if member.is_dir():
                continue
            path = destination / member.filename
            path.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, path.open("xb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
            hashes[member.filename] = sha256_file(path)
    if sha256_file(config.archive) != inspection["archive_sha256"]:
        raise ValueError("Archive changed during extraction")
    write_json(destination.parent / "archive_inspection.json", inspection)
    return hashes
