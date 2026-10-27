"""Discover source files and pair by unique, case-sensitive filename stem."""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


@dataclass(frozen=True)
class Pair:
    """One unambiguous source image and JSON polygon annotation."""

    sample_id: str
    image: Path
    annotation: Path


def discover_pairs(root: Path) -> tuple[list[Pair], list[dict]]:
    """Report unsupported, orphaned, and ambiguous files without guessing matches.

    Wrapper folders are allowed. Repeated stems across folders are ambiguous,
    even if one pairing might appear plausible. No suffix stripping is applied.
    """
    groups: dict[str, dict[str, list[Path]]] = {}
    issues: list[dict] = []
    for role, suffixes in (("images", IMAGE_SUFFIXES), ("annotations", {".json"})):
        folder = root / role
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing extracted directory: {folder}")
        index: dict[str, list[Path]] = defaultdict(list)
        for path in sorted(folder.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"Symlinks are not accepted in extracted data: {path}")
            if not path.is_file():
                continue
            relative = path.relative_to(root).as_posix()
            if path.name == ".DS_Store" or "__MACOSX" in path.parts:
                issues.append({"level": "warning", "sample_id": "", "code": "metadata_file",
                               "detail": relative})
            elif path.suffix.lower() not in suffixes:
                issues.append({"level": "error", "sample_id": path.stem,
                               "code": "unsupported_file", "detail": relative})
            else:
                index[path.stem].append(path)
        groups[role] = index
    pairs = []
    for key in sorted(groups["images"].keys() | groups["annotations"].keys()):
        images = groups["images"].get(key, [])
        annotations = groups["annotations"].get(key, [])
        if len(images) == len(annotations) == 1:
            pairs.append(Pair(key, images[0], annotations[0]))
        else:
            code = "ambiguous_stem" if max(len(images), len(annotations)) > 1 else "missing_pair"
            issues.append({"level": "error", "sample_id": key, "code": code,
                           "detail": f"images={len(images)}, annotations={len(annotations)}"})
    return pairs, issues
