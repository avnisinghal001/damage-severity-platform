"""Decode images, resolve evidenced EXIF frame mismatches, and export RGB copies."""

import hashlib
import warnings
from pathlib import Path

from PIL import Image, ImageOps

from damage_analytics.curation.cleaning import issue
from damage_analytics.curation.config import CleaningPolicy
from damage_analytics.curation.metadata import camera_metadata


def pixel_hash(image: Image.Image) -> str:
    """Fingerprint decoded RGB pixels and dimensions, ignoring file metadata."""
    rgb = image.convert("RGB")
    return hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()


def prepare_image(
    path: Path, document: dict, sample_id: str, policy: CleaningPolicy
) -> tuple[Image.Image, dict, list[dict]]:
    """Return an RGB derivative; infer annotation orientation only from unequal sizes.

    A nonidentity EXIF transform is allowed only when the JSON size equals the
    transformed size and differs from the stored size. Same-size flips/rotations
    are ambiguous and excluded. Originals and annotation coordinates are unchanged.
    """
    decisions = []
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(path) as source:
            source.verify()
        with Image.open(path) as source:
            source.load()
            stored_size, mode, format_name = source.size, source.mode, source.format
            orientation = source.getexif().get(274, 1)
            metadata = {
                "stored_width": stored_size[0],
                "stored_height": stored_size[1],
                "source_mode": mode,
                "source_format": format_name,
                "exif_orientation": orientation,
                "stored_pixel_sha256": pixel_hash(source),
                **camera_metadata(source),
            }
            if Image.registered_extensions().get(path.suffix.lower()) != format_name:
                decisions.append(
                    issue(
                        sample_id,
                        "format_extension_mismatch",
                        "exclude",
                        f"{path.suffix} contains {format_name}",
                    )
                )
            if getattr(source, "n_frames", 1) != 1:
                decisions.append(
                    issue(sample_id, "multiframe_image", "exclude", "Not a single frame")
                )
            if mode not in ("RGB", "L", "RGBA", "LA", "P", "CMYK"):
                decisions.append(issue(sample_id, "unsupported_color_mode", "exclude", mode))
            if "A" in source.getbands() or "transparency" in source.info:
                if source.convert("RGBA").getchannel("A").getextrema() != (255, 255):
                    decisions.append(
                        issue(
                            sample_id,
                            "transparent_pixels",
                            "exclude",
                            "No justified alpha compositing background",
                        )
                    )
            selected = source.copy()
            if orientation != 1:
                oriented = ImageOps.exif_transpose(source)
                declared = (
                    (document.get("imageWidth"), document.get("imageHeight"))
                    if isinstance(document, dict)
                    else (None, None)
                )
                if (
                    policy.exif_policy == "match_annotation_dimensions"
                    and oriented.size == declared
                    and declared != stored_size
                ):
                    selected = oriented
                    decisions.append(
                        issue(
                            sample_id,
                            "exif_frame_alignment",
                            "fix",
                            f"EXIF {orientation}: {stored_size} → {oriented.size}; "
                            "JSON matches oriented frame only; visual review advised",
                        )
                    )
                else:
                    decisions.append(
                        issue(
                            sample_id,
                            "ambiguous_exif_frame",
                            "exclude",
                            f"Stored={stored_size}; EXIF={orientation}; JSON={declared}",
                        )
                    )
            if mode != "RGB":
                decisions.append(
                    issue(
                        sample_id,
                        "rgb_conversion",
                        "fix" if policy.allow_rgb_conversion else "exclude",
                        f"{mode} → RGB in derivative only; no calibrated color claim",
                    )
                )
            rgb = selected.convert("RGB")
            # Do not carry EXIF orientation/ICC metadata into standardized PNGs.
            rgb.info.clear()
    metadata.update({"width": rgb.width, "height": rgb.height, "decoded_sha256": pixel_hash(rgb)})
    return rgb, metadata, decisions
