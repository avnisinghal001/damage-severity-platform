"""Bounded content validation and explicitly recorded image conversion."""

import io
from dataclasses import dataclass

from PIL import Image, ImageOps, UnidentifiedImageError

from damage_analytics.api.config import Settings
from damage_analytics.api.errors import ApiError


@dataclass(frozen=True)
class DecodedUpload:
    """The RGB, EXIF-oriented model input and original encoding information."""

    image: Image.Image
    media_type: str
    source_mode: str


def decode_upload(data: bytes, media_type: str | None, settings: Settings) -> DecodedUpload:
    """Inspect headers before allocation; verify and fully decode only JPEG/PNG images."""
    if not data:
        raise ApiError(422, "empty_image", "The uploaded image is empty")
    if len(data) > settings.max_upload_bytes:
        raise ApiError(413, "upload_too_large", "The uploaded file exceeds the configured limit")
    try:
        with Image.open(io.BytesIO(data)) as source:
            actual = {"JPEG": "image/jpeg", "PNG": "image/png"}.get(source.format)
            if actual is None:
                raise ApiError(415, "unsupported_format", "Only JPEG and PNG content is supported")
            if media_type not in (actual, "application/octet-stream", None):
                raise ApiError(
                    415, "content_type_mismatch", "Content type does not match the image"
                )
            width, height = source.size
            if width * height > settings.max_pixels or max(width, height) > settings.max_dimension:
                raise ApiError(
                    413, "image_too_large", "Image dimensions exceed the configured limit"
                )
            if getattr(source, "n_frames", 1) != 1:
                raise ApiError(422, "animated_image", "Provide a single-frame image")
            mode = source.mode
            if mode not in ("RGB", "L"):
                raise ApiError(422, "unsupported_color_mode", "Provide an RGB or grayscale image")
            source.verify()
        with Image.open(io.BytesIO(data)) as source:
            source.load()
            image = ImageOps.exif_transpose(source).convert("RGB")
            # Stored derivatives contain pixels only, not embedded source EXIF/GPS metadata.
            clean = Image.frombytes("RGB", image.size, image.tobytes())
        return DecodedUpload(clean, actual, mode)
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise ApiError(
            422, "invalid_image", "The image is malformed or cannot be decoded"
        ) from exc
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ApiError(
            413, "image_too_large", "Image dimensions exceed safe decoding limits"
        ) from exc
