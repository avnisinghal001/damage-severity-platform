"""Pillow/NumPy paired transforms, reusable without PyTorch for resize-loss audits."""

from dataclasses import dataclass

import numpy as np
from PIL import Image

from damage_analytics.training.config import IGNORE_INDEX, DataConfig


@dataclass(frozen=True)
class Frame:
    """Letterbox geometry; source_size/crop_box/resized_size use Pillow x/y conventions."""

    source_size: tuple[int, int]
    crop_box: tuple[int, int, int, int]
    resized_size: tuple[int, int]
    left: int
    top: int


def frame_for(
    size: tuple[int, int], target_hw: tuple[int, int], box: tuple[int, int, int, int] | None = None
) -> Frame:
    """Fit a full image or crop inside a target without distorting its aspect ratio."""
    box = box or (0, 0, *size)
    width, height = box[2] - box[0], box[3] - box[1]
    scale = min(target_hw[1] / width, target_hw[0] / height)
    resized = (
        max(1, min(target_hw[1], round(width * scale))),
        max(1, min(target_hw[0], round(height * scale))),
    )
    return Frame(
        size, box, resized, (target_hw[1] - resized[0]) // 2, (target_hw[0] - resized[1]) // 2
    )


def choose_crop(
    mask: Image.Image, config: DataConfig, rng: np.random.Generator
) -> tuple[int, ...]:
    """Sample a source-pixel crop; foreground guidance uses this training mask only.

    Uniformly select a present foreground category, then a pixel of that category.
    Random crop placement includes the chosen pixel; other categories may be cropped
    out intentionally. This is a sampling bias, not restored or invented ground truth.
    """
    width, height = mask.size
    crop_w, crop_h = min(width, config.crop_size[1]), min(height, config.crop_size[0])
    values = np.asarray(mask)
    labels = np.unique(values)
    labels = labels[(labels > 0) & (labels < 9)]
    if len(labels) and rng.random() < config.foreground_crop_probability:
        label = rng.choice(labels)
        candidates = np.flatnonzero(values == label)
        pixel = int(rng.choice(candidates))
        y, x = divmod(pixel, width)
        left = int(rng.integers(max(0, x - crop_w + 1), min(x, width - crop_w) + 1))
        top = int(rng.integers(max(0, y - crop_h + 1), min(y, height - crop_h) + 1))
    else:
        left = int(rng.integers(0, width - crop_w + 1))
        top = int(rng.integers(0, height - crop_h + 1))
    return left, top, left + crop_w, top + crop_h


def transform_mask(mask: Image.Image, frame: Frame, target_hw: tuple[int, int]) -> Image.Image:
    """Nearest-neighbor ID resize, with ignored padding rather than fake background."""
    resized = mask.crop(frame.crop_box).resize(frame.resized_size, Image.Resampling.NEAREST)
    padded = Image.new("L", (target_hw[1], target_hw[0]), IGNORE_INDEX)
    padded.paste(resized, (frame.left, frame.top))
    return padded


def restore_prediction(prediction: np.ndarray, frame: Frame) -> np.ndarray:
    """Unpad a class-ID map and restore native resolution using nearest-neighbor.

    Only full-image frames are valid for validation/inference; crop restoration would
    otherwise invent predictions outside the observed region. Argmax precedes this
    resize, keeping memory bounded for 12-megapixel images and all outputs in 0..8.
    """
    if frame.crop_box != (0, 0, *frame.source_size):
        raise ValueError("Cannot restore a crop as a full-image prediction")
    width, height = frame.resized_size
    values = prediction[frame.top : frame.top + height, frame.left : frame.left + width]
    if values.shape != (height, width) or np.any((values < 0) | (values > 8)):
        raise ValueError("Invalid prediction shape/class IDs")
    return np.asarray(
        Image.fromarray(values.astype(np.uint8)).resize(
            frame.source_size, Image.Resampling.NEAREST
        )
    )


def prepare_pair(
    image: Image.Image,
    mask: Image.Image,
    config: DataConfig,
    *,
    training: bool,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, Frame]:
    """Apply one crop/resize/flip geometry to RGB and mask, then normalize RGB only."""
    if image.mode != "RGB" or mask.mode != "L" or image.size != mask.size:
        raise ValueError("Expected aligned RGB image and L class-ID mask")
    if np.asarray(mask).max() > 8:
        raise ValueError("Source mask contains IDs outside 0..8")
    box = (
        choose_crop(mask, config, rng)
        if training and rng.random() < config.crop_probability
        else None
    )
    frame = frame_for(image.size, config.image_size, box)
    padded = Image.new("RGB", (config.image_size[1], config.image_size[0]))
    padded.paste(
        image.crop(frame.crop_box).resize(frame.resized_size, Image.Resampling.BILINEAR),
        (frame.left, frame.top),
    )
    target = transform_mask(mask, frame, config.image_size)
    if training:
        for probability, operation in (
            (config.horizontal_flip, Image.Transpose.FLIP_LEFT_RIGHT),
            (config.vertical_flip, Image.Transpose.FLIP_TOP_BOTTOM),
        ):
            if rng.random() < probability:
                padded, target = padded.transpose(operation), target.transpose(operation)
    labels = np.array(target, dtype=np.int64)
    rgb = np.asarray(padded, dtype=np.float32) / 255
    rgb = (rgb - np.array(config.mean, dtype=np.float32)) / np.array(config.std, dtype=np.float32)
    rgb[labels == IGNORE_INDEX] = 0
    return np.ascontiguousarray(rgb.transpose(2, 0, 1)), labels, frame
