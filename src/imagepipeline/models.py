"""Core immutable data structures shared by every pipeline stage.

The pipeline converts any supported input into a canonical in-memory
:class:`ImageDocument` (a list of straight, sRGB, premultiplied RGBA frames
with explicit per-frame durations, loop count and canvas size), transforms it
through pure functions and finally encodes it to the requested output format.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

import numpy as np
from PIL import Image


class Format(str, enum.Enum):
    PNG = "png"
    JPEG = "jpeg"
    GIF = "gif"
    WEBP = "webp"

    @property
    def supports_animation(self) -> bool:
        return self in (Format.GIF, Format.WEBP, Format.PNG)

    @property
    def supports_alpha(self) -> bool:
        return self in (Format.PNG, Format.GIF, Format.WEBP)

    @property
    def lossless_default(self) -> bool:
        return self in (Format.PNG, Format.GIF)


class ResizeMode(str, enum.Enum):
    """How requested dimensions are interpreted."""

    NONE = "none"  # keep source size
    FIT = "fit"  # scale to fit inside WxH, preserve aspect ratio
    COVER = "cover"  # scale to cover WxH and center-crop
    STRETCH = "stretch"  # scale exactly to WxH, ignoring aspect ratio


# Canonical frame disposal / blend semantics are already baked into the
# composited canvas frames, so no disposal enum is needed downstream.
# Loop count convention: 0 means loop forever; positive N means N plays.


@dataclass(frozen=True)
class Frame:
    """One fully composited animation frame.

    ``pixels`` is an HxWx4 uint8 array in straight (unassociated) RGBA after
    decode; the resize stage works on a premultiplied copy and un-premultiplies
    afterwards so transparent edges never pick up black fringes.
    """

    pixels: np.ndarray
    duration_ms: int

    def __post_init__(self) -> None:
        if not isinstance(self.pixels, np.ndarray):
            raise TypeError("pixels must be a numpy.ndarray")
        if self.pixels.dtype != np.uint8:
            raise TypeError("pixels must be uint8")
        if self.pixels.ndim != 3 or self.pixels.shape[2] != 4:
            raise ValueError("pixels must have shape (H, W, 4)")
        if int(self.duration_ms) < 0:
            raise ValueError("duration_ms must be non-negative")

    @property
    def size(self) -> tuple[int, int]:
        """Return ``(width, height)``."""
        h, w = self.pixels.shape[:2]
        return w, h

    def to_image(self) -> Image.Image:
        return Image.fromarray(self.pixels, mode="RGBA")


@dataclass(frozen=True)
class ImageDocument:
    """Canonical decoded image: straight RGBA frames, already oriented."""

    frames: tuple[Frame, ...]
    loop: int
    source_format: Format | None = None

    def __post_init__(self) -> None:
        if not self.frames:
            raise ValueError("document must contain at least one frame")
        w, h = self.frames[0].size
        for f in self.frames:
            if f.size != (w, h):
                raise ValueError("all frames must share the canvas size")
        if int(self.loop) < 0:
            raise ValueError("loop must be 0 (infinite) or positive")

    @property
    def is_animated(self) -> bool:
        return len(self.frames) > 1

    @property
    def size(self) -> tuple[int, int]:
        return self.frames[0].size

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    @property
    def duration_ms(self) -> int:
        return sum(f.duration_ms for f in self.frames)


@dataclass(frozen=True)
class TransformRequest:
    """Everything a processing request may specify."""

    target_format: Format
    resize_mode: ResizeMode = ResizeMode.NONE
    width: int | None = None
    height: int | None = None
    # Encoder knobs. None means: use pipeline defaults.
    quality: int | None = None
    lossless: bool | None = None
    # Animated source -> static format policy.
    still: bool = False
    # Override loop count; None keeps the source loop count.
    loop: int | None = None

    def __post_init__(self) -> None:
        if self.width is not None and self.width <= 0:
            raise ValueError("width must be positive")
        if self.height is not None and self.height <= 0:
            raise ValueError("height must be positive")
        if self.quality is not None and not 1 <= self.quality <= 100:
            raise ValueError("quality must be within [1, 100]")
        if self.loop is not None and self.loop < 0:
            raise ValueError("loop must be 0 (infinite) or positive")


@dataclass(frozen=True)
class PipelineResult:
    """Finished, externally visible artifact."""

    data: bytes
    media_type: str
    fmt: Format
    width: int
    height: int
    frame_count: int
    durations_ms: tuple[int, ...]
    loop: int
    request_id: str


@dataclass(frozen=True)
class ProcessedFrame:
    """Intermediate frame produced by the transform stages."""

    pixels: np.ndarray
    duration_ms: int
