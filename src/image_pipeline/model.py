"""Core domain model: frames, decoded images, processing requests/results."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .config import PipelineConfig
from .errors import PipelineError


class Disposal(str, Enum):
    """Frame inter-area disposal semantics (APNG/GIF compatible)."""

    UNSPECIFIED = "unspecified"
    NONE = "none"          # leave frame in place
    BACKGROUND = "background"  # clear to fully transparent
    PREVIOUS = "previous"  # restore prior canvas


class Blending(str, Enum):
    SOURCE = "source"
    OVER = "over"


@dataclass
class Frame:
    """One decoded frame in canvas coordinates.

    image: RGBA uint8 numpy array (canvas height x canvas width x 4),
           fully composited-ready tile placed at (offset_x, offset_y).
           Stored tile may be smaller than canvas; offsets define placement.
    """

    tile: np.ndarray  # uint8 HxWx4 RGBA, this frame's own pixels pre-blend
    duration_ms: int
    disposal: Disposal = Disposal.NONE
    blend: Blending = Blending.OVER
    offset_x: int = 0
    offset_y: int = 0


@dataclass
class DecodedImage:
    frames: list[Frame] = field(default_factory=list)
    loop_count: int = 0  # 0 == infinite (GIF/APNG/WebP convention)
    width: int = 0
    height: int = 0
    is_animated: bool = False
    source_format: str = ""
    background: tuple[int, int, int, int] = (0, 0, 0, 0)


class ResizeMode(str, Enum):
    EXACT = "exact"            # force width/height, may distort
    FIT = "fit"                # scale to fit inside, preserve aspect
    COVER = "cover"            # cover target box, center-crop


@dataclass(frozen=True)
class ProcessRequest:
    source: bytes = b""
    source_id: str = ""       # caller locator (e.g. filename/hash)
    output_format: str = ""   # webp/png/gif/jpeg; empty = source format
    width: int = 0            # 0 = auto from height/aspect
    height: int = 0
    resize_mode: ResizeMode = ResizeMode.FIT
    quality: int | None = None
    lossless: bool | None = None
    request_id: str = ""
    timeout_ms: int = 0       # 0 = config default


@dataclass
class ProcessResult:
    data: bytes
    output_format: str
    width: int
    height: int
    frame_count: int
    loop_count: int
    durations_ms: list[int]
    request_id: str
    source_id: str


@dataclass
class RejectedResult:
    """Explanatory, locatable failure; never carries partial bytes."""

    code: str
    message: str
    request_id: str
    source_id: str

    @classmethod
    def from_error(cls, err: PipelineError) -> "RejectedResult":
        d = err.to_dict()
        return cls(d["code"], d["message"], d["request_id"], d["source_id"])
