"""Decode supported containers into a validated, seekable frame source.

Pillow 12 already implements the frame composition rules for the animated
containers we support:

* GIF: disposal methods 0..3, per-frame transparency and sub-frame extents;
* APNG: dispose_op (none/background/previous) and blend_op (over/source);
* animated WebP: dispose + alpha blending are applied by libwebp.

After ``seek(i); load()`` every frame is therefore the *fully composited*
canvas image.  This module adds what the decoders do not give us uniformly:

1. early container identification and a hard reject on truncation / decode
   errors (frames are pre-walked, so a corrupt last frame is never silently
   dropped and no partial animation is emitted);
2. uniform frame metadata (duration, loop count, canvas size);
3. boundary checks against configured frame / pixel / duration limits;
4. a single EXIF orientation value applied to every frame.
"""

from __future__ import annotations

import io
import struct
from dataclasses import dataclass

from PIL import Image, UnidentifiedImageError

from ..config import Settings
from ..exceptions import InvalidInputError, LimitExceededError, UnsupportedFormatError
from ..models import Format
from .orientation import exif_orientation
from .validation import sniff_signature

# Exceptions raised by Pillow / libc when a container is truncated or its
# bytes are otherwise inconsistent.  struct.error leaks from the GIF parser.
_DECODE_ERRORS: tuple[type[Exception], ...] = (
    UnidentifiedImageError,
    SyntaxError,
    OSError,
    ValueError,
    struct.error,
    IndexError,
)


@dataclass
class FrameMeta:
    index: int
    duration_ms: int
    width: int
    height: int


@dataclass
class DecodedImage:
    image: Image.Image
    fmt: Format
    frame_count: int
    loop: int
    canvas_size: tuple[int, int]
    orientation: int
    frames_meta: tuple[FrameMeta, ...]


def sniff_format(data: bytes) -> Format:
    """Identify the container format from magic bytes or raise early."""
    return sniff_signature(data)


def _normalize_loop(raw) -> int:
    if raw is None:
        return 0
    try:
        loop = int(raw)
    except (TypeError, ValueError):
        return 0
    return max(loop, 0)


def decode(data: bytes, fmt: Format, settings: Settings) -> DecodedImage:
    """Pre-walk and validate the whole image, returning a seekable source."""
    limits = settings.limits
    # Align Pillow's own decompression-bomb guard with the configured pixel
    # bound, so the guard is deterministic and governed by our settings.
    Image.MAX_IMAGE_PIXELS = limits.max_pixels_per_frame
    try:
        im = Image.open(io.BytesIO(data))
        im.load()
    except Image.DecompressionBombError as exc:
        raise LimitExceededError(
            f"decoder refused {fmt.value} image as a decompression bomb: {exc}"
        ) from exc
    except _DECODE_ERRORS as exc:
        raise InvalidInputError(f"cannot decode {fmt.value} image: {exc}") from exc

    if fmt is Format.WEBP and im.format not in ("WEBP", None):
        raise InvalidInputError("signature claims WEBP but decoder disagrees")
    if fmt is Format.GIF and im.format != "GIF":
        raise InvalidInputError("signature claims GIF but decoder disagrees")
    if fmt is Format.PNG and im.format != "PNG":
        raise InvalidInputError("signature claims PNG but decoder disagrees")
    if fmt is Format.JPEG and im.format != "JPEG":
        raise InvalidInputError("signature claims JPEG but decoder disagrees")

    if fmt not in settings.supported_input_formats:
        raise UnsupportedFormatError(f"input format {fmt.value} is not enabled")

    orientation = exif_orientation(im)
    try:
        n_frames = getattr(im, "n_frames", 1)
    except _DECODE_ERRORS as exc:
        raise InvalidInputError(f"cannot enumerate frames: {exc}") from exc
    if n_frames < 1:
        raise InvalidInputError("image contains no frames")
    if n_frames > limits.max_frames:
        raise LimitExceededError(
            f"frame count {n_frames} exceeds max_frames={limits.max_frames}"
        )

    canvas_w, canvas_h = im.size
    if canvas_w <= 0 or canvas_h <= 0:
        raise InvalidInputError(f"illegal canvas size {canvas_w}x{canvas_h}")
    if max(canvas_w, canvas_h) > limits.max_dimension:
        raise LimitExceededError(
            f"canvas {canvas_w}x{canvas_h} exceeds max_dimension="
            f"{limits.max_dimension}"
        )
    if canvas_w * canvas_h > limits.max_pixels_per_frame:
        raise LimitExceededError(
            f"canvas pixel count {canvas_w * canvas_h} exceeds "
            f"max_pixels_per_frame={limits.max_pixels_per_frame}"
        )

    loop = _normalize_loop(im.info.get("loop"))
    metas: list[FrameMeta] = []
    total_duration = 0
    for i in range(n_frames):
        try:
            im.seek(i)
            im.load()
        except Image.DecompressionBombError as exc:
            raise LimitExceededError(
                f"frame {i} exceeds the decoder pixel safety limit: {exc}"
            ) from exc
        except _DECODE_ERRORS as exc:
            raise InvalidInputError(
                f"frame {i} is corrupt or truncated: {exc}"
            ) from exc
        w, h = im.size
        if (w, h) != (canvas_w, canvas_h):
            # Decoders should present the full canvas; guard against a plugin
            # that only exposes a sub-rectangle.
            raise InvalidInputError(
                f"frame {i} size {w}x{h} differs from canvas "
                f"{canvas_w}x{canvas_h}"
            )
        raw_dur = im.info.get("duration")
        if raw_dur is None:
            # Static images have no timing; animated frames without an
            # explicit duration get the pipeline default.
            duration = (
                settings.encoding.default_duration_ms
                if n_frames > 1
                else 0
            )
        else:
            try:
                duration = int(round(float(raw_dur)))
            except (TypeError, ValueError) as exc:
                raise InvalidInputError(
                    f"frame {i} has illegal duration {raw_dur!r}"
                ) from exc
        if duration < 0 or duration > limits.max_frame_duration_ms:
            raise InvalidInputError(
                f"frame {i} duration {duration}ms outside allowed range "
                f"[0, {limits.max_frame_duration_ms}]"
            )
        # A zero-duration frame in an animation is illegal (would change
        # playback timing); a single static frame simply carries no timing.
        if n_frames > 1 and duration < limits.min_frame_duration_ms:
            raise InvalidInputError(
                f"animated frame {i} duration {duration}ms below minimum "
                f"{limits.min_frame_duration_ms}ms"
            )
        total_duration += duration
        if total_duration > limits.max_total_duration_ms:
            raise LimitExceededError(
                f"total duration {total_duration}ms exceeds "
                f"max_total_duration_ms={limits.max_total_duration_ms}"
            )
        metas.append(FrameMeta(i, duration, w, h))

    total_pixels = n_frames * canvas_w * canvas_h
    if total_pixels > limits.max_total_pixels:
        raise LimitExceededError(
            f"total pixel count {total_pixels} exceeds "
            f"max_total_pixels={limits.max_total_pixels}"
        )

    return DecodedImage(
        image=im,
        fmt=fmt,
        frame_count=n_frames,
        loop=loop,
        canvas_size=(canvas_w, canvas_h),
        orientation=orientation,
        frames_meta=tuple(metas),
    )
