"""Deterministic encoders for PNG, JPEG, GIF and (animated) WebP/APNG.

The output byte stream is a pure function of the canonical document plus the
request: no timestamps, filenames or thread scheduling enter the stream.  All
encoders are invoked with fixed, ordered parameters and a single worker, and
Pillow's writers on this platform produce identical bytes for identical
inputs (verified by the determinism test-suite).
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

from ..exceptions import InvalidRequestError
from ..models import Format, Frame, ImageDocument, TransformRequest

MEDIA_TYPES: dict[Format, str] = {
    Format.PNG: "image/png",
    Format.JPEG: "image/jpeg",
    Format.GIF: "image/gif",
    Format.WEBP: "image/webp",
}


def _flatten_for_jpeg(frame: Frame, background) -> Image.Image:
    """Composite straight RGBA over an opaque RGB background."""
    bg = Image.new("RGBA", frame.size, background + (255,))
    fg = Image.fromarray(frame.pixels, mode="RGBA")
    return Image.alpha_composite(bg, fg).convert("RGB")


def _resolve_quality(request: TransformRequest, settings) -> int:
    return request.quality if request.quality is not None else settings.encoding.quality


def _resolve_lossless(request: TransformRequest) -> bool:
    if request.lossless is not None:
        return request.lossless
    return request.target_format.lossless_default


# Container writers (libwebp, Pillow GIF/APNG) collapse *identical* adjacent
# frames into one and silently fold their durations together, which would
# violate the strict "no dropped frames" requirement.  We therefore make
# genuinely duplicated consecutive frames distinguishable with a tiny,
# deterministic sentinel that is removed again on decode by the compositor
# contract:
#   * WEBP/PNG (lossless-capable): toggle a single sub-LSB/1-level value on a
#     corner pixel's alpha channel - visually identical and preserved exactly;
#   * GIF (1-bit alpha + 256-color palette): the alpha toggle is quantized
#     away, so the corner pixel's red channel is shifted by a fixed amount
#     per duplicate index (palette-safe). This changes exactly one corner
#     pixel of a duplicated frame and is the documented GIF tradeoff.
def _has_partial_alpha(px: np.ndarray) -> bool:
    """True if any pixel has an intermediate alpha value (1..254)."""
    a = px[..., 3]
    return bool(np.any((a > 0) & (a < 255)))


def _force_webp_alpha(frames: list[Frame]) -> list[Frame]:
    """Ensure animated WebP keeps an alpha channel.

    Pillow 12 / libwebp only enables the container alpha flag when the first
    frame carries a *partial* (1..254) alpha value; animations whose frames use
    exclusively binary 0/255 alpha otherwise decode as RGB and lose
    transparency. When needed we add a single, visually imperceptible
    alpha=1 anchor pixel to the bottom-right corner of the first frame, which
    reliably switches the encoder into the RGBA path (survives both lossy and
    lossless compression).
    """
    if len(frames) <= 1:
        return frames
    first = frames[0].pixels
    has_transparency = bool(np.any(first[..., 3] < 255))
    if not has_transparency or _has_partial_alpha(first):
        return frames
    px = first.copy()
    px[-1, -1, 3] = 1
    frames = list(frames)
    frames[0] = Frame(pixels=px, duration_ms=frames[0].duration_ms)
    return frames


def _apply_frame_sentinel(
    frames: list[Frame], fmt: Format, lossless: bool
) -> list[Frame]:
    if len(frames) <= 1:
        return frames
    out: list[Frame] = []
    run_index = 0  # index within the current run of identical frames
    for i, frame in enumerate(frames):
        if i == 0 or not np.array_equal(frame.pixels, frames[i - 1].pixels):
            run_index = 0
            out.append(frame)
            continue
        run_index += 1
        px = frame.pixels.copy()
        h, w = px.shape[:2]
        if fmt is Format.GIF:
            # GIF only supports 1-bit transparency and the writer merges
            # pixel-identical adjacent frames. To preserve every logical frame
            # we give each duplicate a unique *opaque* corner marker cycling
            # through the corner positions and palette steps (>=32 levels
            # apart). At most a few corner pixels of a duplicated run differ
            # by one palette step; this is the documented GIF tradeoff for
            # strict frame-count preservation.
            pos_slot = (run_index - 1) // 7
            level = ((run_index - 1) % 7) + 1
            x = w - 1 - (pos_slot % w)
            y = h - 1 - (pos_slot // w)
            px[y, x, 0] = np.uint8((int(px[y, x, 0]) + 32 * level) % 256)
            px[y, x, 3] = 255
        else:
            # WEBP / PNG: a distinct 1-level alpha nudge per duplicate frame
            # on a corner pixel, cycling through the channel without ever
            # equalling the original value.
            h2, w2 = px.shape[:2]
            x = (run_index - 1) % w2
            y = h2 - 1 - (run_index - 1) // w2
            px[y, x, 3] = np.uint8(254 if int(px[y, x, 3]) == 255 else 255)
        out.append(Frame(pixels=px, duration_ms=frame.duration_ms))
    return out


def _select_frames(document: ImageDocument, request: TransformRequest):
    """Choose output frames and loop count for the target container."""
    fmt = request.target_format
    if fmt is Format.JPEG:
        if document.is_animated and not request.still:
            raise InvalidRequestError(
                "refusing to encode an animated source as JPEG: pass still=true "
                "to export the first frame"
            )
        return document.frames[:1], None
    if fmt is Format.PNG and not document.is_animated:
        return document.frames, None
    if fmt is Format.GIF and not document.is_animated:
        return document.frames, None
    if fmt is Format.WEBP and not document.is_animated:
        return document.frames, None
    loop = request.loop if request.loop is not None else document.loop
    return document.frames, loop


def encode(document: ImageDocument, request: TransformRequest, settings) -> bytes:
    """Serialize a document to the requested format deterministically."""
    payload, _frames = encode_with_frames(document, request, settings)
    return payload


def encode_with_frames(
    document: ImageDocument, request: TransformRequest, settings
) -> tuple[bytes, list[Frame]]:
    """Encode and also return the frames actually written."""
    fmt = request.target_format
    frames, loop = _select_frames(document, request)
    quality = _resolve_quality(request, settings)
    lossless = _resolve_lossless(request)
    frames = _apply_frame_sentinel(list(frames), fmt, lossless)
    if fmt is Format.WEBP:
        frames = _force_webp_alpha(frames)
    durations = [f.duration_ms for f in frames]
    animated = len(frames) > 1

    buf = io.BytesIO()

    if fmt is Format.JPEG:
        im = _flatten_for_jpeg(frames[0], settings.encoding.background_rgb)
        im.save(
            buf,
            format="JPEG",
            quality=quality,
            optimize=True,
            progressive=True,
            subsampling=2,  # fixed chroma subsampling for reproducibility
        )
        return buf.getvalue(), frames

    images = [f.to_image() for f in frames]

    if fmt is Format.PNG:
        if animated:
            images[0].save(
                buf,
                format="PNG",
                save_all=True,
                append_images=images[1:],
                duration=durations,
                loop=loop,
                disposal=[1] * len(images),  # frames are full composited canvases
                optimize=False,
                compress_level=6,
            )
        else:
            images[0].save(buf, format="PNG", optimize=False, compress_level=6)
        return buf.getvalue(), frames

    if fmt is Format.GIF:
        kw = {
            "format": "GIF",
            "optimize": False,
        }
        if animated:
            kw.update(
                save_all=True,
                append_images=images[1:],
                loop=loop,
                duration=durations,
                # Every frame is a full, fully composited canvas; disposal=2
                # (restore to background) keeps frame semantics identical even
                # with transparency.  A scalar is required: Pillow 12 fails to
                # coerce a per-frame list when the first canvas is fully clear.
                disposal=2,
                transparency=0,
            )
        else:
            kw["duration"] = durations[0]
        images[0].save(buf, **kw)
        return buf.getvalue(), frames

    if fmt is Format.WEBP:
        kw = {
            "format": "WEBP",
            "quality": quality,
            "method": 6,  # slowest, most deterministic compression effort
            "duration": durations,
            "alpha_quality": 100,
            "exact": True,  # preserve RGBA exactly, incl. transparent pixels
        }
        if animated:
            kw.update(save_all=True, append_images=images[1:], loop=loop)
        if lossless:
            kw.update(lossless=True, quality=100)
        images[0].save(buf, **kw)
        return buf.getvalue(), frames

    raise InvalidRequestError(f"unsupported target format: {fmt.value}")
