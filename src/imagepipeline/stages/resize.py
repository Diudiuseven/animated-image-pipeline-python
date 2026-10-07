"""Deterministic, alpha-aware resizing.

Transparent pixels in a straight-alpha RGBA buffer still carry RGB values;
resizing those directly bleeds their (often black) RGB into partially
transparent edges, producing black fringes.  We therefore:

1. premultiply color by alpha before resampling;
2. resize with a fixed, non-threaded, deterministic filter (LANCZOS);
3. un-premultiply afterwards.

All integer rounding uses the same ``round-half-even`` numpy rules on every
run, so output pixels depend only on inputs.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from ..exceptions import LimitExceededError
from ..config import Settings
from ..models import Frame, ImageDocument, ResizeMode, TransformRequest


def target_size(src_w: int, src_h: int, request: TransformRequest) -> tuple[int, int]:
    """Compute output (w, h) for the requested resize mode."""
    mode = request.resize_mode
    if mode is ResizeMode.NONE:
        return src_w, src_h

    if mode is ResizeMode.STRETCH:
        assert request.width is not None and request.height is not None
        return int(request.width), int(request.height)

    if mode is ResizeMode.COVER:
        assert request.width is not None and request.height is not None
        tw, th = int(request.width), int(request.height)
        scale = max(tw / src_w, th / src_h)
        rw = max(int(round(src_w * scale)), 1)
        rh = max(int(round(src_h * scale)), 1)
        return rw, rh  # caller center-crops to (tw, th)

    # FIT: scale to fit inside the (possibly one-sided) target box.
    assert mode is ResizeMode.FIT
    if request.width is not None and request.height is not None:
        scale = min(request.width / src_w, request.height / src_h)
    elif request.width is not None:
        scale = request.width / src_w
    else:
        assert request.height is not None
        scale = request.height / src_h
    rw = max(int(round(src_w * scale)), 1)
    rh = max(int(round(src_h * scale)), 1)
    return rw, rh


def _resize_rgba(pixels: np.ndarray, w: int, h: int) -> np.ndarray:
    """Premultiplied-alpha LANCZOS resize returning straight RGBA uint8.

    The buffer handed to the resampler stores premultiplied color in 0..255
    units (``rgb * alpha``) together with the straight alpha channel.  After
    resampling, color is un-premultiplied; fully transparent pixels keep zero
    RGB and alpha so they stay invisible regardless of fringe color.
    """
    src = pixels.astype(np.float32)
    a01 = src[..., 3:4] / 255.0
    premult_rgb = src[..., :3] * a01  # 0..255 scaled by coverage
    pm = np.concatenate([premult_rgb, src[..., 3:4]], axis=2)
    im = Image.fromarray(np.round(pm).astype(np.uint8), mode="RGBA")
    resized = im.resize((w, h), Image.Resampling.LANCZOS)
    out = np.asarray(resized, dtype=np.float32)
    alpha = out[..., 3:4]
    a01 = alpha / 255.0
    divisor = np.where(alpha >= 1.0, a01, 1.0)
    rgb = out[..., :3] / divisor
    rgb = np.clip(rgb, 0, 255)
    # Fully transparent pixels carry no visible color: pin RGB to zero so a
    # fringe can never bleed through if a later stage ignores alpha.
    rgb = np.where(alpha >= 1.0, rgb, 0.0)
    result = np.concatenate([rgb, alpha], axis=2)
    return np.round(np.clip(result, 0, 255)).astype(np.uint8)


def _center_crop(pixels: np.ndarray, w: int, h: int) -> np.ndarray:
    src_h, src_w = pixels.shape[:2]
    left = (src_w - w) // 2
    top = (src_h - h) // 2
    return np.ascontiguousarray(pixels[top : top + h, left : left + w])


def resize(
    document: ImageDocument, request: TransformRequest, settings: Settings
) -> ImageDocument:
    """Resize every frame with premultiplied alpha to avoid black fringes."""
    if request.resize_mode is ResizeMode.NONE:
        return document

    src_w, src_h = document.size
    rw, rh = target_size(src_w, src_h, request)
    out_w, out_h = rw, rh
    crop = False
    if request.resize_mode is ResizeMode.COVER:
        out_w, out_h = int(request.width), int(request.height)
        crop = True

    limit = settings.limits.max_target_dimension
    if max(out_w, out_h) > limit:
        raise LimitExceededError(
            f"computed size {out_w}x{out_h} exceeds max_target_dimension={limit}"
        )

    frames: list[Frame] = []
    for frame in document.frames:
        p = _resize_rgba(frame.pixels, rw, rh)
        if crop:
            p = _center_crop(p, out_w, out_h)
        frames.append(Frame(pixels=p, duration_ms=frame.duration_ms))
    return ImageDocument(
        frames=tuple(frames), loop=document.loop, source_format=document.source_format
    )
