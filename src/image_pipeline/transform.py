"""Deterministic per-frame resizing.

A single resampling filter (Lanczos) is used everywhere, so the same
content + same target size always produces identical pixels - regardless
of call order, concurrency or process restart. Transparent edges are
resized in RGBA (no flattening against a background), hence no black
fringes.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from .errors import DimensionTooLargeError, InvalidRequestError
from .model import ResizeMode

# One filter for the whole pipeline; never picked from an implicit default.
FILTER = Image.Resampling.LANCZOS


def compute_target_size(
    src_w: int,
    src_h: int,
    req_w: int,
    req_h: int,
    mode: ResizeMode | str,
    max_dimension: int = 1 << 30,
) -> tuple[int, int]:
    """Compute final (width, height) for the requested resize mode."""
    mode = ResizeMode(mode) if not isinstance(mode, ResizeMode) else mode
    if src_w <= 0 or src_h <= 0:
        raise InvalidRequestError("source has zero dimension")
    if req_w == 0 and req_h == 0:
        return _clamp(src_w, src_h, max_dimension)
    if mode is ResizeMode.EXACT:
        return _clamp(req_w or src_w, req_h or src_h, max_dimension)
    if not (req_w and req_h):
        if mode is ResizeMode.COVER:
            raise InvalidRequestError("cover mode needs both width and height")
        # FIT with a single dimension: derive the other from aspect ratio
        if req_w:
            scale = req_w / src_w
            return _clamp(req_w, max(1, round(src_h * scale)), max_dimension)
        scale = req_h / src_h
        return _clamp(max(1, round(src_w * scale)), req_h, max_dimension)
    if mode is ResizeMode.COVER:
        # Output is exactly the requested box; resize_frames center-crops
        # before scaling, so the target size is the request itself.
        return _clamp(req_w, req_h, max_dimension)
    # FIT inside the box, preserving aspect ratio
    scale = min(req_w / src_w, req_h / src_h)
    tw = max(1, round(src_w * scale))
    th = max(1, round(src_h * scale))
    return _clamp(tw, th, max_dimension)


def _clamp(w: int, h: int, max_dimension: int) -> tuple[int, int]:
    if w > max_dimension or h > max_dimension:
        raise DimensionTooLargeError(f"target {w}x{h} exceeds {max_dimension}")
    return int(w), int(h)


def resize_frames(
    frames: list[np.ndarray],
    src_size: tuple[int, int],
    dst_size: tuple[int, int],
    mode: ResizeMode | str = ResizeMode.FIT,
    crop_box: tuple[int, int] | None = None,
) -> list[np.ndarray]:
    """Resize every frame identically in RGBA.

    For COVER, frames are first center-cropped to the requested aspect
    ratio (crop_box = final w/h), then scaled to exactly dst_size.
    """
    sw, sh = src_size
    dw, dh = dst_size
    mode = ResizeMode(mode) if not isinstance(mode, ResizeMode) else mode
    out: list[np.ndarray] = []
    for arr in frames:
        img = Image.fromarray(arr, "RGBA")
        if crop_box is not None:
            cw, ch = _cover_crop_box(sw, sh, crop_box[0], crop_box[1])
            left = (sw - cw) // 2
            top = (sh - ch) // 2
            img = img.crop((left, top, left + cw, top + ch))
        if img.size != (dw, dh):
            img = img.resize((dw, dh), FILTER)
        out.append(np.asarray(img, dtype=np.uint8))
    return out


def _cover_crop_box(sw: int, sh: int, rw: int, rh: int) -> tuple[int, int]:
    target_ratio = rw / rh
    if sw / sh > target_ratio:
        return max(1, int(round(sh * target_ratio))), sh
    return sw, max(1, int(round(sw / target_ratio)))
