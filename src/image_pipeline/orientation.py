"""EXIF orientation correction, applied to canvas and per-frame tiles."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageOps

_ORIENTATION_TAG = 274  # ExifIFD TIFF tag "Orientation"


def extract_orientation(img: Image.Image) -> int:
    """Return EXIF orientation 1..8; missing/invalid -> 1."""
    try:
        val = int(img.getexif().get(_ORIENTATION_TAG, 1))
    except (TypeError, ValueError):
        return 1
    return val if 1 <= val <= 8 else 1


def apply_orientation(img: Image.Image) -> Image.Image:
    """Bake EXIF orientation into pixels and strip the tag.

    Uses Pillow's exif_transpose (transpose operations only), which is
    lossless for JPEG and exactly deterministic.
    """
    out = ImageOps.exif_transpose(img)
    if out is None:  # pragma: no cover - Pillow always returns an image
        return img
    # Ensure the tag can never leak through alternate info channels.
    exif = out.getexif()
    if _ORIENTATION_TAG in exif:
        del exif[_ORIENTATION_TAG]
    return out


def _map_point(x: float, y: float, cw: int, ch: int, orient: int) -> tuple[int, int]:
    """Map (x,y) from pre-orientation canvas coords to post-orientation."""
    if orient == 2:    # flip horizontal
        return cw - 1 - x, y
    if orient == 3:    # rotate 180
        return cw - 1 - x, ch - 1 - y
    if orient == 4:    # flip vertical
        return x, ch - 1 - y
    if orient == 5:    # transpose
        return y, x
    if orient == 6:    # rotate 90 CW
        return ch - 1 - y, x
    if orient == 7:    # transverse
        return ch - 1 - y, cw - 1 - x
    if orient == 8:    # rotate 90 CCW
        return y, cw - 1 - x
    return x, y


def tile_offset_after_orientation(
    offset_x: int,
    offset_y: int,
    tile_w: int,
    tile_h: int,
    canvas_w: int,
    canvas_h: int,
    orientation: int,
) -> tuple[int, int, int, int]:
    """Map a frame tile's (offset, width, height) through orientation.

    Returns (new_offset_x, new_offset_y, new_tile_w, new_tile_h) in the
    oriented canvas coordinate system.
    """
    orient = orientation if 1 <= orientation <= 8 else 1
    # Corners of the tile box; map all four then take the bounding box.
    corners = [
        (offset_x, offset_y),
        (offset_x + tile_w - 1, offset_y),
        (offset_x, offset_y + tile_h - 1),
        (offset_x + tile_w - 1, offset_y + tile_h - 1),
    ]
    mapped = [_map_point(x, y, canvas_w, canvas_h, orient) for x, y in corners]
    xs = [p[0] for p in mapped]
    ys = [p[1] for p in mapped]
    nx0, ny0 = min(xs), min(ys)
    nx1, ny1 = max(xs), max(ys)
    return nx0, ny0, nx1 - nx0 + 1, ny1 - ny0 + 1
