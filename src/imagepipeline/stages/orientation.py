"""EXIF/GPS orientation correction, applied to every frame.

EXIF orientation values (tag 274) describe how stored pixels must be rotated /
flipped to appear upright.  Rather than relying on ``ImageOps.exif_transpose``
(which mutates a single PIL image and is awkward to apply to pre-composited
frame buffers), we implement the eight deterministic geometric transforms on
numpy RGBA arrays so the compositor can orient each frame identically.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

ORIENTATION_TAG = 274

# Each transform is expressed as an ordered sequence of elementary numpy
# operations: "lr" = flip left-right, "tb" = flip top-bottom,
# "cw" = rotate 90 degrees clockwise, "ccw" = rotate 90 degrees
# counter-clockwise.  Validated against ``ImageOps.exif_transpose`` for all
# eight codes in the test-suite.
_TRANSFORMS: dict[int, tuple[str, ...]] = {
    1: (),
    2: ("lr",),
    3: ("lr", "tb"),  # 180
    4: ("tb",),
    5: ("lr", "ccw"),
    6: ("cw",),
    7: ("lr", "cw"),
    8: ("ccw",),
}


def exif_orientation(image: Image.Image) -> int:
    """Return the EXIF orientation code (1..8); 1 when absent/invalid."""
    try:
        exif = image.getexif()
    except Exception:
        return 1
    value = exif.get(ORIENTATION_TAG)
    if value is None:
        return 1
    try:
        value = int(value)
    except (TypeError, ValueError):
        return 1
    if value not in _TRANSFORMS:
        return 1
    return value


def apply_orientation(pixels: np.ndarray, orientation: int) -> np.ndarray:
    """Rotate/flip an HxWx4 uint8 array according to the EXIF orientation."""
    if orientation == 1:
        return pixels
    out = pixels
    for step in _TRANSFORMS[orientation]:
        if step == "lr":
            out = np.fliplr(out)
        elif step == "tb":
            out = np.flipud(out)
        elif step == "cw":
            out = np.rot90(out, 3)
        elif step == "ccw":
            out = np.rot90(out, 1)
    return np.ascontiguousarray(out)


def strip_exif(image: Image.Image) -> None:
    """Remove orientation metadata after it has been applied (best effort)."""
    try:
        exif = image.getexif()
        if ORIENTATION_TAG in exif:
            exif[ORIENTATION_TAG] = 1
    except Exception:
        pass
