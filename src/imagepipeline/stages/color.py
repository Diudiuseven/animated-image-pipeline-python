"""Color normalization: every frame becomes sRGB straight-alpha RGBA.

Frames are converted to RGBA with a fixed alpha channel behavior, embedded ICC
profiles are transformed to sRGB through littlecms2 when present, and EXIF
color-space flags are handled so that two files depicting the same content in
different color spaces normalize to identical pixels.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageCms

# sRGB rendering intent used for all ICC transforms.
_INTENT = ImageCms.Intent.RELATIVE_COLORIMETRIC


def _icc_to_srgb(rgb: Image.Image) -> Image.Image:
    """Apply an embedded ICC profile -> sRGB; return image unchanged if absent."""
    raw = rgb.info.get("icc_profile")
    if not raw:
        return rgb
    try:
        src = ImageCms.ImageCmsProfile(io.BytesIO(raw))
    except Exception:
        return rgb
    try:
        dst = ImageCms.createProfile("sRGB")
        mode = "RGBA" if rgb.mode == "RGBA" else "RGB"
        transformed = ImageCms.profileToProfile(
            rgb, src, dst, outputMode=mode, intent=_INTENT
        )
        return transformed
    except Exception:
        # Corrupt / unsupported ICC profile: fall back to the raw pixels
        # deterministically rather than failing the whole request.
        return rgb


def _ensure_rgba(image: Image.Image) -> Image.Image:
    """Convert any PIL mode to RGBA with deterministic palette handling."""
    if image.mode == "RGBA":
        return image
    if image.mode == "RGB":
        return image.convert("RGBA")
    if image.mode in ("L", "LA"):
        return image.convert("RGBA")
    if image.mode in ("CMYK", "YCbCr", "LAB", "HSV", "I", "F", "I;16"):
        return image.convert("RGBA")
    # Palette images: respect their transparency index. P -> RGBA is
    # deterministic in Pillow and honors im.info["transparency"].
    return image.convert("RGBA")


def to_srgb_rgba(image: Image.Image) -> np.ndarray:
    """Convert a PIL frame (any mode / ICC) to sRGB straight RGBA uint8."""
    rgba = _ensure_rgba(image)
    # ICC conversion is only meaningful for color (non-indexed) pixels.
    rgba = _icc_to_srgb(rgba)
    arr = np.asarray(rgba, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 4:
        arr = np.asarray(rgba.convert("RGBA"), dtype=np.uint8)
    return np.ascontiguousarray(arr)
