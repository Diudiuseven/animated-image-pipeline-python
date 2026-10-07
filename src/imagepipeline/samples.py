"""Deterministic in-process sample images for local runs and tests.

No network or fixture files are needed: every sample is synthesized from a
fixed seed so byte-identical samples are produced on every run.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageDraw

from .exceptions import InvalidInputError
from .models import Format


def _checker_rgba(w: int, h: int, seed: int = 0) -> np.ndarray:
    """Colorful opaque checkerboard with deterministic per-cell colors."""
    px = np.zeros((h, w, 4), np.uint8)
    for y in range(h):
        for x in range(w):
            v = (x * 37 + y * 73 + seed * 131) % 256
            px[y, x] = (v, (v * 3 + 40) % 256, (255 - v), 255)
    return px


def _animated_transparent_gif(n: int = 4, w: int = 48, h: int = 48) -> bytes:
    """Animated GIF exercising sub-frames, transparency and disposal."""
    frames = []
    for i in range(n):
        im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        cx = int((i + 1) * w / (n + 1))
        d.ellipse([cx - 8, h // 2 - 8, cx + 8, h // 2 + 8], fill=(255, 0, 0, 255))
        d.rectangle([0, 0, w - 1, h - 1], outline=(0, 128, 255, 255))
        frames.append(im)
    buf = io.BytesIO()
    frames[0].save(
        buf,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=120,
        loop=0,
        disposal=2,
        optimize=False,
    )
    return buf.getvalue()


def _animated_webp(n: int = 4, w: int = 48, h: int = 48) -> bytes:
    frames = []
    for i in range(n):
        px = np.zeros((h, w, 4), np.uint8)
        cx = int((i + 1) * w / (n + 1))
        yy, xx = np.ogrid[:h, :w]
        mask = (xx - cx) ** 2 + (yy - h // 2) ** 2 <= 64
        px[mask] = (0, 200, 0, 180)
        frames.append(Image.fromarray(px, "RGBA"))
    buf = io.BytesIO()
    frames[0].save(
        buf,
        format="WEBP",
        save_all=True,
        append_images=frames[1:],
        duration=120,
        loop=0,
        lossless=True,
        quality=100,
        method=6,
        exact=True,
    )
    return buf.getvalue()


def _static_png(w: int = 64, h: int = 48, seed: int = 1) -> bytes:
    px = _checker_rgba(w, h, seed)
    buf = io.BytesIO()
    Image.fromarray(px, "RGBA").save(buf, format="PNG")
    return buf.getvalue()


def _oriented_jpeg(orientation: int = 6, w: int = 64, h: int = 48) -> bytes:
    """JPEG carrying an EXIF orientation tag on a directional marker."""
    px = np.zeros((h, w, 3), np.uint8)
    px[: h // 2, :, :] = (200, 50, 50)  # red top half indicates "up"
    px[h // 2 :, :, :] = (50, 50, 200)
    im = Image.fromarray(px, "RGB")
    exif = Image.Exif()
    exif[274] = orientation
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=90, exif=exif.tobytes())
    return buf.getvalue()


_SAMPLES = {
    "animated_transparent": (_animated_transparent_gif, Format.GIF),
    "animated_webp": (_animated_webp, Format.WEBP),
    "static_png": (_static_png, Format.PNG),
    "oriented_jpeg": (_oriented_jpeg, Format.JPEG),
}


def available_samples() -> tuple[str, ...]:
    return tuple(_SAMPLES)


def generate_sample(kind: str = "animated_transparent") -> tuple[bytes, Format]:
    """Return ``(bytes, format)`` for a named in-memory sample."""
    if kind not in _SAMPLES:
        raise InvalidInputError(
            f"unknown sample {kind!r}; available: {', '.join(_SAMPLES)}"
        )
    builder, fmt = _SAMPLES[kind]
    return builder(), fmt
