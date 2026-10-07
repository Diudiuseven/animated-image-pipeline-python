"""Deterministic in-process sample image generation.

All samples are generated from fixed constants / seeded numpy, so tests
need no external files or network and every run reproduces identical
bytes.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image


def frame_color(i: int) -> tuple[int, int, int]:
    """Distinct, fixed RGB color for frame index i."""
    return (
        (37 * i + 40) & 255,
        (91 * i + 120) & 255,
        (173 * i + 200) & 255,
    )


def _animated_rgba(frames: int, size: tuple[int, int], transparent: bool = False):
    w, h = size
    out = []
    for i in range(frames):
        arr = np.zeros((h, w, 4), dtype=np.uint8)
        c = frame_color(i)
        arr[..., 0], arr[..., 1], arr[..., 2] = c
        arr[..., 3] = 255
        if transparent:
            # moving semi-transparent block over a fully transparent bg
            arr[..., 3] = 0
            bs = max(1, min(w, h) // 2)
            x0 = (i * 2) % max(1, w - bs + 1)
            y0 = (i * 3) % max(1, h - bs + 1)
            arr[y0 : y0 + bs, x0 : x0 + bs, 0:3] = c
            arr[y0 : y0 + bs, x0 : x0 + bs, 3] = 128 + i
        out.append(Image.fromarray(arr, "RGBA"))
    return out


def _save(buf, ims, fmt, duration_ms, loop=0, **kw):
    durations = [duration_ms] * len(ims)
    if len(ims) == 1:
        ims[0].save(buf, format=fmt, **kw)
    else:
        ims[0].save(
            buf, format=fmt, save_all=True, append_images=ims[1:],
            duration=durations, loop=loop, **kw
        )
    return buf.getvalue()


def make_static_png(size: tuple[int, int] = (64, 48)) -> bytes:
    arr = np.zeros((size[1], size[0], 4), dtype=np.uint8)
    yy, xx = np.mgrid[0 : size[1], 0 : size[0]]
    arr[..., 0] = (xx * 4) & 255
    arr[..., 1] = (yy * 5) & 255
    arr[..., 2] = 128
    arr[..., 3] = 255
    buf = io.BytesIO()
    Image.fromarray(arr, "RGBA").save(buf, format="PNG")
    return buf.getvalue()


def make_static_jpeg(size: tuple[int, int] = (64, 48), orientation: int = 1) -> bytes:
    arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    yy, xx = np.mgrid[0 : size[1], 0 : size[0]]
    arr[..., 0] = (xx * 4) & 255
    arr[..., 1] = (yy * 5) & 255
    arr[..., 2] = 128
    im = Image.fromarray(arr, "RGB")
    exif = None
    if orientation != 1:
        e = Image.Exif()
        e[274] = orientation
        exif = e.tobytes()
    buf = io.BytesIO()
    if exif is not None:
        im.save(buf, format="JPEG", quality=92, exif=exif)
    else:
        im.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


def make_animated_gif(
    frames: int = 4, size: tuple[int, int] = (64, 64), duration_ms: int = 100
) -> bytes:
    # Build via palette P images with partial tiles and transparency so
    # frame composition semantics are exercised.
    w, h = size
    pal_images = []
    palette = [0, 0, 0]
    for i in range(frames):
        palette.extend(frame_color(i))
    palette.extend([0] * (256 * 3 - len(palette)))
    for i in range(frames):
        im = Image.new("P", (w, h), 0)
        im.putpalette(palette)
        # paint index i+1 into a moving region at image level
        tile = Image.new("P", (max(1, w // 2), max(1, h // 2)), i + 1)
        x0 = (i * 4) % max(1, w - w // 2 + 1)
        y0 = (i * 7) % max(1, h - h // 2 + 1)
        im.paste(tile, (x0, y0))
        pal_images.append(im)
    buf = io.BytesIO()
    pal_images[0].save(
        buf, format="GIF", save_all=True, append_images=pal_images[1:],
        duration=[duration_ms] * frames, loop=0, transparency=0,
        disposal=[2] * frames, optimize=False,
    )
    return buf.getvalue()


def make_animated_webp(
    frames: int = 4, size: tuple[int, int] = (64, 64), duration_ms: int = 100
) -> bytes:
    ims = _animated_rgba(frames, size, transparent=True)
    return _save(
        io.BytesIO(), ims, "WEBP", duration_ms, loop=0,
        lossless=True, method=6, minimize_size=False,
    )


def make_animated_png(
    frames: int = 4, size: tuple[int, int] = (64, 64), duration_ms: int = 100
) -> bytes:
    ims = _animated_rgba(frames, size, transparent=True)
    return _save(
        io.BytesIO(), ims, "PNG", duration_ms, loop=0,
        disposal=[1] * frames, blend=[1] * frames, compress_level=6,
    )


def make_truncated_gif() -> bytes:
    """A GIF cut mid-LZW-stream: container valid, frame data incomplete."""
    full = make_animated_gif(frames=3)
    return full[: int(len(full) * 0.6)]


def make_garbage() -> bytes:
    return b"this is not an image file at all" + bytes(range(256))


def make_p3_jpeg(p3_icc: bytes, size: tuple[int, int] = (64, 48)) -> bytes:
    """Mid-tone in-gamut JPEG tagged with a wide-gamut (P3) ICC profile."""
    import numpy as _np

    rng = _np.random.RandomState(7)
    arr = (40 + rng.rand(size[1], size[0], 3) * 170).astype("uint8")
    im = Image.fromarray(arr, "RGB")
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=95, icc_profile=p3_icc)
    return buf.getvalue()
