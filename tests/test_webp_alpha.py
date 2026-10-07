"""Regression: animated WebP must retain an alpha channel for binary-alpha
animations (Pillow 12/libwebp only enable container alpha on partial alpha)."""

import io

import numpy as np
from PIL import Image, ImageDraw

from imagepipeline.config import Settings
from imagepipeline.models import Format, TransformRequest
from imagepipeline.pipeline import process


def _disposal_gif(disposal):
    W = H = 48
    frames = []
    for x in (4, 20):
        im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        ImageDraw.Draw(im).rectangle([x, 8, x + 10, 40], fill=(0, 128, 255, 255))
        frames.append(im)
    buf = io.BytesIO()
    frames[0].save(
        buf,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=100,
        loop=0,
        disposal=[disposal, disposal],
        optimize=False,
    )
    return buf.getvalue()


def test_binary_alpha_animation_keeps_webp_alpha_channel():
    data = _disposal_gif(2)
    result = process(data, TransformRequest(Format.WEBP, lossless=True), Settings())
    out = Image.open(io.BytesIO(result.data))
    assert out.mode == "RGBA"


def test_do_not_dispose_accumulates_without_ghost_to_webp():
    data = _disposal_gif(1)
    result = process(data, TransformRequest(Format.WEBP, lossless=True), Settings())
    out = Image.open(io.BytesIO(result.data))
    assert out.mode == "RGBA"
    out.seek(1)
    out.load()
    f1 = np.asarray(out)
    # Both rectangles visible (accumulated) ...
    assert f1[20, 8, 3] == 255 and tuple(f1[20, 8, :3]) == (0, 128, 255)
    assert f1[20, 25, 3] == 255 and tuple(f1[20, 25, :3]) == (0, 128, 255)
    # ... and the gap between them stays transparent (no residue/ghost).
    assert f1[20, 16, 3] == 0


def test_resize_binary_alpha_animation_keeps_alpha():
    data = _disposal_gif(2)
    from imagepipeline.models import ResizeMode

    req = TransformRequest(
        Format.WEBP, ResizeMode.FIT, width=24, height=24, lossless=True
    )
    result = process(data, req, Settings())
    out = Image.open(io.BytesIO(result.data))
    assert out.mode == "RGBA"
