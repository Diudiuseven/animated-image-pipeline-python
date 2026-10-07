"""EXIF orientation must be applied to every animation frame."""

import io

import numpy as np
from PIL import Image

from imagepipeline.config import Settings
from imagepipeline.models import Format, TransformRequest
from imagepipeline.pipeline import process


def _oriented_animated_webp(orientation=6):
    H, W = 20, 40  # stored landscape
    frames = []
    for i in range(3):
        a = np.zeros((H, W, 4), np.uint8)
        a[: H // 2, :, :3] = (200, 50, 50)  # red top half
        a[H // 2 :, :, :3] = (50, 50, 200)  # blue bottom half
        a[:, :, 3] = 255
        a[0, 0, :3] = (i * 60, i * 60, 0)  # keep frames distinct
        frames.append(Image.fromarray(a, "RGBA"))
    exif = Image.Exif()
    exif[274] = orientation
    buf = io.BytesIO()
    frames[0].save(
        buf,
        format="WEBP",
        save_all=True,
        append_images=frames[1:],
        duration=100,
        loop=0,
        lossless=True,
        exif=exif.tobytes(),
    )
    return buf.getvalue()


def test_orientation_applies_to_all_frames():
    data = _oriented_animated_webp(orientation=6)  # rotate CW
    result = process(data, TransformRequest(Format.WEBP, lossless=True), Settings())
    assert (result.width, result.height) == (20, 40)  # 40x20 -> 20x40
    assert result.frame_count == 3
    out = Image.open(io.BytesIO(result.data))
    for i in range(out.n_frames):
        out.seek(i)
        out.load()
        g = np.asarray(out)
        assert g.shape == (40, 20, 4)
        # Red top row becomes the right column; blue bottom the left column.
        assert tuple(g[5, -1, :3]) == (200, 50, 50)
        assert tuple(g[5, 0, :3]) == (50, 50, 200)


def test_output_carries_no_stale_orientation():
    data = _oriented_animated_webp(orientation=6)
    result = process(data, TransformRequest(Format.WEBP, lossless=True), Settings())
    out = Image.open(io.BytesIO(result.data))
    assert out.getexif().get(274, 1) in (None, 1)
