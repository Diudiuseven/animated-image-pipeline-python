"""Oversized animation bounds and multi-frame transparency / ghosting."""

import io

import numpy as np
import pytest
from PIL import Image, ImageDraw

from imagepipeline.config import Limits, Settings
from imagepipeline.exceptions import LimitExceededError
from imagepipeline.models import Format, TransformRequest
from imagepipeline.pipeline import process
from imagepipeline.samples import generate_sample


def _gif(n, size, dur=100, disposal=2):
    w, h = size
    frames = []
    for i in range(n):
        im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        x = (i * 7) % max(w - 10, 1)
        d.rectangle([x, 0, x + 8, h - 1], fill=(i * 30 % 255, 80, 200, 220))
        frames.append(im)
    buf = io.BytesIO()
    frames[0].save(
        buf,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=dur,
        loop=0,
        disposal=disposal,
        optimize=False,
    )
    return buf.getvalue()


def test_too_many_frames_rejected_before_processing():
    data = _gif(n=12, size=(32, 32))
    s = Settings(limits=Limits(max_frames=10))
    with pytest.raises(LimitExceededError) as info:
        process(data, TransformRequest(Format.GIF), s)
    assert info.value.code == "limit_exceeded"


def test_too_large_pixels_rejected():
    data = _gif(n=2, size=(64, 64))  # 4096 px/frame
    s = Settings(limits=Limits(max_pixels_per_frame=4000))
    with pytest.raises(LimitExceededError):
        process(data, TransformRequest(Format.GIF), s)


def test_too_many_total_pixels_rejected():
    data = _gif(n=8, size=(32, 32))  # 8192 total
    s = Settings(limits=Limits(max_total_pixels=8000))
    with pytest.raises(LimitExceededError):
        process(data, TransformRequest(Format.GIF), s)


def test_oversized_animation_does_not_block_batch():
    """One oversized GIF fails; other requests in the batch still succeed."""
    import asyncio

    from imagepipeline.executor import PipelineExecutor

    big = _gif(n=6, size=(32, 32))
    ok, _ = generate_sample("static_png")
    s = Settings(limits=Limits(max_frames=5, max_concurrent_requests=8))

    async def scenario():
        ex = PipelineExecutor(s)
        big_task = ex.submit(big, TransformRequest(Format.GIF), "big")
        good_task = ex.submit(ok, TransformRequest(Format.PNG), "good")
        results = await asyncio.gather(big_task, good_task, return_exceptions=True)
        await ex.aclose()
        return results

    results = asyncio.run(scenario())
    assert isinstance(results[0], LimitExceededError)
    assert results[1].frame_count == 1


def test_partial_alpha_composite_has_no_black_fringe_or_ghost():
    """Semi-transparent moving rectangles blend without black edges/residue."""
    w = h = 40
    frames = []
    positions = [6, 18]
    for x in positions:
        im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        ImageDraw.Draw(im).rectangle(
            [x, 8, x + 12, 32], fill=(255, 0, 0, 128)
        )
        frames.append(im)
    buf = io.BytesIO()
    frames[0].save(
        buf,
        format="WEBP",
        save_all=True,
        append_images=frames[1:],
        duration=100,
        loop=0,
        lossless=True,
        quality=100,
        method=6,
        exact=True,
    )
    data = buf.getvalue()
    result = process(data, TransformRequest(Format.WEBP, lossless=True), Settings())

    out = Image.open(io.BytesIO(result.data))
    out.seek(1)
    out.load()
    f1 = np.asarray(out).astype(np.int16)
    # The first rectangle's area (which is not covered by frame 2) must be
    # fully cleared - no red residue / ghosting from the previous frame.
    leftover = f1[8:32, 6:18]
    assert (leftover[..., 3] == 0).all()
    # Inside the second rectangle alpha must be ~128, not black-polluted.
    inside = f1[8:32, 18:30]
    assert (inside[..., 3] > 100).all()
    # Red channel dominates (no black/color bleed): red >= green, blue.
    assert (inside[..., 0] >= inside[..., 1]).all()
    assert (inside[..., 0] >= inside[..., 2]).all()
