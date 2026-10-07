"""APNG animation fidelity and illegal frame-duration rejection."""

import io

import numpy as np
import pytest
from PIL import Image

from imagepipeline.config import Limits, Settings
from imagepipeline.exceptions import InvalidInputError
from imagepipeline.models import Format, TransformRequest
from imagepipeline.pipeline import process
from imagepipeline.stages import decoder as dec


def _apng(n=3, size=(20, 20), durations=None, loop=1):
    frames = []
    for i in range(n):
        a = np.zeros(size + (4,), np.uint8)
        a[:, :, 0] = i * 70 % 255
        a[:, :, 3] = 255
        frames.append(Image.fromarray(a, "RGBA"))
    buf = io.BytesIO()
    frames[0].save(
        buf,
        format="PNG",
        save_all=True,
        append_images=frames[1:],
        duration=durations or [90, 140, 210],
        loop=loop,
    )
    return buf.getvalue()


def test_apng_frame_count_duration_loop():
    data = _apng()
    result = process(data, TransformRequest(Format.PNG), Settings())
    assert result.frame_count == 3
    assert list(result.durations_ms) == [90, 140, 210]
    assert result.loop == 1


def test_apng_to_webp_keeps_frames():
    data = _apng(n=3)
    result = process(data, TransformRequest(Format.WEBP, lossless=True), Settings())
    assert result.frame_count == 3
    out = Image.open(io.BytesIO(result.data))
    assert out.n_frames == 3


def test_zero_duration_is_illegal():
    # Build a GIF with a 0-duration frame by crafting through PIL.
    frames = [Image.new("RGBA", (8, 8), (i * 50, 0, 0, 255)) for i in range(2)]
    buf = io.BytesIO()
    frames[0].save(
        buf,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=[0, 100],
        disposal=[2, 2],
    )
    s = Settings(limits=Limits(min_frame_duration_ms=1))
    with pytest.raises(InvalidInputError):
        dec.decode(buf.getvalue(), Format.GIF, s)


def test_excessive_frame_duration_rejected():
    # Animated WebP carries millisecond durations well beyond GIF's range.
    f0 = Image.new("RGBA", (8, 8), (255, 0, 0, 255))
    f1 = Image.new("RGBA", (8, 8), (0, 255, 0, 255))
    buf = io.BytesIO()
    f0.save(
        buf,
        format="WEBP",
        save_all=True,
        append_images=[f1],
        duration=[999999, 100],
        lossless=True,
    )
    s = Settings(limits=Limits(max_frame_duration_ms=60000))
    with pytest.raises(InvalidInputError):
        dec.decode(buf.getvalue(), Format.WEBP, s)


@pytest.mark.parametrize("loop_value", [0, 1, 3])
def test_loop_count_preserved_across_formats(loop_value):
    data = _apng(n=2, durations=[100, 100], loop=loop_value)
    for target in (Format.GIF, Format.WEBP, Format.PNG):
        result = process(
            data,
            TransformRequest(target, lossless=True),
            Settings(),
        )
        out = Image.open(io.BytesIO(result.data))
        assert out.info.get("loop") == loop_value, (target, out.info.get("loop"))
        assert result.loop == loop_value
