import io

import pytest
from PIL import Image

from imagepipeline.config import Settings
from imagepipeline.exceptions import InvalidInputError, InvalidRequestError
from imagepipeline.models import Format, ResizeMode, TransformRequest
from imagepipeline.pipeline import process
from imagepipeline.samples import generate_sample


def test_gif_to_webp_preserves_frames_and_timing():
    data, _ = generate_sample("animated_transparent")
    src = Image.open(io.BytesIO(data))
    src_durs = []
    for i in range(src.n_frames):
        src.seek(i)
        src.load()
        src_durs.append(src.info["duration"])

    result = process(data, TransformRequest(Format.WEBP, lossless=True), Settings())
    assert result.frame_count == src.n_frames
    assert list(result.durations_ms) == src_durs
    out = Image.open(io.BytesIO(result.data))
    assert out.n_frames == src.n_frames


def test_resize_pipeline():
    data, _ = generate_sample("static_png")
    req = TransformRequest(Format.PNG, ResizeMode.FIT, width=32, height=24)
    result = process(data, req, Settings())
    assert (result.width, result.height) == (32, 24)
    Image.open(io.BytesIO(result.data)).load()


def test_byte_determinism_across_repeated_processing():
    data, _ = generate_sample("animated_webp")
    req = TransformRequest(Format.GIF)
    a = process(data, req, Settings(), request_id="a").data
    b = process(data, req, Settings(), request_id="b").data
    c = process(data, req, Settings(), request_id="c").data
    assert a == b == c


def test_orientation_correction_e2e():
    import numpy as np

    data, _ = generate_sample("oriented_jpeg")
    # After correction (orientation=6) the stored red top half ends up on the
    # right column and the canvas rotates from 64x48 to 48x64.
    result = process(data, TransformRequest(Format.JPEG, still=True), Settings())
    out = Image.open(io.BytesIO(result.data)).convert("RGB")
    assert out.size == (48, 64)
    arr = np.asarray(out)
    # Right edge red-dominant, left edge blue-dominant.
    assert tuple(arr[10, -1]) > (arr[10, -1][1], arr[10, -1][2], 0)
    assert arr[10, -1, 0] > arr[10, -1, 2]
    assert arr[10, 0, 2] > arr[10, 0, 0]


def test_corrupt_input_rejected_before_output():
    data, _ = generate_sample("animated_transparent")
    with pytest.raises(InvalidInputError):
        process(data[:30], TransformRequest(Format.WEBP), Settings())
    with pytest.raises(InvalidInputError):
        process(b"not an image", TransformRequest(Format.PNG), Settings())


def test_animated_to_jpeg_requires_still():
    data, _ = generate_sample("animated_transparent")
    with pytest.raises(InvalidRequestError):
        process(data, TransformRequest(Format.JPEG), Settings())
    result = process(data, TransformRequest(Format.JPEG, still=True), Settings())
    assert result.frame_count == 1
