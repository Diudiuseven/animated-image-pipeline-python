import io

import numpy as np
import pytest
from PIL import Image

from imagepipeline.config import Settings
from imagepipeline.exceptions import InvalidRequestError
from imagepipeline.models import Format, Frame, ImageDocument, TransformRequest


def _doc(n=3, size=(16, 16), color=(0, 100, 200, 255), loop=2, dur=100, vary=False):
    w, h = size
    frames = []
    for i in range(n):
        c = ((i * 70) % 255, color[1], color[2], color[3]) if vary else color
        frames.append(Frame(np.full((h, w, 4), c, np.uint8), duration_ms=dur))
    return ImageDocument(frames=tuple(frames), loop=loop)


@pytest.mark.parametrize("fmt", [Format.PNG, Format.GIF, Format.WEBP])
def test_animated_roundtrip_keeps_frames_durations_loop(fmt):
    # Identical frames are the case container writers like to merge; the
    # encoder must still preserve frame count and per-frame durations.
    doc = _doc(n=3, loop=2, dur=100)
    from imagepipeline.stages.encoder import encode

    out = encode(doc, TransformRequest(fmt), Settings())
    im = Image.open(io.BytesIO(out))
    assert im.n_frames == 3
    durs = []
    for i in range(im.n_frames):
        im.seek(i)
        im.load()
        durs.append(im.info["duration"])
    assert durs == [100, 100, 100]
    assert im.info.get("loop") == 2


@pytest.mark.parametrize("fmt", [Format.PNG, Format.GIF, Format.WEBP])
def test_distinct_identical_mix_keeps_all_frames(fmt):
    from imagepipeline.stages.encoder import encode

    w = h = 16
    frames = [
        Frame(np.full((h, w, 4), (10, 20, 30, 255), np.uint8), 100),
        Frame(np.full((h, w, 4), (90, 20, 30, 255), np.uint8), 150),
        Frame(np.full((h, w, 4), (90, 20, 30, 255), np.uint8), 200),
    ]
    doc = ImageDocument(frames=tuple(frames), loop=1)
    out = encode(doc, TransformRequest(fmt), Settings())
    im = Image.open(io.BytesIO(out))
    assert im.n_frames == 3
    durs = []
    for i in range(3):
        im.seek(i)
        im.load()
        durs.append(im.info["duration"])
    assert durs == [100, 150, 200]


def test_static_to_animated_format_is_single_frame():
    doc = _doc(n=1)
    for fmt in (Format.GIF, Format.WEBP):
        out = __import__("imagepipeline.stages.encoder", fromlist=["encode"]).encode(
            doc, TransformRequest(fmt), Settings()
        )
        im = Image.open(io.BytesIO(out))
        assert getattr(im, "n_frames", 1) == 1


def test_animated_jpeg_requires_still_flag():
    doc = _doc(n=2)
    with pytest.raises(InvalidRequestError):
        __import__("imagepipeline.stages.encoder", fromlist=["encode"]).encode(
            doc, TransformRequest(Format.JPEG), Settings()
        )
    out = __import__("imagepipeline.stages.encoder", fromlist=["encode"]).encode(
        doc, TransformRequest(Format.JPEG, still=True), Settings()
    )
    Image.open(io.BytesIO(out)).load()  # decodable


def test_transparency_flattened_for_jpeg():
    doc = _doc(n=1, color=(255, 0, 0, 128))
    from imagepipeline.stages.encoder import encode

    out = encode(doc, TransformRequest(Format.JPEG), Settings())
    rgb = np.asarray(Image.open(io.BytesIO(out)).convert("RGB"))
    # Red over the default white background -> light pink, not black.
    assert rgb[0, 0, 0] > rgb[0, 0, 1]
    assert rgb[0, 0, 1] > 100


def test_gif_preserves_transparency():
    doc = _doc(n=2, color=(10, 20, 30, 0))
    from imagepipeline.stages.encoder import encode

    out = encode(doc, TransformRequest(Format.GIF), Settings())
    im = Image.open(io.BytesIO(out))
    im.seek(1)
    im.load()
    assert np.asarray(im.convert("RGBA"))[0, 0, 3] == 0


def test_encoding_is_byte_deterministic():
    from imagepipeline.stages.encoder import encode

    doc = _doc(n=3, color=(1, 2, 3, 200))
    req = TransformRequest(Format.WEBP, quality=75)
    a = encode(doc, req, Settings())
    b = encode(doc, req, Settings())
    assert a == b
    req2 = TransformRequest(Format.GIF)
    assert encode(doc, req2, Settings()) == encode(doc, req2, Settings())
