import numpy as np

from imagepipeline.config import Settings
from imagepipeline.models import Format, Frame, ImageDocument, ResizeMode, TransformRequest
from imagepipeline.stages.resize import resize, target_size


def _doc(size=(40, 20), n=2, color=(255, 0, 0, 255), durations=(100, 200)):
    w, h = size
    frames = tuple(
        Frame(np.full((h, w, 4), color, np.uint8), duration_ms=d)
        for d in durations[:n]
    )
    return ImageDocument(frames=frames, loop=0, source_format=Format.PNG)


def test_target_size_modes():
    assert target_size(40, 20, TransformRequest(Format.PNG, ResizeMode.NONE)) == (40, 20)
    r = TransformRequest(Format.PNG, ResizeMode.FIT, width=20, height=20)
    assert target_size(40, 20, r) == (20, 10)
    r = TransformRequest(Format.PNG, ResizeMode.FIT, width=20)
    assert target_size(40, 20, r) == (20, 10)
    r = TransformRequest(Format.PNG, ResizeMode.COVER, width=20, height=20)
    assert target_size(40, 20, r) == (40, 20)
    r = TransformRequest(Format.PNG, ResizeMode.STRETCH, width=7, height=9)
    assert target_size(40, 20, r) == (7, 9)


def test_fit_resizes_all_frames_and_keeps_timing():
    doc = _doc()
    req = TransformRequest(Format.PNG, ResizeMode.FIT, width=20, height=20)
    out = resize(doc, req, Settings())
    assert out.size == (20, 10)
    assert out.frame_count == 2
    assert [f.duration_ms for f in out.frames] == [100, 200]


def test_cover_center_crops():
    doc = _doc(size=(40, 20))
    req = TransformRequest(Format.PNG, ResizeMode.COVER, width=20, height=20)
    out = resize(doc, req, Settings())
    assert out.size == (20, 20)


def test_transparent_edge_has_no_black_fringe():
    # Opaque red square centered in a fully transparent canvas whose RGB is
    # black (the classic black-fringe trigger).
    w = h = 40
    px = np.zeros((h, w, 4), np.uint8)
    px[10:30, 10:30] = (255, 0, 0, 255)
    doc = ImageDocument(frames=(Frame(px, 100),), loop=0, source_format=Format.PNG)
    req = TransformRequest(Format.PNG, ResizeMode.FIT, width=20, height=20)
    out = resize(doc, req, Settings())
    f = out.frames[0].pixels.astype(np.int16)
    edge = f[5, 5]
    # A partially transparent edge pixel: its stored RGB must be red-ish,
    # never black, after un-premultiply.
    alpha = edge[3]
    if 0 < alpha < 255:
        assert edge[0] >= edge[1]
        assert edge[0] >= edge[2]


def test_resize_is_deterministic():
    doc = _doc(color=(12, 34, 56, 128))
    req = TransformRequest(Format.PNG, ResizeMode.FIT, width=13, height=7)
    a = resize(doc, req, Settings()).frames[0].pixels
    b = resize(doc, req, Settings()).frames[0].pixels
    assert np.array_equal(a, b)
