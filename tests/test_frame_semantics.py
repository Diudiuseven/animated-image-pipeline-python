"""Requirement 1 & 4: frame count/duration/loop fidelity and disposal/blend
compositing without black edges or ghosts."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image
import pytest

from image_pipeline.codec_io import decode
from image_pipeline.config import DEFAULT_CONFIG
from image_pipeline.model import Blending, DecodedImage, Disposal, Frame
from image_pipeline.pipeline import process
from image_pipeline.model import ProcessRequest
from image_pipeline.render import alpha_over, composite_frames


def solid(size, rgba):
    w, h = size
    a = np.zeros((h, w, 4), dtype=np.uint8)
    a[..., :] = rgba
    return a


# ---- raw compositor semantics ------------------------------------------------

def test_alpha_over_integer_examples():
    dst = solid((1, 1), (255, 255, 255, 255))
    src = solid((1, 1), (0, 0, 0, 128))
    out = alpha_over(dst, src)[0, 0]
    assert out[3] == 255
    assert tuple(out[:3]) == (127, 127, 127)


def test_transparent_source_leaves_destination():
    dst = solid((2, 2), (99, 88, 77, 255))
    src = solid((2, 2), (0, 0, 0, 0))
    assert np.array_equal(alpha_over(dst, src), dst)


def test_background_disposal_clears_region():
    d = DecodedImage(width=2, height=2)
    d.frames.append(Frame(solid((2, 2), (255, 0, 0, 255)), 10, Disposal.NONE, Blending.SOURCE))
    d.frames.append(Frame(solid((1, 1), (0, 255, 0, 255)), 10, Disposal.BACKGROUND, Blending.SOURCE, 1, 1))
    out = composite_frames(d)
    assert tuple(out[1][1, 1]) == (0, 255, 0, 255)
    # A third frame observes the cleared (transparent) region, no ghost.
    d.frames.append(Frame(solid((1, 1), (0, 0, 255, 255)), 10, Disposal.NONE, Blending.SOURCE, 0, 0))
    out = composite_frames(d)
    assert tuple(out[2][1, 1]) == (0, 0, 0, 0)
    assert tuple(out[2][0, 0]) == (0, 0, 255, 255)


def test_previous_disposal_restores_canvas():
    d = DecodedImage(width=2, height=2)
    d.frames.append(Frame(solid((2, 2), (255, 0, 0, 255)), 10, Disposal.NONE, Blending.SOURCE))
    d.frames.append(Frame(solid((1, 1), (0, 255, 0, 255)), 10, Disposal.PREVIOUS, Blending.SOURCE, 1, 1))
    d.frames.append(Frame(solid((1, 1), (0, 0, 255, 255)), 10, Disposal.NONE, Blending.SOURCE, 0, 0))
    out = composite_frames(d)
    assert tuple(out[2][1, 1]) == (255, 0, 0, 255)  # red restored, no green ghost


def test_half_alpha_over_opaque_no_black():
    d = DecodedImage(width=1, height=1)
    d.frames.append(Frame(solid((1, 1), (255, 0, 0, 255)), 10, Disposal.NONE, Blending.SOURCE))
    d.frames.append(Frame(solid((1, 1), (0, 0, 255, 128)), 10, Disposal.NONE, Blending.OVER))
    out = composite_frames(d)
    px = tuple(out[1][0, 0])
    assert px == (127, 0, 128, 255)


# ---- frame fidelity through decode/transcode --------------------------------

@pytest.mark.parametrize("fmt_suffix", ["gif", "webp", "png"])
def test_frame_count_durations_loop_preserved(animated_gif, animated_webp, animated_png, fmt_suffix):
    src = {"gif": animated_gif, "webp": animated_webp, "png": animated_png}[fmt_suffix]
    d = decode(src, DEFAULT_CONFIG)
    assert len(d.frames) == 4
    assert [f.duration_ms for f in d.frames] == [100, 100, 100, 100]
    assert d.loop_count == 0
    # transcode to every animated target: count must stay 4
    for out_fmt in ("PNG", "WEBP", "GIF"):
        r = process(ProcessRequest(source=src, output_format=out_fmt, request_id="x"))
        im = Image.open(io.BytesIO(r.data))
        assert im.n_frames == 4, (fmt_suffix, out_fmt)
        assert r.durations_ms == [100, 100, 100, 100]
        assert r.loop_count == 0


def test_nonzero_loop_preserved():
    b0 = Image.new("RGBA", (8, 8), (255, 0, 0, 255))
    b1 = Image.new("RGBA", (8, 8), (0, 0, 255, 255))
    buf = io.BytesIO()
    b0.save(buf, format="WEBP", save_all=True, append_images=[b1], duration=120, loop=7, lossless=True)
    r = process(ProcessRequest(source=buf.getvalue(), output_format="WEBP", request_id="x"))
    assert r.loop_count == 7 and r.durations_ms == [120, 120]


def test_apng_partial_update_tile_offset():
    b0 = Image.new("RGBA", (8, 8), (255, 0, 0, 255))
    b1 = Image.new("RGBA", (3, 2), (0, 255, 0, 255))
    buf = io.BytesIO()
    b0.save(buf, format="PNG", save_all=True, append_images=[b1], duration=100,
            disposal=[0, 0], blend=[0, 0])
    d = decode(buf.getvalue(), DEFAULT_CONFIG)
    f1 = d.frames[1]
    assert (f1.tile.shape[1], f1.tile.shape[0]) == (3, 2)
    out = composite_frames(d)
    # default APNG location is (0,0); green tile replaces top-left, red stays elsewhere
    assert tuple(out[1][5, 5]) == (255, 0, 0, 255)
    assert tuple(out[1][0, 0]) == (0, 255, 0, 255)


def test_apng_nonzero_offset_tile_placed_correctly():
    import zlib, struct as st

    def chunk(typ, body):
        return (
            st.pack(">I", len(body)) + typ + body
            + st.pack(">I", zlib.crc32(typ + body) & 0xFFFFFFFF)
        )

    def fctl(seq, w, h, x, y, dis=0, bl=0):
        return chunk(
            b"fcTL",
            st.pack(">5I", seq, w, h, x, y) + st.pack(">HHBB", 1, 10, dis, bl),
        )

    def image_chunk(kind, w, h, color, seq=None):
        arr = np.zeros((h, w, 4), dtype=np.uint8)
        arr[..., :3] = color
        arr[..., 3] = 255
        raw = b"".join(b"\x00" + arr[row].tobytes() for row in range(h))
        comp = zlib.compress(raw)
        return chunk(kind, comp if kind == b"IDAT" else st.pack(">I", seq) + comp)

    W = H = 16
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", st.pack(">IIBBBBB", W, H, 8, 6, 0, 0, 0))
        + chunk(b"acTL", st.pack(">II", 2, 0))
        + fctl(0, W, H, 0, 0) + image_chunk(b"IDAT", W, H, (255, 0, 0))
        + fctl(1, 4, 3, 5, 2) + image_chunk(b"fdAT", 4, 3, (0, 255, 0), 2)
        + chunk(b"IEND", b"")
    )
    d = decode(png, DEFAULT_CONFIG)
    assert (d.frames[1].offset_x, d.frames[1].offset_y) == (5, 2)
    out = composite_frames(d)
    assert tuple(out[1][2, 5]) == (0, 255, 0, 255)
    assert tuple(out[1][2, 4]) == (255, 0, 0, 255)  # left of tile untouched
    assert tuple(out[1][7, 7]) == (255, 0, 0, 255)  # outside tile untouched
