"""Requirement 2: EXIF orientation is baked in, including on every frame."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

from image_pipeline.codec_io import decode
from image_pipeline.config import DEFAULT_CONFIG
from image_pipeline.orientation import (
    apply_orientation,
    extract_orientation,
    tile_offset_after_orientation,
)


def _jpeg_with_orientation(orientation: int, size=(40, 24)) -> bytes:
    arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    yy, xx = np.mgrid[0 : size[1], 0 : size[0]]
    arr[..., 0] = (xx * 6) & 255
    arr[..., 1] = (yy * 10) & 255
    im = Image.fromarray(arr, "RGB")
    exif = Image.Exif()
    exif[274] = orientation
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=95, exif=exif.tobytes())
    return buf.getvalue()


def test_extract_orientation_default():
    im = Image.new("RGB", (4, 4))
    assert extract_orientation(im) == 1


import pytest


@pytest.mark.parametrize("orient,expected", [
    (1, (40, 24)),
    (2, (40, 24)),
    (3, (40, 24)),
    (4, (40, 24)),
    (5, (24, 40)),
    (6, (24, 40)),
    (7, (24, 40)),
    (8, (24, 40)),
])
def test_decode_bakes_orientation(orient, expected):
    d = decode(_jpeg_with_orientation(orient), DEFAULT_CONFIG)
    assert (d.width, d.height) == expected
    assert len(d.frames) == 1


def test_orientation_tag_stripped():
    im = Image.open(io.BytesIO(_jpeg_with_orientation(6)))
    out = apply_orientation(im)
    assert out.getexif().get(274) is None


def test_orientation_matches_pillow():
    for orient in range(1, 9):
        raw = _jpeg_with_orientation(orient)
        im = Image.open(io.BytesIO(raw))
        ref = np.asarray(apply_orientation(im))
        d = decode(raw, DEFAULT_CONFIG)
        got = d.frames[0].tile
        # alpha channel differs (we add opaque alpha); compare RGB
        assert got.shape[:2] == ref.shape[:2]
        assert np.array_equal(got[..., :3], np.asarray(ref)[..., :3])


def test_tile_offset_mapping_known_values():
    # 90 CW on 100x50 canvas, tile (10,20,30x8)
    x, y, w, h = tile_offset_after_orientation(10, 20, 30, 8, 100, 50, 6)
    assert (x, y, w, h) == (22, 10, 8, 30)
    # identity
    assert tile_offset_after_orientation(10, 20, 30, 8, 100, 50, 1) == (10, 20, 30, 8)
