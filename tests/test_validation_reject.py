"""Requirement 5: corrupt/illegal/oversize inputs are rejected early with
distinguishable, locatable reasons and never produce partial images."""

from __future__ import annotations

import io

import pytest
from PIL import Image

from image_pipeline import samples
from image_pipeline.config import Limits, PipelineConfig
from image_pipeline.model import ProcessRequest
from image_pipeline.pipeline import process_or_reject


def rej(data, cfg=None, **kw):
    return process_or_reject(
        ProcessRequest(source=data, source_id="src.bin", request_id="req-1", **kw),
        cfg or PipelineConfig(),
    )


def test_empty_input():
    assert rej(b"").code == "EMPTY_INPUT"


def test_unrecognized_signature():
    assert rej(samples.make_garbage()).code == "UNSUPPORTED_FORMAT"


@pytest.mark.parametrize("maker,out", [
    (samples.make_static_png, "PNG"),
    (samples.make_static_jpeg, "JPEG"),
    (samples.make_animated_webp, "WEBP"),
    (samples.make_animated_gif, "GIF"),
])
def test_truncated_inputs_rejected(maker, out):
    data = maker()
    r = rej(data[: int(len(data) * 0.45)], output_format=out)
    assert r.code == "TRUNCATED_OR_UNDECODABLE"
    assert r.request_id == "req-1" and r.source_id == "src.bin"


def test_truncated_gif_specific(animated_gif=None):
    r = rej(samples.make_truncated_gif(), output_format="GIF")
    assert r.code == "TRUNCATED_OR_UNDECODABLE"


def test_oversized_input_bytes():
    cfg = PipelineConfig(limits=Limits(max_bytes=16))
    r = rej(samples.make_static_png(), cfg)
    assert r.code == "INVALID_REQUEST"


def test_frame_count_limit(tight_config):
    r = rej(samples.make_animated_gif(frames=6, size=(32, 32)), tight_config, output_format="WEBP")
    assert r.code == "FRAME_COUNT_EXCEEDED"


def test_pixel_limit(tight_config):
    r = rej(samples.make_animated_gif(frames=2, size=(96, 96)), tight_config)
    assert r.code == "PIXEL_LIMIT_EXCEEDED"


def test_dimension_too_large(animated_gif):
    r = rej(animated_gif, width=999999)
    assert r.code == "DIMENSION_TOO_LARGE"


def test_bad_quality(animated_gif):
    r = rej(animated_gif, quality=0)
    assert r.code == "INVALID_REQUEST"
    r = rej(animated_gif, quality=101)
    assert r.code == "INVALID_REQUEST"


def test_unsupported_output_format(animated_gif):
    r = rej(animated_gif, output_format="TIFF")
    assert r.code == "UNSUPPORTED_FORMAT"


def test_negative_dimensions(animated_gif):
    r = rej(animated_gif, width=-5)
    assert r.code == "INVALID_REQUEST"


def test_multi_frame_to_jpeg_rejected(animated_gif):
    r = rej(animated_gif, output_format="JPEG")
    # no silent frame dropping: animated -> jpeg is unsupported
    assert r.code == "UNSUPPORTED_FORMAT"


def test_error_locator_fields_present():
    r = rej(b"")
    assert r.request_id and r.source_id


def test_single_corrupt_frame_does_not_drop_animation(tight_config):
    """A damaged frame must fail the whole image, not silently drop frames."""
    # truncated gif must fail rather than decode fewer frames
    data = samples.make_animated_gif(frames=3, size=(24, 24))
    cut = data[: int(len(data) * 0.7)]
    r = rej(cut, output_format="GIF")
    assert r.code == "TRUNCATED_OR_UNDECODABLE"


def _make_apng_with_duration_ms(duration_ms: int) -> bytes:
    import struct
    import zlib

    import numpy as np

    def chunk(t, b):
        return (
            struct.pack(">I", len(b)) + t + b
            + struct.pack(">I", zlib.crc32(t + b) & 0xFFFFFFFF)
        )

    def fctl(seq, ms):
        # duration_ms = delay_num*1000/delay_den; stay within u16 fields
        if ms <= 65535:
            num, den = ms, 1000
        else:
            num, den = min(ms // 1000, 65535), 1  # express in whole seconds
        return chunk(
            b"fcTL",
            struct.pack(">5I", seq, 8, 8, 0, 0)
            + struct.pack(">HHBB", num, den, 0, 0),
        )

    arr = np.zeros((8, 8, 4), dtype=np.uint8)
    arr[..., 3] = 255
    raw = chunk(b"IDAT", zlib.compress(b"".join(b"\x00" + r.tobytes() for r in arr)))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 8, 8, 8, 6, 0, 0, 0))
        + chunk(b"acTL", struct.pack(">II", 1, 0))
        + fctl(0, duration_ms)
        + raw
        + chunk(b"IEND", b"")
    )


def test_excessive_frame_duration_rejected(tight_config):
    cfg = PipelineConfig(limits=Limits(max_frame_duration_ms=500))
    bad = _make_apng_with_duration_ms(999_999)
    assert rej(bad, cfg, output_format="PNG").code == "INVALID_FRAME_METADATA"
    ok = _make_apng_with_duration_ms(250)
    assert hasattr(rej(ok, cfg, output_format="PNG"), "data")
