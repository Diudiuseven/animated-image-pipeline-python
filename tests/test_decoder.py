import io

import pytest
from PIL import Image

from imagepipeline.config import Limits, Settings
from imagepipeline.exceptions import InvalidInputError, LimitExceededError
from imagepipeline.models import Format
from imagepipeline.stages import decoder as dec


def _gif_bytes(n=3, size=(16, 16), durations=None, loop=0):
    frames = []
    for i in range(n):
        im = Image.new("RGBA", size, (i * 80, 255 - i * 80, 128, 255))
        frames.append(im)
    buf = io.BytesIO()
    save_kw = {
        "format": "GIF",
        "loop": loop,
    }
    durations = durations or [100] * n
    if n > 1:
        save_kw["save_all"] = True
        save_kw["append_images"] = frames[1:]
        save_kw["disposal"] = [2] * n
        save_kw["duration"] = durations
    else:
        save_kw["duration"] = durations[0]
    frames[0].save(buf, **save_kw)
    return buf.getvalue()


def test_decode_static_png():
    im = Image.new("RGBA", (10, 12), (1, 2, 3, 255))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    d = dec.decode(buf.getvalue(), Format.PNG, Settings())
    assert d.frame_count == 1
    assert d.canvas_size == (10, 12)
    assert d.orientation == 1


def test_decode_animated_gif_metadata():
    data = _gif_bytes(n=3, durations=[70, 130, 200], loop=2)
    d = dec.decode(data, Format.GIF, Settings())
    assert d.frame_count == 3
    assert d.loop == 2
    assert [m.duration_ms for m in d.frames_meta] == [70, 130, 200]
    assert all(m.width == 16 and m.height == 16 for m in d.frames_meta)


def test_truncated_bytes_rejected():
    data = _gif_bytes(n=3)
    with pytest.raises(InvalidInputError):
        dec.decode(data[: len(data) // 2], Format.GIF, Settings())
    with pytest.raises(InvalidInputError):
        dec.decode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 40, Format.PNG, Settings())


def test_garbage_rejected():
    with pytest.raises(InvalidInputError):
        dec.decode(b"GIF89a" + b"\x00" * 100, Format.GIF, Settings())


def test_frame_limit():
    data = _gif_bytes(n=5)
    s = Settings(limits=Limits(max_frames=4))
    with pytest.raises(LimitExceededError):
        dec.decode(data, Format.GIF, s)


def test_pixel_limit():
    data = _gif_bytes(n=1, size=(64, 64))
    s = Settings(limits=Limits(max_pixels_per_frame=1000))
    with pytest.raises(LimitExceededError):
        dec.decode(data, Format.GIF, s)


def test_total_pixel_limit():
    data = _gif_bytes(n=4, size=(32, 32))  # 1024 pixels/frame * 4 = 4096
    s = Settings(limits=Limits(max_total_pixels=4000))
    with pytest.raises(LimitExceededError):
        dec.decode(data, Format.GIF, s)


def test_decompression_bomb_rejected_as_limit():
    import struct
    import zlib

    def chunk(typ, data):
        c = typ + data
        return struct.pack(">I", len(data)) + c + struct.pack(
            ">I", zlib.crc32(c)
        )

    def png(w, h):
        ihdr = struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)
        idat = zlib.compress(b"\x00" + b"\x00" * (w * 4))
        return (
            b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", idat)
            + chunk(b"IEND", b"")
        )

    # Declared pixels far beyond any configured bound.
    with pytest.raises(LimitExceededError):
        dec.decode(png(100000, 100000), Format.PNG, Settings())
    # Edge beyond max_dimension but with a small body is still rejected.
    with pytest.raises(LimitExceededError):
        dec.decode(png(20000, 2), Format.PNG, Settings())
