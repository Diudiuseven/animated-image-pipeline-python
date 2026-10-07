"""WebP container frame-metadata parser.

Pillow does not expose per-frame offset, disposal or blend for animated
WebP. Those live in RIFF ANMF chunks; we parse the container for metadata
and still use Pillow/libwebp for pixel decoding.

ANMF frame header (little endian):
    frame_x     24 bits (3 bytes, low 24)
    frame_y     24 bits
    width-1     24 bits
    height-1    24 bits
    duration    24 bits
    flags        8 bits:
       bit 0 disposal: 0 NONE (keep canvas), 1 BACKGROUND (clear)
       bit 1 blend:    0 use alpha blending (OVER), 1 do not blend (SOURCE)
ANIM chunk: bg color RGBA + 16-bit loop count (0 = infinite).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass


class WebpParseError(Exception):
    pass


@dataclass
class WebpFrameMeta:
    x: int
    y: int
    width: int
    height: int
    duration_ms: int
    disposal: int  # 0 NONE, 1 BACKGROUND
    blend: int     # 0 OVER, 1 SOURCE


def _iter_riff(data: bytes):
    if data[:4] != b"RIFF" or data[8:12] != b"WEBP":
        raise WebpParseError("not a WebP/RIFF file")
    pos = 12
    end = len(data)
    while pos + 8 <= end:
        fourcc = data[pos : pos + 4]
        (length,) = struct.unpack("<I", data[pos + 4 : pos + 8])
        body = data[pos + 8 : pos + 8 + length]
        if len(body) != length:
            raise WebpParseError(f"truncated chunk {fourcc!r}")
        yield fourcc, body
        pos += 8 + length + (length & 1)  # RIFF pads odd chunks


def parse_webp_meta(data: bytes) -> tuple[int, int, int, tuple[int, int, int, int], list[WebpFrameMeta]]:
    """Return (canvas_w, canvas_h, loop, background, frame metas)."""
    canvas_w = canvas_h = 0
    loop = 0
    background = (0, 0, 0, 0)
    metas: list[WebpFrameMeta] = []
    for fourcc, body in _iter_riff(data):
        if fourcc == b"VP8X":
            flags = body[0]
            canvas_w = int.from_bytes(body[4:7], "little") + 1
            canvas_h = int.from_bytes(body[7:10], "little") + 1
        elif fourcc == b"ANIM":
            background = tuple(body[0:4])
            loop = struct.unpack("<H", body[4:6])[0]
        elif fourcc == b"ANMF":
            x = int.from_bytes(body[0:3], "little") & 0x3FFF
            y = int.from_bytes(body[3:6], "little") & 0x3FFF
            fw = (int.from_bytes(body[6:9], "little") & 0x3FFF) + 1
            fh = (int.from_bytes(body[9:12], "little") & 0x3FFF) + 1
            duration = int.from_bytes(body[12:15], "little")
            flags = body[15]
            metas.append(
                WebpFrameMeta(
                    x=x,
                    y=y,
                    width=fw,
                    height=fh,
                    duration_ms=duration,
                    disposal=flags & 1,
                    blend=(flags >> 1) & 1,
                )
            )
    if canvas_w == 0 or canvas_h == 0:
        raise WebpParseError("missing VP8X canvas dimensions")
    if not metas:
        raise WebpParseError("animated WebP has no ANMF frames")
    return canvas_w, canvas_h, loop, background, metas
