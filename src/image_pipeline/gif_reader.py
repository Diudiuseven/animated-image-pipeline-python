"""Stateless GIF parser.

Pillow\'s GIF decoder composites frames during seek(), which makes raw
per-frame tiles, durations and disposal hard to recover reliably. This
parser reads the GIF container itself (header, logical screen, NETSCAPE
loop, per-frame GCE + image descriptor) and decodes each frame\'s LZW
indices onto a *fresh* image, so every frame is independent and exact.

Only container parsing is hand-rolled; LZW decompression still uses
Pillow\'s battle-tested \'gif\' decoder.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass
class GifFrame:
    indices: np.ndarray  # uint8, tile HxW, palette indices
    x: int
    y: int
    width: int
    height: int
    duration_ms: int
    disposal: int       # 0..3 GIF raw values
    transparent_index: int | None
    interlace: bool
    palette: list[tuple[int, int, int]]  # effective RGB palette


class GifParseError(Exception):
    pass


def _read_subblocks(fp: io.BytesIO) -> bytearray:
    data = bytearray()
    while True:
        size = fp.read(1)
        if not size:
            raise GifParseError("truncated sub-block terminator")
        n = size[0]
        if n == 0:
            break
        chunk = fp.read(n)
        if len(chunk) != n:
            raise GifParseError("truncated sub-block data")
        data.extend(chunk)
    return data


def _parse_palette(fp: io.BytesIO, size: int) -> list[tuple[int, int, int]]:
    raw = fp.read(size * 3)
    if len(raw) != size * 3:
        raise GifParseError("truncated color table")
    return [(raw[i * 3], raw[i * 3 + 1], raw[i * 3 + 2]) for i in range(size)]


def parse_gif(data: bytes) -> tuple[int, int, int, list[GifFrame]]:
    """Return (canvas_w, canvas_h, loop_count, frames).

    loop_count 0 means infinite; a GIF without NETSCAPE extension yields 1
    (play once). Raises GifParseError on structural problems.
    """
    fp = io.BytesIO(data)
    header = fp.read(6)
    if header not in (b"GIF87a", b"GIF89a"):
        raise GifParseError("not a GIF")
    screen = fp.read(7)
    if len(screen) != 7:
        raise GifParseError("truncated logical screen descriptor")
    cw = int.from_bytes(screen[0:2], "little")
    ch = int.from_bytes(screen[2:4], "little")
    flags = screen[4]
    bg_index = screen[5]
    global_table = None
    if flags & 0x80:
        n = 2 ** ((flags & 0x07) + 1)
        global_table = _parse_palette(fp, n)

    loop_count = 1  # default: play once
    frames: list[GifFrame] = []

    pending_duration = 0  # absent GCE -> 0 delay (we do not invent 100ms)
    pending_disposal = 0
    pending_transparent: int | None = None

    while True:
        marker = fp.read(1)
        if not marker:
            raise GifParseError("truncated: missing trailer")
        m = marker[0]
        if m == 0x3B:  # trailer
            break
        if m == 0x21:  # extension
            label = fp.read(1)
            if not label:
                raise GifParseError("truncated extension label")
            if label[0] == 0xFF:
                # Application extension: FF 0B <11-byte app id> sub-blocks
                app_len_b = fp.read(1)
                if not app_len_b:
                    raise GifParseError("truncated application extension")
                app_id = fp.read(app_len_b[0])
                if len(app_id) != app_len_b[0]:
                    raise GifParseError("truncated application identifier")
                body = bytes(app_id) + bytes(_read_subblocks(fp))
            else:
                body = bytes(_read_subblocks(fp))
            if label[0] == 0xF9 and len(body) >= 4:
                gce_flags = body[0]
                pending_duration = int.from_bytes(body[1:3], "little") * 10
                pending_disposal = (gce_flags >> 2) & 0x07
                if gce_flags & 1:
                    pending_transparent = body[3]
                else:
                    pending_transparent = None
            elif label[0] == 0xFF and body[:11] == b"NETSCAPE2.0":
                # Sub-block data after the 11-byte app id is
                # 0x01 <loops:uint16-le> 0x00 (the 0x03 block-size byte is
                # consumed by the sub-block reader).
                if len(body) >= 14 and body[11] == 0x01:
                    loop_count = int.from_bytes(body[12:14], "little")
        elif m == 0x2C:  # image descriptor
            desc = fp.read(9)
            if len(desc) != 9:
                raise GifParseError("truncated image descriptor")
            x = int.from_bytes(desc[0:2], "little")
            y = int.from_bytes(desc[2:4], "little")
            w = int.from_bytes(desc[4:6], "little")
            h = int.from_bytes(desc[6:8], "little")
            iflags = desc[8]
            table = global_table
            if iflags & 0x80:
                n = 2 ** ((iflags & 0x07) + 1)
                table = _parse_palette(fp, n)
            if table is None:
                raise GifParseError("frame without any color table")
            min_code = fp.read(1)
            if not min_code:
                raise GifParseError("missing LZW minimum code size")
            # args already carry the min code size; the byte stream fed to
            # the decoder starts immediately AFTER that byte.
            lzw_start = fp.tell()
            # Gather the LZW data (skip sub-block framing) length so Pillow
            # can stream it from the original buffer at the given offset.
            lzw_data = bytearray()
            while True:
                size = fp.read(1)
                if not size:
                    raise GifParseError("truncated LZW sub-block size")
                n = size[0]
                if n == 0:
                    break
                chunk = fp.read(n)
                if len(chunk) != n:
                    raise GifParseError("truncated LZW data")
                lzw_data.extend(chunk)
            indices = _lzw_decode(
                data, lzw_start, x, y, w, h, cw, ch, bool(iflags & 0x40),
                min_code[0],
            )
            frames.append(
                GifFrame(
                    indices=indices,
                    x=x,
                    y=y,
                    width=w,
                    height=h,
                    duration_ms=pending_duration,
                    disposal=pending_disposal if pending_disposal else 0,
                    transparent_index=pending_transparent,
                    interlace=bool(iflags & 0x40),
                    palette=table,
                )
            )
            pending_disposal = 0
            pending_transparent = None
            pending_duration = 0
        else:
            raise GifParseError(f"unknown GIF block marker: 0x{m:02x}")

    if not frames:
        raise GifParseError("GIF contains no frames")
    return cw, ch, loop_count, frames


def _lzw_decode(
    data: bytes,
    lzw_start: int,
    x: int,
    y: int,
    w: int,
    h: int,
    cw: int,
    ch: int,
    interlace: bool,
    min_code: int,
) -> np.ndarray:
    """Decode one frame onto a FRESH P canvas and crop to the tile box.

    The native 'gif' decoder lays pixels out with full-canvas row stride
    and paints exactly the extents rectangle, so the decoder target must
    be canvas-sized; we then crop the tile out as palette indices.
    """
    if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > cw or y + h > ch:
        raise GifParseError("illegal frame geometry")
    if not 2 <= min_code <= 12:
        raise GifParseError(f"illegal LZW minimum code size: {min_code}")
    decoder = Image._getdecoder("P", "gif", (min_code, interlace, -1), ())
    core = Image.core.fill("P", (cw, ch), 0)
    decoder.setimage(core, (x, y, x + w, y + h))
    _buf, errcode = decoder.decode(data[lzw_start:])
    if errcode != 0:
        raise GifParseError(
            f"frame LZW decode failed at tile ({x},{y},{w}x{h}), code={errcode}"
        )
    arr = np.asarray(core, dtype=np.uint8).reshape(ch, cw)
    return np.ascontiguousarray(arr[y : y + h, x : x + w])
