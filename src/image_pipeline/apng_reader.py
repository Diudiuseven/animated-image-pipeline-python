"""Stateless APNG parser.

Pillow\'s APNG seek() accumulates compositing state and rescales
semi-transparent frame pixels, so exact per-frame recovery is done here
without it:

  * IHDR  -> canvas size / pixel format;
  * acTL  -> frame count + loop count;
  * fcTL  -> per-frame width/height/offset, delay, disposal, blend;
  * IDAT  -> default image (frame 0); fdAT -> later frame image data.

PNG zlib inflate and row filtering are implemented directly (8-bit
RGBA/RGB/Gray/GA), giving unpremultiplied raw frame pixels.
"""

from __future__ import annotations

import io
import struct
import zlib
from dataclasses import dataclass

import numpy as np
from PIL import Image

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class APngParseError(Exception):
    pass


@dataclass
class APngFrame:
    width: int
    height: int
    x: int
    y: int
    duration_ms: int
    disposal: int   # 0 NONE, 1 BACKGROUND, 2 PREVIOUS
    blend: int      # 0 SOURCE, 1 OVER
    rgba: np.ndarray  # uint8 HxWx4, unpremultiplied raw frame


def _iter_chunks(data: bytes):
    if data[:8] != PNG_SIGNATURE:
        raise APngParseError("not a PNG")
    pos = 8
    seen_iend = False
    while pos + 8 <= len(data):
        (length,) = struct.unpack(">I", data[pos : pos + 4])
        ctype = data[pos + 4 : pos + 8]
        b0 = pos + 8
        b1 = b0 + length
        if b1 + 4 > len(data):
            raise APngParseError(f"truncated chunk {ctype!r}")
        body = data[b0:b1]
        expected = struct.unpack(">I", data[b1 : b1 + 4])[0]
        if zlib.crc32(ctype + body) & 0xFFFFFFFF != expected:
            raise APngParseError(f"CRC mismatch in chunk {ctype!r}")
        yield ctype, body
        pos = b1 + 4
        if ctype == b"IEND":
            seen_iend = True
            break
    if not seen_iend:
        raise APngParseError("missing IEND")


def _unfilter(raw: bytes, width: int, height: int, channels: int) -> np.ndarray:
    bpp = channels
    stride = width * bpp + 1
    if len(raw) != stride * height:
        raise APngParseError(
            f"decompressed size {len(raw)} != expected {stride*height}"
        )
    out = np.zeros((height, width, bpp), dtype=np.uint8)
    prev = np.zeros(width * bpp, dtype=np.uint8)
    pos = 0
    for y in range(height):
        ftype = raw[pos]
        line = np.frombuffer(raw[pos + 1 : pos + stride], dtype=np.uint8).copy()
        pos += stride
        if ftype == 1:
            for x in range(bpp, len(line)):
                line[x] = (int(line[x]) + int(line[x - bpp])) & 0xFF
        elif ftype == 2:
            line = (line.astype(np.uint16) + prev.astype(np.uint16)).astype(np.uint8)
        elif ftype == 3:
            for x in range(len(line)):
                a = int(line[x - bpp]) if x >= bpp else 0
                line[x] = (int(line[x]) + ((a + int(prev[x])) // 2)) & 0xFF
        elif ftype == 4:
            for x in range(len(line)):
                a = int(line[x - bpp]) if x >= bpp else 0
                b = int(prev[x])
                c = int(prev[x - bpp]) if x >= bpp else 0
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                pr = a if pa <= pb and pa <= pc else (b if pb <= pc else c)
                line[x] = (int(line[x]) + pr) & 0xFF
        elif ftype != 0:
            raise APngParseError(f"unsupported PNG filter type {ftype}")
        out[y] = line.reshape(width, bpp)
        prev = line
    return out


def _expand_channels(arr: np.ndarray, color_type: int) -> np.ndarray:
    """Expand gray/RGB/GA 8-bit arrays to RGBA."""
    if color_type == 6:
        return arr
    h, w, c = arr.shape
    if color_type == 2:  # RGB
        alpha = np.full((h, w, 1), 255, dtype=np.uint8)
        return np.concatenate([arr, alpha], axis=2)
    if color_type == 4:  # gray+alpha
        gray = arr[:, :, 0:1]
        return np.concatenate([np.repeat(gray, 3, axis=2), arr[:, :, 1:2]], axis=2)
    if color_type == 0:  # gray
        gray = np.repeat(arr, 3, axis=2)
        alpha = np.full((h, w, 1), 255, dtype=np.uint8)
        return np.concatenate([gray, alpha], axis=2)
    raise APngParseError(f"unsupported PNG color type {color_type}")


_CHANNELS = {0: 1, 2: 3, 4: 2, 6: 4}


def parse_apng(data: bytes) -> tuple[int, int, int, list[APngFrame], bool]:
    """Return (canvas_w, canvas_h, loop_count, frames, is_animated)."""
    cw = ch = 0
    color_type = bit_depth = interlace = None
    loop = 0
    num_frames: int | None = None
    is_animated = False
    pending_ctl: dict | None = None
    image_data: list[bytes] = []
    source: str | None = None
    frames: list[APngFrame] = []

    def flush():
        nonlocal image_data, source, pending_ctl
        if pending_ctl is None:
            return
        m = pending_ctl
        channels = _CHANNELS[color_type]
        raw = zlib.decompress(b"".join(image_data))
        arr = _unfilter(raw, m["width"], m["height"], channels)
        rgba = _expand_channels(arr, color_type)
        frames.append(
            APngFrame(
                m["width"], m["height"], m["x"], m["y"], m["duration_ms"],
                m["disposal"], m["blend"], rgba,
            )
        )
        image_data = []
        source = None
        pending_ctl = None

    for ctype, body in _iter_chunks(data):
        if ctype == b"IHDR":
            cw, ch, bit_depth, color_type, _comp, _filt, interlace = struct.unpack(
                ">IIBBBBB", body
            )
            if interlace != 0:
                raise APngParseError("interlaced PNG not supported")
            if bit_depth != 8:
                raise APngParseError(f"only 8-bit PNG supported, got depth {bit_depth}")
            if color_type not in _CHANNELS:
                raise APngParseError(f"unsupported PNG color type {color_type}")
        elif ctype == b"acTL":
            is_animated = True
            num_frames, loop = struct.unpack(">II", body[:8])
        elif ctype == b"fcTL":
            flush()
            (seq, fw, fh, fx, fy, dn, dd, dispose_op, blend_op) = struct.unpack(
                ">IIIIIHHBB", body[:26]
            )
            denominator = dd if dd else 100
            pending_ctl = {
                "seq": seq, "width": fw, "height": fh, "x": fx, "y": fy,
                "duration_ms": int(round(dn * 1000 / denominator)),
                "disposal": dispose_op, "blend": blend_op,
            }
        elif ctype == b"IDAT":
            if pending_ctl is not None:
                image_data.append(body)
                source = "IDAT"
            else:
                # static PNG (no preceding fcTL)
                image_data.append(body)
                source = "IDAT"
        elif ctype == b"fdAT":
            if pending_ctl is None:
                raise APngParseError("fdAT without fcTL")
            image_data.append(body[4:])  # strip 4-byte sequence number
            source = "fdAT"
        elif ctype == b"IEND":
            if is_animated:
                flush()

    if not is_animated:
        channels = _CHANNELS[color_type]
        raw = zlib.decompress(b"".join(image_data))
        rgba = _expand_channels(_unfilter(raw, cw, ch, channels), color_type)
        frames.append(APngFrame(cw, ch, 0, 0, 0, 0, 0, rgba))
        return cw, ch, 0, frames, False

    if num_frames is not None and len(frames) != num_frames:
        raise APngParseError(
            f"acTL declares {num_frames} frames but {len(frames)} image blocks found"
        )
    if not frames:
        raise APngParseError("APNG contains no frames")
    return cw, ch, loop, frames, True
