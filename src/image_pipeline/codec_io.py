"""Deterministic decode/encode for static and animated formats.

Supported containers: PNG (incl. APNG), GIF, animated WebP, JPEG, BMP, TIFF.

Decode contract (see requirement 1 & 5):
  * every frame is fully decoded up-front and validated: count, geometry,
    duration and disposal/blend metadata are checked against limits before
    the image is accepted - no frame is ever dropped silently;
  * GIF uses a stateless custom container parser (gif_reader) so that raw
    per-frame tiles with transparency and disposal are recovered exactly;
  * all output tiles are sRGB RGBA numpy arrays placed at their true
    canvas offsets.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

from .color import to_srgb_rgba
from .config import PipelineConfig
from .errors import (
    DecodeError,
    FrameCountExceededError,
    FrameMetadataError,
    PixelLimitExceededError,
    UnsupportedFormatError,
)
from .gif_reader import GifParseError, parse_gif
from .apng_reader import APngParseError, parse_apng
from .webp_reader import WebpParseError, parse_webp_meta
from .model import Blending, DecodedImage, Disposal, Frame
from .orientation import (
    apply_orientation,
    extract_orientation,
    tile_offset_after_orientation,
)

SUPPORTED_INPUT_FORMATS: frozenset[str] = frozenset(
    {"PNG", "GIF", "WEBP", "JPEG", "BMP", "TIFF"}
)
SUPPORTED_OUTPUT_FORMATS: frozenset[str] = frozenset({"PNG", "GIF", "WEBP", "JPEG"})

_GIF_DISPOSAL = {
    0: Disposal.NONE,         # unspecified -> leave (matches browsers)
    1: Disposal.NONE,         # do not dispose
    2: Disposal.BACKGROUND,   # restore to background (transparent)
    3: Disposal.PREVIOUS,     # restore previous
}
_APNG_DISPOSAL = {0: Disposal.NONE, 1: Disposal.BACKGROUND, 2: Disposal.PREVIOUS}
_APNG_BLEND = {0: Blending.SOURCE, 1: Blending.OVER}


def _check_duration(value: float, index: int, source_id: str) -> int:
    lim_msg = "frame"
    if value is None or not np.isfinite(value):
        raise FrameMetadataError(
            f"{lim_msg} {index}: missing/non-finite duration", source_id=source_id
        )
    d = int(round(float(value)))
    if d < 0:
        raise FrameMetadataError(
            f"frame {index}: negative duration {value}", source_id=source_id
        )
    return d


def decode(data: bytes, config: PipelineConfig, *, source_id: str = "") -> DecodedImage:
    """Decode and fully validate all frames; never silently drops one."""
    from .validation import sniff_format

    fmt = sniff_format(data)
    if fmt not in SUPPORTED_INPUT_FORMATS:
        raise UnsupportedFormatError(f"unsupported input format: {fmt}", source_id=source_id)
    try:
        if fmt == "GIF":
            return _decode_gif(data, config, source_id)
        if fmt == "WEBP":
            return _decode_webp(data, config, source_id)
        if fmt == "PNG":
            return _decode_png(data, config, source_id)
        return _decode_pil_static(data, fmt, config, source_id)
    except (UnsupportedFormatError, FrameMetadataError, FrameCountExceededError,
            PixelLimitExceededError, DecodeError):
        raise
    except (GifParseError, APngParseError, WebpParseError, OSError, ValueError, EOFError) as exc:
        raise DecodeError(f"{fmt} decode failed: {exc}", source_id=source_id) from exc


def _validate_canvas(w: int, h: int, config: PipelineConfig, source_id: str) -> None:
    if w <= 0 or h <= 0:
        raise FrameMetadataError(f"illegal canvas size {w}x{h}", source_id=source_id)
    if w * h > config.limits.max_canvas_pixels:
        raise PixelLimitExceededError(
            f"canvas {w}x{h} ({w*h}px) exceeds {config.limits.max_canvas_pixels}",
            source_id=source_id,
        )


def _validate_dims(w: int, h: int, config: PipelineConfig, source_id: str) -> None:
    if w <= 0 or h <= 0:
        raise FrameMetadataError(f"illegal tile size {w}x{h}", source_id=source_id)
    if w * h > config.limits.max_frame_pixels:
        raise PixelLimitExceededError(
            f"tile {w}x{h} ({w*h}px) exceeds {config.limits.max_frame_pixels}",
            source_id=source_id,
        )


def _decode_gif(data: bytes, config: PipelineConfig, source_id: str) -> DecodedImage:
    cw, ch, loop, gframes = parse_gif(data)
    _validate_canvas(cw, ch, config, source_id)
    if len(gframes) > config.limits.max_frame_count:
        raise FrameCountExceededError(
            f"{len(gframes)} frames exceed limit {config.limits.max_frame_count}",
            source_id=source_id,
        )
    decoded = DecodedImage(
        loop_count=loop, width=cw, height=ch,
        is_animated=len(gframes) > 1, source_format="GIF",
    )
    for i, gf in enumerate(gframes):
        _validate_dims(gf.width, gf.height, config, source_id)
        if gf.duration_ms > config.limits.max_frame_duration_ms:
            raise FrameMetadataError(
                f"frame {i}: duration {gf.duration_ms}ms exceeds maximum",
                source_id=source_id,
            )
        if gf.x + gf.width > cw or gf.y + gf.height > ch:
            raise FrameMetadataError(
                f"frame {i}: tile outside canvas", source_id=source_id
            )
        # Build a P/RGBA tile via the effective palette + transparent index.
        tile = _gif_indices_to_rgba(gf)
        decoded.frames.append(
            Frame(
                tile=tile,
                duration_ms=gf.duration_ms,
                disposal=_GIF_DISPOSAL.get(gf.disposal, Disposal.NONE),
                # GIF always paints via simple replace semantics; our
                # compositor handles alpha in SOURCE tiles directly.
                blend=Blending.SOURCE,
                offset_x=gf.x,
                offset_y=gf.y,
            )
        )
    return decoded


def _gif_indices_to_rgba(gf) -> np.ndarray:
    h, w = gf.indices.shape
    pal = np.asarray(gf.palette, dtype=np.uint8)
    if len(pal) < 256:
        pal = np.vstack([pal, np.zeros((256 - len(pal), 3), dtype=np.uint8)])
    rgb = pal[gf.indices]
    alpha = np.full((h, w), 255, dtype=np.uint8)
    if gf.transparent_index is not None:
        alpha[gf.indices == gf.transparent_index] = 0
    return np.dstack([rgb, alpha]).astype(np.uint8)


def _decode_webp(data: bytes, config: PipelineConfig, source_id: str) -> DecodedImage:
    try:
        cw, ch, loop, background, metas = parse_webp_meta(data)
    except WebpParseError:
        # Single-frame (possibly lossy) WebP without VP8X/ANMF chunks.
        return _decode_pil_static(data, "WEBP", config, source_id)
    _validate_canvas(cw, ch, config, source_id)
    if len(metas) > config.limits.max_frame_count:
        raise FrameCountExceededError(
            f"{len(metas)} frames exceed limit {config.limits.max_frame_count}",
            source_id=source_id,
        )
    im = Image.open(io.BytesIO(data))
    n = getattr(im, "n_frames", 1)
    if n != len(metas):
        raise DecodeError(
            f"WebP frame count mismatch: container={len(metas)} decoder={n}",
            source_id=source_id,
        )
    decoded = DecodedImage(
        loop_count=loop, width=cw, height=ch, is_animated=len(metas) > 1,
        source_format="WEBP", background=background,
    )
    for i, meta in enumerate(metas):
        _validate_dims(meta.width, meta.height, config, source_id)
        if meta.x + meta.width > cw or meta.y + meta.height > ch:
            raise FrameMetadataError(
                f"frame {i}: tile outside canvas", source_id=source_id
            )
        if meta.duration_ms <= 0:
            raise FrameMetadataError(
                f"frame {i}: non-positive duration {meta.duration_ms}",
                source_id=source_id,
            )
        if meta.duration_ms > config.limits.max_frame_duration_ms:
            raise FrameMetadataError(
                f"frame {i}: duration {meta.duration_ms}ms exceeds maximum",
                source_id=source_id,
            )
        im.seek(i)
        im.load()
        full = np.asarray(to_srgb_rgba(im))
        # libwebp hands back the current frame's pixels; crop to the tile
        # box declared by the ANMF chunk.
        tile = np.ascontiguousarray(
            full[meta.y : meta.y + meta.height, meta.x : meta.x + meta.width]
        )
        decoded.frames.append(
            Frame(
                tile=tile,
                duration_ms=meta.duration_ms,
                disposal=Disposal.BACKGROUND if meta.disposal else Disposal.NONE,
                blend=Blending.SOURCE if meta.blend else Blending.OVER,
                offset_x=meta.x,
                offset_y=meta.y,
            )
        )
    return decoded


def _decode_png(data: bytes, config: PipelineConfig, source_id: str) -> DecodedImage:
    cw, ch, loop, aframes, animated = parse_apng(data)
    _validate_canvas(cw, ch, config, source_id)
    if len(aframes) > config.limits.max_frame_count:
        raise FrameCountExceededError(
            f"{len(aframes)} frames exceed limit {config.limits.max_frame_count}",
            source_id=source_id,
        )
    # Orientation only applies to the static default image (EXIF in PNG).
    base_im = Image.open(io.BytesIO(data))
    orientation = extract_orientation(base_im) if not animated else 1
    canvas_w, canvas_h = cw, ch
    if orientation in (5, 6, 7, 8):
        canvas_w, canvas_h = ch, cw
    decoded = DecodedImage(
        loop_count=loop, width=canvas_w, height=canvas_h,
        is_animated=animated, source_format="PNG",
    )
    for i, af in enumerate(aframes):
        _validate_dims(af.width, af.height, config, source_id)
        if af.x + af.width > cw or af.y + af.height > ch:
            raise FrameMetadataError(
                f"frame {i}: tile outside canvas", source_id=source_id
            )
        duration = af.duration_ms
        if animated:
            if duration <= 0:
                raise FrameMetadataError(
                    f"frame {i}: non-positive duration {duration}",
                    source_id=source_id,
                )
            if duration > config.limits.max_frame_duration_ms:
                raise FrameMetadataError(
                    f"frame {i}: duration {duration}ms exceeds maximum",
                    source_id=source_id,
                )
        tile = af.rgba
        ox, oy = af.x, af.y
        tw, th = af.width, af.height
        if orientation != 1:
            tile, ox, oy, tw, th = _orient_tile(
                tile, ox, oy, tw, th, cw, ch, orientation
            )
        decoded.frames.append(
            Frame(
                tile=tile,
                duration_ms=duration,
                disposal=_APNG_DISPOSAL.get(af.disposal, Disposal.NONE),
                blend=_APNG_BLEND.get(af.blend, Blending.OVER),
                offset_x=ox,
                offset_y=oy,
            )
        )
    return decoded


def _decode_pil_static(
    data: bytes, fmt: str, config: PipelineConfig, source_id: str
) -> DecodedImage:
    """Decode JPEG/BMP/TIFF single-frame image with orientation + color."""
    im = Image.open(io.BytesIO(data))
    orientation = extract_orientation(im)
    w, h = im.size
    cw, ch = (h, w) if orientation in (5, 6, 7, 8) else (w, h)
    _validate_canvas(cw, ch, config, source_id)
    img = apply_orientation(im)
    try:
        img.load()
    except Exception as exc:  # noqa: BLE001
        raise DecodeError(f"{fmt} pixel decode failed: {exc}", source_id=source_id) from exc
    tile = np.asarray(to_srgb_rgba(img))
    decoded = DecodedImage(
        loop_count=0, width=cw, height=ch, is_animated=False, source_format=fmt,
    )
    decoded.frames.append(
        Frame(tile=tile, duration_ms=0, disposal=Disposal.NONE, blend=Blending.OVER)
    )
    return decoded





def encode(
    decoded: DecodedImage,
    composited: list,
    req,
    config: PipelineConfig,
    dst_size: tuple[int, int],
):
    """Deterministically encode normalized+resized composited frames.

    ``composited`` are full-canvas RGBA numpy arrays (already resized to
    ``dst_size``). Output bytes are a pure function of (pixels, durations,
    loop_count, format, encoding params) - no timestamps, no thread order.
    """
    from .model import ProcessResult

    fmt = (req.output_format or decoded.source_format or "PNG").upper()
    if fmt == "JPG":
        fmt = "JPEG"
    if fmt not in SUPPORTED_OUTPUT_FORMATS:
        raise UnsupportedFormatError(f"unsupported output format: {fmt}")
    frames = [Image.fromarray(a, "RGBA") for a in composited]
    durations = [f.duration_ms for f in decoded.frames]
    loop = decoded.loop_count
    animated = len(frames) > 1
    buf = io.BytesIO()

    if fmt == "JPEG":
        if animated:
            # JPEG has no animation: encode the first frame only is illegal
            # per requirement 1 (would drop frames); force a real choice.
            raise UnsupportedFormatError(
                "cannot encode multi-frame image as single-frame JPEG"
            )
        rgb = frames[0].convert("RGB")
        q = req.quality if req.quality is not None else config.jpeg_quality
        rgb.save(buf, format="JPEG", quality=int(q), subsampling=0, optimize=False)
    elif fmt == "PNG":
        if animated:
            frames[0].save(
                buf, format="PNG", save_all=True, append_images=frames[1:],
                duration=durations, loop=loop, compress_level=config.png_compress_level,
                disposal=[0] * len(frames), blend=[0] * len(frames),
                optimize=False,
            )
        else:
            frames[0].save(
                buf, format="PNG", compress_level=config.png_compress_level,
                optimize=False,
            )
    elif fmt == "GIF":
        if not animated:
            pal = _quantize_one(frames[0], config.gif_colors)
            pal.save(buf, format="GIF", optimize=False)
        else:
            quantized = [_quantize_one(fr, config.gif_colors) for fr in frames]
            quantized[0].save(
                buf, format="GIF", save_all=True,
                append_images=quantized[1:], duration=durations, loop=loop,
                optimize=False, disposal=[2] * len(frames),
            )
    elif fmt == "WEBP":
        lossless = (
            req.lossless if req.lossless is not None else config.webp_lossless
        )
        q = req.quality if req.quality is not None else config.webp_quality
        if animated:
            # libwebp requires every frame at canvas size; composited
            # frames already are.
            frames[0].save(
                buf, format="WEBP", save_all=True, append_images=frames[1:],
                duration=durations, loop=loop, quality=int(q),
                lossless=bool(lossless), method=6, minimize_size=False,
                allow_mixed=False,
            )
        else:
            frames[0].save(
                buf, format="WEBP", quality=int(q), lossless=bool(lossless),
                method=6,
            )
    data = buf.getvalue()
    return ProcessResult(
        data=data, output_format=fmt, width=dst_size[0], height=dst_size[1],
        frame_count=len(frames), loop_count=loop if animated else 0,
        durations_ms=durations if animated else [],
        request_id=req.request_id, source_id=req.source_id,
    )


def _quantize_one(im: Image.Image, colors: int) -> Image.Image:
    """Deterministic palette quantization with one transparency index.

    FASTOCTREE is the only RGB(A) quantizer that accepts alpha; we give it
    a fixed 256-entry target and no dithering, so the output is stable.
    Fully (and thresholded semi-) transparent pixels share index 0 and are
    flagged via the GIF ``transparency`` entry - no black halos.
    """
    rgba = im.convert("RGBA")
    alpha = np.asarray(rgba.getchannel("A"))
    q = rgba.quantize(
        colors=colors,
        method=Image.Quantize.FASTOCTREE,
        dither=Image.Dither.NONE,
    )
    # Rebuild RGBA from the quantized palette, then force a single
    # transparency value where source alpha was below threshold.
    rebuilt = np.asarray(q.convert("RGBA")).copy()
    transparent = alpha < 128
    rebuilt[transparent, 3] = 0
    rebuilt[~transparent, 3] = 255
    # Quantize again so all transparent pixels collapse onto one palette
    # entry; median cut on RGB after flattening transparent RGB to black.
    flat_rgb = rebuilt.copy()
    flat_rgb[transparent, 0:3] = 0
    rgb_im = Image.fromarray(flat_rgb[:, :, 0:3], "RGB")
    out = rgb_im.quantize(
        colors=colors,
        method=Image.Quantize.MEDIANCUT,
        dither=Image.Dither.NONE,
    )
    # Determine the index used for the black transparent entries and set it
    # as the palette transparency.
    out_arr = np.array(out)  # writable copy
    if transparent.any():
        # black (0,0,0) index
        pal = np.asarray(out.getpalette(), dtype=np.uint8).reshape(-1, 3)
        black_idx = int(np.flatnonzero((pal == [0, 0, 0]).all(axis=1))[0])
        out_arr[transparent] = black_idx
        out.putdata(out_arr.flatten().tolist())
        out.info["transparency"] = black_idx
    return out
