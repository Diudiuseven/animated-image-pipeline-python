"""Turn a validated :class:`DecodedImage` into a canonical ImageDocument.

The container decoders in Pillow 12 already apply disposal / blending /
sub-frame offsets (GIF, APNG) or libwebp composition (animated WebP), so each
``seek(i); load()`` presents the *fully composited* canvas.  Here we:

1. normalize every composited frame to straight sRGB RGBA;
2. apply the single EXIF orientation value to every frame;
3. carry over the validated per-frame durations and loop count.

No frame is dropped, duplicated or reordered.
"""

from __future__ import annotations

from ..models import Frame, ImageDocument
from .color import to_srgb_rgba
from .orientation import apply_orientation


def composite(decoded) -> ImageDocument:
    """Build fully composited, oriented, sRGB straight-RGBA canvas frames."""
    im = decoded.image
    frames: list[Frame] = []
    for meta in decoded.frames_meta:
        im.seek(meta.index)
        im.load()
        # Work on a per-frame copy so the seekable source is never mutated and
        # each frame is independent of the decoder's internal state.
        frame_im = im.copy()
        pixels = to_srgb_rgba(frame_im)
        pixels = apply_orientation(pixels, decoded.orientation)
        frames.append(Frame(pixels=pixels, duration_ms=meta.duration_ms))
    return ImageDocument(
        frames=tuple(frames),
        loop=decoded.loop,
        source_format=decoded.fmt,
    )
