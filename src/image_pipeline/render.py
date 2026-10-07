"""Frame compositor: disposal/blend semantics, no black edges or ghosts.

Canvas model (matches APNG / GIF / animated WebP rendering):
  * canvas starts fully transparent;
  * each frame covers a tile at (offset_x, offset_y);
  * blend OVER  -> straight-alpha source-over compositing in the tile box;
  * blend SOURCE-> tile pixels replace the tile box (sparse/dirty update);
  * disposal NONE       -> canvas persists;
  * disposal BACKGROUND -> tile box cleared to transparent after display;
  * disposal PREVIOUS   -> canvas restored to the pre-frame snapshot.

All math is integer-based on unpremultiplied straight alpha, so fully
transparent pixels contribute zero color (no black fringe) and stale
pixels from earlier frames can never survive a BACKGROUND/PREVIOUS reset.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from .model import Blending, DecodedImage, Disposal


def alpha_over(dst: np.ndarray, src: np.ndarray) -> np.ndarray:
    """src over dst for equally-shaped HxWx4 uint8 arrays, integer math."""
    s = src.astype(np.uint32)
    d = dst.astype(np.uint32)
    sa = s[..., 3:4]
    da = d[..., 3:4]
    sr = s[..., 0:1]
    sg = s[..., 1:2]
    sb = s[..., 2:3]
    inv_a = 255 - sa
    # Result alpha: ao = as + ad*(1-as), integer with rounding
    out_a = sa + ((da * inv_a + 127) // 255)
    safe_a = np.where(out_a == 0, 1, out_a)
    # Straight-alpha compositing:
    #   co*ao = cs*as + cd*ad*(1-as)
    # Compute numerator in 1/255 units, then divide by ao (units cancel).
    num = s[..., 0:3] * sa + d[..., 0:3] * ((da * inv_a + 127) // 255)
    out_rgb = (num + (safe_a // 2)) // safe_a
    out = np.concatenate([out_rgb, out_a], axis=2)
    zero_mask = out_a == 0  # HxWx1, broadcasts over channels
    out = np.where(np.broadcast_to(zero_mask, out.shape), 0, out)
    return out.astype(np.uint8)


def _paste_tile(canvas: np.ndarray, frame) -> None:
    h, w = canvas.shape[0], canvas.shape[1]
    th, tw = frame.tile.shape[0], frame.tile.shape[1]
    x0, y0 = frame.offset_x, frame.offset_y
    x1, y1 = x0 + tw, y0 + th
    # clip to canvas
    sx0 = max(0, -x0)
    sy0 = max(0, -y0)
    dx0, dy0 = max(0, x0), max(0, y0)
    dx1, dy1 = min(w, x1), min(h, y1)
    if dx0 >= dx1 or dy0 >= dy1:
        return
    src_tile = frame.tile[sy0 : sy0 + (dy1 - dy0), sx0 : sx0 + (dx1 - dx0)]
    region = canvas[dy0:dy1, dx0:dx1]
    if frame.blend is Blending.SOURCE:
        canvas[dy0:dy1, dx0:dx1] = src_tile
    else:
        canvas[dy0:dy1, dx0:dx1] = alpha_over(region, src_tile)


def _clear_tile(canvas: np.ndarray, frame) -> None:
    h, w = canvas.shape[0], canvas.shape[1]
    th, tw = frame.tile.shape[0], frame.tile.shape[1]
    x0, y0 = frame.offset_x, frame.offset_y
    dx0, dy0 = max(0, x0), max(0, y0)
    dx1 = min(w, x0 + tw)
    dy1 = min(h, y0 + th)
    if dx0 < dx1 and dy0 < dy1:
        canvas[dy0:dy1, dx0:dx1] = 0


def composite_frames(decoded: DecodedImage) -> list[np.ndarray]:
    """Render every displayed frame to a full-canvas HxWx4 RGBA uint8 image."""
    if decoded.width <= 0 or decoded.height <= 0:
        return []
    canvas = np.zeros((decoded.height, decoded.width, 4), dtype=np.uint8)
    result: list[np.ndarray] = []
    for frame in decoded.frames:
        saved = None
        if frame.disposal is Disposal.PREVIOUS:
            saved = canvas.copy()
        _paste_tile(canvas, frame)
        result.append(canvas.copy())
        if frame.disposal is Disposal.BACKGROUND:
            _clear_tile(canvas, frame)
        elif frame.disposal is Disposal.PREVIOUS:
            canvas = saved
        # NONE / UNSPECIFIED -> leave canvas
    return result


def composite_to_pillow(decoded: DecodedImage) -> list[Image.Image]:
    return [Image.fromarray(a, "RGBA") for a in composite_frames(decoded)]
