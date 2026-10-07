"""Color normalization: every frame normalized to sRGB, 8-bit RGBA.

Determinism rules:
  * Images without an ICC profile are already assumed sRGB (the de-facto
    web default); no hidden conversion is applied.
  * Embedded ICC profiles are converted to a single canonical sRGB profile
    with relative-colorimetric intent + black-point compensation, giving
    identical output pixels for identical content regardless of the
    source color space.
  * Alpha is always preserved; conversion happens on the color channels
    only, so transparent regions never turn black.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from PIL import Image, ImageCms

# ICC color space signatures (bytes 16:20 of an ICC profile).
_ICC_RGB = b"RGB "
_ICC_GRAY = b"GRAY"
_ICC_CMYK = b"CMYK"


@lru_cache(maxsize=1)
def _sRGB_profile_obj() -> ImageCms.ImageCmsProfile:
    return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))


@lru_cache(maxsize=1)
def srgb_icc_profile() -> bytes:
    """Canonical sRGB ICC profile bytes (used to tag output when wanted)."""
    return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


def _is_srgb_profile(icc_bytes: bytes) -> bool:
    """Decide whether an embedded profile is already canonical sRGB.

    Profile descriptions are not trustworthy (a modified matrix profile can
    keep the original description), so we compare the actual colorants and
    tone curve against the sRGB reference instead of relying on text.
    """
    try:
        cand = ImageCms.getOpenProfile(__io(icc_bytes))
        ref = _sRGB_profile_obj()
        if not getattr(cand.profile, "is_matrix_shaper", False):
            return False
        # colorants report ((X,Y,Z),(x,y,Y)); compare the XYZ triples and
        # the media white point XYZ.
        for attr in (
            "red_colorant",
            "green_colorant",
            "blue_colorant",
            "media_white_point",
        ):
            if not _colorant_close(getattr(cand.profile, attr), getattr(ref.profile, attr)):
                return False
        return True
    except Exception:  # noqa: BLE001
        return False


def _colorant_close(a, b, tol: float = 1e-5) -> bool:
    try:
        va = [float(x) for x in tuple(a)[0]]
        vb = [float(x) for x in tuple(b)[0]]
    except Exception:  # noqa: BLE001
        return False
    return len(va) == len(vb) and all(abs(x - y) <= tol for x, y in zip(va, vb))


def __io(b: bytes):
    import io

    return io.BytesIO(b)


def _convert_icc_to_srgb(rgba: Image.Image, icc_bytes: bytes) -> Image.Image:
    """Apply ICC -> sRGB on color channels, alpha untouched."""
    src = ImageCms.getOpenProfile(__io(icc_bytes))
    dst = _sRGB_profile_obj()
    sig = bytes(icc_bytes[16:20]) if len(icc_bytes) >= 20 else b""
    if sig == _ICC_GRAY:
        in_mode = "L"
    elif sig == _ICC_CMYK:
        in_mode = "RGB"  # callers pre-convert CMYK->RGB
    else:
        in_mode = "RGB"
    has_alpha = rgba.mode == "RGBA"
    if has_alpha:
        alpha = rgba.getchannel("A")
        rgb = rgba.convert("RGB")
        t = ImageCms.buildTransform(
            src, dst, "RGB", "RGB",
            renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
            flags=ImageCms.Flags.BLACKPOINTCOMPENSATION,
        )
        out = ImageCms.applyTransform(rgb, t)
        out.putalpha(alpha)
        return out
    base = rgba if rgba.mode == in_mode else rgba.convert(in_mode)
    t = ImageCms.buildTransform(
        src, dst, in_mode, "RGB",
        renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
        flags=ImageCms.Flags.BLACKPOINTCOMPENSATION,
    )
    return ImageCms.applyTransform(base, t).convert("RGBA")


def to_srgb_rgba(img: Image.Image) -> Image.Image:
    """Convert any Pillow image (any mode, with/without ICC) to RGBA sRGB.

    The result has no embedded ICC profile: pixels are canonical sRGB.
    Palette/LA/P transparency is expanded without compositing over black.
    """
    im = img
    icc = im.info.get("icc_profile")

    if im.mode == "CMYK":
        # Pillow's CMYK->RGB is deterministic; apply embedded profile after
        # only if present, otherwise the path below handles raw pixels.
        im = im.convert("RGB")
    elif im.mode in ("P", "PA"):
        # preserve transparency from palette tRNS / PA alpha
        im = im.convert("RGBA")
    elif im.mode == "L":
        im = im.convert("RGBA")
    elif im.mode == "LA":
        im = im.convert("RGBA")
    elif im.mode in ("RGB", "RGBA"):
        pass
    elif im.mode in ("I", "F", "I;16"):
        im = im.convert("RGBA")
    elif im.mode == "YCbCr":
        im = im.convert("RGB")
    else:
        im = im.convert("RGBA")

    if icc and not _is_srgb_profile(icc):
        try:
            im = _convert_icc_to_srgb(im if im.mode == "RGBA" else im.convert("RGBA"), icc)
        except ImageCms.PyCMSError:
            # Unusable/embedded profile: fall back to treating pixels as sRGB
            # rather than failing; result remains deterministic.
            im = im.convert("RGBA")
    else:
        im = im.convert("RGBA")
    return im


def normalize_array(arr: np.ndarray, icc_profile: bytes | None) -> np.ndarray:
    """Apply ICC->sRGB to an HxWx3/4 uint8 array; returns HxWx4 RGBA uint8."""
    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        raise ValueError("expected HxWx3 or HxWx4 uint8 array")
    mode = "RGBA" if arr.shape[2] == 4 else "RGB"
    img = Image.fromarray(arr.astype(np.uint8), mode)
    if icc_profile:
        img.info["icc_profile"] = icc_profile
    return np.asarray(to_srgb_rgba(img))
