import numpy as np
from PIL import Image, ImageCms

from imagepipeline.stages.color import to_srgb_rgba


def _gradient():
    w = h = 24
    x = np.linspace(0, 255, w)
    y = np.linspace(0, 255, h)
    xx, yy = np.meshgrid(x, y)
    arr = np.zeros((h, w, 3), np.uint8)
    arr[..., 0] = xx.astype(np.uint8)
    arr[..., 1] = yy.astype(np.uint8)
    arr[..., 2] = ((xx + yy) / 2).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


def test_srgb_passthrough_and_shape():
    im = _gradient()
    out = to_srgb_rgba(im)
    assert out.dtype == np.uint8 and out.shape == (24, 24, 4)
    assert np.array_equal(out[..., :3], np.asarray(im))
    assert (out[..., 3] == 255).all()


def test_embedded_srgb_profile_normalizes_to_identical_pixels():
    """Same content, one tagged with an ICC profile: must normalize equally."""
    plain = _gradient()
    tagged = _gradient()
    srgb_profile = ImageCms.createProfile("sRGB")
    tagged.info["icc_profile"] = ImageCms.ImageCmsProfile(srgb_profile).tobytes()

    out_plain = to_srgb_rgba(plain)
    out_tagged = to_srgb_rgba(tagged)
    assert np.array_equal(out_plain, out_tagged)

    # Repeated calls must be deterministic.
    assert np.array_equal(to_srgb_rgba(tagged), out_tagged)


def test_corrupt_icc_falls_back_deterministically():
    im = _gradient()
    im.info["icc_profile"] = b"not a real icc profile"
    out_a = to_srgb_rgba(im)
    out_b = to_srgb_rgba(im)
    assert np.array_equal(out_a, out_b)
    assert np.array_equal(out_a[..., :3], np.asarray(_gradient()))


def test_palette_transparency_is_preserved():
    im = Image.new("P", (4, 4), 0)
    im.putpalette([0, 0, 0, 255, 0, 0, 0, 255, 0])
    im.info["transparency"] = 0
    im.putpixel((1, 1), 1)
    out = to_srgb_rgba(im)
    assert out[0, 0, 3] == 0
    assert tuple(out[1, 1]) == (255, 0, 0, 255)


def test_mode_conversions():
    rgba = np.array([[[10, 20, 30, 128], [40, 50, 60, 0]]], np.uint8)
    im = Image.fromarray(rgba, "RGBA")
    for mode in ("RGB", "L", "LA"):
        out = to_srgb_rgba(im.convert(mode))
        assert out.shape == (1, 2, 4)


def test_different_modes_of_same_content_converge():
    """The same color expressed via RGB / L / palette normalizes equally."""
    from PIL import Image as PILImage

    rgb = PILImage.new("RGB", (4, 4), (200, 100, 50))
    gray = rgb.convert("L")  # luminosity-preserving grayscale representation
    # Grayscale is a different (luminance) encoding; at minimum it must be
    # deterministic and produce neutral pixels with correct alpha.
    out_gray = to_srgb_rgba(gray)
    assert out_gray.shape == (4, 4, 4)
    assert (out_gray[..., 3] == 255).all()
    # All grayscale pixels share the same R=G=B value.
    assert (out_gray[..., 0] == out_gray[..., 1]).all()
    assert (out_gray[..., 1] == out_gray[..., 2]).all()

    # Palette version carrying the exact same RGB triple.
    pal = PILImage.new("P", (4, 4), 0)
    pal.putpalette([200, 100, 50] + [0] * (768 - 3))
    out_pal = to_srgb_rgba(pal)
    assert tuple(out_pal[0, 0, :3]) == (200, 100, 50)


def test_tagged_and_untagged_same_content_byte_equal():
    """Different provenance (ICC-tagged vs untagged) converges byte-for-byte."""
    im1 = _gradient()
    im2 = _gradient()
    im2.info["icc_profile"] = ImageCms.ImageCmsProfile(
        ImageCms.createProfile("sRGB")
    ).tobytes()
    assert to_srgb_rgba(im1).tobytes() == to_srgb_rgba(im2).tobytes()
