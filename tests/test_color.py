"""Requirement 2: different color spaces of one scene normalize identically,
deterministically, and alpha is preserved (no black fringe)."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageCms

from image_pipeline.color import srgb_icc_profile, to_srgb_rgba


def test_no_icc_equals_embedded_srgb():
    rng = np.random.RandomState(0)
    arr = (rng.rand(32, 32, 3) * 255).astype(np.uint8)
    a = np.asarray(to_srgb_rgba(Image.fromarray(arr, "RGB")))
    im = Image.fromarray(arr, "RGB")
    im.info["icc_profile"] = srgb_icc_profile()
    b = np.asarray(to_srgb_rgba(im))
    assert np.array_equal(a, b)


def test_different_color_spaces_same_scene_normalize_consistently(p3_icc):
    """Same in-gamut scene tagged sRGB vs P3 -> equal sRGB pixels."""
    srgb = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))
    p3 = ImageCms.getOpenProfile(io.BytesIO(p3_icc))
    # Mid-tone content safely inside both gamuts (no clipping).
    rng = np.random.RandomState(7)
    base = (40 + rng.rand(64, 64, 3) * 170).astype(np.uint8)
    scene = Image.fromarray(base, "RGB")
    p3_encoded = ImageCms.applyTransform(
        scene, ImageCms.buildTransform(srgb, p3, "RGB", "RGB")
    )
    assert int(np.asarray(p3_encoded).min()) >= 2
    assert int(np.asarray(p3_encoded).max()) <= 253  # stayed in-gamut
    p3_encoded.info["icc_profile"] = p3_icc

    srgb_im = scene.copy()
    srgb_im.info["icc_profile"] = srgb_icc_profile()

    a = np.asarray(to_srgb_rgba(srgb_im))[..., :3].astype(np.int16)
    b = np.asarray(to_srgb_rgba(p3_encoded))[..., :3].astype(np.int16)
    delta = np.abs(a - b)
    # One deterministic lcms hop each way; allow only sub-LSB quantization
    # noise and require zero systematic bias.
    assert delta.mean() < 1.0
    assert abs(float((b - a).reshape(-1, 3).mean())) < 0.05


def test_normalizer_matches_single_lcms_transform(p3_icc):
    srgb = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))
    p3 = ImageCms.getOpenProfile(io.BytesIO(p3_icc))
    rng = np.random.RandomState(31)
    arr = (rng.rand(48, 48, 3) * 255).astype(np.uint8)
    im = Image.fromarray(arr, "RGB")
    im.info["icc_profile"] = p3_icc
    got = np.asarray(to_srgb_rgba(im))[..., :3]
    ref = np.asarray(
        ImageCms.applyTransform(
            Image.fromarray(arr, "RGB"),
            ImageCms.buildTransform(
                p3,
                srgb,
                "RGB",
                "RGB",
                renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
                flags=ImageCms.Flags.BLACKPOINTCOMPENSATION,
            ),
        )
    )
    assert np.array_equal(got, ref)


def test_fully_transparent_pixels_never_become_black():
    arr = np.zeros((8, 8, 4), dtype=np.uint8)
    arr[..., 0:3] = (10, 20, 30)
    arr[..., 3] = 0  # fully transparent
    out = np.asarray(to_srgb_rgba(Image.fromarray(arr, "RGBA")))
    assert (out[..., 3] == 0).all()


def test_deterministic_repeat(p3_icc):
    rng = np.random.RandomState(9)
    arr = (rng.rand(20, 20, 3) * 255).astype(np.uint8)
    outs = []
    for _ in range(4):
        im = Image.fromarray(arr, "RGB")
        im.info["icc_profile"] = p3_icc
        outs.append(np.asarray(to_srgb_rgba(im)))
    for o in outs[1:]:
        assert np.array_equal(outs[0], o)
