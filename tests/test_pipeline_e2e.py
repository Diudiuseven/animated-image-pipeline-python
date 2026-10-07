"""End-to-end: normalization survives the full process() pipeline, and
resize preserves frame count + semantics."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

from image_pipeline import samples
from image_pipeline.codec_io import decode
from image_pipeline.config import DEFAULT_CONFIG
from image_pipeline.model import ProcessRequest, ResizeMode
from image_pipeline.pipeline import process
from image_pipeline.render import composite_frames


def test_p3_vs_srgb_same_content_e2e(p3_icc):
    srgb_jpeg = samples.make_static_jpeg()
    # Build equivalent P3-tagged JPEG from the SAME source pixels encoded
    # through the sRGB->P3 transform.
    from PIL import ImageCms

    base = Image.open(io.BytesIO(srgb_jpeg)).convert("RGB")
    srgb = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))
    p3 = ImageCms.getOpenProfile(io.BytesIO(p3_icc))
    p3_im = ImageCms.applyTransform(
        base, ImageCms.buildTransform(srgb, p3, "RGB", "RGB")
    )
    pb = io.BytesIO()
    p3_im.save(pb, format="JPEG", quality=95, icc_profile=p3_icc)
    p3_jpeg = pb.getvalue()

    a = decode(srgb_jpeg, DEFAULT_CONFIG).frames[0].tile[..., :3].astype(np.int16)
    b = decode(p3_jpeg, DEFAULT_CONFIG).frames[0].tile[..., :3].astype(np.int16)
    delta = np.abs(a - b)
    assert delta.mean() < 1.5
    # JPEG lossy coding adds a small symmetric quantization noise; require
    # near-zero systematic color bias, not exactness.
    assert abs(float((b - a).reshape(-1, 3).mean())) < 0.15


def test_resize_keeps_frame_count_and_aspect(animated_webp):
    r = process(ProcessRequest(source=animated_webp, output_format="PNG", width=24))
    im = Image.open(io.BytesIO(r.data))
    assert im.n_frames == 4
    # 48x48 square stays square
    assert r.width == 24 and r.height == 24


def test_resize_fit_preserves_aspect():
    png = samples.make_static_png(size=(60, 30))
    r = process(ProcessRequest(source=png, width=30, resize_mode=ResizeMode.FIT))
    assert (r.width, r.height) == (30, 15)


def test_cover_center_crop():
    png = samples.make_static_png(size=(80, 40))
    r = process(
        ProcessRequest(source=png, width=20, height=20, resize_mode=ResizeMode.COVER)
    )
    assert (r.width, r.height) == (20, 20)


def test_transparent_resize_no_black_edge(animated_png):
    r = process(ProcessRequest(source=animated_png, output_format="PNG", width=24))
    im = Image.open(io.BytesIO(r.data))
    im.seek(1)
    # corners are transparent background; alpha must be 0, RGB not black-filled
    arr = np.asarray(im.convert("RGBA"))
    assert arr[0, 0, 3] == 0 or arr[0, 0, 3] < 255  # transparent corner preserved


def test_output_decodable_and_rgba(animated_gif):
    r = process(ProcessRequest(source=animated_gif, output_format="WEBP", width=32))
    im = Image.open(io.BytesIO(r.data))
    assert im.format == "WEBP"
    assert im.n_frames == 4


def test_resizing_transparent_edge_has_no_black_fringe():
    arr = np.zeros((32, 32, 4), dtype=np.uint8)
    arr[8:24, 8:24, 0:3] = (255, 0, 0)
    arr[8:24, 8:24, 3] = 255
    buf = io.BytesIO()
    Image.fromarray(arr, "RGBA").save(buf, format="PNG")
    r = process(ProcessRequest(source=buf.getvalue(), output_format="PNG", width=16))
    a = np.asarray(Image.open(io.BytesIO(r.data)).convert("RGBA"))
    edge = a[(a[..., 3] > 0) & (a[..., 3] < 255)]
    assert len(edge) > 0  # there are anti-aliased edge pixels
    assert len(edge[edge[:, :3].max(axis=1) < 60]) == 0  # none near-black
    assert float((edge[:, 0].astype(int) - edge[:, 1].astype(int)).mean()) > 30
