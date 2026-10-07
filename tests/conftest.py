"""Shared pytest fixtures: deterministic sample images + ICC profiles.

Nothing here touches the network or external services; every byte is
generated in-process and fixed.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest
from PIL import Image, ImageCms

from image_pipeline import samples
from image_pipeline.config import Limits, PipelineConfig


D50 = np.array([0.9642, 1.0, 0.8249])
P3_PRIMARIES = ((0.708, 0.292), (0.170, 0.797), (0.131, 0.046))


def _xyz_type(v) -> bytes:
    return b"XYZ " + b"\x00" * 4 + b"".join(
        struct.pack(">i", int(round(float(c) * 65536))) for c in v
    )


def _primaries_to_d50_xyz(primaries, white=D50) -> np.ndarray:
    (rx, ry), (gx, gy), (bx, by) = primaries
    P = np.array(
        [
            [rx / ry, gx / gy, bx / by],
            [1, 1, 1],
            [(1 - rx - ry) / ry, (1 - gx - gy) / gy, (1 - bx - by) / by],
        ]
    )
    return P @ np.diag(np.linalg.solve(P, white))


def make_p3_icc() -> bytes:
    """Build a valid wide-gamut (Display-P3 primaries, sRGB TRC, D65) ICC.

    Derived from lcms' canonical sRGB profile by replacing only the three
    20-byte XYZ colorant tags with D50-adapted P3 colorants; chromatic
    adaptation (chad) and the sRGB parametric tone curve are retained.
    """
    srgb = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    n = struct.unpack(">I", srgb[128:132])[0]
    tag_at: dict[bytes, tuple[int, int]] = {}
    for i in range(n):
        o = 132 + i * 12
        tag_at[srgb[o : o + 4]] = struct.unpack(">II", srgb[o + 4 : o + 12])
    M = _primaries_to_d50_xyz(P3_PRIMARIES)
    out = bytearray(srgb)
    for name, col in ((b"rXYZ", M[:, 0]), (b"gXYZ", M[:, 1]), (b"bXYZ", M[:, 2])):
        off, size = tag_at[name]
        assert size == 20
        out[off : off + 20] = _xyz_type([float(x) for x in col])
    return bytes(out)


@pytest.fixture(scope="session")
def p3_icc() -> bytes:
    return make_p3_icc()


@pytest.fixture
def tight_config() -> PipelineConfig:
    return PipelineConfig(
        limits=Limits(
            max_frame_count=3,
            max_frame_pixels=4096,
            max_canvas_pixels=4096,
            max_bytes=1 << 20,
            max_decode_workers=2,
            max_encode_workers=2,
            max_inflight=8,
            queue_capacity=4,
        )
    )


@pytest.fixture
def animated_gif() -> bytes:
    return samples.make_animated_gif(frames=4, size=(48, 48))


@pytest.fixture
def animated_webp() -> bytes:
    return samples.make_animated_webp(frames=4, size=(48, 48))


@pytest.fixture
def animated_png() -> bytes:
    return samples.make_animated_png(frames=4, size=(48, 48))
