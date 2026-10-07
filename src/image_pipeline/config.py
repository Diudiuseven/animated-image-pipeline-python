"""Configuration: hard resource bounds and encoding defaults.

All bounds are process-local and explicit; nothing is discovered from
external services.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Limits:
    # Decode-side bounds
    max_bytes: int = 64 * 1024 * 1024  # max accepted encoded input size
    max_frame_count: int = 500
    max_frame_pixels: int = 64_000_000  # width*height per frame after decode
    max_canvas_pixels: int = 64_000_000  # full composited canvas
    min_frame_duration_ms: int = 0
    max_frame_duration_ms: int = 60_000
    max_dimension: int = 16384  # max output width/height
    min_dimension: int = 1
    # Concurrency bounds
    max_decode_workers: int = 2
    max_encode_workers: int = 2
    max_inflight: int = 16
    queue_capacity: int = 64
    default_timeout_ms: int = 30_000


@dataclass(frozen=True)
class PipelineConfig:
    limits: Limits = Limits()

    # Output encode parameter defaults per format
    webp_quality: int = 80
    webp_lossless: bool = False
    png_compress_level: int = 6
    jpeg_quality: int = 85
    gif_colors: int = 256

    # Normalization target
    target_color_space: str = "sRGB"


DEFAULT_CONFIG = PipelineConfig()
