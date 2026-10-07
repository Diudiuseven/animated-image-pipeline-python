"""Pipeline configuration: capacity bounds, timeouts and encoding defaults."""

from __future__ import annotations

from dataclasses import dataclass

from .models import Format


class QueuePolicy:
    REJECT = "reject"  # fail immediately with queue_rejected
    QUEUE = "queue"  # wait up to queue_timeout for a slot


@dataclass(frozen=True)
class Limits:
    # Request body / source image size.
    max_source_bytes: int = 64 * 1024 * 1024
    # Per-image frame and pixel bounds (checked before heavy processing).
    max_frames: int = 500
    max_dimension: int = 16384
    max_pixels_per_frame: int = 40_000_000
    max_total_pixels: int = 500_000_000
    # Frame timing bounds in milliseconds.
    min_frame_duration_ms: int = 1
    max_frame_duration_ms: int = 60_000
    max_total_duration_ms: int = 600_000
    # Output / request bounds.
    max_target_dimension: int = 8192
    # In-flight admission control.
    max_concurrent_requests: int = 8
    max_concurrent_decodes: int = 2
    max_concurrent_encodes: int = 2
    queue_policy: str = QueuePolicy.REJECT
    queue_timeout_s: float = 2.0
    # Per-request wall-clock budget.
    processing_timeout_s: float = 30.0


@dataclass(frozen=True)
class EncodingDefaults:
    quality: int = 85
    loop: int = 0  # infinite
    default_duration_ms: int = 100  # for outputs where a frame lacks timing
    # Flat background used when flattening transparency for JPEG.
    background_rgb: tuple[int, int, int] = (255, 255, 255)


@dataclass(frozen=True)
class Settings:
    limits: Limits = Limits()
    encoding: EncodingDefaults = EncodingDefaults()
    supported_input_formats: tuple[Format, ...] = (
        Format.PNG,
        Format.JPEG,
        Format.GIF,
        Format.WEBP,
    )


DEFAULT_SETTINGS = Settings()
