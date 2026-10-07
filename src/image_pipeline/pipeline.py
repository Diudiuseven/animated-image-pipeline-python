"""End-to-end orchestration.

Stages, in order:
  1. early validation of bytes + request parameters (before any decode);
  2. full decode (orientation + color normalization happen per frame
     inside decode);
  3. full-canvas compositing with disposal/blend semantics;
  4. deterministic resize (single fixed filter);
  5. deterministic encode.

The result is a pure function of (source bytes, request parameters):
same input -> byte-identical output at any concurrency, ordering or after
restart. On failure a RejectedResult with a stable code and locators is
produced; no partial bytes ever leave the pipeline.
"""

from __future__ import annotations

from .codec_io import decode, encode
from .config import DEFAULT_CONFIG, PipelineConfig
from .errors import PipelineError
from .model import ProcessRequest, ProcessResult, RejectedResult, ResizeMode
from .render import composite_frames
from .transform import compute_target_size, resize_frames
from .validation import validate_process_request, validate_request_bytes


def decode_stage(req: ProcessRequest, cfg: PipelineConfig):
    """Validate + decode + composite. Returns (decoded, composited).

    This is the CPU/memory-heavy input side and is what the decode worker
    pool bounds. Raises PipelineError on any problem.
    """
    validate_request_bytes(req.source, cfg)
    validate_process_request(req, cfg)
    decoded = decode(req.source, cfg, source_id=req.source_id)
    composited = composite_frames(decoded)
    if not composited:
        raise PipelineError("decoded image has no frames")
    return decoded, composited


def encode_stage(req: ProcessRequest, cfg: PipelineConfig, decoded, composited):
    """Deterministic resize + encode of already-composited frames.

    Bounded by the encode worker pool. Pure function of its inputs.
    """
    mode = ResizeMode(req.resize_mode)
    dst_w, dst_h = compute_target_size(
        decoded.width, decoded.height, req.width, req.height, mode,
        max_dimension=cfg.limits.max_dimension,
    )
    if dst_w * dst_h > cfg.limits.max_canvas_pixels:
        from .errors import PixelLimitExceededError

        raise PixelLimitExceededError(
            f"output {dst_w}x{dst_h} exceeds pixel limit",
            request_id=req.request_id,
            source_id=req.source_id,
        )
    crop_box = (
        (req.width, req.height)
        if mode is ResizeMode.COVER and req.width and req.height
        else None
    )
    resized = resize_frames(
        composited, (decoded.width, decoded.height), (dst_w, dst_h), mode, crop_box
    )
    return encode(decoded, resized, req, cfg, (dst_w, dst_h))


def process(req: ProcessRequest, config: PipelineConfig | None = None) -> ProcessResult:
    """Synchronously process one request; raises PipelineError on failure."""
    cfg = config or DEFAULT_CONFIG
    decoded, composited = decode_stage(req, cfg)
    return encode_stage(req, cfg, decoded, composited)


def process_or_reject(
    req: ProcessRequest, config: PipelineConfig | None = None
) -> ProcessResult | RejectedResult:
    """Return RejectedResult (never raise) so callers get stable reasons."""
    try:
        return process(req, config)
    except PipelineError as exc:
        if not exc.request_id:
            exc.request_id = req.request_id
        if not exc.source_id:
            exc.source_id = req.source_id
        return RejectedResult.from_error(exc)
    except Exception as exc:  # noqa: BLE001 - last-resort, still locatable
        return RejectedResult(
            code="INTERNAL",
            message=f"{type(exc).__name__}: {exc}",
            request_id=req.request_id,
            source_id=req.source_id,
        )
