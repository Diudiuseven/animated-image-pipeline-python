"""Pure, in-process processing pipeline.

Stages share no mutable state between requests, so the output is a pure
function of (source bytes, request, settings): orientation correction and
color normalization happen before resizing, every frame is
composited/resized identically, and encoding uses fixed deterministic
parameters.
"""

from __future__ import annotations

from .config import DEFAULT_SETTINGS, Settings
from .exceptions import (
    InternalProcessingError,
    InvalidRequestError,
    PipelineError,
)
from .models import PipelineResult, TransformRequest
from .stages import compositor, decoder, encoder, resize, validation
from .stages.encoder import MEDIA_TYPES


def run_stages(
    data: bytes,
    request: TransformRequest,
    settings: Settings = DEFAULT_SETTINGS,
    request_id: str = "",
    decode_fn=None,
    encode_fn=None,
) -> tuple[bytes, object, object]:
    """Pure synchronous stage sequence.

    ``decode_fn`` / ``encode_fn`` optionally replace the heavy decode and
    encode calls so a host executor can place them in separately bounded
    worker pools.  They default to the in-module implementations.

    Returns ``(payload_bytes, document_after_resize, written_frames)``.
    Raises a :class:`PipelineError` subclass for every failure; never returns
    partial output.
    """
    decode_fn = decode_fn or decoder.decode
    encode_fn = encode_fn or encoder.encode_with_frames

    # 1. Parameter validation before touching the image.
    validation.validate_request(request, settings)
    validation.validate_source_size(data, settings)

    # 2. Container identification + full decode pre-walk. Corrupt frames,
    #    illegal durations and limit violations are rejected here, before any
    #    artifact is produced.
    fmt = decoder.sniff_format(data)
    decoded = decode_fn(data, fmt, settings)

    # 3. Canonicalize: composited, oriented, sRGB straight-RGBA frames.
    document = compositor.composite(decoded)

    # 4. Static vs animated compatibility with the target container.
    if request.target_format.name == "JPEG" and document.is_animated and not request.still:
        raise InvalidRequestError(
            "refusing to encode an animated source as JPEG: pass still=true "
            "to export the first frame"
        )

    # 5. Deterministic alpha-aware resize.
    document = resize.resize(document, request, settings)

    # 6. Deterministic encode.
    payload, written_frames = encode_fn(document, request, settings)
    return payload, document, written_frames


def process(
    data: bytes,
    request: TransformRequest,
    settings: Settings = DEFAULT_SETTINGS,
    request_id: str = "",
    decode_fn=None,
    encode_fn=None,
) -> PipelineResult:
    """Run decode -> orient -> color-normalize -> resize -> encode."""
    try:
        payload, document, written_frames = run_stages(
            data, request, settings, request_id,
            decode_fn=decode_fn, encode_fn=encode_fn,
        )
    except PipelineError as exc:
        # Make every failure locatable to the specific request/source even
        # when a stage raised before the id could be attached.
        if not exc.request_id:
            exc.request_id = request_id or None
        raise
    except Exception as exc:  # pragma: no cover - defensive boundary
        raise InternalProcessingError(
            f"unexpected failure during processing: {exc}",
            request_id=request_id or None,
        ) from exc
    w, h = document.size
    return PipelineResult(
        data=payload,
        media_type=MEDIA_TYPES[request.target_format],
        fmt=request.target_format,
        width=w,
        height=h,
        frame_count=len(written_frames),
        durations_ms=tuple(f.duration_ms for f in written_frames),
        loop=request.loop if request.loop is not None else document.loop,
        request_id=request_id,
    )
