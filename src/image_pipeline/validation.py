"""Early rejection of bad inputs and out-of-bounds requests.

All checks happen before any frame is decoded/encoded, so a rejected
request never yields a partial image. Reasons carry a stable ErrorCode.
"""

from __future__ import annotations

import io

from PIL import Image

from .config import PipelineConfig
from .errors import (
    DimensionTooLargeError,
    EmptyInputError,
    InvalidRequestError,
    UnsupportedFormatError,
)
from .model import ProcessRequest, ResizeMode

# Magic-byte signatures for formats we accept.
_SIGNATURES: tuple[tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "PNG"),
    (b"GIF87a", "GIF"),
    (b"GIF89a", "GIF"),
    (b"RIFF", "WEBP"),  # additionally require b"WEBP" at offset 8
    (b"\xff\xd8\xff", "JPEG"),
    (b"BM", "BMP"),
    (b"II*\x00", "TIFF"),
    (b"MM\x00*", "TIFF"),
)

OUTPUT_FORMATS: frozenset[str] = frozenset({"PNG", "GIF", "WEBP", "JPEG"})


def sniff_format(data: bytes) -> str:
    """Return Pillow format name from magic bytes, or raise UnsupportedFormatError."""
    if not data:
        raise EmptyInputError("empty input: zero bytes")
    for sig, name in _SIGNATURES:
        if data.startswith(sig):
            if name == "WEBP":
                if len(data) >= 12 and data[8:12] != b"WEBP":
                    break
                return "WEBP"
            return name
    raise UnsupportedFormatError(f"unrecognized file signature: {data[:8]!r}")


def validate_request_bytes(data: bytes, config: PipelineConfig) -> str:
    """Reject empty/oversized/unrecognized inputs before decode.

    Returns the sniffed Pillow format name. Also asks Pillow to open+verify
    the header (no pixel decode) to reject structurally broken containers.
    """
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise EmptyInputError("input must be bytes")
    raw = bytes(data)
    if not raw:
        raise EmptyInputError("empty input: zero bytes")
    if len(raw) > config.limits.max_bytes:
        raise InvalidRequestError(
            f"input size {len(raw)} exceeds limit {config.limits.max_bytes}"
        )
    fmt = sniff_format(raw)
    # Header-only verification: verify() checks structure without pixel
    # decode; animated containers are re-verified frame by frame in codec_io.
    try:
        im = Image.open(io.BytesIO(raw))
        im.verify()
    except Exception as exc:  # noqa: BLE001 - any failure here means bad input
        from .errors import DecodeError

        raise DecodeError(
            f"header verification failed: {exc}",
            source_id=getattr(data, "source_id", ""),
        ) from exc
    return fmt


def validate_process_request(req: ProcessRequest, config: PipelineConfig) -> None:
    """Reject illegal dimensions/encoding parameters before processing."""
    lim = config.limits
    if req.width < 0 or req.height < 0:
        raise InvalidRequestError("width/height must be non-negative")
    if req.width == 0 and req.height == 0:
        pass  # both 0 => keep original size, that is legal
    for name, v in (("width", req.width), ("height", req.height)):
        if v > lim.max_dimension:
            raise DimensionTooLargeError(
                f"requested {name} {v} exceeds max {lim.max_dimension}",
                request_id=req.request_id,
                source_id=req.source_id,
            )
        if v != 0 and v < lim.min_dimension:
            raise InvalidRequestError(
                f"{name} {v} below minimum {lim.min_dimension}",
                request_id=req.request_id,
                source_id=req.source_id,
            )
    try:
        ResizeMode(req.resize_mode)
    except ValueError as exc:
        raise InvalidRequestError(
            f"unknown resize_mode: {req.resize_mode!r}",
            request_id=req.request_id,
            source_id=req.source_id,
        ) from exc
    if req.output_format:
        fmt = req.output_format.upper()
        if fmt not in OUTPUT_FORMATS:
            raise UnsupportedFormatError(
                f"unsupported output format: {req.output_format!r}",
                request_id=req.request_id,
                source_id=req.source_id,
            )
    if req.quality is not None and not (1 <= req.quality <= 100):
        raise InvalidRequestError(
            f"quality must be in 1..100, got {req.quality}",
            request_id=req.request_id,
            source_id=req.source_id,
        )
    if req.timeout_ms < 0:
        raise InvalidRequestError(
            "timeout_ms must be non-negative",
            request_id=req.request_id,
            source_id=req.source_id,
        )
