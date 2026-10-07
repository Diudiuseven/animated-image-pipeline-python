"""Early validation of request parameters and raw input boundaries.

Everything here runs *before* any expensive decode work so that oversized,
truncated or disallowed requests are rejected up front with a distinguishable
error code; no partial artifact is ever produced.
"""

from __future__ import annotations

from ..config import Settings
from ..exceptions import InvalidInputError, InvalidRequestError
from ..models import Format, ResizeMode, TransformRequest

# Minimal container signatures (magic bytes). Matched by PIL at decode time,
# but checked here for an early, format-specific rejection.
_MAGIC: tuple[tuple[bytes, Format], ...] = (
    (b"\x89PNG\r\n\x1a\n", Format.PNG),
    (b"\xff\xd8\xff", Format.JPEG),
    (b"GIF87a", Format.GIF),
    (b"GIF89a", Format.GIF),
    (b"RIFF", Format.WEBP),  # additionally require b"WEBP" at offset 8
)


def validate_request(request: TransformRequest, settings: Settings) -> None:
    """Reject requests that violate parameter or dimension boundaries."""
    limits = settings.limits
    mode = request.resize_mode

    if mode is ResizeMode.NONE:
        if request.width is not None or request.height is not None:
            msg = "width/height must not be set when resize_mode=none"
            raise InvalidRequestError(msg)
        return

    if request.width is None and request.height is None:
        msg = f"at least one of width/height is required for resize_mode={mode.value}"
        raise InvalidRequestError(msg)

    for name, value in (("width", request.width), ("height", request.height)):
        if value is not None and value > limits.max_target_dimension:
            msg = (
                f"{name}={value} exceeds max_target_dimension="
                f"{limits.max_target_dimension}"
            )
            raise InvalidRequestError(msg)

    if mode is ResizeMode.COVER and (request.width is None or request.height is None):
        msg = "cover resize requires both width and height"
        raise InvalidRequestError(msg)
    if mode is ResizeMode.STRETCH and (request.width is None or request.height is None):
        msg = "stretch resize requires both width and height"
        raise InvalidRequestError(msg)


def validate_source_size(data: bytes, settings: Settings) -> None:
    """Reject sources exceeding the raw byte size bound."""
    if len(data) == 0:
        raise InvalidInputError("empty input: no image bytes received")
    if len(data) > settings.limits.max_source_bytes:
        msg = (
            f"source size {len(data)} bytes exceeds "
            f"max_source_bytes={settings.limits.max_source_bytes}"
        )
        raise InvalidInputError(msg)


def sniff_signature(data: bytes) -> Format:
    """Identify a supported container from magic bytes or raise early."""
    for magic, fmt in _MAGIC:
        if data.startswith(magic):
            if fmt is Format.WEBP and data[8:12] != b"WEBP":
                raise InvalidInputError("RIFF container is not a WEBP image")
            return fmt
    raise InvalidInputError(
        "unrecognized image signature: supported formats are png/jpeg/gif/webp"
    )
