"""Stable, distinguishable, localizable error types."""

from __future__ import annotations

from enum import Enum


class ErrorCode(str, Enum):
    """Machine-readable error codes; stable across runs."""

    EMPTY_INPUT = "EMPTY_INPUT"
    UNSUPPORTED_FORMAT = "UNSUPPORTED_FORMAT"
    TRUNCATED_OR_UNDECODABLE = "TRUNCATED_OR_UNDECODABLE"
    INVALID_FRAME_METADATA = "INVALID_FRAME_METADATA"
    FRAME_COUNT_EXCEEDED = "FRAME_COUNT_EXCEEDED"
    PIXEL_LIMIT_EXCEEDED = "PIXEL_LIMIT_EXCEEDED"
    INVALID_REQUEST = "INVALID_REQUEST"
    DIMENSION_TOO_LARGE = "DIMENSION_TOO_LARGE"
    ENCODE_FAILED = "ENCODE_FAILED"
    CANCELLED = "CANCELLED"
    TIMEOUT = "TIMEOUT"
    QUEUE_FULL = "QUEUE_FULL"
    INTERNAL = "INTERNAL"


class PipelineError(Exception):
    """Base error; carries stable code + request/source locator."""

    code: ErrorCode = ErrorCode.INTERNAL

    def __init__(self, message: str = "", *, request_id: str = "", source_id: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.request_id = request_id
        self.source_id = source_id

    def to_dict(self) -> dict[str, str]:
        return {
            "code": self.code.value,
            "message": self.message,
            "request_id": self.request_id,
            "source_id": self.source_id,
        }


class EmptyInputError(PipelineError):
    code = ErrorCode.EMPTY_INPUT


class UnsupportedFormatError(PipelineError):
    code = ErrorCode.UNSUPPORTED_FORMAT


class DecodeError(PipelineError):
    code = ErrorCode.TRUNCATED_OR_UNDECODABLE


class FrameMetadataError(PipelineError):
    code = ErrorCode.INVALID_FRAME_METADATA


class FrameCountExceededError(PipelineError):
    code = ErrorCode.FRAME_COUNT_EXCEEDED


class PixelLimitExceededError(PipelineError):
    code = ErrorCode.PIXEL_LIMIT_EXCEEDED


class InvalidRequestError(PipelineError):
    code = ErrorCode.INVALID_REQUEST


class DimensionTooLargeError(PipelineError):
    code = ErrorCode.DIMENSION_TOO_LARGE


class EncodeError(PipelineError):
    code = ErrorCode.ENCODE_FAILED


class CancelledError(PipelineError):
    code = ErrorCode.CANCELLED


class TimeoutError(PipelineError):  # noqa: A001 - explicit domain error
    code = ErrorCode.TIMEOUT


class QueueFullError(PipelineError):
    code = ErrorCode.QUEUE_FULL
