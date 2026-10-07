"""Pipeline error hierarchy.

All failures carry a stable machine-readable ``code`` plus a human readable
message and optional request/source identifiers so that every failure can be
located to a specific request and source image.
"""

from __future__ import annotations


class PipelineError(Exception):
    """Base class for every error raised by the pipeline."""

    code: str = "pipeline_error"

    def __init__(
        self,
        message: str,
        *,
        request_id: str | None = None,
        source_id: str | None = None,
        detail: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.request_id = request_id
        self.source_id = source_id
        self.detail = detail

    def to_dict(self) -> dict[str, str | None]:
        return {
            "code": self.code,
            "message": self.message,
            "detail": self.detail,
            "request_id": self.request_id,
            "source_id": self.source_id,
        }


class InvalidInputError(PipelineError):
    """Input bytes are unusable (truncated, undecodable, illegal metadata)."""

    code = "invalid_input"


class UnsupportedFormatError(PipelineError):
    """Container/codec combination is not supported."""

    code = "unsupported_format"


class InvalidRequestError(PipelineError):
    """Request parameters are invalid or outside the allowed boundaries."""

    code = "invalid_request"


class LimitExceededError(PipelineError):
    """A configured capacity boundary was crossed."""

    code = "limit_exceeded"


class QueueRejectedError(PipelineError):
    """No capacity available and the queue policy is ``reject``."""

    code = "queue_rejected"


class ProcessingTimeoutError(PipelineError):
    """Processing did not finish within the allotted timeout."""

    code = "processing_timeout"


class CancelledError(PipelineError):
    """Processing was cancelled by the caller or because the task was aborted."""

    code = "cancelled"


class InternalProcessingError(PipelineError):
    """An unexpected failure during decode/transform/encode."""

    code = "internal_processing_error"
