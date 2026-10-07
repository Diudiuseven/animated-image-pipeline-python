"""Bounded asynchronous executor around the synchronous pipeline.

Capacity controls (all configurable in :class:`Settings`):

* ``max_concurrent_requests``  - global in-flight admission;
* ``max_concurrent_decodes``   - bounded decode thread pool;
* ``max_concurrent_encodes``   - bounded encode thread pool;
* queue policy                 - reject immediately or wait up to
  ``queue_timeout_s`` for a slot;
* ``processing_timeout_s``     - per-request wall-clock budget.

The CPU-bound pipeline runs in worker threads; cancellation or timeout
abandons the worker (its result is never published) and the only externally
visible outcome is an error. No half-written artifact or shared state can
leak because each request owns its own bytes and canonical document.
"""

from __future__ import annotations

import asyncio
import concurrent.futures

from .config import QueuePolicy, Settings
from .exceptions import (
    CancelledError as PipelineCancelledError,
    InvalidRequestError,
    ProcessingTimeoutError,
    QueueRejectedError,
)
from .models import PipelineResult, TransformRequest
from .stages import compositor, decoder, resize as resize_mod, validation, encoder
from .stages.encoder import MEDIA_TYPES


class PipelineExecutor:
    """Async façade enforcing global and per-stage capacity bounds."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or Settings()
        limits = self._settings.limits
        self._global = asyncio.Semaphore(limits.max_concurrent_requests)
        self._decode_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=limits.max_concurrent_decodes,
            thread_name_prefix="imgp-decode",
        )
        self._encode_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=limits.max_concurrent_encodes,
            thread_name_prefix="imgp-encode",
        )
        # Compositing/color/resize are bounded by the global admission count.
        self._compute_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=limits.max_concurrent_requests,
            thread_name_prefix="imgp-compute",
        )
        self._closed = False
        self._inflight = 0

    @property
    def inflight(self) -> int:
        return self._inflight

    async def _acquire_admission(self, request_id: str) -> None:
        limits = self._settings.limits
        if limits.queue_policy == QueuePolicy.QUEUE:
            try:
                await asyncio.wait_for(
                    self._global.acquire(), timeout=limits.queue_timeout_s
                )
            except TimeoutError as exc:
                raise QueueRejectedError(
                    "queue capacity exhausted within queue_timeout_s",
                    request_id=request_id,
                ) from exc
            return
        if self._global.locked():
            raise QueueRejectedError(
                "at capacity and queue policy is reject",
                request_id=request_id,
            )
        await self._global.acquire()

    async def submit(
        self, data: bytes, request: TransformRequest, request_id: str
    ) -> PipelineResult:
        """Submit one request, honoring queue policy and timeout."""
        if self._closed:
            raise PipelineCancelledError(
                "executor is closed", request_id=request_id
            )
        acquired = False
        loop = asyncio.get_running_loop()
        try:
            await self._acquire_admission(request_id)
            acquired = True
            self._inflight += 1
            return await asyncio.wait_for(
                self._orchestrate(loop, data, request, request_id),
                timeout=self._settings.limits.processing_timeout_s,
            )
        except asyncio.CancelledError as exc:
            raise PipelineCancelledError(
                "request was cancelled", request_id=request_id
            ) from exc
        except TimeoutError as exc:
            raise ProcessingTimeoutError(
                f"processing exceeded "
                f"{self._settings.limits.processing_timeout_s}s",
                request_id=request_id,
            ) from exc
        finally:
            if acquired:
                self._global.release()
                self._inflight -= 1

    async def _orchestrate(
        self, loop, data: bytes, request: TransformRequest, request_id: str
    ) -> PipelineResult:
        from .exceptions import InternalProcessingError, PipelineError

        settings = self._settings
        try:
            return await self._run_pipeline(
                loop, data, request, settings, request_id
            )
        except PipelineError as exc:
            if not exc.request_id:
                exc.request_id = request_id or None
            raise
        except Exception as exc:  # pragma: no cover - defensive boundary
            raise InternalProcessingError(
                f"unexpected failure during processing: {exc}",
                request_id=request_id or None,
            ) from exc

    async def _run_pipeline(self, loop, data, request, settings, request_id) -> PipelineResult:
        validation.validate_request(request, settings)
        validation.validate_source_size(data, settings)
        fmt = decoder.sniff_format(data)

        decoded = await loop.run_in_executor(
            self._decode_pool, decoder.decode, data, fmt, settings
        )
        document = await loop.run_in_executor(
            self._compute_pool, compositor.composite, decoded
        )
        if request.target_format.name == "JPEG" and document.is_animated and not request.still:
            raise InvalidRequestError(
                "refusing to encode an animated source as JPEG: pass still=true"
            )
        document = await loop.run_in_executor(
            self._compute_pool, resize_mod.resize, document, request, settings
        )
        payload, written = await loop.run_in_executor(
            self._encode_pool,
            encoder.encode_with_frames,
            document,
            request,
            settings,
        )
        w, h = document.size
        return PipelineResult(
            data=payload,
            media_type=MEDIA_TYPES[request.target_format],
            fmt=request.target_format,
            width=w,
            height=h,
            frame_count=len(written),
            durations_ms=tuple(f.duration_ms for f in written),
            loop=request.loop if request.loop is not None else document.loop,
            request_id=request_id,
        )

    async def aclose(self) -> None:
        """Reject new requests and release worker threads (non-blocking)."""
        self._closed = True
        for pool in (self._decode_pool, self._compute_pool, self._encode_pool):
            pool.shutdown(wait=False, cancel_futures=True)
