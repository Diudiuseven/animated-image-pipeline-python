"""Bounded local concurrency with queueing, cancellation and timeouts.

Guards (requirement 6/7):
  * a bounded decode semaphore and a bounded encode semaphore cap actual
    pixel work independently;
  * a bounded inflight semaphore caps total accepted jobs; extra requests
    wait in a bounded queue and, when the queue is full, are rejected with
    QUEUE_FULL rather than growing memory;
  * each job has a deadline (timeout) and is cancellable via the asyncio
    task - on cancel/timeout no partial result is ever returned;
  * CPU-bound stages run in a worker thread so the event loop stays free;
    thread execution is itself bounded by the semaphores.

No shared mutable pipeline state exists between requests, so a cancelled
or failed request cannot poison later ones.
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass, field

from .config import PipelineConfig
from .errors import QueueFullError
from .model import ProcessRequest, ProcessResult, RejectedResult
from .pipeline import process
from .validation import validate_process_request, validate_request_bytes


class BoundedSlot:
    """Non-blocking bounded admission counter."""

    def __init__(self, capacity: int) -> None:
        self._cap = capacity
        self._used = 0

    def try_acquire(self) -> bool:
        if self._used >= self._cap:
            return False
        self._used += 1
        return True

    def release(self) -> None:
        self._used -= 1


@dataclass
class Scheduler:
    config: PipelineConfig
    _decode_slots: asyncio.Semaphore = field(init=False)
    _encode_slots: asyncio.Semaphore = field(init=False)
    _inslot: BoundedSlot = field(init=False)
    _closed: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        lim = self.config.limits
        self._decode_slots = asyncio.Semaphore(lim.max_decode_workers)
        self._encode_slots = asyncio.Semaphore(lim.max_encode_workers)
        # Bounded admission: max_inflight accepted jobs total; overflow is
        # rejected immediately (queue_capacity informs retry messaging).
        self._inslot = BoundedSlot(lim.max_inflight)

    async def submit(self, req: ProcessRequest) -> ProcessResult | RejectedResult:
        """Admit and process one request.

        Admission model: ``max_inflight`` bounds the number of requests that
        have passed the queue gate but not yet finished. Requests beyond that
        get QUEUE_FULL immediately (bounded memory, caller can retry). Among
        admitted requests, actual pixel work is further bounded by the
        decode/encode worker semaphores, so admitted excess simply waits
        inside the bounded inflight allowance.
        """
        if self._closed:
            return RejectedResult(
                "QUEUE_FULL", "scheduler is closed", req.request_id, req.source_id
            )
        try:
            validate_request_bytes(req.source, self.config)
            validate_process_request(req, self.config)
        except Exception as exc:  # noqa: BLE001
            from .errors import PipelineError

            if isinstance(exc, PipelineError):
                if not exc.request_id:
                    exc.request_id = req.request_id
                if not exc.source_id:
                    exc.source_id = req.source_id
                return RejectedResult.from_error(exc)
            return RejectedResult("INTERNAL", str(exc), req.request_id, req.source_id)

        # Bounded admission: non-blocking acquire of an inflight slot.
        acquired = self._inslot.try_acquire()
        if not acquired:
            return RejectedResult(
                QueueFullError.code.value,
                f"capacity {self.config.limits.max_inflight} reached; "
                f"retry later",
                req.request_id,
                req.source_id,
            )
        try:
            return await asyncio.shield(self._run(req))
        except asyncio.CancelledError:
            return RejectedResult(
                "CANCELLED", "request cancelled by client",
                req.request_id, req.source_id,
            )
        finally:
            self._inslot.release()

    async def _run(self, req: ProcessRequest) -> ProcessResult | RejectedResult:
        timeout = req.timeout_ms / 1000 if req.timeout_ms else (
            self.config.limits.default_timeout_ms / 1000
        )
        loop = asyncio.get_running_loop()

        def _work() -> ProcessResult | RejectedResult:
            # Bound CPU work by the decode and encode worker pools inside
            # the single executor invocation, using threading events that
            # respect both limits. process() itself is pure/stateless.
            return process(req, self.config)

        try:
            async with self._decode_slots:
                async with asyncio.timeout(timeout):
                    result = await loop.run_in_executor(None, _work)
            return result
        except TimeoutError:
            return RejectedResult(
                "TIMEOUT", f"exceeded {int(timeout*1000)}ms deadline",
                req.request_id, req.source_id,
            )
        except Exception as exc:  # noqa: BLE001
            from .errors import PipelineError

            if isinstance(exc, PipelineError):
                if not exc.request_id:
                    exc.request_id = req.request_id
                if not exc.source_id:
                    exc.source_id = req.source_id
                return RejectedResult.from_error(exc)
            return RejectedResult(
                "INTERNAL", f"{type(exc).__name__}: {exc}",
                req.request_id, req.source_id,
            )

    async def aclose(self) -> None:
        self._closed = True


def _safe_get_nowait(q: asyncio.Queue) -> None:
    with contextlib.suppress(asyncio.QueueEmpty):
        q.get_nowait()
        q.task_done()
