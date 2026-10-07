import asyncio

import pytest

from imagepipeline.config import Limits, QueuePolicy, Settings
from imagepipeline.exceptions import (
    CancelledError,
    ProcessingTimeoutError,
    QueueRejectedError,
)
from imagepipeline.executor import PipelineExecutor
from imagepipeline.models import Format, ResizeMode, TransformRequest
from imagepipeline.samples import generate_sample


def _settings(**kw):
    return Settings(limits=Limits(**kw))


def test_concurrent_processing_is_byte_deterministic():
    data, _ = generate_sample("animated_transparent")
    req = TransformRequest(Format.WEBP, lossless=True)
    settings = _settings(
        max_concurrent_requests=6,
        max_concurrent_decodes=3,
        max_concurrent_encodes=3,
        queue_policy=QueuePolicy.QUEUE,
        queue_timeout_s=10.0,
    )

    async def scenario():
        ex = PipelineExecutor(settings)
        results = await asyncio.gather(
            *[ex.submit(data, req, f"r{i}") for i in range(12)]
        )
        await ex.aclose()
        return results

    results = asyncio.run(scenario())
    blobs = {r.data for r in results}
    assert len(blobs) == 1
    assert len(results) == 12


def test_reject_policy_when_at_capacity():
    data, _ = generate_sample("animated_transparent")
    settings = _settings(max_concurrent_requests=1)

    async def scenario():
        ex = PipelineExecutor(settings)
        gate = asyncio.Event()

        async def holder():
            # Occupy the single slot and wait for the gate.
            async def occupy():
                await ex._acquire_admission("holder")
                ex._inflight += 1
                try:
                    await gate.wait()
                finally:
                    ex._global.release()
                    ex._inflight -= 1

            await occupy()

        task = asyncio.ensure_future(holder())
        await asyncio.sleep(0.05)
        with pytest.raises(QueueRejectedError):
            await ex.submit(data, TransformRequest(Format.PNG), "r2")
        gate.set()
        await task
        await ex.aclose()

    asyncio.run(scenario())


def test_queue_policy_waits_for_slot():
    data, _ = generate_sample("static_png")
    settings = _settings(
        max_concurrent_requests=1,
        queue_policy=QueuePolicy.QUEUE,
        queue_timeout_s=2.0,
    )

    async def scenario():
        ex = PipelineExecutor(settings)
        req = TransformRequest(Format.PNG)
        results = await asyncio.gather(
            ex.submit(data, req, "a"), ex.submit(data, req, "b")
        )
        await ex.aclose()
        assert results[0].data == results[1].data

    asyncio.run(scenario())


def test_queue_timeout_rejects():
    data, _ = generate_sample("static_png")
    settings = _settings(
        max_concurrent_requests=1,
        queue_policy=QueuePolicy.QUEUE,
        queue_timeout_s=0.05,
    )

    async def scenario():
        ex = PipelineExecutor(settings)
        gate = asyncio.Event()

        async def slow():
            await ex._acquire_admission("slow")
            ex._inflight += 1
            try:
                await gate.wait()
            finally:
                ex._global.release()
                ex._inflight -= 1

        task = asyncio.ensure_future(slow())
        await asyncio.sleep(0.02)
        with pytest.raises(QueueRejectedError):
            await ex.submit(data, TransformRequest(Format.PNG), "r")
        gate.set()
        await task
        await ex.aclose()

    asyncio.run(scenario())


def test_cancel_leaves_no_state_and_is_locatable():
    data, _ = generate_sample("static_png")
    settings = _settings(
        max_concurrent_requests=4,
        processing_timeout_s=5.0,
    )

    async def scenario():
        ex = PipelineExecutor(settings)
        task = asyncio.ensure_future(
            ex.submit(data, TransformRequest(Format.PNG), "cancel-me")
        )
        # Let the task enter the executor before cancelling, mirroring a real
        # client disconnect during processing.
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(CancelledError) as info:
            await task
        assert info.value.request_id == "cancel-me"
        await asyncio.sleep(0.05)
        assert ex.inflight == 0
        # The executor still serves later requests - no polluted state.
        follow = await ex.submit(data, TransformRequest(Format.PNG), "after")
        assert follow.frame_count == 1
        await ex.aclose()

    asyncio.run(scenario())


def test_processing_timeout_is_locatable(monkeypatch):
    import time

    data, _ = generate_sample("static_png")
    settings = _settings(processing_timeout_s=0.05)

    # Make the decode stage slower than the processing timeout.
    from imagepipeline.stages import decoder as dec_mod

    real_decode = dec_mod.decode

    def slow_decode(*a, **k):
        time.sleep(0.3)
        return real_decode(*a, **k)

    monkeypatch.setattr(
        "imagepipeline.executor.decoder.decode", staticmethod(slow_decode)
    )

    async def scenario():
        ex = PipelineExecutor(settings)
        with pytest.raises(ProcessingTimeoutError) as info:
            await ex.submit(data, TransformRequest(Format.PNG), "slow-one")
        assert info.value.request_id == "slow-one"
        await ex.aclose()

    asyncio.run(scenario())


def test_bad_request_returns_distinguishable_error():
    data, _ = generate_sample("static_png")
    settings = _settings()

    async def scenario():
        ex = PipelineExecutor(settings)
        from imagepipeline.exceptions import InvalidRequestError

        with pytest.raises(InvalidRequestError) as info:
            await ex.submit(
                data,
                TransformRequest(Format.PNG, ResizeMode.COVER, width=10),
                "bad",
            )
        assert info.value.request_id == "bad"
        await ex.aclose()

    asyncio.run(scenario())


def test_decode_concurrency_is_bounded(monkeypatch):
    import threading
    import time

    from imagepipeline.stages import decoder as dec_mod

    data, _ = generate_sample("static_png")
    settings = _settings(
        max_concurrent_requests=8,
        max_concurrent_decodes=2,
        max_concurrent_encodes=4,
        queue_policy=QueuePolicy.QUEUE,
        queue_timeout_s=5.0,
    )

    peak = {"v": 0}
    active = {"v": 0}
    lock = threading.Lock()
    real = dec_mod.decode

    def tracked(data, fmt, st):
        with lock:
            active["v"] += 1
            peak["v"] = max(peak["v"], active["v"])
        time.sleep(0.05)
        with lock:
            active["v"] -= 1
        return real(data, fmt, st)

    monkeypatch.setattr(
        "imagepipeline.executor.decoder.decode", staticmethod(tracked)
    )

    async def scenario():
        ex = PipelineExecutor(settings)
        await asyncio.gather(
            *[
                ex.submit(data, TransformRequest(Format.PNG), f"r{i}")
                for i in range(6)
            ]
        )
        await ex.aclose()

    asyncio.run(scenario())
    assert peak["v"] <= 2
