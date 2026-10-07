"""Requirement 6 & 7: bounded capacity, oversized animations isolated,
cancellation, timeout, and absence of cross-request state pollution."""

from __future__ import annotations

import threading

import asyncio
import pytest

import image_pipeline.concurrency as conc
import image_pipeline.pipeline as pipe
from image_pipeline import samples
from image_pipeline.config import Limits, PipelineConfig
from image_pipeline.concurrency import Scheduler
from image_pipeline.model import ProcessRequest


def _cfg(**kw):
    base = dict(
        max_decode_workers=1, max_encode_workers=1, max_inflight=2,
        queue_capacity=2, default_timeout_ms=30000,
        max_frame_count=500, max_frame_pixels=10_000_000,
        max_canvas_pixels=10_000_000,
    )
    base.update(kw)
    return PipelineConfig(limits=Limits(**base))


def test_worker_concurrency_is_bounded(animated_gif):
    cfg = _cfg(max_decode_workers=1, max_inflight=4)
    gate = threading.Event()
    lock = threading.Lock()
    peak = {"n": 0}
    real = conc.process

    def blocked(req, config=None):
        with lock:
            peak["n"] += 1
        gate.wait(timeout=10)
        with lock:
            peak["n"] -= 1
        return real(req, config)

    conc.process = blocked
    pipe.process = blocked

    async def scenario():
        sch = Scheduler(cfg)
        try:
            tasks = [
                asyncio.ensure_future(
                    sch.submit(ProcessRequest(source=animated_gif, request_id=f"t{i}", output_format="PNG"))
                )
                for i in range(4)
            ]
            await asyncio.sleep(0.5)
            assert peak["n"] <= 1
            gate.set()
            res = await asyncio.gather(*tasks)
            assert all(hasattr(r, "data") for r in res)
        finally:
            conc.process = real
            pipe.process = real
            await sch.aclose()

    asyncio.run(scenario())


def test_queue_full_rejected(animated_gif):
    cfg = _cfg(max_inflight=2)
    gate = threading.Event()
    real = conc.process

    def blocked(req, config=None):
        gate.wait(timeout=10)
        return real(req, config)

    conc.process = blocked
    pipe.process = blocked

    async def scenario():
        sch = Scheduler(cfg)
        try:
            held = [
                asyncio.ensure_future(
                    sch.submit(ProcessRequest(source=animated_gif, request_id=f"h{i}", output_format="PNG"))
                )
                for i in range(2)
            ]
            await asyncio.sleep(0.5)
            r = await sch.submit(ProcessRequest(source=animated_gif, request_id="extra"))
            assert r.code == "QUEUE_FULL"
            gate.set()
            out = await asyncio.gather(*held)
            assert all(hasattr(x, "data") for x in out)
        finally:
            conc.process = real
            pipe.process = real
            await sch.aclose()

    asyncio.run(scenario())


def test_timeout_returns_locatable_error(animated_gif):
    sch = Scheduler(_cfg())
    gate = threading.Event()
    real = conc.process

    def slow(req, config=None):
        gate.wait(timeout=5)
        return real(req, config)

    conc.process = slow
    pipe.process = slow
    try:
        r = asyncio.run(
            sch.submit(ProcessRequest(source=animated_gif, request_id="slow", timeout_ms=100))
        )
        assert r.code == "TIMEOUT" and r.request_id == "slow"
        gate.set()
    finally:
        conc.process = real
        pipe.process = real


def test_cancellation_then_no_pollution(animated_gif):
    sch = Scheduler(_cfg())
    gate = threading.Event()
    real = conc.process

    def blocked(req, config=None):
        gate.wait(timeout=5)
        return real(req, config)

    conc.process = blocked
    pipe.process = blocked
    try:

        async def scenario():
            t = asyncio.ensure_future(
                sch.submit(ProcessRequest(source=animated_gif, request_id="cancel"))
            )
            await asyncio.sleep(0.4)
            t.cancel()
            r = await t
            assert r.code == "CANCELLED" and r.request_id == "cancel"
            gate.set()
            await asyncio.sleep(0.1)
            return True

        assert asyncio.run(scenario())
    finally:
        conc.process = real
        pipe.process = real

    # follow-up request still works and is deterministic
    a = asyncio.run(sch.submit(ProcessRequest(source=animated_gif, request_id="after1", output_format="PNG")))
    b = asyncio.run(sch.submit(ProcessRequest(source=animated_gif, request_id="after2", output_format="PNG")))
    assert hasattr(a, "data") and a.data == b.data


def test_oversized_animation_rejected_not_starving_batch(tight_config):
    """One over-frame-count animation is rejected while normal jobs proceed."""
    big = samples.make_animated_gif(frames=6, size=(24, 24))
    normal = samples.make_animated_gif(frames=2, size=(24, 24))

    async def scenario():
        sch = Scheduler(tight_config)
        bad = sch.submit(ProcessRequest(source=big, request_id="big", output_format="WEBP"))
        good = sch.submit(ProcessRequest(source=normal, request_id="ok", output_format="WEBP"))
        rb, rg = await asyncio.gather(bad, good)
        await sch.aclose()
        assert rb.code == "FRAME_COUNT_EXCEEDED"
        assert hasattr(rg, "data") and rg.frame_count == 2

    asyncio.run(scenario())
