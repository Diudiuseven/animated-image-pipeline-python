"""Requirement 3: output is a pure function of source + request.

Byte-identical across repeated calls, animated transcode round trips,
different processing order, high concurrency, and (simulated) restarts.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import random

import pytest
from PIL import Image

from image_pipeline.concurrency import Scheduler
from image_pipeline.config import PipelineConfig, Limits
from image_pipeline.model import ProcessRequest
from image_pipeline.pipeline import process


def _h(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


@pytest.mark.parametrize("out_fmt", ["PNG", "WEBP", "GIF"])
def test_repeated_calls_byte_identical(animated_gif, out_fmt):
    outs = {
        _h(process(ProcessRequest(source=animated_gif, output_format=out_fmt,
                                   width=40, request_id=f"r{i}")).data)
        for i in range(5)
    }
    assert len(outs) == 1


@pytest.mark.parametrize("out_fmt", ["PNG", "WEBP"])
def test_deterministic_across_sources(animated_gif, animated_webp, animated_png, out_fmt):
    for src in (animated_gif, animated_webp, animated_png):
        a = process(ProcessRequest(source=src, output_format=out_fmt, width=32, request_id="a")).data
        b = process(ProcessRequest(source=src, output_format=out_fmt, width=32, request_id="b")).data
        assert a == b


def test_shuffled_order_identical(animated_gif, animated_webp):
    reqs = [
        ProcessRequest(source=animated_gif, output_format="PNG", width=40, request_id="g"),
        ProcessRequest(source=animated_webp, output_format="PNG", width=40, request_id="w"),
    ] * 6
    random.Random(123).shuffle(reqs)
    outputs = {}
    for r in reqs:
        outputs.setdefault(r.request_id, set()).add(_h(process(r).data))
    assert all(len(v) == 1 for v in outputs.values())
    # and both orderings agree
    g = process(ProcessRequest(source=animated_gif, output_format="PNG", width=40)).data
    assert _h(g) == next(iter(outputs["g"]))


async def _run_concurrent(src, n, workers):
    cfg = PipelineConfig(
        limits=Limits(
            max_decode_workers=workers, max_encode_workers=workers,
            max_inflight=n, queue_capacity=n, default_timeout_ms=60000,
        )
    )
    sch = Scheduler(cfg)
    tasks = [
        sch.submit(ProcessRequest(source=src, output_format="WEBP", width=36,
                                   request_id=f"c{i}", source_id="s"))
        for i in range(n)
    ]
    results = await asyncio.gather(*tasks)
    await sch.aclose()
    return results


def test_high_concurrency_byte_identical(animated_gif):
    results = asyncio.run(_run_concurrent(animated_gif, 32, 2))
    assert all(hasattr(r, "data") for r in results)
    digests = {_h(r.data) for r in results}
    assert len(digests) == 1


def test_restart_equivalence(animated_png, monkeypatch):
    """Simulate a fresh process by clearing module caches, then reprocess."""
    r1 = process(ProcessRequest(source=animated_png, output_format="WEBP", quality=70))
    # clear lru caches (color profile cache) like a restart would
    from image_pipeline import color

    color._sRGB_profile_obj.cache_clear()
    color.srgb_icc_profile.cache_clear()
    r2 = process(ProcessRequest(source=animated_png, output_format="WEBP", quality=70))
    assert r1.data == r2.data


def test_request_id_does_not_affect_bytes(animated_gif):
    a = process(ProcessRequest(source=animated_gif, output_format="PNG", request_id="aaa")).data
    b = process(ProcessRequest(source=animated_gif, output_format="PNG", request_id="zzz")).data
    assert a == b
