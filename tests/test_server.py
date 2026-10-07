"""Requirement: local upload/process/download API with structured errors."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from image_pipeline import samples
from image_pipeline.config import Limits, PipelineConfig
from image_pipeline.server import create_app


@pytest.fixture
def client():
    cfg = PipelineConfig(limits=Limits(max_inflight=8, queue_capacity=8))
    app = create_app(cfg)
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert "max_frame_count" in r.json()["limits"]


def test_process_raw_webp(client, animated_gif):
    r = client.post(
        "/images/process-raw",
        params={"output_format": "webp", "width": "32"},
        content=animated_gif,
        headers={"Content-Type": "application/octet-stream"},
    )
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/webp"
    assert int(r.headers["x-frame-count"]) == 4
    assert "X-Request-Id" in r.headers


def test_process_multipart_png(client, animated_gif):
    r = client.post(
        "/images/process",
        files={"file": ("anim.gif", animated_gif, "image/gif")},
        data={"output_format": "png", "width": "16"},
    )
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"


def test_garbage_rejected_with_json(client):
    r = client.post(
        "/images/process-raw",
        params={"output_format": "webp"},
        content=b"not an image",
    )
    assert r.status_code == 400
    body = r.json()
    assert body["code"] in ("UNSUPPORTED_FORMAT", "EMPTY_INPUT")
    assert body["message"] and body["request_id"]


def test_oversized_dimension_400(client, animated_gif):
    r = client.post(
        "/images/process-raw", params={"width": "99999999"}, content=animated_gif
    )
    assert r.status_code == 400 and r.json()["code"] == "DIMENSION_TOO_LARGE"


def test_empty_body_400(client):
    r = client.post("/images/process-raw", content=b"")
    assert r.status_code == 400 and r.json()["code"] == "EMPTY_INPUT"


def test_deterministic_endpoint_output(client, animated_png):
    def call():
        return client.post(
            "/images/process-raw",
            params={"output_format": "webp", "quality": "75"},
            content=animated_png,
        ).content

    assert call() == call()
