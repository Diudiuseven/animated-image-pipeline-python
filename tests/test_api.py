from fastapi.testclient import TestClient

from imagepipeline.api import create_app
from imagepipeline.samples import generate_sample


def _client():
    return TestClient(create_app())


def test_health_reports_limits():
    with _client() as c:
        r = c.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["limits"]["max_frames"] > 0


def test_samples_list_and_download():
    with _client() as c:
        names = c.get("/samples").json()["samples"]
        assert "animated_transparent" in names
        r = c.get("/samples/animated_transparent")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/gif"
    assert len(r.content) > 0


def test_process_endpoint_gif_to_webp():
    data, _ = generate_sample("animated_transparent")
    with _client() as c:
        r = c.post(
            "/process",
            files={"file": ("x.gif", data, "image/gif")},
            data={"target_format": "webp", "lossless": "true"},
        )
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "image/webp"
    assert int(r.headers["x-frame-count"]) == 4
    assert r.headers["x-request-id"]


def test_process_with_resize():
    data, _ = generate_sample("static_png")
    with _client() as c:
        r = c.post(
            "/process",
            files={"file": ("x.png", data, "image/png")},
            data={
                "target_format": "png",
                "resize_mode": "fit",
                "width": "16",
                "height": "16",
            },
        )
    assert r.status_code == 200
    assert r.headers["x-width"] == "16"
    assert r.headers["x-height"] == "12"


def test_garbage_upload_is_rejected_with_locatable_error():
    with _client() as c:
        r = c.post(
            "/process",
            files={"file": ("bad.bin", b"not an image", "application/octet-stream")},
            data={"target_format": "png"},
        )
    assert r.status_code == 422
    body = r.json()["error"]
    assert body["code"] == "invalid_input"
    assert body["request_id"]


def test_animated_jpeg_without_still_rejected():
    data, _ = generate_sample("animated_transparent")
    with _client() as c:
        r = c.post(
            "/process",
            files={"file": ("x.gif", data, "image/gif")},
            data={"target_format": "jpeg"},
        )
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "invalid_request"
