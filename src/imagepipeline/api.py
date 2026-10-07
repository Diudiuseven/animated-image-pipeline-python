"""FastAPI HTTP surface for the local image processing service.

Endpoints:
* ``GET  /health``          - liveness and active configuration bounds;
* ``GET  /samples``         - list the in-process sample images;
* ``GET  /samples/{name}``  - download a synthesized sample;
* ``POST /process``         - multipart upload + transform parameters.

Every pipeline error is mapped to a stable JSON body with a machine-readable
``code``, a human message and the request/source identifiers so failures are
distinguishable and locatable. No partial artifact is ever returned: an error
response contains no image bytes.
"""

from __future__ import annotations

import uuid
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, Response

from .config import DEFAULT_SETTINGS, Settings
from .exceptions import PipelineError
from .executor import PipelineExecutor
from .models import Format, ResizeMode, TransformRequest
from .samples import available_samples, generate_sample


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or DEFAULT_SETTINGS
    app = FastAPI(title="animated-image-pipeline", version="0.1.0")
    app.state.settings = settings
    # The executor owns asyncio primitives, so build it lazily once an event
    # loop is running (FastAPI startup).
    app.state.executor: PipelineExecutor | None = None

    def _executor() -> PipelineExecutor:
        if app.state.executor is None:
            app.state.executor = PipelineExecutor(settings)
        return app.state.executor

    def _fail(exc: PipelineError) -> HTTPException:
        status = 422
        code = exc.code
        if code in ("queue_rejected",):
            status = 429
        elif code in ("processing_timeout",):
            status = 408
        elif code in ("cancelled",):
            status = 499
        elif code in ("internal_processing_error",):
            status = 500
        return HTTPException(status_code=status, detail=exc.to_dict())

    @app.exception_handler(PipelineError)
    async def _on_pipeline_error(_request, exc: PipelineError):
        http = _fail(exc)
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=http.status_code, content={"error": http.detail})

    @app.get("/health")
    async def health():
        lim = settings.limits
        return {
            "status": "ok",
            "limits": {
                "max_source_bytes": lim.max_source_bytes,
                "max_frames": lim.max_frames,
                "max_dimension": lim.max_dimension,
                "max_pixels_per_frame": lim.max_pixels_per_frame,
                "max_total_pixels": lim.max_total_pixels,
                "max_concurrent_requests": lim.max_concurrent_requests,
                "max_concurrent_decodes": lim.max_concurrent_decodes,
                "max_concurrent_encodes": lim.max_concurrent_encodes,
                "queue_policy": lim.queue_policy,
            },
            "inflight": _executor().inflight,
        }

    @app.get("/samples")
    async def list_samples():
        return {"samples": list(available_samples())}

    @app.get("/samples/{name}")
    async def get_sample(name: str):
        data, fmt = generate_sample(name)
        return Response(content=data, media_type=f"image/{fmt.value}")

    @app.post("/process")
    async def process_image(
        file: bytes = File(...),
        target_format: Literal["png", "jpeg", "gif", "webp"] = Form(...),
        resize_mode: Literal["none", "fit", "cover", "stretch"] = Form("none"),
        width: int | None = Form(None),
        height: int | None = Form(None),
        quality: int | None = Form(None),
        lossless: bool | None = Form(None),
        still: bool = Form(False),
        loop: int | None = Form(None),
    ):
        request_id = uuid.uuid4().hex
        request = TransformRequest(
            target_format=Format(target_format),
            resize_mode=ResizeMode(resize_mode),
            width=width,
            height=height,
            quality=quality,
            lossless=lossless,
            still=still,
            loop=loop,
        )
        result = await _executor().submit(file, request, request_id)
        return Response(
            content=result.data,
            media_type=result.media_type,
            headers={
                "X-Request-Id": result.request_id,
                "X-Frame-Count": str(result.frame_count),
                "X-Width": str(result.width),
                "X-Height": str(result.height),
            },
        )

    return app


app = create_app()
