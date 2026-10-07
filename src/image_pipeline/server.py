"""Local FastAPI service: upload -> process -> download bytes.

Endpoints:
  GET  /health
  POST /images/process        multipart upload + query/form params
  POST /images/process-json   raw body + query params

Responses are either the encoded image bytes (Content-Type image/*) or a
JSON object {code,message,request_id,source_id} with a 4xx status. No
partial image bytes are ever returned.
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, Query, Request, UploadFile
from fastapi.responses import JSONResponse, Response

from .concurrency import Scheduler
from .config import DEFAULT_CONFIG, PipelineConfig
from .model import ProcessRequest, ResizeMode

_CONTENT_TYPES = {
    "PNG": "image/png",
    "GIF": "image/gif",
    "WEBP": "image/webp",
    "JPEG": "image/jpeg",
}


def create_app(config: PipelineConfig | None = None) -> FastAPI:
    cfg = config or DEFAULT_CONFIG
    scheduler = Scheduler(cfg)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            yield
        finally:
            await scheduler.aclose()

    app = FastAPI(title="animated-image-pipeline", version="0.1.0", lifespan=lifespan)
    app.state.scheduler = scheduler
    app.state.config = cfg

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "limits": _limits_dict(cfg)}

    async def _handle(
        source: bytes,
        source_id: str,
        output_format: str,
        width: int,
        height: int,
        resize_mode: str,
        quality: int | None,
        lossless: bool | None,
        timeout_ms: int,
    ):
        request_id = uuid.uuid4().hex
        req = ProcessRequest(
            source=source,
            source_id=source_id,
            output_format=output_format or "",
            width=width,
            height=height,
            resize_mode=ResizeMode(resize_mode),
            quality=quality,
            lossless=lossless,
            request_id=request_id,
            timeout_ms=timeout_ms,
        )
        result = await scheduler.submit(req)
        # RejectedResult -> structured JSON error
        if hasattr(result, "code") and not hasattr(result, "data"):
            status = 400
            if result.code in ("QUEUE_FULL",):
                status = 503
            elif result.code in ("TIMEOUT",):
                status = 504
            elif result.code in ("CANCELLED",):
                status = 499
            return JSONResponse(
                status_code=status,
                content={
                    "code": result.code,
                    "message": result.message,
                    "request_id": result.request_id,
                    "source_id": result.source_id,
                },
            )
        media = _CONTENT_TYPES.get(result.output_format, "application/octet-stream")
        ext = result.output_format.lower()
        headers = {
            "Content-Disposition": f'attachment; filename="{request_id}.{ext}"',
            "X-Request-Id": request_id,
            "X-Frame-Count": str(result.frame_count),
        }
        return Response(content=result.data, media_type=media, headers=headers)

    @app.post("/images/process")
    async def process_upload(
        file: UploadFile = File(...),
        output_format: str = Form(""),
        width: int = Form(0),
        height: int = Form(0),
        resize_mode: str = Form("fit"),
        quality: int | None = Form(None),
        lossless: bool | None = Form(None),
        timeout_ms: int = Form(0),
    ):
        source = await file.read()
        return await _handle(
            source, file.filename or "upload",
            output_format, width, height, resize_mode, quality, lossless, timeout_ms,
        )

    @app.post("/images/process-raw")
    async def process_raw(
        request: Request,
        output_format: str = Query(""),
        width: int = Query(0),
        height: int = Query(0),
        resize_mode: str = Query("fit"),
        quality: int | None = Query(None),
        lossless: bool | None = Query(None),
        timeout_ms: int = Query(0),
        source_id: str = Query("raw-body"),
    ):
        source = await request.body()
        return await _handle(
            source, source_id,
            output_format, width, height, resize_mode, quality, lossless, timeout_ms,
        )

    return app


def _limits_dict(cfg: PipelineConfig) -> dict:
    lim = cfg.limits
    return {
        "max_bytes": lim.max_bytes,
        "max_frame_count": lim.max_frame_count,
        "max_frame_pixels": lim.max_frame_pixels,
        "max_decode_workers": lim.max_decode_workers,
        "max_encode_workers": lim.max_encode_workers,
        "max_inflight": lim.max_inflight,
        "queue_capacity": lim.queue_capacity,
    }


app = create_app()
