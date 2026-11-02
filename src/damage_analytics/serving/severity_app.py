"""Independent FastAPI Phi-Net service. Run with uvicorn --factory; no backend model loads."""

import os
import re
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import BoundedSemaphore

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile

from damage_analytics.api.errors import ApiError
from damage_analytics.api.middleware import RequestMiddleware
from damage_analytics.api.schemas import IMAGE_UPLOAD_OPENAPI
from damage_analytics.api.uploads import decode_upload
from damage_analytics.serving.contracts import SeverityPrediction


@dataclass(frozen=True)
class SeveritySettings:
    checkpoint: Path
    checkpoint_sha256: str
    device: str = "cpu"
    cpu_threads: int = 1
    # A bounded JPEG upload can expand to a ~12 MB lossless 4 MP RGB PNG in transit.
    max_upload_bytes: int = 16 * 1024**2
    max_pixels: int = 4_000_000
    max_dimension: int = 8192
    upload_timeout_seconds: int = 30

    def __post_init__(self):
        if not re.fullmatch(r"[0-9a-f]{64}", self.checkpoint_sha256):
            raise ValueError("SEVERITY_CHECKPOINT_SHA256 must pin a trusted model")
        if self.device not in ("cpu", "mps") or any(
            type(v) is not int or v < 1
            for v in (
                self.cpu_threads,
                self.max_upload_bytes,
                self.max_pixels,
                self.max_dimension,
                self.upload_timeout_seconds,
            )
        ):
            raise ValueError("Invalid severity device/resource limits")

    @classmethod
    def from_env(cls):
        limits = {
            name: int(os.getenv(f"SEVERITY_{name.upper()}", str(default)))
            for name, default in (
                ("max_upload_bytes", 16 * 1024**2),
                ("max_pixels", 4_000_000),
                ("max_dimension", 8192),
                ("upload_timeout_seconds", 30),
            )
        }
        return cls(
            checkpoint=Path(os.environ["SEVERITY_CHECKPOINT"]),
            checkpoint_sha256=os.environ["SEVERITY_CHECKPOINT_SHA256"],
            device=os.getenv("SEVERITY_DEVICE", "cpu"),
            cpu_threads=int(os.getenv("SEVERITY_CPU_THREADS", "1")),
            **limits,
        )


def create_app(settings: SeveritySettings | None = None, *, predictor_factory=None) -> FastAPI:
    settings = settings or SeveritySettings.from_env()

    def initialize():
        if predictor_factory:
            return predictor_factory(settings)
        import torch

        from damage_analytics.serving.classifiers import SeverityClassifier
        from damage_analytics.training.runtime import configure_runtime

        torch.set_num_threads(settings.cpu_threads)
        device = configure_runtime(42, settings.device, False)
        return SeverityClassifier(settings.checkpoint, settings.checkpoint_sha256, device)

    @asynccontextmanager
    async def lifespan(app):
        app.state.predictor = await run_in_threadpool(initialize)
        try:
            yield
        finally:
            app.state.predictor = None

    app = FastAPI(title="Phi-Net component damage-level service", version="1.0", lifespan=lifespan)
    app.add_middleware(RequestMiddleware, settings=settings)
    gate = BoundedSemaphore(1)

    @app.exception_handler(ApiError)
    async def error(request: Request, exc: ApiError):
        return JSONResponse(
            {
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                    "request_id": getattr(request.state, "request_id", "unavailable"),
                }
            },
            status_code=exc.status,
        )

    @app.exception_handler(Exception)
    async def unexpected(request: Request, exc: Exception):
        return JSONResponse(
            {
                "error": {
                    "code": "model_error",
                    "message": "Severity inference failed",
                    "request_id": getattr(request.state, "request_id", "unavailable"),
                }
            },
            status_code=500,
        )

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/ready")
    async def ready():
        return {"status": "ready", "checkpoint_sha256": settings.checkpoint_sha256}

    @app.get("/models/current")
    async def metadata(request: Request):
        return request.app.state.predictor.metadata

    def infer(request, data, media_type):
        if not gate.acquire(blocking=False):
            raise ApiError(503, "inference_busy", "Severity worker is busy")
        try:
            upload = decode_upload(data, media_type, settings)
            result = request.app.state.predictor.predict(upload.image)
            return SeverityPrediction.model_validate(result)
        finally:
            gate.release()

    @app.post("/predict", response_model=SeverityPrediction, openapi_extra=IMAGE_UPLOAD_OPENAPI)
    async def predict(request: Request):
        async with request.form(max_files=1, max_fields=0) as form:
            items = list(form.multi_items())
            if len(items) != 1 or items[0][0] != "file" or not isinstance(items[0][1], UploadFile):
                raise ApiError(422, "invalid_upload", "Provide one JPEG/PNG file field named file")
            data = await items[0][1].read(settings.max_upload_bytes + 1)
            media_type = items[0][1].content_type
        return await run_in_threadpool(infer, request, data, media_type)

    return app
