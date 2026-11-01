"""FastAPI application factory; startup loads one model and never executes migrations."""

import logging
from collections.abc import Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException

from damage_analytics.api.config import Settings
from damage_analytics.api.errors import ApiError
from damage_analytics.api.inference import Predictor, Segmenter
from damage_analytics.api.logging import configure_logging
from damage_analytics.api.middleware import RequestMiddleware
from damage_analytics.api.repository import PostgresRepository, Repository
from damage_analytics.api.routes import router
from damage_analytics.api.service import InspectionService
from damage_analytics.api.storage import ArtifactStore

logger = logging.getLogger("damage_analytics.api")


def create_app(
    settings: Settings | None = None,
    *,
    repository: Repository | None = None,
    predictor_factory: Callable[[Settings], Predictor] | None = None,
) -> FastAPI:
    """Construct an app with injectable boundaries for tests; imports do not load weights."""
    settings = settings or Settings.from_env()
    configure_logging()
    repository = repository or PostgresRepository(
        settings.database_url, settings.database_timeout_seconds
    )
    predictor_factory = predictor_factory or Segmenter

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        def initialize() -> InspectionService:
            predictor = predictor_factory(settings)
            return InspectionService(
                settings, predictor, repository, ArtifactStore(settings.storage_dir)
            )

        app.state.service = await run_in_threadpool(initialize)
        logger.info("model_loaded")
        try:
            yield
        finally:
            app.state.service = None  # Release the worker's model reference on shutdown.
            logger.info("worker_stopped")

    app = FastAPI(title="Damage Analytics API", version="0.1.0", lifespan=lifespan)
    app.add_middleware(RequestMiddleware, settings=settings)
    app.include_router(router)

    def error(request: Request, status: int, code: str, message: str) -> JSONResponse:
        headers = {"Retry-After": "1"} if status == 503 else {}
        if status == 500:  # ServerErrorMiddleware sends outside the user middleware stack.
            headers.update(
                {
                    "X-Request-ID": getattr(request.state, "request_id", "unavailable"),
                    "X-Content-Type-Options": "nosniff",
                    "Cache-Control": "no-store",
                }
            )
        return JSONResponse(
            {
                "error": {
                    "code": code,
                    "message": message,
                    "request_id": getattr(request.state, "request_id", "unavailable"),
                }
            },
            status_code=status,
            headers=headers,
        )

    @app.exception_handler(ApiError)
    async def domain_error(request: Request, exc: ApiError) -> JSONResponse:
        return error(request, exc.status, exc.code, exc.message)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError) -> JSONResponse:
        return error(request, 422, "invalid_request", "Request fields are invalid")

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException) -> JSONResponse:
        return error(request, exc.status_code, "http_error", "Request could not be accepted")

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        logger.error(
            "request_failed",
            extra={
                "request_id": getattr(request.state, "request_id", None),
                "error_type": type(exc).__name__,
            },
        )
        return error(request, 500, "internal_error", "The request could not be completed")

    return app
