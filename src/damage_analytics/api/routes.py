"""Thin HTTP routes; decoding, model work, filesystem IO and SQL run off the event loop."""

from datetime import datetime
from typing import Literal
from uuid import UUID

from fastapi import APIRouter, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile

from damage_analytics.api.errors import ApiError
from damage_analytics.api.repository import Inspection
from damage_analytics.api.schemas import (
    IMAGE_UPLOAD_OPENAPI,
    ErrorResponse,
    HistoryPage,
    InspectionResponse,
    ModelMetadata,
    Readiness,
)
from damage_analytics.api.service import InspectionService

router = APIRouter(
    responses={code: {"model": ErrorResponse} for code in (400, 404, 408, 413, 415, 422, 500, 503)}
)


def _service(request: Request) -> InspectionService:
    return request.app.state.service


def _response(record: Inspection) -> InspectionResponse:
    """Return stable artifact URLs; never expose storage paths or user filenames."""
    base = f"/inspections/{record.id}/artifacts"
    return InspectionResponse(
        id=record.id,
        created_at=record.created_at,
        width=record.width,
        height=record.height,
        upload_sha256=record.upload_sha256,
        source_media_type=record.source_media_type,
        source_mode=record.source_mode,
        image={"url": f"{base}/image", "sha256": record.image_sha256},
        mask={"url": f"{base}/mask", "sha256": record.mask_sha256},
        detected_classes=[
            row
            for row in record.analytics["classes"]
            if row["class_id"] > 0 and row["area_pixels"] > 0
        ],
        model=record.model_metadata,
        analytics=record.analytics,
        features=record.features,
        classification=record.classification,
        severity=record.severity,
    )


@router.get("/health")
async def health() -> dict[str, str]:
    """Liveness only; database connectivity does not determine whether the process is alive."""
    return {"status": "ok"}


@router.get("/ready", response_model=Readiness, responses={503: {"model": Readiness}})
async def ready(request: Request) -> JSONResponse:
    """Check the loaded model and migration/schema readiness without running inference."""
    database = await run_in_threadpool(_service(request).database_ready)
    gateway = _service(request).severity
    severity = await run_in_threadpool(gateway.ready) if gateway else None
    healthy = database and severity is not False
    return JSONResponse(
        {
            "status": "ready" if healthy else "not_ready",
            "model": True,
            "database": database,
            "severity": severity,
        },
        status_code=200 if healthy else 503,
    )


@router.get("/models/current", response_model=ModelMetadata)
async def model_metadata(request: Request) -> dict:
    """Describe the exact resident checkpoint, preprocessing, taxonomy and scientific scope."""
    return _service(request).predictor.metadata


@router.post(
    "/inspections",
    response_model=InspectionResponse,
    status_code=201,
    openapi_extra=IMAGE_UPLOAD_OPENAPI,
)
async def predict(request: Request) -> InspectionResponse:
    """Accept exactly one multipart file, with bounded bytes and no extra form fields."""
    service = _service(request)
    async with request.form(max_files=1, max_fields=0) as form:
        items = list(form.multi_items())
        if len(items) != 1 or items[0][0] != "file" or not isinstance(items[0][1], UploadFile):
            raise ApiError(422, "invalid_upload", "Provide exactly one file field named file")
        upload = items[0][1]
        data = await upload.read(service.settings.max_upload_bytes + 1)
        media_type = upload.content_type
    record = await run_in_threadpool(
        service.predict, data, media_type, request_id=request.state.request_id
    )
    return _response(record)


@router.get("/inspections", response_model=HistoryPage)
async def history(
    request: Request,
    limit: int = Query(default=20, ge=1, le=100),
    before: datetime | None = None,
    before_id: UUID | None = None,
) -> HistoryPage:
    if (before is None) != (before_id is None) or (before is not None and before.tzinfo is None):
        raise ApiError(422, "invalid_cursor", "Use both a timezone-aware before and before_id")
    cursor = (before, before_id) if before is not None else None
    rows = await run_in_threadpool(_service(request).history, limit, cursor)
    has_more = len(rows) == limit
    return HistoryPage(
        items=rows,
        next_before=rows[-1]["created_at"] if has_more else None,
        next_before_id=rows[-1]["id"] if has_more else None,
    )


@router.get("/inspections/{identifier}", response_model=InspectionResponse)
async def inspection(identifier: UUID, request: Request) -> InspectionResponse:
    """Retrieve the stored result without loading the model again or repeating prediction."""
    return _response(await run_in_threadpool(_service(request).get, identifier))


@router.get("/inspections/{identifier}/artifacts/{kind}", response_class=FileResponse)
async def artifact(identifier: UUID, kind: Literal["image", "mask"], request: Request):
    """Serve only validated, database-backed PNG keys, never a client-supplied file path."""
    service = _service(request)
    record = await run_in_threadpool(service.get, identifier)
    key, fingerprint = (
        (record.image_key, record.image_sha256)
        if kind == "image"
        else (record.mask_key, record.mask_sha256)
    )
    try:
        path = await run_in_threadpool(service.storage.file, identifier, key, kind, fingerprint)
    except (OSError, ValueError) as exc:
        raise ApiError(503, "artifact_unavailable", "The stored artifact is unavailable") from exc
    return FileResponse(path, media_type="image/png", filename=path.name)
