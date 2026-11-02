"""Inspection, history and model contracts with separate crack/Phi-Net results."""

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field

from damage_analytics.serving.contracts import (
    CrackPrediction,
    SeverityPrediction,
    SeverityUnavailable,
)

IMAGE_UPLOAD_OPENAPI = {
    "requestBody": {
        "required": True,
        "content": {
            "multipart/form-data": {
                "schema": {
                    "type": "object",
                    "required": ["file"],
                    "properties": {"file": {"type": "string", "format": "binary"}},
                }
            }
        },
    }
}


class ClassMapping(BaseModel):
    class_id: int = Field(ge=0, le=8)
    source_label: str


class ModelMetadata(BaseModel):
    checkpoint_sha256: str
    checkpoint_epoch: int
    dataset: dict[str, Any]
    architecture: dict[str, Any]
    preprocessing: dict[str, Any]
    class_mapping: list[ClassMapping]
    device: Literal["cpu", "mps"]
    training_code_version: str | None
    confidence: str
    alignment: str
    severity_scope: str


class ArtifactReference(BaseModel):
    url: str
    sha256: str
    media_type: Literal["image/png"] = "image/png"


class DetectedClass(BaseModel):
    class_id: int = Field(ge=1, le=8)
    source_label: str
    area_pixels: int = Field(gt=0)
    image_area_percent: float = Field(ge=0, le=100)
    connected_regions: int = Field(ge=1)


class InspectionResponse(BaseModel):
    id: UUID
    created_at: datetime
    status: Literal["completed"] = "completed"
    width: int
    height: int
    upload_sha256: str
    source_media_type: str
    source_mode: str
    image: ArtifactReference
    mask: ArtifactReference
    detected_classes: list[DetectedClass]
    model: ModelMetadata
    analytics: dict[str, Any]  # Legacy/dashboard pixel view; Task 09 features stay separate.
    features: dict[str, Any] | None = None  # Validated Task 09 native-resolution report.
    classification: CrackPrediction | None = None
    severity: SeverityPrediction | SeverityUnavailable | None = None


class HistoryItem(BaseModel):
    id: UUID
    created_at: datetime
    width: int
    height: int
    classification: CrackPrediction | None = None
    severity: SeverityPrediction | SeverityUnavailable | None = None


class HistoryPage(BaseModel):
    items: list[HistoryItem]
    next_before: datetime | None = None
    next_before_id: UUID | None = None


class Readiness(BaseModel):
    status: Literal["ready", "not_ready"]
    model: bool
    database: bool
    severity: bool | None = None


class ErrorDetail(BaseModel):
    code: str
    message: str
    request_id: str


class ErrorResponse(BaseModel):
    error: ErrorDetail
