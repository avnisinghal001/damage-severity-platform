"""HTTP contracts around the preserved Task 04 analytics document."""

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field


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
    analytics: dict[str, Any]  # Task 04 schema: full regions, complexes and uncertainty retained.


class Readiness(BaseModel):
    status: Literal["ready", "not_ready"]
    model: bool
    database: bool


class ErrorDetail(BaseModel):
    code: str
    message: str
    request_id: str


class ErrorResponse(BaseModel):
    error: ErrorDetail
