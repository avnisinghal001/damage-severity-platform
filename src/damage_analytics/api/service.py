"""Synchronous application service, invoked in a worker thread by the async HTTP routes."""

import hashlib
import logging
from datetime import UTC, datetime
from threading import BoundedSemaphore
from uuid import UUID, uuid4

from damage_analytics.api.config import Settings
from damage_analytics.api.errors import ApiError
from damage_analytics.api.inference import Predictor
from damage_analytics.api.repository import Inspection, Repository
from damage_analytics.api.storage import ArtifactStore
from damage_analytics.api.uploads import decode_upload
from damage_analytics.serving.severity_client import SeverityGateway

logger = logging.getLogger("damage_analytics.api")


class InspectionService:
    """Serialize resident-model inference and reject busy requests without queuing."""

    def __init__(
        self,
        settings: Settings,
        predictor: Predictor,
        repository: Repository,
        storage: ArtifactStore,
        severity: SeverityGateway | None = None,
    ) -> None:
        self.settings, self.predictor = settings, predictor
        self.repository, self.storage = repository, storage
        self.severity = severity
        self._gate = BoundedSemaphore(1)

    def database_ready(self) -> bool:
        """Translate dependency failures into readiness status without leaking connection URLs."""
        try:
            return self.repository.ready()
        except Exception:
            return False

    def predict(
        self, data: bytes, media_type: str | None, *, request_id: str | None = None
    ) -> Inspection:
        """Validate, infer, publish, then commit metadata; retain files on ambiguous DB failure."""
        if not self._gate.acquire(blocking=False):
            raise ApiError(503, "inference_busy", "This worker is busy; retry shortly")
        try:
            upload = decode_upload(data, media_type, self.settings)
            if not self.database_ready():
                raise ApiError(503, "database_unavailable", "Inspection storage is not ready")
            identifier, created_at = uuid4(), datetime.now(UTC)
            prediction = self.predictor.predict(upload.image)
            severity = self.severity.predict(upload.image) if self.severity else None
            artifacts = self.storage.publish(identifier, upload.image, prediction.mask)
            record = Inspection(
                identifier,
                created_at,
                hashlib.sha256(data).hexdigest(),
                artifacts.image_key,
                artifacts.image_sha256,
                upload.image.width,
                upload.image.height,
                upload.media_type,
                upload.source_mode,
                artifacts.mask_key,
                artifacts.mask_sha256,
                self.predictor.metadata,
                prediction.analytics,
                prediction.features,
                prediction.crack,
                severity,
            )
            try:
                self.repository.save(record)
            except Exception as exc:
                # A lost commit acknowledgement may mean rows actually committed. Deleting
                # files here could break a committed inspection. Operators reconcile by UUID.
                logger.error(
                    "persistence_failed",
                    extra={
                        "inspection_id": str(identifier),
                        "error_type": type(exc).__name__,
                        "request_id": request_id,
                    },
                )
                raise ApiError(
                    503, "database_unavailable", "Inspection could not be confirmed"
                ) from exc
            return record
        finally:
            self._gate.release()

    def history(self, limit: int, before: tuple | None = None) -> list[dict]:
        try:
            return self.repository.history(limit, before)
        except Exception as exc:
            raise ApiError(
                503, "database_unavailable", "Inspection history is unavailable"
            ) from exc

    def get(self, identifier: UUID) -> Inspection:
        """Fetch persisted results without rerunning inference."""
        try:
            record = self.repository.get(identifier)
        except Exception as exc:
            raise ApiError(
                503, "database_unavailable", "Inspection storage is unavailable"
            ) from exc
        if record is None:
            raise ApiError(404, "inspection_not_found", "Inspection was not found")
        return record
