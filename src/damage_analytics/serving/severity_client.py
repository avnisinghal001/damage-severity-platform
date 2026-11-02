"""Bounded HTTP boundary; failures are unavailable states, never fabricated predictions."""

import hashlib
import io
import logging
from typing import Protocol

import httpx
from PIL import Image
from pydantic import ValidationError

from damage_analytics.api.config import Settings
from damage_analytics.serving.contracts import SeverityPrediction, SeverityUnavailable

logger = logging.getLogger("damage_analytics.api")


class SeverityGateway(Protocol):
    def predict(self, image: Image.Image) -> dict: ...
    def ready(self) -> bool: ...
    def close(self) -> None: ...


class SeverityClient:
    def __init__(self, settings: Settings, *, client: httpx.Client | None = None):
        self.digest = settings.severity_checkpoint_sha256
        self.client = client or httpx.Client(
            base_url=settings.severity_url.rstrip("/"),
            timeout=httpx.Timeout(
                settings.severity_timeout_seconds, connect=2.0, write=10.0, pool=2.0
            ),
            follow_redirects=False,
            trust_env=False,
        )

    def ready(self) -> bool:
        try:
            response = self.client.get("/ready", timeout=2.0)
            return (
                response.status_code == 200
                and response.json().get("status") == "ready"
                and response.json().get("checkpoint_sha256") == self.digest
            )
        except (httpx.HTTPError, ValueError, TypeError, AttributeError):
            return False

    def predict(self, image: Image.Image) -> dict:
        encoded = io.BytesIO()
        image.save(encoded, format="PNG")
        try:
            with self.client.stream(
                "POST", "/predict", files={"file": ("image.png", encoded.getvalue(), "image/png")}
            ) as response:
                response.raise_for_status()
                content = bytearray()
                for chunk in response.iter_bytes():
                    content.extend(chunk)
                    if len(content) > 128 * 1024:
                        raise ValueError("Severity response exceeded the contract size limit")
            result = SeverityPrediction.model_validate_json(content)
            if (
                result.width != image.width
                or result.height != image.height
                or result.input_rgb_sha256 != hashlib.sha256(image.tobytes()).hexdigest()
                or result.model.checkpoint_sha256 != self.digest
            ):
                raise ValueError("Severity response input or checkpoint provenance differs")
            return result.model_dump(mode="json")
        except httpx.HTTPError:
            reason, message = (
                "service_unavailable",
                "Severity service is unavailable; no prediction returned",
            )
        except (ValidationError, ValueError, TypeError):
            reason, message = (
                "invalid_service_response",
                "Severity response failed validation; no prediction returned",
            )
        logger.warning("severity_unavailable", extra={"reason": reason})
        return SeverityUnavailable(reason=reason, message=message).model_dump(mode="json")

    def close(self) -> None:
        self.client.close()
