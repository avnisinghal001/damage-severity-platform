"""Validated classification contracts shared by the backend and severity HTTP service."""

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from damage_analytics.task10.config import CLASS_NAMES


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Provenance(Contract):
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    epoch: int = Field(ge=1)
    architecture: Literal["resnet18"] = "resnet18"
    dataset: dict
    preprocessing: dict
    device: Literal["cpu", "mps"]
    limitations: list[str]


class Probability(Contract):
    class_id: int = Field(ge=0, le=3)
    label: str
    probability: float = Field(ge=0, le=1)


class SeverityPrediction(Contract):
    status: Literal["available"] = "available"
    class_id: int = Field(ge=0, le=3)
    label: Literal["Heavy", "Minor", "Moderate", "Undamaged"]
    confidence: float = Field(ge=0, le=1)
    calibrated: Literal[False] = False
    probabilities: list[Probability] = Field(min_length=4, max_length=4)
    normalized_entropy: float = Field(ge=0, le=1)
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    input_rgb_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model: Provenance
    scope: Literal["visible_image_component_damage_level"] = "visible_image_component_damage_level"

    @model_validator(mode="after")
    def consistent(self):
        scores = [p.probability for p in self.probabilities]
        if (
            [(p.class_id, p.label) for p in self.probabilities] != list(enumerate(CLASS_NAMES))
            or not math.isclose(sum(scores), 1, abs_tol=1e-5)
            or self.class_id != max(range(4), key=lambda i: scores[i])
            or self.label != CLASS_NAMES[self.class_id]
            or not math.isclose(self.confidence, scores[self.class_id], abs_tol=1e-6)
        ):
            raise ValueError("Severity IDs, labels, decision and probabilities disagree")
        return self


class SeverityUnavailable(Contract):
    status: Literal["unavailable"] = "unavailable"
    reason: Literal["service_unavailable", "invalid_service_response", "not_configured"]
    message: str


class CrackPrediction(Contract):
    class_id: Literal[0, 1]
    label: Literal["no_crack", "crack"]
    crack_probability: float = Field(ge=0, le=1)
    calibrated: Literal[False] = False
    decision_threshold: float = Field(ge=0, le=1.000001)
    threshold_provenance: dict
    model: Provenance
    scope: Literal["crack_presence_only_not_general_damage"] = (
        "crack_presence_only_not_general_damage"
    )

    @model_validator(mode="after")
    def consistent(self):
        predicted = int(self.crack_probability >= self.decision_threshold)
        if self.class_id != predicted or self.label != ("no_crack", "crack")[predicted]:
            raise ValueError("Crack decision differs from frozen checkpoint threshold")
        return self
