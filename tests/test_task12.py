"""Integrated HTTP contracts, native geometry, failures and opt-in PostgreSQL persistence."""

# ruff: noqa: E402 -- Check optional dependencies before importing serving modules.
import hashlib
import io
import json
import os

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")
from fastapi.testclient import TestClient
from pydantic import ValidationError
from test_api_core import FakePredictor, FakeRepository, image_bytes

from damage_analytics.api.app import create_app
from damage_analytics.api.config import Settings
from damage_analytics.api.inference import Prediction
from damage_analytics.api.repository import PostgresRepository
from damage_analytics.serving.contracts import CrackPrediction, SeverityPrediction
from damage_analytics.serving.segmentation import compatibility_analytics
from damage_analytics.serving.severity_app import SeveritySettings
from damage_analytics.serving.severity_app import create_app as severity_app
from damage_analytics.serving.severity_client import SeverityClient
from damage_analytics.task09.config import FeaturePolicy
from damage_analytics.task09.features import extract_features


def provenance(digest="2" * 64):
    return {
        "checkpoint_sha256": digest,
        "epoch": 1,
        "architecture": "resnet18",
        "dataset": {"release_id": "synthetic"},
        "preprocessing": {},
        "device": "cpu",
        "limitations": ["Synthetic fixture; no safety claim"],
    }


class MemoryRepository(FakeRepository):
    def history(self, limit, before=None):
        records = sorted(self.records.values(), key=lambda r: (r.created_at, r.id), reverse=True)
        if before:
            records = [r for r in records if (r.created_at, r.id) < before]
        return [
            {
                key: getattr(r, key)
                for key in ("id", "created_at", "width", "height", "classification", "severity")
            }
            for r in records[:limit]
        ]


class FixtureSeverity:
    metadata = provenance()

    def __init__(self):
        self.calls = 0

    def predict(self, image):
        self.calls += 1
        return SeverityPrediction(
            class_id=2,
            label="Moderate",
            confidence=0.6,
            probabilities=[
                {"class_id": i, "label": name, "probability": p}
                for i, (name, p) in enumerate(
                    zip(
                        ["Heavy", "Minor", "Moderate", "Undamaged"],
                        [0.1, 0.2, 0.6, 0.1],
                        strict=True,
                    )
                )
            ],
            normalized_entropy=0.8,
            width=image.width,
            height=image.height,
            input_rgb_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
            model=self.metadata,
        )


class FixtureIntegrated(FakePredictor):
    def predict(self, image):
        self.calls += 1
        mask = np.zeros((image.height, image.width), dtype=np.uint8)
        mask[1:3, 2:5] = 3
        policy = FeaturePolicy()
        features, _ = extract_features(
            mask,
            policy,
            maximum_softmax=np.full(mask.shape, 0.8),
            normalized_entropy=np.full(mask.shape, 0.2),
        )
        crack = CrackPrediction(
            class_id=1,
            label="crack",
            crack_probability=0.7,
            decision_threshold=0.4,
            threshold_provenance={"selected_on": "validation"},
            model=provenance("1" * 64),
        ).model_dump(mode="json")
        return Prediction(mask, compatibility_analytics(features, policy), features, crack)


@pytest.fixture
def integration(tmp_path):
    settings = Settings(
        "postgresql://unused/testing",
        tmp_path / "seg.pt",
        "0" * 64,
        storage_dir=tmp_path / "artifacts",
        crack_checkpoint=tmp_path / "crack.pt",
        crack_checkpoint_sha256="1" * 64,
        severity_url="http://severity",
        severity_checkpoint_sha256="2" * 64,
    )
    severity_settings = SeveritySettings(tmp_path / "severity.pt", "2" * 64)
    model, segmentation, repository, loads = (
        FixtureSeverity(),
        FixtureIntegrated(),
        MemoryRepository(),
        [],
    )

    def severity_factory(_):
        loads.append("severity")
        return model

    def integrated_factory(_):
        loads.append("segmentation_and_crack")
        return segmentation

    with TestClient(severity_app(severity_settings, predictor_factory=severity_factory)) as remote:

        def transport(request):
            response = remote.request(
                request.method, request.url.path, headers=request.headers, content=request.read()
            )
            return httpx.Response(
                response.status_code,
                content=response.content,
                headers=response.headers,
                request=request,
            )

        client = SeverityClient(
            settings,
            client=httpx.Client(
                base_url="http://severity", transport=httpx.MockTransport(transport)
            ),
        )
        app = create_app(
            settings,
            repository=repository,
            predictor_factory=integrated_factory,
            severity_client_factory=lambda _: client,
        )
        with TestClient(app) as browser:
            yield browser, remote, repository, segmentation, model, loads, client
        assert client.client.is_closed


def test_upload_http_severity_persist_retrieve_history_and_no_reload(integration):
    browser, _, repository, model, severity, loads, _ = integration
    for _ in range(2):
        response = browser.post(
            "/inspections", files={"file": ("x.png", image_bytes(), "image/png")}
        )
        assert response.status_code == 201, response.text
        result = response.json()
        assert result["classification"]["label"] == "crack"
        assert result["severity"]["class_id"] == 2 and result["severity"]["calibrated"] is False
        assert result["features"]["schema_version"] == "damage-features-v1"
        assert result["analytics"]["regions"][0]["bbox_xyxy_exclusive"] == [2, 1, 5, 3]
        assert result["analytics"]["regions"][0]["centroid_xy_pixel_index"] == [3.0, 1.5]
        assert browser.get(f"/inspections/{result['id']}").json() == result
        raw = browser.get(result["mask"]["url"])
        with Image.open(io.BytesIO(raw.content)) as mask:
            assert mask.size == (8, 6) and np.count_nonzero(np.array(mask)) == 6
    assert len(repository.records) == 2 and model.calls == severity.calls == 2
    assert sorted(loads) == ["segmentation_and_crack", "severity"]
    page = browser.get("/inspections?limit=1").json()
    next_page = browser.get(
        "/inspections",
        params={"limit": 1, "before": page["next_before"], "before_id": page["next_before_id"]},
    ).json()
    assert len(page["items"]) == len(next_page["items"]) == 1
    assert page["items"][0]["id"] != next_page["items"][0]["id"]
    assert browser.get("/inspections?limit=101").status_code == 422
    assert browser.get("/inspections?before=2026-01-01").status_code == 422
    assert browser.get("/ready").status_code == 200


def test_service_outage_is_persisted_as_unavailable_without_fake_prediction(integration):
    browser, _, repository, _, _, _, gateway = integration
    gateway.client.close()

    def offline(request):
        raise httpx.ConnectError("private connection details", request=request)

    gateway.client = httpx.Client(
        base_url="http://severity", transport=httpx.MockTransport(offline)
    )
    assert browser.get("/ready").status_code == 503
    response = browser.post("/inspections", files={"file": ("x.png", image_bytes(), "image/png")})
    assert response.status_code == 201
    result = response.json()
    assert result["severity"]["status"] == "unavailable" and "label" not in result["severity"]
    assert result["classification"]["label"] == "crack" and repository.records
    assert "private" not in response.text


@pytest.mark.parametrize("fault", ["pin", "pixels", "mapping", "calibrated", "size", "malformed"])
def test_untrusted_severity_response_never_becomes_a_prediction(tmp_path, fault):
    settings = Settings(
        "postgresql://unused/test",
        tmp_path / "seg.pt",
        "0" * 64,
        storage_dir=tmp_path / "storage",
        crack_checkpoint=tmp_path / "crack.pt",
        crack_checkpoint_sha256="1" * 64,
        severity_url="http://severity",
        severity_checkpoint_sha256="2" * 64,
    )
    image = Image.new("RGB", (4, 3))
    document = FixtureSeverity().predict(image).model_dump(mode="json")
    if fault == "pin":
        document["model"]["checkpoint_sha256"] = "3" * 64
    if fault == "pixels":
        document["input_rgb_sha256"] = "f" * 64
    if fault == "mapping":
        document["label"] = "Heavy"
    if fault == "calibrated":
        document["calibrated"] = True
    if fault == "size":
        document["width"] += 1
    content = b"invalid" if fault == "malformed" else json.dumps(document).encode()
    client = SeverityClient(
        settings,
        client=httpx.Client(
            base_url="http://severity",
            transport=httpx.MockTransport(lambda _: httpx.Response(200, content=content)),
        ),
    )
    try:
        result = client.predict(image)
        assert result["status"] == "unavailable" and result["reason"] == "invalid_service_response"
        assert "confidence" not in result
    finally:
        client.close()


def test_severity_upload_validation_and_no_reload(integration):
    _, remote, _, _, severity, loads, _ = integration
    before = severity.calls
    contract = remote.get("/openapi.json").json()["paths"]["/predict"]["post"]["requestBody"]
    assert contract["content"]["multipart/form-data"]["schema"]["required"] == ["file"]
    assert remote.get("/health").json() == {"status": "ok"}
    assert (
        remote.post("/predict", files={"file": ("x.png", b"bad", "image/png")}).status_code == 422
    )
    assert (
        remote.post(
            "/predict", files={"file": ("x.png", image_bytes("RGBA"), "image/png")}
        ).status_code
        == 422
    )
    assert (
        remote.post("/predict", files={"file": ("x.png", image_bytes(), "image/jpeg")}).status_code
        == 415
    )
    assert severity.calls == before and loads.count("severity") == 1


def test_lossless_internal_png_expansion_of_valid_jpeg(integration):
    _, remote, _, _, _, _, _ = integration
    # A legal external JPEG can exceed the external byte limit after lossless transport.
    pixels = np.random.default_rng(17).integers(0, 256, (2000, 2000, 3), dtype=np.uint8)
    jpeg = io.BytesIO()
    Image.fromarray(pixels).save(jpeg, format="JPEG", quality=95)
    assert len(jpeg.getvalue()) < 10 * 1024**2
    png = io.BytesIO()
    with Image.open(io.BytesIO(jpeg.getvalue())) as image:
        image.convert("RGB").save(png, format="PNG")
    assert len(png.getvalue()) > 10 * 1024**2
    response = remote.post(
        "/predict", files={"file": ("canonical.png", png.getvalue(), "image/png")}
    )
    assert response.status_code == 200, response.text
    assert response.json()["width"] == response.json()["height"] == 2000


def test_heavy_zero_is_a_nominal_class_and_invalid_probabilities_rejected():
    value = FixtureSeverity().predict(Image.new("RGB", (2, 2))).model_dump(mode="json")
    value.update(class_id=0, label="Heavy", confidence=0.7)
    for row, probability in zip(value["probabilities"], [0.7, 0.1, 0.1, 0.1], strict=True):
        row["probability"] = probability
    assert SeverityPrediction.model_validate(value).label == "Heavy"
    value["probabilities"][0]["label"] = "Undamaged"
    with pytest.raises(ValidationError):
        SeverityPrediction.model_validate(value)


def test_invalid_exif_rejected_and_orientation_applied_once(integration):
    browser, _, _, _, _, _, _ = integration
    for orientation, expected in [(6, (6, 8)), (999, None)]:
        image = Image.new("RGB", (8, 6))
        exif = Image.Exif()
        exif[274] = orientation
        stream = io.BytesIO()
        image.save(stream, format="JPEG", exif=exif)
        response = browser.post(
            "/inspections", files={"file": ("x.jpg", stream.getvalue(), "image/jpeg")}
        )
        if expected:
            result = response.json()
            assert response.status_code == 201 and (result["width"], result["height"]) == expected
            assert (result["severity"]["width"], result["severity"]["height"]) == expected
        else:
            assert response.status_code == 422


def test_integrated_settings_require_all_pins(tmp_path):
    with pytest.raises(ValueError, match="Integrated serving needs"):
        Settings(
            "postgresql://unused/test",
            tmp_path / "m.pt",
            "0" * 64,
            storage_dir=tmp_path / "out",
            crack_checkpoint=tmp_path / "crack.pt",
        )


@pytest.mark.skipif(
    not os.getenv("DAMAGE_TASK12_TEST_DATABASE_URL"), reason="Opt-in disposable PostgreSQL"
)
def test_live_postgres_jsonb_roundtrip_and_history(tmp_path):
    from damage_analytics.api.service import InspectionService
    from damage_analytics.api.storage import ArtifactStore
    from damage_analytics.serving.migrate import migrate

    url = os.environ["DAMAGE_TASK12_TEST_DATABASE_URL"]
    migrate(url)
    assert migrate(url) == []
    repository = PostgresRepository(url)
    assert repository.ready()
    settings = Settings(url, tmp_path / "m.pt", "0" * 64, storage_dir=tmp_path / "artifacts")
    service = InspectionService(
        settings, FixtureIntegrated(), repository, ArtifactStore(settings.storage_dir)
    )
    record = service.predict(image_bytes(), "image/png")
    try:
        assert repository.get(record.id) == record
        from damage_analytics.api.routes import _response

        assert _response(repository.get(record.id)).model_dump(mode="json") == _response(
            record
        ).model_dump(mode="json")
        assert record.features and record.classification and record.severity is None
        assert record.id in {row["id"] for row in repository.history(100)}
    finally:
        with repository._connection() as connection:
            connection.execute("DELETE FROM inspections WHERE id=%s", (record.id,))
            assert (
                connection.execute(
                    "SELECT 1 FROM predictions WHERE inspection_id=%s", (record.id,)
                ).fetchone()
                is None
            )


@pytest.mark.skipif(
    not os.getenv("DAMAGE_TASK12_TEST_DATABASE_URL"), reason="Opt-in disposable PostgreSQL"
)
def test_additive_migration_preserves_version_one_records():
    from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
    from uuid import uuid4

    import psycopg
    from psycopg import sql

    from damage_analytics.serving.migrate import migrate

    original = os.environ["DAMAGE_TASK12_TEST_DATABASE_URL"]
    schema = "task12_migration_" + uuid4().hex
    with psycopg.connect(original, autocommit=True) as connection:
        connection.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    parts = urlsplit(original)
    query = dict(parse_qsl(parts.query))
    query["options"] = "-csearch_path=" + schema
    isolated = urlunsplit(parts._replace(query=urlencode(query)))
    identifier = uuid4()
    try:
        with psycopg.connect(isolated, autocommit=True) as connection:
            connection.execute(open("migrations/001_initial.sql").read(), prepare=False)
            connection.execute(
                """INSERT INTO inspections
                (id,created_at,upload_sha256,image_key,image_sha256,width,height,source_media_type,source_mode)
                VALUES (%s,now(),%s,'legacy-image',%s,3,2,'image/png','RGB')""",
                (identifier, "0" * 64, "0" * 64),
            )
            connection.execute(
                """INSERT INTO predictions
                (inspection_id,mask_key,mask_sha256,checkpoint_sha256,model_metadata,analytics)
                VALUES (%s,'legacy-mask',%s,%s,'{}','{}')""",
                (identifier, "0" * 64, "0" * 64),
            )
        assert migrate(isolated) == [2]
        assert migrate(isolated) == []
        with psycopg.connect(isolated) as connection:
            row = connection.execute(
                "SELECT mask_key,features,classification,severity "
                "FROM predictions WHERE inspection_id=%s",
                (identifier,),
            ).fetchone()
            assert row == ("legacy-mask", None, None, None)
    finally:
        with psycopg.connect(original, autocommit=True) as connection:
            connection.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))
