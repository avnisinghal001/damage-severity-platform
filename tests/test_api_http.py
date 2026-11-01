"""HTTP integration with a fake repository/model; skipped when API extras are absent."""

# Optional framework imports intentionally follow module-level dependency checks.
# ruff: noqa: E402

from concurrent.futures import ThreadPoolExecutor
from threading import Event
from uuid import uuid4

import pytest

pytest.importorskip("fastapi", reason="Backend dependencies are not installed")
pytest.importorskip("httpx", reason="HTTP test dependency is not installed")
pytest.importorskip("python_multipart", reason="Multipart dependency is not installed")
from fastapi.testclient import TestClient
from test_api_core import FakePredictor, FakeRepository, image_bytes

from damage_analytics.api.app import create_app
from damage_analytics.api.config import Settings


@pytest.fixture
def client(tmp_path):
    settings = Settings(
        "postgresql://unused/testing",
        tmp_path / "unused.pt",
        "0" * 64,
        storage_dir=tmp_path / "artifacts",
    )
    repository, model, loads = FakeRepository(), FakePredictor(), []

    def factory(_settings):
        loads.append(1)
        return model

    app = create_app(settings, repository=repository, predictor_factory=factory)
    with TestClient(app) as browser:
        yield browser, repository, model, loads
    assert app.state.service is None


def test_upload_retrieval_artifact_and_single_startup_load(client):
    browser, _, model, loads = client
    result = browser.post(
        "/inspections", files={"file": ("../../bad.png", image_bytes(), "image/png")}
    )
    assert result.status_code == 201
    body = result.json()
    assert body["detected_classes"][0]["class_id"] == 3
    assert body["analytics"]["overall_structural_severity"]["score"] is None
    assert browser.get(f"/inspections/{body['id']}").json() == body
    assert browser.get(body["mask"]["url"]).headers["content-type"] == "image/png"
    assert browser.get("/models/current").json()["checkpoint_sha256"] == "0" * 64
    assert loads == [1] and model.calls == 1


def test_health_readiness_errors_and_request_ids(client):
    browser, repository, _, _ = client
    assert browser.get("/health").status_code == 200
    assert browser.get("/ready").status_code == 200
    repository.available = False
    assert browser.get("/ready").status_code == 503
    assert browser.get("/health").status_code == 200
    for path, status in ((f"/inspections/{uuid4()}", 404), ("/inspections/not-a-uuid", 422)):
        response = browser.get(path)
        assert response.status_code == status
        assert response.json()["error"]["request_id"] == response.headers["x-request-id"]


def test_invalid_multipart_content_and_extra_files(client):
    browser, _, model, _ = client
    assert browser.post("/inspections").status_code == 422
    assert (
        browser.post("/inspections", files={"file": ("x.png", b"bad", "image/png")}).status_code
        == 422
    )
    result = browser.post(
        "/inspections",
        files=[
            ("file", ("x.png", image_bytes(), "image/png")),
            ("file", ("y.png", image_bytes(), "image/png")),
        ],
    )
    assert result.status_code == 400 and model.calls == 0


def test_body_limit_without_content_length(tmp_path):
    settings = Settings(
        "postgresql://unused/testing",
        tmp_path / "unused.pt",
        "0" * 64,
        storage_dir=tmp_path / "files",
        max_upload_bytes=1,
    )
    app = create_app(
        settings, repository=FakeRepository(), predictor_factory=lambda _: FakePredictor()
    )
    with TestClient(app) as browser:
        response = browser.post("/inspections", content=iter([b"x" * 65538]))
        assert response.status_code == 413
        assert response.json()["error"]["code"] == "upload_too_large"


def test_health_remains_available_and_concurrent_upload_is_rejected(client):
    browser, _, model, _ = client
    started, release = Event(), Event()
    original = model.predict

    def delayed(image):
        started.set()
        assert release.wait(3)
        return original(image)

    model.predict = delayed
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(
            browser.post, "/inspections", files={"file": ("x.png", image_bytes(), "image/png")}
        )
        assert started.wait(3)
        try:
            assert browser.get("/health").status_code == 200
            busy = browser.post(
                "/inspections", files={"file": ("x.png", image_bytes(), "image/png")}
            )
            assert busy.status_code == 503 and busy.headers["retry-after"] == "1"
        finally:
            release.set()
        assert first.result().status_code == 201
