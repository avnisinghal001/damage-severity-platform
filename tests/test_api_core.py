"""Quick serving tests without FastAPI, PostgreSQL, real checkpoints or network access."""

import io
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Event
from uuid import uuid4

import numpy as np
import pytest
from PIL import Image

from damage_analytics.analytics.config import RegionPolicy, UncertaintyPolicy
from damage_analytics.analytics.regions import analyze_mask
from damage_analytics.api.config import Settings
from damage_analytics.api.errors import ApiError
from damage_analytics.api.inference import Prediction
from damage_analytics.api.repository import PostgresRepository
from damage_analytics.api.service import InspectionService
from damage_analytics.api.storage import ArtifactStore
from damage_analytics.api.uploads import decode_upload
from damage_analytics.labels import CLASS_NAMES


@pytest.fixture
def api_settings(tmp_path):
    return Settings(
        "postgresql://example:secret@localhost/testing",
        tmp_path / "model.pt",
        "0" * 64,
        storage_dir=tmp_path / "artifacts",
    )


def image_bytes(mode="RGB", size=(8, 6), format="PNG"):
    stream = io.BytesIO()
    Image.new(mode, size).save(stream, format=format)
    return stream.getvalue()


class FakeRepository:
    def __init__(self):
        self.records = {}
        self.available = True

    def ready(self):
        return self.available

    def save(self, record):
        self.records[record.id] = record

    def get(self, identifier):
        return self.records.get(identifier)


class FakePredictor:
    def __init__(self):
        self.calls = 0
        self.metadata = {
            "checkpoint_sha256": "0" * 64,
            "checkpoint_epoch": 1,
            "dataset": {"release_id": "synthetic"},
            "architecture": {},
            "preprocessing": {},
            "class_mapping": [
                {"class_id": i, "source_label": name} for i, name in enumerate(CLASS_NAMES)
            ],
            "device": "cpu",
            "training_code_version": None,
            "confidence": "Uncalibrated",
            "alignment": "Native oriented pixels",
            "severity_scope": "No safety score",
        }

    def predict(self, image):
        self.calls += 1
        mask = np.zeros((image.height, image.width), dtype=np.uint8)
        mask[1:3, 2:5] = 3
        return Prediction(mask, analyze_mask(mask, RegionPolicy(), UncertaintyPolicy()))


def make_service(settings):
    return InspectionService(
        settings, FakePredictor(), FakeRepository(), ArtifactStore(settings.storage_dir)
    )


@pytest.mark.parametrize("format,media", [("PNG", "image/png"), ("JPEG", "image/jpeg")])
def test_valid_content_and_grayscale_conversion(api_settings, format, media):
    decoded = decode_upload(image_bytes("L", format=format), media, api_settings)
    assert decoded.image.mode == "RGB" and decoded.image.size == (8, 6)
    assert decoded.source_mode == "L" and decoded.media_type == media


@pytest.mark.parametrize(
    "data,media,status",
    [
        (b"", "image/png", 422),
        (b"not an image", "image/jpeg", 422),
        (image_bytes()[:-12], "image/png", 422),
        (image_bytes(), "image/jpeg", 415),
        (image_bytes("RGBA"), "image/png", 422),
        (image_bytes(format="GIF"), "image/gif", 415),
    ],
)
def test_invalid_inputs_fail_explicitly(api_settings, data, media, status):
    with pytest.raises(ApiError) as raised:
        decode_upload(data, media, api_settings)
    assert raised.value.status == status


def test_file_and_dimension_limits(api_settings):
    for settings in (
        replace(api_settings, max_upload_bytes=1),
        replace(api_settings, max_pixels=2),
        replace(api_settings, max_dimension=2),
    ):
        with pytest.raises(ApiError) as raised:
            decode_upload(image_bytes(), "image/png", settings)
        assert raised.value.status == 413


def test_exif_orientation_and_metadata_removal(api_settings):
    source = Image.new("RGB", (8, 6))
    exif = Image.Exif()
    exif[274] = 6
    stream = io.BytesIO()
    source.save(stream, format="JPEG", exif=exif)
    decoded = decode_upload(stream.getvalue(), "image/jpeg", api_settings)
    assert decoded.image.size == (6, 8) and not decoded.image.getexif()


def test_service_persists_native_masks_and_fetch_does_not_predict(api_settings):
    service = make_service(api_settings)
    record = service.predict(image_bytes(), "image/png")
    assert service.get(record.id) == record and service.predictor.calls == 1
    assert record.analytics["classes"][3]["area_pixels"] == 6
    assert record.analytics["overall_structural_severity"]["score"] is None
    path = service.storage.file(record.id, record.mask_key, "mask", record.mask_sha256)
    with Image.open(path) as mask:
        assert mask.mode == "L" and mask.size == (8, 6)
        assert set(np.asarray(mask).ravel()) == {0, 3}
    with pytest.raises(ApiError) as raised:
        service.get(uuid4())
    assert raised.value.status == 404


def test_storage_rejects_traversal_misalignment_tampering_and_overwrite(api_settings):
    store = ArtifactStore(api_settings.storage_dir)
    identifier = uuid4()
    image = Image.new("RGB", (4, 3))
    artifacts = store.publish(identifier, image, np.zeros((3, 4), np.uint8))
    with pytest.raises(ValueError):
        store.file(identifier, "../../secret", "image", artifacts.image_sha256)
    with pytest.raises(FileExistsError):
        store.publish(identifier, image, np.zeros((3, 4), np.uint8))
    with pytest.raises(ValueError, match="match"):
        store.publish(uuid4(), image, np.zeros((1, 4), np.uint8))
    (store.root / artifacts.mask_key).write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash"):
        store.file(identifier, artifacts.mask_key, "mask", artifacts.mask_sha256)


def test_database_failure_retains_published_files_for_commit_reconciliation(api_settings):
    service = make_service(api_settings)

    def failed(_record):
        raise RuntimeError("secret database details")

    service.repository.save = failed
    with pytest.raises(ApiError) as raised:
        service.predict(image_bytes(), "image/png")
    assert raised.value.status == 503 and "secret" not in str(raised.value)
    assert len(list(service.storage.root.glob("*/prediction_ids.png"))) == 1


def test_unready_database_prevents_inference(api_settings):
    service = make_service(api_settings)
    service.repository.available = False
    with pytest.raises(ApiError):
        service.predict(image_bytes(), "image/png")
    assert service.predictor.calls == 0


def test_concurrent_inference_is_rejected_and_gate_recovers(api_settings):
    service = make_service(api_settings)
    started, release = Event(), Event()
    original = service.predictor.predict

    def delayed(image):
        started.set()
        assert release.wait(3)
        return original(image)

    service.predictor.predict = delayed
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(service.predict, image_bytes(), "image/png")
        assert started.wait(3)
        try:
            with pytest.raises(ApiError) as raised:
                service.predict(image_bytes(), "image/png")
            assert raised.value.code == "inference_busy"
        finally:
            release.set()
        assert first.result().id
    assert service.predict(image_bytes(), "image/png").id


def test_settings_protect_prior_tasks_and_hide_credentials(api_settings):
    assert "secret" not in repr(api_settings)
    for name in ("outputs/task04", "data", "src"):
        with pytest.raises(ValueError, match="overlap"):
            replace(api_settings, storage_dir=__import__("pathlib").Path(name))
    with pytest.raises(ValueError):
        replace(api_settings, checkpoint_sha256="unpinned")


def test_repository_uses_bound_parameters_and_single_transaction(api_settings, monkeypatch):
    record = make_service(api_settings).predict(image_bytes(), "image/png")
    calls, exits = [], []

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, kind, *_args):
            exits.append(kind)

        def execute(self, query, parameters):
            calls.append((query, parameters))

    repository = PostgresRepository(api_settings.database_url)
    monkeypatch.setattr(repository, "_connection", Connection)
    repository.save(record)
    assert len(calls) == 2 and exits == [None]
    assert str(record.id) not in calls[0][0] and calls[0][1][0] == record.id
    assert calls[1][1][3] == record.model_metadata["checkpoint_sha256"]
    calls.clear()

    class FailedConnection(Connection):
        def execute(self, query, parameters):
            super().execute(query, parameters)
            if len(calls) == 2:
                raise RuntimeError("second insert failed")

    monkeypatch.setattr(repository, "_connection", FailedConnection)
    with pytest.raises(RuntimeError, match="second insert"):
        repository.save(record)
    assert exits[-1] is RuntimeError  # The same connection context receives the failure.


def test_small_resident_checkpoint_roundtrip(training_release, api_settings, tmp_path):
    torch = pytest.importorskip("torch")
    from damage_analytics.api.inference import Segmenter
    from damage_analytics.io import sha256_file
    from damage_analytics.training.checkpoints import save_checkpoint
    from damage_analytics.training.model import LightweightUNet

    checkpoint = tmp_path / "synthetic.pt"
    model = LightweightUNet(training_release.model)
    optimizer = torch.optim.AdamW(model.parameters())
    save_checkpoint(
        checkpoint,
        model,
        optimizer,
        epoch=1,
        score=0.25,
        metadata={
            "purpose": "training",
            "dataset": {"release_id": "synthetic"},
            "config": training_release.snapshot(),
        },
        architecture=training_release.model,
    )
    settings = replace(
        api_settings,
        checkpoint=checkpoint,
        checkpoint_sha256=sha256_file(checkpoint),
        cpu_threads=1,
    )
    threads, deterministic = torch.get_num_threads(), torch.are_deterministic_algorithms_enabled()
    try:
        resident = Segmenter(settings)
        first = resident.predict(Image.new("RGB", (12, 8)))
        second = resident.predict(Image.new("RGB", (12, 8)))
        assert first.mask.shape == (8, 12) and np.array_equal(first.mask, second.mask)
        assert all(not parameter.requires_grad for parameter in resident.model.parameters())
        assert resident.metadata["checkpoint_sha256"] == sha256_file(checkpoint)
        with pytest.raises(ValueError, match="pin"):
            Segmenter(replace(settings, checkpoint_sha256="f" * 64))
    finally:
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(deterministic)
