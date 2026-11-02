"""Task 10 archive, label, leakage, loader, metric and training contract tests."""

# ruff: noqa: E402 -- Skip optional Torch before importing the training modules.

import csv
import hashlib
import io
import json
import pickle
import shutil
from dataclasses import replace
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from damage_analytics.task10 import checkpoints
from damage_analytics.task10.archives import checked_header, inspect
from damage_analytics.task10.cli import main, parser
from damage_analytics.task10.config import CLASS_NAMES, TrainConfig, load_config
from damage_analytics.task10.data import SeverityDataset, loader
from damage_analytics.task10.encoding import CAFFE_OFFSETS, recovered_channels, rgb_from_array
from damage_analytics.task10.leakage import investigate, visual_views
from damage_analytics.task10.metrics import metrics
from damage_analytics.task10.models import build_model
from damage_analytics.task10.prepare import attach_source_evidence, prepare
from damage_analytics.task10.release import open_release
from damage_analytics.task10.smoke import synthetic_archive
from damage_analytics.task10.splits import development_split


@pytest.fixture(scope="module")
def fixture_release(tmp_path_factory):
    root = tmp_path_factory.mktemp("phi7")
    archive = root / "fixture.zip"
    synthetic_archive(archive)
    config = replace(
        load_config(Path("config/task10.toml")),
        archive=archive,
        output_dir=root / "outputs/task10",
        expected_image_shape=(32, 32, 3),
        encoding_verified=True,
        encoding_evidence="Synthetic known BGR fixture",
        training=TrainConfig(epochs=2, batch_size=8, image_size=32, patience=2),
    )
    torch.set_num_threads(1)
    return config, open_release(prepare(config), full=True)


def rewritten(config, destination, updates):
    with ZipFile(config.archive) as original, ZipFile(destination, "w") as new:
        for member in original.infolist():
            value = original.read(member)
            new.writestr(member.filename, updates.get(member.filename, value))
        for name, value in updates.items():
            if name not in original.namelist():
                new.writestr(name, value)
    return replace(config, archive=destination)


def npy_bytes(array):
    stream = io.BytesIO()
    np.save(stream, array)
    return stream.getvalue()


def row(identifier, *, partition="train", label=0, digest=None, source=""):
    return {
        "sample_id": identifier,
        "official_partition": partition,
        "class_id": label,
        "valid": True,
        "label_valid": True,
        "raw_sha256": digest or identifier,
        "recovered_sha256": digest or identifier,
        "source_group": source,
        "disposition": "development_eligible"
        if partition == "train"
        else "official_test_preserved",
    }


def evidence(value=0, std=20):
    return {"phash": value, "dhash": value, "thumbnail": [100] * 256, "gray_std": std}


def test_archive_registration_and_alignment(fixture_release):
    config, release = fixture_release
    report = inspect(config)
    assert [r["label"] for r in report["class_mapping"]] == list(CLASS_NAMES)
    assert CLASS_NAMES[0] == "Heavy" and CLASS_NAMES[3] == "Undamaged"
    assert report["headers"]["task7_y_train.npy"]["shape"] == [40, 4]
    assert len(release.samples("train")) == 32
    assert len(release.samples("validation")) == 8
    assert sum(r["split"] == "official_test" for r in release.rows) == 8
    assert all(not r["source_group"] for r in release.rows)
    provenance = release.metadata["inputs"]["code_sha256"]
    assert "task10/archives.py" in provenance and "archives.py" in provenance
    assert provenance["task10/archives.py"] != provenance["archives.py"]


@pytest.mark.parametrize(
    "name,value",
    [
        ("../escape", b"bad"),
        ("task7/extra.txt", b"unexpected"),
        ("task7/README.txt", b"0: Undamaged state (UD)\n"),
        ("task7/task7_y_train.npy", npy_bytes(np.zeros((39, 4), np.float32))),
        ("task7/task7_X_train.npy", npy_bytes(np.zeros((40, 32, 32, 3), np.float64))),
    ],
)
def test_reject_bad_archive_schema(fixture_release, tmp_path, name, value):
    config, _ = fixture_release
    changed = rewritten(config, tmp_path / "invalid.zip", {name: value})
    with pytest.raises(ValueError):
        inspect(changed)


@pytest.mark.parametrize(
    "values", [np.array([{}], object), np.asfortranarray(np.zeros((2, 3), np.float32))]
)
def test_reject_unsafe_npy(values):
    raw = npy_bytes(values)
    with pytest.raises(ValueError):
        checked_header(io.BytesIO(raw), len(raw))


def test_reject_truncated_body():
    raw = npy_bytes(np.zeros((2, 3), np.float32))[:-1]
    with pytest.raises(ValueError, match="body byte length"):
        checked_header(io.BytesIO(raw), len(raw))


def test_rgb_inverse_exact_and_no_double_normalization():
    rgb = np.array([[[250, 34, 17], [0, 255, 128]]], dtype=np.uint8)
    stored = rgb[..., ::-1].astype(np.float32) - CAFFE_OFFSETS
    assert np.array_equal(rgb_from_array(stored), rgb)
    channels, error = recovered_channels(stored)
    assert error == 0 and np.array_equal(channels, rgb[..., ::-1])


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 300.0, 0.1])
def test_inverse_rejects_unsupported_values(bad):
    values = np.zeros((2, 2, 3), np.float32) - CAFFE_OFFSETS
    values[0, 0, 0] += bad
    with pytest.raises(ValueError):
        recovered_channels(values)


def test_invalid_label_quarantined_not_guessed(fixture_release, tmp_path):
    config, _ = fixture_release
    with ZipFile(config.archive) as archive:
        labels = np.load(io.BytesIO(archive.read("task7/task7_y_train.npy")))
    labels[0] = [0.5, 0.5, 0, 0]
    changed = rewritten(
        config, tmp_path / "soft.zip", {"task7/task7_y_train.npy": npy_bytes(labels)}
    )
    changed = replace(changed, output_dir=tmp_path / "outputs/task10")
    release = open_release(prepare(changed))
    invalid = release.rows[0]
    assert not invalid["valid"] and invalid["class_id"] is None
    assert invalid["split"] == "quarantine"
    assert "invalid_nonfinite_or_non_one_hot_label" in invalid["issues"]
    assert (release.root / "issues.csv").read_text().count(invalid["sample_id"]) == 1


def test_duplicate_conflicting_labels_never_relabeled(fixture_release):
    config, _ = fixture_release
    rows = [row("a", label=0, digest="same"), row("b", label=1, digest="same")]
    _, _, near, report = investigate(rows, {}, config)
    assert not near and report["raw_or_recovered_label_conflict_samples"] == 2
    assert [r["class_id"] for r in rows] == [0, 1]
    assert all(r["disposition"] == "quarantine_exact_image_label_conflict" for r in rows)


def test_test_connected_crop_candidate_transitively_quarantines(fixture_release):
    config, _ = fixture_release
    rows = [
        row("a", source="documented_component"),
        row("b", source="documented_component"),
        row("t", partition="test"),
    ]
    views = {"b": [("top_left", evidence())], "t": [("full", evidence())]}
    _, _, near, report = investigate(rows, views, config)
    assert len(near) == 1 and near[0]["left_view"] == "top_left"
    assert near[0]["decision"] == near[0]["human_evidence"] == ""
    assert near[0]["status"] == "unresolved_visual_candidate"
    assert report["phash_search"]["node_distance_evaluations"] > 0
    assert all(r["disposition"] == "quarantine_official_test_related_component" for r in rows[:2])
    assert rows[2]["disposition"] == "official_test_preserved"
    assert len({r["leakage_component"] for r in rows}) == 1
    assert rows[2]["source_group"] == ""  # A graph component is not an invented source ID.


def test_low_information_not_identity_evidence(fixture_release):
    config, _ = fixture_release
    rows = [row("a"), row("b")]
    views = {i: [("full", evidence(std=0))] for i in ("a", "b")}
    _, _, near, report = investigate(rows, views, config)
    assert near == [] and report["low_information_views_not_used_for_near_identity"] == 2


def test_screening_budget_marked_incomplete(fixture_release):
    config, _ = fixture_release
    rows = [row(i) for i in ("a", "b", "c")]
    views = {i: [("full", evidence())] for i in ("a", "b", "c")}
    _, _, near, report = investigate(rows, views, replace(config, maximum_near_pairs=1))
    assert len(near) == 1 and report["screening_truncated"]


def test_gray_views_invariant_to_unverified_channel_reversal():
    channels = np.random.default_rng(5).integers(0, 256, (32, 32, 3), dtype=np.uint8)
    assert visual_views(channels, 0.75) == visual_views(channels[..., ::-1], 0.75)


def test_grouped_split_reproducible_and_no_component_overlap():
    rows = []
    for label in range(4):
        for group in range(6):
            for repeat in range(2):
                item = row(f"{label}-{group}-{repeat}", label=label)
                item["leakage_component"] = f"evidence-{label}-{group}"
                rows.append(item)
    first = development_split(rows, 0.2, 42)
    assert first == development_split(list(reversed(rows)), 0.2, 42)
    groups = {
        fold: {r["leakage_component"] for r in rows if first[r["sample_id"]] == fold}
        for fold in ("train", "validation")
    }
    assert not groups["train"] & groups["validation"]
    for fold in groups:
        assert {r["class_id"] for r in rows if first[r["sample_id"]] == fold} == set(range(4))
    for item in rows:
        if item["class_id"] == 0:
            item["leakage_component"] = "only_heavy_component"
    with pytest.raises(ValueError, match="infeasible"):
        development_split(rows, 0.2, 42)


def test_source_evidence_must_be_real_supplied_and_nonempty(tmp_path):
    path = tmp_path / "sources.csv"
    path.write_text("sample_id,source_group,evidence\na,invented,\n")
    with pytest.raises(ValueError):
        attach_source_evidence([row("a")], path)
    path.write_text("sample_id,source_group,evidence\na,component7,publisher mapping row 7\n")
    rows = [row("a")]
    attach_source_evidence(rows, path)
    assert rows[0]["source_group"] == "component7"


def test_preparation_idempotent_and_raw_unchanged(fixture_release):
    config, release = fixture_release
    digest = hashlib.sha256(config.archive.read_bytes()).hexdigest()
    before = (release.root / "checksums.json").read_bytes()
    assert prepare(config) == release.root
    assert before == (release.root / "checksums.json").read_bytes()
    assert hashlib.sha256(config.archive.read_bytes()).hexdigest() == digest


def test_manifest_tamper_rejected(fixture_release, tmp_path):
    _, release = fixture_release
    shutil.copytree(release.root, tmp_path / "release")
    path = tmp_path / "release/manifests/train.csv"
    path.write_text(path.read_text().replace("Heavy", "Undamaged"))
    with pytest.raises(ValueError, match="metadata changed"):
        open_release(tmp_path / "release")


def test_dataloader_shapes_ids_deterministic_validation_and_no_test(fixture_release):
    config, release = fixture_release
    batches = loader(release, "validation", config.training, 42)
    images, targets, identifiers = next(iter(batches))
    assert images.shape == (8, 3, 32, 32) and images.dtype == torch.float32
    assert set(targets.tolist()) == set(range(4)) and len(identifiers) == 8
    assert torch.isfinite(images).all()
    assert torch.equal(images, next(iter(batches))[0])
    dataset = batches.dataset
    assert isinstance(dataset._x, np.memmap)
    assert all(r["official_partition"] == "train" for r in dataset.rows)
    restored = pickle.loads(pickle.dumps(dataset))
    assert restored._x is None and torch.equal(dataset[0][0], restored[0][0])
    with pytest.raises(ValueError, match="official test"):
        SeverityDataset(release, "official_test", config.training, 42)


def test_unverified_decoder_default_refused(fixture_release):
    config, release = fixture_release
    metadata = json.loads(json.dumps(release.metadata))
    metadata["inputs"]["settings"]["encoding_verified"] = False
    unverified = replace(release, metadata=metadata)
    with pytest.raises(ValueError, match="unverified"):
        SeverityDataset(unverified, "train", config.training, 42)
    assert (
        len(
            SeverityDataset(unverified, "train", config.training, 42, allow_inferred_encoding=True)
        )
        == 32
    )


def test_row_bytes_tamper_rejected(fixture_release, tmp_path):
    config, release = fixture_release
    shutil.copytree(release.root, tmp_path / "release")
    copied = open_release(tmp_path / "release")
    item = copied.samples("validation")[0]
    x = np.load(copied.file(item["image_array"], verify=False), mmap_mode="r+")
    x[item["array_index"], 0, 0, 0] += 1
    x.flush()
    del x
    with pytest.raises(ValueError, match="changed since preparation"):
        SeverityDataset(copied, "validation", config.training, 42)[0]


def test_metrics_heavy_zero_is_not_background():
    y = [0, 0, 1, 2, 3]
    scores = np.eye(4)[[0, 1, 1, 2, 3]]
    report = metrics(y, scores)
    assert report["accuracy"] == 0.8
    assert report["balanced_accuracy"] == 0.875
    assert report["macro_f1"] == pytest.approx(5 / 6)
    assert report["classes"][0]["recall"] == 0.5
    assert report["confusion_matrix"][0] == [1, 1, 0, 0]
    assert metrics([0], np.eye(4)[[0]])["balanced_accuracy"] is None
    with pytest.raises(ValueError):
        metrics([0], [[0.5, 0.5, 0.5, 0.5]])


def test_resnet_four_class_forward_backward_without_download(monkeypatch):
    import torchvision.models as tv

    original = tv.resnet18

    def no_weights(**kwargs):
        assert kwargs["weights"] is None
        return original(**kwargs)

    monkeypatch.setattr(tv, "resnet18", no_weights)
    model = build_model(pretrained=False)
    logits = model(torch.randn(2, 3, 32, 32))
    assert logits.shape == (2, 4)
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 3])).backward()
    assert model.fc.weight.grad is not None


def test_checkpoint_roundtrip_and_release_pin(tmp_path):
    model = build_model(pretrained=False, synthetic=True)
    metadata = {
        "class_names": list(CLASS_NAMES),
        "dataset": {"release_id": "fixture"},
        "purpose": "synthetic_smoke",
    }
    checkpoints.save(tmp_path / "best.pt", model, metadata, 1, 0.5)
    loaded, _ = checkpoints.load(tmp_path / "best.pt", torch.device("cpu"), metadata["dataset"])
    values = torch.randn(2, 3, 16, 16)
    assert torch.equal(model(values), loaded(values))
    with pytest.raises(ValueError, match="compatible"):
        checkpoints.load(tmp_path / "best.pt", torch.device("cpu"), {"release_id": "different"})


def test_tiny_training_selection_evaluation_end_to_end(fixture_release, tmp_path):
    from damage_analytics.task10.engine import evaluate, train

    config, release = fixture_release
    config = replace(config, output_dir=tmp_path / "outputs/task10")
    with pytest.raises(ValueError, match="provisional"):
        train(config, release, "refused", synthetic=True)
    experiment = train(config, release, "tiny", synthetic=True, allow_provisional=True)
    report_path = evaluate(config, release, experiment / "best.pt")
    report = json.loads((report_path / "metrics.json").read_text())
    assert report["split"] == "validation" and not report["test_evaluated"]
    assert report["training_purpose"] == "synthetic_smoke"
    assert sum(sum(r) for r in report["metrics"]["confusion_matrix"]) == 8
    assert (report_path / "confusion_matrix.png").stat().st_size > 0
    history = json.loads((experiment / "history.json").read_text())
    assert len(history) == 2 and all(np.isfinite(r["train_loss"]) for r in history)
    with (experiment / "validation_predictions.csv").open() as stream:
        predictions = list(csv.DictReader(stream))
    assert len(predictions) == 8 and "probability_id_0" in predictions[0]
    assert not any(r["sample_id"].startswith("phi7-test") for r in predictions)
    with pytest.raises(FileExistsError):
        train(config, release, "tiny", synthetic=True, allow_provisional=True)


def test_no_cli_test_performance_branch():
    with pytest.raises(SystemExit):
        parser().parse_args(["evaluate", "--release", "x", "--checkpoint", "x", "--split", "test"])
    assert main(["validate", "--release", "missing-task10-fixture"]) == 2


def test_integrity_review_never_renders_official_test(fixture_release, tmp_path):
    from damage_analytics.task10.review import build_review

    _, release = fixture_release
    # No near pairs occur in independent random fixtures, so this also checks empty reports.
    folder = build_review(release, tmp_path, "review", limit=0)
    result = json.loads((folder / "summary.json").read_text())
    assert (
        result["official_test_images_rendered"] == 0 and not result["test_performance_inspected"]
    )
    assert (folder / "class_distribution.csv").exists()
    assert not list(folder.glob("*.png"))
    with pytest.raises(FileExistsError):
        build_review(release, tmp_path, "review")


def test_hard_readiness_blocker_cannot_be_overridden(fixture_release, tmp_path):
    from damage_analytics.task10.engine import train

    config, release = fixture_release
    config = replace(config, output_dir=tmp_path)
    quality = {**release.quality, "technical_training_ready": False}
    with pytest.raises(ValueError, match="hard readiness blockers"):
        train(
            config,
            replace(release, quality=quality),
            "blocked",
            synthetic=True,
            allow_provisional=True,
        )


def test_review_with_mixed_pairs_only_opens_training_pixels(
    fixture_release, tmp_path, monkeypatch
):
    from damage_analytics.task10.review import build_review

    _, release = fixture_release
    a, b = release.samples("train")[:2]
    test = next(r for r in release.rows if r["official_partition"] == "test")
    pairs = [
        {
            "left": a["sample_id"],
            "right": other["sample_id"],
            "left_view": "center",
            "right_view": "full",
            "decision": "",
            "human_evidence": "",
        }
        for other in (b, test)
    ]
    inventory = tmp_path / "near.json"
    inventory.write_text(json.dumps(pairs))
    original = type(release).file
    opened = []

    def file_spy(self, relative, **kwargs):
        opened.append(relative)
        assert "X_test" not in relative and "y_test" not in relative
        return (
            inventory if relative == "near_candidates.json" else original(self, relative, **kwargs)
        )

    monkeypatch.setattr(type(release), "file", file_spy)
    output = build_review(release, tmp_path, "mixed", limit=2)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["candidate_pairs"] == 2 and summary["train_only_sheets_pairs"] == 1
    assert summary["official_test_images_rendered"] == 0
    assert len(list(output.glob("*.png"))) == 1
    assert a["image_array"] in opened
