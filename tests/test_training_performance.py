"""Numerical/operation regressions, not fragile hardware-dependent speed assertions."""

# Optional training dependency must be checked before importing torch-based modules.
# ruff: noqa: E402

import json
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from damage_analytics.training.cache import PairCache
from damage_analytics.training.config import IGNORE_INDEX, PerformanceConfig
from damage_analytics.training.dataset import ReplaySampler, SegmentationDataset, make_loader
from damage_analytics.training.engine import run_epoch, train_model
from damage_analytics.training.losses import SegmentationLoss
from damage_analytics.training.manifests import open_release
from damage_analytics.training.profiling import profile_pipeline, spaced_samples
from damage_analytics.training.selection import SelectionState, validation_due
from damage_analytics.training.timing import StageTimer


@pytest.fixture(autouse=True)
def small_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def legacy_loss(logits, target, weights, dice_weight):
    """Independent transcription of the original loss for values and gradient parity."""
    valid = target != IGNORE_INDEX
    ce = F.cross_entropy(logits, target, weight=weights, ignore_index=IGNORE_INDEX)
    truth = F.one_hot(target.masked_fill(~valid, 0), 9).permute(0, 3, 1, 2).float()
    truth = truth * valid.unsqueeze(1)
    probabilities = logits.softmax(1) * valid.unsqueeze(1)
    intersection = (probabilities * truth).sum((0, 2, 3))
    denominator = (probabilities + truth).sum((0, 2, 3))
    present = truth.sum((0, 2, 3)) > 0
    present[0] = False
    dice = 1 - (2 * intersection[present] + 1e-6) / (denominator[present] + 1e-6)
    return ce + dice_weight * (dice.mean() if bool(present.any()) else logits.sum() * 0)


@pytest.mark.parametrize("classes", [1, 3, 9])
@pytest.mark.parametrize("dice_weight", [0.0, 0.5])
def test_loss_and_gradients_match_original(classes, dice_weight):
    torch.manual_seed(7)
    original = torch.randn(2, 9, 9, 11, requires_grad=True)
    optimized = original.detach().clone().requires_grad_()
    target = torch.randint(classes, (2, 9, 11))
    target[:, :2] = IGNORE_INDEX
    weights = torch.linspace(0.1, 3.0, 9)
    expected = legacy_loss(original, target, weights, dice_weight)
    actual = SegmentationLoss(weights, dice_weight)(
        optimized, target, supervised_pixels=int((target != IGNORE_INDEX).sum())
    )
    expected.backward()
    actual.backward()
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(optimized.grad, original.grad, rtol=1e-6, atol=1e-8)


def test_shuffle_replays_historical_order_across_completed_epochs():
    dataset = TensorDataset(torch.arange(13))
    historical = DataLoader(
        dataset, batch_size=4, shuffle=True, generator=torch.Generator().manual_seed(42)
    )
    sampler = ReplaySampler(dataset, 42)
    for _ in range(4):
        expected = torch.cat([batch[0] for batch in historical]).tolist()
        assert list(sampler) == expected


def test_cache_avoids_decoding_but_detects_changed_source(training_release, monkeypatch):
    import damage_analytics.training.cache as caching

    release = open_release(training_release)
    sample = release.samples("train")[0]
    real = caching.read_pair
    calls = []

    def read(*args):
        calls.append(1)
        return real(*args)

    monkeypatch.setattr(caching, "read_pair", read)
    cache = PairCache(1)
    first = cache.read(release, sample)
    assert cache.read(release, sample) is first
    assert len(calls) == 1 and cache.hits == 1 and cache.bytes <= cache.budget
    path = release.root / sample.image
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="changed"):
        cache.read(release, sample)


def test_cache_budget_and_disabled_mode(training_release):
    release = open_release(training_release)
    samples = release.samples("train") + release.samples("validation")
    cache = PairCache(1)
    cache.budget = samples[0].width * samples[0].height * 4
    for sample in samples:
        cache.read(release, sample)
        assert cache.bytes <= cache.budget
    assert len(cache.pairs) == 1
    disabled = PairCache(0)
    disabled.read(release, samples[0])
    disabled.read(release, samples[0])
    assert disabled.hits == 0 and disabled.bytes == 0 and disabled.misses == 2


def test_cached_and_uncached_paired_transforms_are_identical(training_release):
    release = open_release(training_release)
    args = (release, release.samples("train"), training_release.data)
    left = SegmentationDataset(*args, training=True, seed=42)
    right = SegmentationDataset(*args, training=True, seed=42, cache_mb=1)
    for epoch in range(4):
        left.set_epoch(epoch)
        right.set_epoch(epoch)
        for name in ("image", "mask"):
            assert torch.equal(left[0][name], right[0][name])


def test_persistent_workers_receive_epochs_and_keep_same_shuffle(training_release):
    release = open_release(training_release)
    rows = tuple(replace(release.samples("train")[0], sample_id=str(i)) for i in range(6))
    config = replace(training_release.data, horizontal_flip=0.5, crop_probability=0.5)
    args = (release, rows, config)
    local = SegmentationDataset(*args, training=True, seed=42)
    shared = SegmentationDataset(*args, training=True, seed=42, cache_mb=1)
    one = make_loader(local, 2, 0, 42, shuffle=True)
    many = make_loader(shared, 2, 2, 42, shuffle=True, persistent_workers=True)
    for epoch in (1, 2, 3):
        local.set_epoch(epoch)
        shared.set_epoch(epoch)
        a, b = list(one), list(many)
        assert [[s.sample_id for s in x["samples"]] for x in a] == [
            [s.sample_id for s in x["samples"]] for x in b
        ]
        for left, right in zip(a, b, strict=True):
            assert torch.equal(left["images"], right["images"])
            assert torch.equal(left["masks"], right["masks"])


def test_native_validation_metrics_unchanged_and_no_barriers_unprofiled(
    training_release, monkeypatch
):
    import damage_analytics.training.timing as timing
    from damage_analytics.training.model import LightweightUNet

    release = open_release(training_release)
    dataset = SegmentationDataset(
        release, release.samples("validation"), training_release.data, training=False, seed=42
    )
    loader = make_loader(dataset, 1, 0, 42, shuffle=False)
    model = LightweightUNet(training_release.model)
    criterion = SegmentationLoss(torch.ones(9), 0.5)
    calls = []
    monkeypatch.setattr(timing, "synchronize", lambda device: calls.append(device))
    result = run_epoch(model, loader, criterion, torch.device("cpu"))
    assert calls == []
    timer = StageTimer(torch.device("cpu"))
    profiled = run_epoch(model, loader, criterion, torch.device("cpu"), timer=timer)
    assert result == profiled and calls
    assert result["metrics"]["evaluated_pixels"] == 12 * 8
    assert "native_metrics" in timer.report()


def test_selection_cadence_and_patience_count_only_real_checks():
    assert [e for e in range(1, 8) if validation_due(e, 7, 3)] == [1, 3, 6, 7]
    state = SelectionState()
    assert state.observe(1, 0.4, 0.01)
    assert state.observe(3, 0.405, 0.01)  # Best selection differs from significant improvement.
    assert state.best_epoch == 3 and state.stale == 1 and state.checks == 2
    assert not state.observe(6, 0.39, 0.01)
    assert state.stale == 2


def test_sparse_validation_history_has_no_invented_scores(training_release):
    config = replace(
        training_release,
        optimization=replace(
            training_release.optimization, epochs=4, validation_every=3, patience=8
        ),
        performance=PerformanceConfig(plot_every=3),
    )
    folder = train_model(config, "sparse", exploratory=True)
    history = json.loads((folder / "history.json").read_text())
    assert [row["epoch"] for row in history if row["validated"]] == [1, 3, 4]
    assert history[1]["validation_loss"] is None
    assert history[1]["validation_foreground_miou"] is None
    assert all(row["training_seconds"] > 0 and row["artifact_seconds"] > 0 for row in history)
    status = json.loads((folder / "status.json").read_text())
    assert status["validation_checks"] == 3 and status["exploratory"]
    payload = torch.load(folder / "best.pt", weights_only=True)
    assert payload["epoch"] in (1, 3, 4)
    assert payload["metadata"]["purpose"] == "exploratory_training"


def test_bounded_profile_writes_no_model_run_and_never_accesses_test(training_release, tmp_path):
    path = profile_pipeline(
        training_release,
        tmp_path / "profile",
        samples=1,
        repeats=1,
        warmup_batches=0,
        cpu_threads=1,
    )
    report = json.loads((path / "profile.json").read_text())
    assert not report["test_accessed"] and not report["checkpoints_written"]
    assert set(report["sample_ids"]) == {"train", "validation"}
    assert not list(path.rglob("*.pt"))
    assert report["cases"][0]["stages"]["train"]["stages"]["backward"]["calls"] == 1
    with pytest.raises(FileExistsError):
        profile_pipeline(training_release, path, samples=1, repeats=1)
    assert spaced_samples(tuple(range(100)), 3) == (0, 49, 99)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Host MPS required")
def test_mps_loss_gradient_parity_without_changing_model_determinism():
    old = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(False)
        logits = torch.randn(1, 9, 8, 8, device="mps", requires_grad=True)
        target = torch.ones(1, 8, 8, device="mps", dtype=torch.long)
        left = legacy_loss(logits, target, torch.ones(9, device="mps"), 0.5)
        a = torch.autograd.grad(left, logits)[0]
        right = SegmentationLoss(torch.ones(9), 0.5).to("mps")(logits, target)
        b = torch.autograd.grad(right, logits)[0]
        torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
    finally:
        torch.use_deterministic_algorithms(old)
