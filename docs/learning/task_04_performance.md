# Task 04: measure training performance before running more experiments

This extends [Task 04](task_04.md). Performance means elapsed time and resource use here,
not segmentation quality. No full experiment was launched during implementation.

## What was inspected

The existing `control_s42` completed 50 epochs on **CPU**, with strict determinism,
four CPU threads, batch size four, zero loader workers, and validation every epoch.
Its best validation foreground mIoU is 0.2431215908 at epoch 48. The original Task 03
model and this control remain untouched. File timestamps suggest about 2 h 46 min
between metadata creation and completion, but the old history has no epoch timers.
Do not treat that approximation as a controlled performance measurement.

The training loop, dataset adapter, paired transforms, losses, optimizer, native
validation, checkpoints, device selection, and experiment comparison were inspected.
The training split contains 1,132 images and validation contains 243. No final-test
images, masks, scores, or galleries were used for this work.

## Measurements already obtained

Host: Apple silicon macOS, Python 3.14.5, PyTorch 2.15.0. Benchmarks use the same
evenly spaced 64 training and 64 validation samples, batch size four, two warm-up
batches, and three timed passes. The initial model is seeded, then temporary updates
are made; none are saved as a training checkpoint. OS file caches are warm/uncontrolled.
Numbers below are **medians of end-to-end passes**, not sums of synchronized stages.

| Execution | Training seconds / 64 images | Validation seconds / 64 images |
| --- | ---: | ---: |
| Original CPU, four threads | 7.566 | 3.040 |
| Optimized CPU, four threads | 6.889 | 2.631 |
| Optimized CPU, one thread | 10.395 | 4.467 |
| Optimized CPU, four threads, channels last | 6.018 | 2.136 |
| Optimized MPS, strict determinism disabled | 1.431 | 0.683 |

The direct CPU change measured about 1.10× training throughput and 1.16× validation
throughput. Optional channels-last CPU storage measured about 1.26× and 1.42× relative
to the original. MPS measured about 5.29× training throughput relative to that CPU
reference, **with a different reproducibility policy**. These are short measurements
on this host, not promised full-run speedups or accuracy improvements. Thermal state,
background work, allocator retention, and workload composition affect results.

The original stage probe attributed about 91% of training time to forward/backward
computation and about 6% to waiting for data. Native confusion accumulation was small.
CPU prediction extraction was needlessly slow: the optimized NumPy argmax probe took
about 0.037 seconds for 64 images while the old tensor argmax/host-conversion probe
was much slower. Both choose the first maximum class and restore the same class-ID
map; no native target pixels are skipped.

Raw measurements and their exact source hashes live in `outputs/task04/performance/`.
`before_host_cpu4` is the host reference; `before_cpu` is a separate sandbox measurement
and should not be mixed into the host comparison. Instrumented pre-optimization source
snapshots are retained in `reference_sources/`. Initial strict-MPS diagnostic failures
are also retained rather than presented as successful profiles.

The full loader-only sweep used all **1,132 training images**, two passes per setting:

| Workers | Persistent | Cache MiB/process | First pass seconds | Second pass seconds |
| ---: | :---: | ---: | ---: | ---: |
| 0 | No | 0 | 6.974 | 6.348 |
| 2 | No | 0 | 14.945 | 15.065 |
| 2 | Yes | 0 | 5.666 | 4.653 |
| 4 | No | 0 | 25.500 | 24.787 |
| 4 | Yes | 0 | 4.901 | 3.193 |
| 0 | No | 128 | 7.424 | 6.579 |
| 2 | No | 128 | 14.864 | 14.941 |
| 2 | Yes | 128 | 4.873 | 3.694 |
| 4 | No | 128 | 24.157 | 24.212 |
| 4 | Yes | 128 | 4.606 | 2.612 |

Nonpersistent process teardown is particularly costly on this host. The single-process
128 MiB cache had **zero hits over both complete passes**: sequential scanning evicts
earlier pairs before reuse. Worker-cache locality can differ, and real shuffled training
can have different hit rates. A loader-only gain does not establish a training gain.

End-to-end MPS checks, batch size four and no cache:

| Workers | Persistent | Training median seconds | Validation median seconds | Sampled process-tree RSS GiB |
| ---: | :---: | ---: | ---: | ---: |
| 0 | No | 1.431 | 0.683 | 0.492 |
| 2 | Yes | 1.122 | 0.517 | 1.395 |
| 4 | Yes | 1.096 | 0.438 | 2.084 |

Two persistent workers are the recommended compromise. Their first training pass took
2.000 seconds and later passes took 1.116/1.122 seconds: startup is not hidden. Four
workers bought little extra training throughput and used about 0.69 GiB more sampled
process memory. Recommended MPS training throughput was about **57 images/second** in
this subset probe, versus about **8.46 images/second** for the original CPU reference.
The unchanged native-validation algorithm still evaluates every pixel.

The MPS batch sweep with zero workers measured training medians of **1.675 / 1.452 /
1.388 seconds** for batch sizes **2 / 4 / 8**. Sampled MPS allocated peaks were about
**189 / 370 / 748 MiB**, and driver peaks about **1,148 / 1,155 / 2,185 MiB**. The small
gain at eight does not justify automatically changing optimization dynamics and memory
requirements. Keep four initially. A 384² probe with two persistent workers took
**2.408 seconds training / 0.872 seconds validation**; sampled MPS allocation was about
852 MiB and driver allocation about 2,127 MiB. It ran successfully, but remains a
more expensive experimental intervention rather than an automatic speed optimization.

Original CPU sampled process-tree RSS was about **2.36 GiB**; the channels-last CPU
probe observed about **2.17 GiB**. MPS RSS alone omits relevant driver/allocator memory;
the two-worker run also observed about 381 MiB allocated and 1,082 MiB driver memory.
Do not sum these overlapping unified-memory measurements.

## Why MPS determinism fails

The sandbox initially hid MPS. A host-level probe found it available. Autograd anomaly
tracing identified backward through **bilinear upsampling** in the unchanged U-Net:
`UnsafeIndexBackward0` ultimately calls `index_put_with_accumulate_mps`, which this
PyTorch build rejects under strict deterministic algorithms.

The architecture was not changed to avoid this operation. CPU determinism stays the
safe default. `--device mps --nondeterministic` is an explicit alternative for profiling
or manual training; seeds are still recorded, but bitwise repeatability is not promised.
There is no silent CPU fallback and no automatic disabling of determinism.

The Dice expression was independently simplified to a fixed nine-channel reduction.
For each class, D_c = (2 * intersection_c + epsilon) / (denominator_c + epsilon).
The loss averages (1 - D_c) only over present foreground classes. A zero/one selector
now performs that average without dynamic tensor indexing or reading a device boolean.
The class set, smoothing epsilon, CE weights, padding policy, and formula are unchanged.
CPU and MPS value/gradient parity tests compare it with the original expression.
This simplification does **not** fix the separate model upsampling limitation.

## Implemented optimizations and their limits

* Count supervised pixels on the already available CPU target, avoiding a GPU scalar
  reduction/readback. Reuse one loss readback for finite checking and history accounting.
  Finite-loss and finite-gradient protections remain before optimizer updates.
* Use inference mode during validation. On CPU, use NumPy argmax on the existing logits;
  on MPS, argmax stays on-device and only the ID map is copied back.
* Support channels-last tensor **storage**. Shapes remain NCHW and the U-Net is unchanged.
  Convolution reduction order can change floating-point rounding. It is opt-in and must
  be shared by a matched comparison cohort, not mixed with the historical control.
* Support 0/2/4 loader workers, bounded prefetch, and persistent spawned workers.
  A shared CPU epoch scalar prevents workers from reusing stale augmentation epochs.
  A separate shuffle generator replays the historical completed-epoch sample order
  regardless of worker lifetime; this is regression-tested against the old loader.
* Support an optional per-process LRU cache of **native decoded pairs**. Augmentations
  are still regenerated from seed/epoch/index. Every cache hit hashes both sealed files;
  a modified file is rejected. No raw or curated source file is altered.
* Save history and the last checkpoint each epoch, but allow less frequent plot rendering.
  New histories include training, validation and artifact elapsed times. Epoch boundaries
  are synchronized on MPS for those times; ordinary batches have no profiling barriers.

The full training split requires about 1.22 GiB of raw RGB+mask payload before PIL,
prefetch, tensors or activations. A 128 MiB cache cannot hold it. Cache budgets apply
**per dataset per process**: training and validation pools, worker replicas and model
activations all contribute to memory use. Caching is disabled by default.

## Validation scheduling and honest short runs

`[schedule]` in `config/task04.toml` controls the common epoch budget, batch size and
validation cadence. `validation_every = 1` preserves the original selection protocol.
For a cadence of three over seven epochs, validation runs at epochs **1, 3, 6 and 7**.
The first and final epochs always run full validation. No score is fabricated or carried
forward into skipped rows; their validation fields are null. A skipped epoch cannot
replace `best.pt`. Its `last.pt` validation score is null.

Early-stopping patience counts **completed validation checks**. Skipped epochs never
increase it. Sparse validation necessarily changes which intermediate models can be
selected; it is not scientifically interchangeable with checking every epoch. Use the
same cadence, epoch cap, batch size and early-stopping rule across a comparison cohort.
Final evaluation always uses the complete frozen validation set at native resolution.

`train --exploratory-epochs 5` runs five epochs over the full frozen training split,
not an easier data subset. Checkpoints are tagged `exploratory_training`. They can be
validated and compared with equally budgeted exploratory runs, but cannot masquerade
as the completed 50-epoch control or be passed to Task 03's final-test command.

## Commands, in order

Use the existing environment; installation and downloads are unnecessary. Commands
below use fresh report/run IDs. Existing IDs refuse overwrite. Run outside a restricted
sandbox when measuring host MPS or process-tree RSS.

```bash
source .venv/bin/activate

# Current CPU stages: decoding, transforms, data wait, transfer, forward, loss,
# backward, clipping, optimizer, native validation and memory.
python -m damage_analytics.analytics profile \
  --report-id local_cpu --samples 64 --repeats 3

# No model training: full-loader worker/persistence/cache sweep, two epochs each.
python -m damage_analytics.analytics profile \
  --report-id local_loaders --suite loaders --samples 1132 --repeats 2

# Explicit MPS policy; temporary updates only, no checkpoint writes.
python -m damage_analytics.analytics profile \
  --report-id local_mps --device mps --nondeterministic --samples 64 --repeats 3

# Batch sizes 2/4/8. Batch-size changes affect optimization; this is only a speed probe.
python -m damage_analytics.analytics profile \
  --report-id local_mps_batches --suite batches --device mps --nondeterministic \
  --samples 64 --repeats 3

# Check the more expensive resolution before committing to its training run.
python -m damage_analytics.analytics profile \
  --report-id local_mps_384 --experiment resolution_384 \
  --device mps --nondeterministic --workers 2 --persistent-workers --samples 64 --repeats 3

# Optional alternatives: run sequentially, never concurrently with another benchmark.
python -m damage_analytics.analytics profile \
  --report-id local_cpu_channels --channels-last --samples 64 --repeats 3
python -m damage_analytics.analytics profile \
  --report-id local_mps_workers2 --device mps --nondeterministic \
  --workers 2 --persistent-workers --samples 64 --repeats 3

python -m damage_analytics.analytics profile-summary \
  --profile-id local_cpu --profile-id local_loaders --profile-id local_mps \
  --profile-id local_mps_batches --profile-id local_mps_384 --report-id local_summary

python -m json.tool outputs/task04/performance/local_summary/summary.json
python -m pytest -q
python -m ruff check src tests
```

To benchmark a different cache budget, add `--cache-mb 128` to a stage profile.
For a different CPU thread count add `--cpu-threads 2`. Worker sweeps have cold/startup
and subsequent-epoch timings; stage profiles separate barriers from end-to-end passes.

## Efficient plan for remaining segmentation experiments

1. Preserve `control_s42` as the historical CPU control. Its training need not be repeated
   to inspect or evaluate it: `python -m damage_analytics.analytics evaluate --run-id control_s42`.
2. Choose one measured runtime policy. Keep batch size four initially; increased batch
   size changes the optimization path. Keep validation every epoch unless all runs
   deliberately share a different exploratory schedule. Keep workers/cache off unless
   **end-to-end** throughput improves, not just a loader-only figure. On this host,
   `--workers 2 --persistent-workers` has demonstrated that improvement. The flags
   override centralized `num_workers` / `[performance].persistent_workers` defaults
   and are recorded in the run. CPU defaults remain conservative.
3. Run the following matched five-epoch pilots manually. They were not launched during
   implementation. Early behavior does not establish a final winning model.

```bash
python -m damage_analytics.analytics train --experiment control \
  --run-id perf_control_e5 --exploratory-epochs 5 --device mps --nondeterministic \
  --workers 2 --persistent-workers
python -m damage_analytics.analytics train --experiment capped_weights \
  --run-id perf_capped_e5 --exploratory-epochs 5 --device mps --nondeterministic \
  --workers 2 --persistent-workers
python -m damage_analytics.analytics train --experiment resolution_384 \
  --run-id perf_resolution_e5 --exploratory-epochs 5 --device mps --nondeterministic \
  --workers 2 --persistent-workers

python -m damage_analytics.analytics compare \
  --run-id perf_control_e5 --run-id perf_capped_e5 --run-id perf_resolution_e5 \
  --comparison-id perf_pilots_e5
```

4. Inspect histories, validation per-class recall/IoU, memory, and failure galleries.
   If the profiles and pilots justify longer runs, run a **new matched current control**
   and the promising variant(s) with the same full budget and runtime policy:

```bash
python -m damage_analytics.analytics train --experiment control \
  --run-id perf_control_s42 --device mps --nondeterministic \
  --workers 2 --persistent-workers
python -m damage_analytics.analytics train --experiment capped_weights \
  --run-id perf_capped_s42 --device mps --nondeterministic \
  --workers 2 --persistent-workers
python -m damage_analytics.analytics train --experiment resolution_384 \
  --run-id perf_resolution_s42 --device mps --nondeterministic \
  --workers 2 --persistent-workers
python -m damage_analytics.analytics compare \
  --run-id perf_control_s42 --run-id perf_capped_s42 --run-id perf_resolution_s42 \
  --comparison-id perf_full_s42
```

Do not use the old CPU control as a matched control for new MPS variants. Comparison
checks reject different training code, backend, determinism, selection schedules,
and performance policies. Historical source hashes are not weakened to avoid rerunning
a scientifically necessary matched reference. To retain strict CPU determinism, omit
the two MPS flags from **all** cohort commands instead.

The already-used final test remains excluded. There is no Task 04 test command.

## Reading outputs and memory numbers

* `performance/<id>/profile.json`: exact samples/settings/source hashes, decoding probes,
  every measured pass, synchronized stage statistics and memory.
* `performance/<id>/failure.json`: a diagnostic failure, when profiling actually started.
* `performance/<summary-id>/timings.csv` and `summary.json`: readable measured tables.
* `performance/optimization_report.json`: implementation outcomes, profiles, test results,
  selected runtime recommendation and preservation evidence.
* `experiments/<new-run-id>/history.csv`: per-epoch timings and explicit validation gaps.
* Existing Task 04 validation/comparison paths remain unchanged.

Process-tree RSS is sampled roughly every 250 ms and may double-count shared pages.
It includes worker processes but can miss short spikes. Parent high-water is process
lifetime, including earlier cases in a batch/worker sweep. MPS allocator and driver
measurements are separate; MPS and CPU share physical memory, so **do not add driver
bytes and RSS**. These are useful observations, not a proof that a larger batch will
fit under every workload. Run a resolution-specific probe and watch memory pressure.

The profiler inserts MPS barriers around individual stages to attribute asynchronous
work correctly. Those barriers perturb overlap. Throughput passes synchronize only at
whole-pass boundaries. Normal training only synchronizes epoch timing boundaries;
finite checks, gradient clipping and validation readbacks retain their necessary waits.

## New modules and regression checks

| Module | Responsibility |
| --- | --- |
| `training/epochs.py` | One optimized pass; native metrics and optional stage hooks |
| `training/engine.py` | Training orchestration, timings, checkpoints and selection |
| `training/selection.py` | Explicit validation cadence and patience bookkeeping |
| `training/cache.py` | Byte-bounded verified native-pair LRU |
| `training/dataset.py` | Epoch-sharing workers and historical shuffle replay |
| `training/timing.py` | Opt-in device-aware stage timers |
| `training/memory.py` | Best-effort process-tree and MPS memory observations |
| `training/profile_steps.py` | Decode, transform and whole-pass probes |
| `training/profiling.py` | Bounded benchmark orchestration, no real experiment writes |
| `training/profile_reports.py` | CSV/JSON timing summaries |

Regression tests check values and gradients against the original loss, source changes
on cache hits, cache bounds, paired transforms, persistent-worker epoch propagation,
historical shuffle order, native metrics, absence of profiling barriers in an ordinary
pass, sparse selection, final validation, profile isolation and MPS loss parity when
hardware is available. CI does not assert a fragile number of milliseconds.

Actual verification: **174 tests and 21 subtests passed, with no skips**, including host
MPS parity and real spawned workers. Ruff and whitespace checks passed. The optimized
epoch evaluator independently re-evaluated all 243 native validation masks for the
existing control: **54,435,143 pixels**, an exactly matching confusion matrix and
foreground mIoU **0.24312159076729617**. This verifies aggregate evaluation parity;
it is not a claim about new training performance or unseen data. Original baseline and
control artifacts, architecture source, paired transforms, and the release seal were
verified unchanged. See `performance/native_validation_verification.json` and
`performance/preservation_verified.json`.

## Interview and viva questions

1. Why can unsynchronized GPU timers measure submission instead of computation?
2. Why should stage sums not substitute for end-to-end throughput?
3. How can persistent workers silently repeat the same augmentation epoch?
4. Why can persistence change shuffling when sampler and worker seeds share a generator?
5. Why does caching a 64-image subset exaggerate benefits on 1,132 images?
6. How does a per-worker cache differ from one shared cache?
7. Why does a faster batch size require a new matched training cohort?
8. Why does skipping validation change the set of selectable checkpoints?
9. Why is patience measured in validation checks rather than skipped epochs?
10. Why are deterministic CPU and nondeterministic MPS runs different research protocols?
11. Why must performance tests check work/correctness rather than fixed wall-time thresholds?
12. Why is test-set speed profiling still excluded from iterative development here?

Primary references: [PyTorch performance tuning](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide)
and [DataLoader implementation and persistence](https://github.com/pytorch/pytorch/blob/main/torch/utils/data/dataloader.py).
CUDA-specific pinning and stream advice is not blindly applied to MPS.
