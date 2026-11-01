# Task 09: segmentation evaluation and damage features

Task 09 is isolated under `src/damage_analytics/task09/` and `outputs/task09/`. It does
not modify Task 08 training code, dependencies, running processes, serving endpoints,
existing checkpoint formats, datasets or frozen splits. Current verification uses one
CPU thread; no MPS workload or full real training was launched.

## Executed evidence

* `outputs/task09/reviews/validation_001/REPORT.md`: dataset audit and native validation results.
* `.../audit/`: class statistics, decoded sample list, issues and resize diagnostics.
* `.../evaluations/{unet_001,control_s42}/`: per-class/per-image metrics, confusion matrices,
  authenticated prediction masks and targeted false-positive/false-negative panels.
* `.../poor_class_diagnostics.json`: dominant missed-as and false-positive source categories.
* `outputs/task09/inference/validation_example_cpu/`: one actual CPU validation-image
  inference, native masks, uncertainty maps and validated damage-feature JSON.
* `outputs/task09/plans/matched_models_001.json`: manual experiment protocol.
* `outputs/task09/implementation_report.json`: exact executed checks and limitations.

The release is pinned by Task 03's existing ID and checksum manifest. All split/hash
metadata was checked, including test membership solely for integrity QA. Fresh checks
decoded/rerasterized all 243 validation samples plus 24 deterministic training samples
covering classes/spatial extremes. No issues were found in those 267 checks. This is
not a fresh full-decode audit of all 1,617 samples. Full **development** decoding is an
optional later command; test pixels, test annotations and test metrics remain excluded.

Historical curation excluded 224 samples. Reasons are nonexclusive: malformed pixel
rings/self-intersections, cross-class overlap, out-of-bounds vertices and conflicting
duplicate annotations. A mask consistent with polygons does not prove human annotation
completeness or semantic correctness. Original source labels remain unchanged; project
IDs 0–8 and display colors are not published annotation IDs/colors. Background is the
polygon complement, not an intact-concrete or non-concrete category.

Native saved validation predictions were hash-checked, matched to every validation ID,
and re-scored against decoded native ground truth. The recomputed confusion matrices
match the historical results: original U-Net foreground mIoU 23.9906%, control 24.3122%.
This is an authenticated cached-prediction evaluation, not a newly executed full forward
pass. A separate fresh evaluation command supports both architectures. The historical
MPS and CPU baselines are contextual comparisons, not evidence of a causal improvement.

Rare classes have few validation images (crushing and deformation: seven each; minor
spalling: twelve). Inspect precision, recall and FP/FN panels together. Class imbalance,
foreground sampling, annotation ambiguity and feature capacity are possible contributors;
none was isolated causally by this implementation. The resize audit measures actual
ground-truth resampling loss, not a model accuracy bound or proof cracks are preserved.

## Modules and responsibilities

| Module | Responsibility |
|---|---|
| `config.py` | Reuse the pinned Task 03 source configuration; central Task 09 policies and matched settings |
| `audit.py` | Sealed manifest/split checks, development statistics, polygon/mask equality, resize audit |
| `evaluation.py` | Exact validation population, native metrics, cached replay, fresh evaluation and targeted failure panels |
| `visualization.py`, `reports.py` | Original/GT/prediction/error panels, confusion diagnostics and evidence-based report |
| `models.py` | Original U-Net or TorchVision LR-ASPP MobileNetV3-Large with nine replacement heads |
| `checkpoints.py` | Separate restricted Task 09 checkpoint format, plus read-only old checkpoint support |
| `experiments.py` | Matched manual training, train-only weights, early stopping and guarded comparisons |
| `geometry.py` | Exact RLE pixel-edge perimeter, centroid and covariance descriptors |
| `features.py`, `schema.py` | Raw/retained region accounting and validated versioned JSON invariants |
| `pipeline.py` | Load-once model, checkpoint-bound preprocessing, native score restoration and feature artifacts |
| `smoke.py`, `cli.py` | Bounded CPU synthetic checks and explicit commands; no final-test command |

Most training mathematics is reused directly: Task 03 paired transforms, Dataset/Loader,
loss, class weighting, native metrics, epoch execution and selection state; Task 04's RLE
components and scalar restoration are reused. No shared Task 01–08 Python module was
edited for Task 09. No verified historical mask-interpolation or loss-formula defect was
found. New checks reject nonfinite logits, unsupported normalization and invalid JSON;
they are guardrails, not evidence that historical results were incorrect.

## Native features and filtering

The structured pipeline performs:

```text
RGB → EXIF-oriented image → saved normalization/letterbox → 9-class logits
    → argmax → unpad → nearest-native IDs
    → class components + raw/retained coverage + geometry + uncalibrated score summaries
    → validated damage-features-v1 JSON, raw IDs, reporting-filtered IDs and score maps
```

`Segmenter` loads once per instance and can be reused by another Python caller. It is
not integrated into the existing API/dashboard. Independent instances are independent
models; do not share an instance across concurrent training/inference calls without
serialization. `extract_features(mask, policy)` also works without a model; provenance
and uncertainty are then explicitly unknown.

Class percentages use **whole image pixel count**, not foreground count or physical
surface area. Raw counts always sum to image width × height. Per-class components use
configurable 4/8 connectivity. Boxes are `[x_min,y_min,x_max_exclusive,y_max_exclusive]`;
the top-left pixel center is `(0.5,0.5)`. All geometry is in the EXIF-oriented original
image frame for inference; mask-only callers get an explicitly supplied-mask frame.

Perimeter counts exposed four-neighbor pixel edges, including holes. Diagonally touching
pixels can be one region under 8-connectivity while retaining their exposed edges. Moment
axes equal `4*sqrt(population covariance eigenvalue)`. Their orientation uses image x/y
with y downward, and has 180-degree ambiguity. Isotropic/single-pixel orientation and
degenerate axis ratios are null. These descriptors do **not** measure crack width, crack
length, material loss volume or physical damage instances. Semantic-class interfaces
contribute per-class boundaries; summing perimeters is not a physical damage outline.

Filtering defaults to minimum one pixel: nothing is removed. To investigate reporting
speckles, explicitly change `[features]` settings, for example:

```toml
minimum_region_pixels = 8
minimum_by_class = {"1" = 1, "2" = 1} # preserve small crack regions under this heuristic
```

This affects a separately saved reporting mask and region table only. Raw model masks,
raw percentages and segmentation metrics remain unchanged. Below-size regions are
**candidates**, not verified artifacts; thresholds have no physical interpretation.
Region table caps declare unreported retained counts/pixels, so truncation never silently
changes coverage. No morphology, hole filling, label merging or ground-truth editing occurs.

Maximum softmax and normalized entropy are restored with the same unpad/nearest geometry
as IDs. They are descriptive and uncalibrated. Scores absent for mask-only input remain
null. No OOD rejection, calibrated confidence, overall severity or structural safety is
inferred. Minor/moderate spalling remain source segmentation labels, not image-level grades.

## Exact commands

Run from the repository root using the existing environment. No dependencies were
installed or changed. TorchVision is already part of Task 07's declared classification
extra; do not install or change packages while Task 08 training is running.

```bash
source .venv/bin/activate

# Bounded CPU verification; one thread and zero loader workers.
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m pytest -q tests/test_task09.py tests/test_analytics_regions.py \
  tests/test_training_metrics.py tests/test_training_data.py
python -m ruff check src/damage_analytics/task09 tests/test_task09.py
python -m damage_analytics.task09 smoke --run-id local_smoke

# Read-only native cached validation replay; new report IDs prevent overwrites.
python -m damage_analytics.task09 replay --report-id local_validation
cat outputs/task09/reviews/local_validation/REPORT.md
open outputs/task09/reviews/local_validation/evaluations/control_s42/failures
python -m damage_analytics.task09 plan --report-id local_plan

# Actual one-image CPU inference. The selected image belongs to frozen validation.
python -m damage_analytics.task09 infer --device cpu --report-id local_example \
  --checkpoint outputs/task04/experiments/control_s42/best.pt \
  --image outputs/task02/versions/rc1841-curated-v1-d34771dbb662406a/images/000016.png
cat outputs/task09/inference/local_example/damage_features.json
```

**Only after Task 08 training has finished**, optional full development QA and the manual
matched experiments can run. No command below was launched against real training data
during implementation:

```bash
# Full train/validation decoding/rerasterization, never test pixels.
python -m damage_analytics.task09 audit --report-id full_development --full-development

# Fresh baseline validation forward pass if required; CPU or explicitly MPS after Task 08.
python -m damage_analytics.task09 evaluate --device mps \
  --checkpoint outputs/task04/experiments/control_s42/best.pt

# New matched runs; ImageNet normalization for BOTH, not the old checkpoint's 0.5/0.5.
python -m damage_analytics.task09 train --model unet \
  --run-id unet_matched_s42 --device mps
python -m damage_analytics.task09 train --model lraspp_mobilenet_v3_large \
  --run-id lraspp_matched_s42 --device mps

# Same raw native evaluation rules and all 243 validation samples.
python -m damage_analytics.task09 evaluate --device mps \
  --checkpoint outputs/task09/experiments/unet_matched_s42/best.pt
python -m damage_analytics.task09 evaluate --device mps \
  --checkpoint outputs/task09/experiments/lraspp_matched_s42/best.pt
python -m damage_analytics.task09 compare --report-id matched_s42 \
  --run-id unet_matched_s42 --run-id lraspp_matched_s42
```

Training starts fresh with seed 42, batch 4, 25-epoch ceiling, AdamW constant LR 0.0003,
training-native inverse-square-root weights and CE+0.5 Dice. Validation is every epoch;
patience eight counts validation checks. Best pooled native foreground mIoU selects the
checkpoint. Both new models share ImageNet normalization and spatial/crop/flip policies;
historical U-Net normalization remains checkpoint-bound and unchanged.

LR-ASPP alone requests its pretrained COCO/VOC weights during the explicit real training
command. Both output heads are newly initialized for RC IDs. Normal loading, evaluation,
inference and smoke use `weights=None` **and** `weights_backbone=None`, preventing accidental
downloads. Default frozen BatchNorm running moments address small batches; affine parameters
remain trainable. U-Net uses its existing GroupNorm. Architecture and initialization are
a combined system comparison, not a pure pretraining ablation. LR-ASPP has more parameters
than this project's tiny U-Net and its coarse features may hurt cracks; no speedup or
accuracy improvement is assumed.

Replace `--device mps` with `--device cpu` if needed; there is no silent backend fallback.
MPS kernels and real training remain to be verified manually. Determinism is disabled to
avoid known MPS bilinear-backward restrictions; seeds/code/configuration/device are recorded,
but bitwise cross-device equality is not promised. Run matched cohorts on the same backend.
Checkpoints support inference/model selection, not optimizer-resume training. New IDs are
required for reruns; no existing baseline or output is overwritten.

The CPU smoke deliberately uses a tiny U-Net fixture (base channels 2, depth 2) and the
full LR-ASPP architecture with no pretrained weights. The real matched U-Net retains
Task 03's base channels 16/depth 3 configuration. Smoke scores are not real-data results.

## Expected outputs

```text
outputs/task09/
  implementation_report.json
  reviews/{report_id}/REPORT.md, summary.json, poor_class_diagnostics.json
    audit/{audit.json,class_statistics.csv,checked_samples.csv,issues.csv,resize_occurrences.csv}
    evaluations/{baseline}/{metrics.json,per_class.csv,per_image_class_metrics.csv,predictions.csv}
      confusion_matrix.{npy,png}, per_image_confusion.npz, predictions/*.png
      failure_selection.json, failures/class_{id}/*.png
  audits/{report_id}/...                         # optional independent/full development QA
  plans/{report_id}.json
  experiments/{run_id}/{run.json,history.json,history.csv,learning_curves.png,best.pt,last.pt,status.json}
  evaluations/{checkpoint_hash}/validation/... # fresh evaluation, never final test
  comparisons/{report_id}.json
  inference/{report_id}/damage_features.json, prediction_ids.png, reporting_filtered_ids.png
    uncalibrated_scores.npz, comparison.png
  smoke/{run_id}/...                           # synthetic model/feature artifacts only
```

All failures are relative to supplied source annotations. Rare-class counts and source
identity uncertainty limit interpretation. No new test cohort exists here; the already-used
final test is excluded. Use class-wise metrics and failure images, not foreground mIoU alone,
before adopting a new model. Any annotation correction needs expert evidence and a versioned
curation decision that preserves frozen test membership.
