# Task 03: learn nine-class semantic segmentation with a small U-Net

Historical implementation snapshot: a baseline was subsequently trained and its final
test executed. [Task 04](task_04.md) records the current verification and further
validation-only development. Do not reuse Task 03's used test for Task 04 selection.

## 1. What Task 03 does

Task 02 prepared a versioned dataset. Task 03 connects its existing train/validation/test
CSV files to a from-scratch PyTorch segmentation model. It provides preprocessing,
training, validation, inference, explicit final test evaluation, diagnostics, and tests.
It does not change annotations, recreate splits, download pretrained weights, or create
severity targets. A segmentation prediction is not a structural safety assessment.

The pinned release is `rc1841-curated-v1-d34771dbb662406a`. It contains 1,132 train,
243 validation, and 242 test samples. These numbers are inherited, not newly randomized.
The unresolved source identities and category terminology from Task 02 still matter.
Read [dataset.md](../dataset.md) before interpreting future model performance.

The current environment has NumPy/Pillow/Matplotlib/pytest/Ruff but **no PyTorch**.
Dataset integrity, preprocessing, resizing analysis, and numeric metrics were checked.
Model/DataLoader/backprop/checkpoint/training/evaluation execution remains unverified.
No dependencies were installed, no model was trained, and no test model evaluation occurred.

## 2. Images, masks and tensors

A **tensor** is a multidimensional numeric array. A photograph starts as height × width ×
three RGB channels. PyTorch convolution layers expect channels before spatial dimensions.
For a configured batch of four 256×256 images:

```text
images:  [4, 3, 256, 256]   float32, normalized RGB
targets: [4,    256, 256]   int64, one category ID per pixel
logits:  [4, 9, 256, 256]   float32, nine unnormalized scores per pixel
argmax:  [4,    256, 256]   integer predicted category IDs
```

The first dimension is batch size. The final batch can be smaller; no samples are
dropped. Height and width are distinct from Pillow's `(width, height)` image-size order.

Source categories remain exactly Task 01/02's IDs 0–8. `Microcrack` is ID 1 and
`Structural crack` is ID 2; the integers are names, not severity rankings. The mask is
not normalized, converted to RGB for training, or treated as a continuous intensity.
Zero means the unannotated/background complement, not verified undamaged concrete.

Padding is represented by **255 in targets only**. This tells the losses to ignore
artificial padding. It is not another category: the network still has nine outputs.
Saved source and predicted masks contain only IDs 0–8.

## 3. Authenticate the exact existing data

`manifests.py` first compares `checksums.json` with the SHA-256 pinned in
`config/task03.toml`. It checks sealed release/quality metadata, exact category mapping,
split hashes, unique IDs, pixel-count totals, and group isolation. It does not follow
Task 02's mutable `latest.json`. This prevents a different release from being selected
silently when someone later curates another version.

The actual CSV fields include `sample_id`, `image`, `mask`, `width`, `height`, `group_id`,
`split`, file fingerprints, and `pixels_0` through `pixels_8`. Image/mask paths are
relative to the release folder. For example, a valid row references
`images/000016.png` and its paired `masks/000016.png`, not a guessed source filename.

Selected pairs are checked for RGB/L modes, equal dimensions, known IDs, expected pixel
counts, containment, and file fingerprints. Full integrity verification also hashes
sealed test artifacts as bytes. That is a data-integrity operation: it does not decode
test targets, audit test resizing, run predictions, or use test metrics for decisions.

Normal sample access refuses `test` unless explicitly authorized by the final-test
evaluation path. The guard prevents accidental routine use, not deliberate protocol
violations by someone writing separate code or copying raw test images elsewhere.

## 4. Resize without inventing annotation values

Images have different aspect ratios and resolutions. Stretching every image directly
to a square would distort crack widths and shapes. `transforms.py` instead fits the
image inside the target rectangle and adds centered padding, called **letterboxing**.
Integer rounding can change the fitted dimension by a pixel; recorded geometry makes
the transformation explicit.

RGB uses bilinear interpolation, appropriate for continuous color intensities. Masks
use nearest-neighbor interpolation so IDs remain integers from the existing class set.
Bilinear mask resizing could create fractional or misleading category values.

RGB is converted to `[0,1]`, then each channel uses `(value - mean) / std`. Defaults
are predefined mean/std 0.5, giving approximately `[-1,1]`. These are not ImageNet
statistics and are not estimated from test data. Padding becomes exactly zero after
normalization; padded target pixels are ignored by both losses.

### Thin cracks and rare pixels

Resizing a very thin region to a small grid can remove it even with nearest-neighbor.
`audit.py` measures this on train/validation only by resizing ground-truth masks and
restoring them to their native grid. It reports entire-category disappearance and
per-class native-pixel recall/IoU. This is **mask-resampling fidelity, not model accuracy,
an architectural upper bound, or proof that every connected fragment survives**.

Actual 256×256 audit results:

| Category | Train pixel recall | Validation pixel recall | Entire image-category occurrences lost |
| --- | ---: | ---: | ---: |
| Microcrack | 95.6384% | 95.0950% | 0 in both splits |
| Structural crack | 97.6983% | 98.4477% | 0 in both splits |
| Minor spalling | 97.0153% | 96.3013% | 0 in both splits |
| Concrete crushing | 99.5560% | 99.6026% | 0 in both splits |
| Structural deformation | 99.4358% | 98.8897% | 0 in both splits |

All foreground categories survive at image-category level, but pixel geometry is not
perfectly preserved. The synthetic tests demonstrate a one-pixel crack that disappears
under aggressive resizing. Examine `resize_occurrences.csv` before changing resolution.

### Training-only crops and paired augmentation

For larger source images, configurable source-pixel crops preserve more local detail
than shrinking the full image. The default proposes crops on half the training draws.
On a crop draw, foreground guidance is used with probability 0.8: choose a present
foreground category uniformly, then one of its pixels, and a crop containing that pixel.
Other draws use random placement. Crop dimensions are capped at the actual image size.

The image and mask receive exactly the same crop and flips. A flip applied to only one
would corrupt supervision. The configured horizontal flip probability is 0.5; vertical
flips are off by default. Evaluation has no random flips or crops.

Foreground guidance intentionally changes the training distribution and may remove
other regions outside a crop. It does not permanently discard a sample/class or change
source labels. The audit distinguishes crop-out context from disappearance caused by
resizing. Its one-crop-per-training-image diagnostic is not a simulation of every epoch.
Whether cropping improves a learned model remains unverified; compare ablations using
validation only. If the image is smaller than the source crop size, the crop may simply
be the full image.

## 5. Dataset and DataLoader

`dataset.py` wraps selected frozen rows in `SegmentationDataset`. Each item returns a
tensor pair, sample metadata and letterbox geometry. Validation items also keep the
native target on the CPU. A custom collator stacks fixed-size tensors and keeps
variable-size native masks as a list, so large masks are not stacked or sent to MPS.

The DataLoader controls batches and training shuffle. Validation stays ordered.
`num_workers=0` is the simple default; worker spawning is optional. Worker seeds and
the shuffle generator are explicit. Augmentation uses seed + epoch + sample position,
making it independent of worker scheduling. Workers are not persistent, so they see
the current epoch. A fixed run is reproducible within the supported runtime, subject
to backend numerical/determinism limitations.

`inspect-loader` reports real PyTorch tensor shapes/dtypes, IDs, and sample names.
It does not substitute NumPy shapes when PyTorch is missing; it exits with a clear error.

## 6. CNNs and our lightweight U-Net

A **convolution** slides a small learned filter over nearby pixels. Early filters can
learn local edges/textures; deeper features combine information across larger regions.
Filters are learned from the training loss, not assigned handwritten crack rules.
ReLU adds nonlinearity. Pooling reduces spatial resolution while the encoder gathers
context. These are the main ingredients of a convolutional neural network, or CNN.

The [original U-Net paper](https://arxiv.org/abs/1505.04597) describes a contracting path
and an expanding path for localization. This project implements a smaller adaptation,
not a reproduction of that paper's biomedical results. There is no pretrained backbone.

```text
RGB                       3 × 256 × 256
Encoder + skip           16 × 256 × 256
Pool + encoder + skip    32 × 128 × 128
Pool + encoder + skip    64 ×  64 ×  64
Pool + bottleneck       128 ×  32 ×  32
Upsample + skip + conv   64 ×  64 ×  64
Upsample + skip + conv   32 × 128 × 128
Upsample + skip + conv   16 × 256 × 256
1×1 classification head   9 × 256 × 256
```

**Skip connections** concatenate high-resolution encoder features with decoder
features, helping preserve location details. Bilinear upsampling enlarges feature
maps; convolutions learn how to combine them. A 1×1 head predicts nine scores at
every pixel. `argmax` chooses the largest score when an ID map is needed.

`model.py` uses GroupNorm because small batches are expected. It does not depend on
large-batch running statistics. Depth, base channels and normalization groups are
configurable; spatial settings must be divisible by `2**depth`. There is no inference
softmax requirement when only argmax is used, because softmax preserves score ordering.

## 7. Forward pass, losses, backward pass and optimizer

The **forward pass** maps a batch through the network to logits. A **loss** measures
how those logits disagree with the target. PyTorch autograd builds the computation
graph. `loss.backward()` computes gradients: how parameter changes affect the loss.
The optimizer then updates parameters. Gradients are cleared before the next batch;
otherwise they would accumulate unintentionally.

`losses.py` combines:

- **Weighted cross entropy:** at each valid pixel, reward a high log-probability for
  its true class. The weighted sum is normalized by target-weight mass. Pass raw logits,
  not a manually softmaxed input; cross entropy includes the needed log-softmax.
- **Optional foreground soft Dice:** compare differentiable foreground probabilities
  with one-hot targets. Average over foreground classes actually present in that batch.
  Ignore padding and background in this Dice average. An all-background batch still has
  cross-entropy supervision; its foreground Dice term is zero.

Dice weight 0 disables the extra term. A small smoothing constant avoids division
instability. Cross entropy still penalizes false-positive logits for absent categories.

`weights.py` uses only the selected training manifest's native pixel counts. The default
uses inverse square-root class frequency, normalizes present-class weights, and caps
extreme weights. It does not read validation/test counts to fit weights. Counts are
an approximation to the transformed distribution because cropping/nearest resizing
changes which pixels appear. The weights and counts are written explicitly.

`engine.py` uses AdamW with configured learning rate and weight decay. AdamW adapts
updates using gradient statistics and applies decoupled weight decay. Gradient clipping
limits the global gradient norm; nonfinite losses/gradients stop the run. There is no
automatic mixed precision or silent backend fallback in this baseline.

Epoch loss is a valid-pixel-weighted mean of batch losses. With weighted CE and
batch-present Dice, it is not identical to a single global loss over every native pixel;
batch composition can affect it. Numeric segmentation metrics use a different, pooled
native-grid calculation described below.

## 8. Validation, early stopping and native-grid metrics

Validation disables gradients and augmentation. It is used for learning-rate/loss/model
decisions and checkpoint selection. Training never opens the test DataLoader.

The best checkpoint is every **strict** improvement in validation foreground mIoU.
Early-stopping patience uses its own `min_delta` threshold, so a tiny improvement can
still be saved as best while failing to reset patience. This avoids confusing the
best actual score with the threshold for continuing training.

The model predicts on the configured grid. Evaluation removes padding, takes class
IDs and nearest-restores them to original resolution, then compares with the native
ground truth. This avoids resizing the evaluation target and hiding lost crack pixels.
Argmax occurs before native resizing, which keeps memory bounded on 12-megapixel images.
This declared prediction-restoration convention differs from interpolating full native
logits; do not compare protocols as if they were identical.

`metrics.py` adds confusion counts over the entire split before computing ratios. Rows
are ground truth and columns are prediction. For one class:

```text
IoU       = TP / (TP + FP + FN)
Dice      = 2 TP / (2 TP + FP + FN)
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
```

For example, one correct crack pixel, one missed crack pixel, and one false crack
pixel give IoU 1/3 and Dice/precision/recall 1/2. Precision is undefined when nothing
is predicted for a class; recall is undefined when no ground truth exists. These
values serialize as JSON null rather than misleading perfect scores or NaN.

A class absent from both target and prediction is excluded from overlap means.
A false-positive-only class has IoU/Dice zero and remains in the mean. Foreground
mIoU averages IDs 1–8 with defined IoU; background is excluded from averaging, but
foreground predictions on background still count as false positives.
Do not average per-batch IoUs: small and large batches/images would receive incorrect
relative weight. Pixel accuracy alone can look good when background dominates.

## 9. Checkpoints, code versions and experiments

Each new run ID gets its own directory; reusing an existing ID is refused.
`experiments.py` records the exact release/seal/split hashes, selected row IDs, config,
seed, device, source-code hashes, Git commit, actual Python/package versions, class
weights, and actual parameter count. These are more informative than a seed alone.

`checkpoints.py` atomically replaces `best.pt`/`last.pt` with model state, architecture,
optimizer state, epoch, selection score, metadata and CPU torch RNG state. Loading
uses explicit restricted `weights_only=True`, CPU mapping, strict state matching,
and expected dataset identity. Use trusted local checkpoints. Optimizer state is
preserved for future work, but exact interrupted-run resume is not implemented.

Evaluation/inference use preprocessing and loss settings saved inside the checkpoint,
not potentially edited current hyperparameters. Current execution provenance and the
checkpoint content hash are recorded with results. Runs sharing files should be
executed sequentially.

`runtime.py` chooses available Apple MPS or CPU. An explicitly requested unavailable
MPS device fails clearly. Seeds cover Python, NumPy, torch, loader shuffling and paired
augmentation. Deterministic algorithms are requested by default; unsupported operations
may fail. CPU/MPS/version changes are not promised bitwise-identical. See the official
[MPS notes](https://docs.pytorch.org/docs/main/notes/mps.html),
[reproducibility notes](https://docs.pytorch.org/docs/main/notes/randomness.html), and
[serialization notes](https://docs.pytorch.org/docs/main/notes/serialization.html).

## 10. Smoke checks, tiny overfitting and final test isolation

The synthetic smoke check makes simple colored half-image targets, performs three CPU
AdamW steps, checks finite gradients/decreasing loss, and verifies exact checkpoint
roundtrip outputs. It never uses the real release. It is a plumbing check, not learned
concrete-damage performance.

The overfit command selects a tiny prefix of **training** rows, disables crop/flip
augmentation, and trains/evaluates that same subset. It tests whether the pipeline can
memorize examples. Its apparent validation fields are explicitly marked training-subset
diagnostics in provenance. A configurable target of 0.90 foreground mIoU is a sanity
threshold, not a held-out benchmark. Failure can reflect bugs, insufficient optimization,
capacity, or resize/representation limitations; inspect masks and losses before claiming
an implementation is wrong. Overfit checkpoints are refused by final-test evaluation.

After normal training, `validate` reports loss and native-grid metrics and makes fixed-prefix
original/GT/prediction comparisons. It does not choose attractive examples by score.
`predict` produces a native-size class-ID PNG and comparison for an RGB image. Known
test-release image paths/hashes are refused so routine inference does not preview them.

Only `final-test` consumes test pixels/targets. It requires a completed normal run's
validation-selected best checkpoint and writes a non-overwritable `test_final` directory
keyed by checkpoint hash. Failed attempts are also recorded. It cannot prevent someone
from repeatedly running separate models against test, so research discipline still
matters: freeze choices first and never tune from test performance. No real final-test
evaluation has been run here.

## 11. Every module's place in the workflow

| Module | Responsibility |
| --- | --- |
| `__init__.py`, `__main__.py` | Package identity and guarded CLI execution |
| `cli.py` | Explicit data/diagnostic/train/validation/inference/final-test commands |
| `config.py` | TOML settings, typed objects, path/value validation, ignored padding |
| `manifests.py` | Pinned seals, CSV schemas, split identity and verified selected pairs |
| `transforms.py` | Shared letterbox/crop/flip geometry, RGB normalization and native restoration |
| `dataset.py` | Tensor conversion, collation, loader seeding and epoch-aware transforms |
| `audit.py` | Train/validation resize-loss evidence and saved training-only weights |
| `weights.py` | Training pixel counts and explicit capped class weighting |
| `model.py` | Lightweight nine-class U-Net, convolution blocks and skip connections |
| `losses.py` | Padding-aware weighted CE and optional present-foreground Dice |
| `metrics.py` | Integer confusion aggregation and correct undefined-class conventions |
| `runtime.py` | Device choice and seeded deterministic-operation setup |
| `checkpoints.py` | Atomic portable state saves and strict restricted loads |
| `experiments.py` | Unique output directories and actual execution provenance |
| `engine.py` | Epoch loops, gradients, optimizer, history, selection and early stopping |
| `evaluation.py` | Checkpoint-bound validation and explicitly authorized final test |
| `inference.py` | Native-size class-ID prediction for one RGB image |
| `plots.py` | Curves, confusion figures and labeled image/mask comparisons |
| `smoke.py` | Independent synthetic CPU optimization/checkpoint diagnostic |

## 12. Exact command sequence

Run from the repository root. These installation instructions were **not executed**.
Use an available PyTorch wheel compatible with your Python/OS; the current Python
environment has no torch, so runtime compatibility is not certified.

```bash
if [ ! -d .venv ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate
python -m pip install -e '.[dev,training]'

python -m damage_analytics.training verify-data
python -m damage_analytics.training audit-data
python -m damage_analytics.training inspect-loader
python -m damage_analytics.training smoke
python -m pytest -q
python -m ruff check src tests

python -m damage_analytics.training overfit --run-id overfit_001
python -m damage_analytics.training train --run-id unet_001
python -m damage_analytics.training validate \
  --checkpoint outputs/task03/experiments/unet_001/best.pt
python -m damage_analytics.training predict \
  --checkpoint outputs/task03/experiments/unet_001/best.pt \
  --image outputs/task02/versions/rc1841-curated-v1-d34771dbb662406a/images/000016.png \
  --prediction-id validation_example_001
```

The pinned release is already present. If it is missing, recover that exact Task 02
snapshot and verify it; do not silently create another split or change the release pin.
Do not run curation merely because adding training code would produce a new Task 02 code hash.
To change device/config, global options precede the subcommand:

```bash
python -m damage_analytics.training --config config/task03.toml --device cpu train --run-id cpu_001
```

After all choices are frozen, run the final evaluation separately:

```bash
python -m damage_analytics.training final-test \
  --checkpoint outputs/task03/experiments/unet_001/best.pt
```

Inspect actual preflight artifacts with:

```bash
python -m json.tool outputs/task03/data_verification.json
python -m json.tool outputs/task03/data_audit/resize_audit.json
python -m json.tool outputs/task03/data_audit/training_class_weights.json
```

After a real run, inspect `experiments/unet_001/status.json`, `history.csv`,
`learning_curves.png`, and the hash-keyed evaluation folder. The [README output
tree](../../README.md#task-03-architecture-and-output-locations) lists every artifact.
Repeated run IDs/evaluations refuse overwriting; existing artifacts remain evidence.

## 13. Actual tests and remaining work

The complete suite passed **112 tests and 21 subtests**, with **eight PyTorch tests
skipped** because torch is absent. Ruff passed. The new dependency-free tests cover
release pins/tampering, paired flips, ignored padding, crop guidance, nearest IDs,
native restoration, thin-pixel destruction, serialized resize audits, train-only
weights, configuration validation, and hand-computed confusion metrics.

The eight executable-but-skipped tests cover DataLoader tensors, forward/backward,
loss gradients/ignored padding, restricted checkpoint roundtrip/dataset binding,
synthetic optimization, synthetic train/validate/infer/final-test integration,
overfit/test isolation, and unavailable-MPS handling. Synthetic final-test fixtures
never refer to the real test set.

`inspect-loader` and `smoke` currently return exit code 2 with a missing-torch message.
No model checkpoint, training curve, learned metric, prediction, or test evaluation
has been generated. After permitted installation, run all eight skipped checks, the
smoke command, tiny overfitting, and CPU/MPS checks before trusting a baseline experiment.
Keep Task 02's physical-source/annotation limitations visible in every research report.
`outputs/task03/implementation_checks.json` records the actual checks and blockers.

## 14. Interview / viva questions with project-specific answers

1. **Why segmentation instead of image classification?** A single image can contain
   several damage categories, and the target describes their pixel locations.
2. **Why nine outputs?** Eight verbatim damage labels plus unannotated background;
   no severity target or extra padding class is added.
3. **Why is a target not a three-channel color image?** CE needs one integer category
   per pixel. Display colors are only a visualization convention.
4. **Why nearest-neighbor masks?** It preserves IDs. Bilinear blending can corrupt
   category values, and even nearest can remove very thin regions during downsampling.
5. **Why ignore padding?** Artificial square-fill pixels are not annotated concrete.
   Supervising them as background would bias the task.
6. **What do U-Net skips do?** Bring fine spatial encoder information into the decoder
   alongside deeper context, useful for narrow crack localization.
7. **Why GroupNorm?** Expected batches are small; this baseline avoids BatchNorm's
   dependence on running batch statistics.
8. **Why not softmax before cross entropy?** CE expects logits and applies a stable
   internal log-softmax. Softmax is used for differentiable Dice probabilities.
9. **Why use both CE and Dice?** CE gives pixelwise supervision; foreground Dice can
   emphasize region overlap. It is optional and must be judged on validation.
10. **Why are weights training-only?** Estimating them from held-out targets would
    let evaluation data influence the training objective.
11. **Why not average batch IoUs?** Ratios are nonlinear and batch populations differ.
    Sum confusion counts first, then compute dataset-level metrics.
12. **Why native-resolution evaluation?** Resizing targets can erase thin cracks and
    make results look easier. Native targets expose the configured prediction's losses.
13. **Does foreground mIoU ignore background mistakes?** No. Predicting a damage class
    on background contributes false positives even though background IoU is not averaged.
14. **Does tiny-subset overfitting prove generalization?** No; it checks memorization
    and processing/optimization plumbing on training examples only.
15. **Why freeze test until the end?** Repeated test-guided decisions turn the test
    into another validation set and bias the claimed generalization result.
16. **Does a fixed seed guarantee identical MPS and CPU results?** No. Runtime/backend
    kernels and versions can differ; record the actual environment and deterministic settings.
17. **Can damage area predict structural safety here?** No. Segmentation categories
    and pixel fractions are not independently validated severity or safety labels.
