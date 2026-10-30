# Damage Analytics and Severity Prediction Platform

Task 01 provides the dataset foundation for future concrete damage research:
safe ingestion, validation, exploratory analysis, and sample visualization.
It does not train models, infer severity, or implement an API or frontend.

Task 02 adds explicit polygon/image cleaning policies, exact and near-duplicate
evidence, conservative grouping, reproducible multi-label splits, and verified,
immutable dataset releases. Start with the [Task 02 workflow](#task-02-curation-and-splitting)
or the [beginner lesson](docs/learning/task_02.md). A technically valid release does
not prove expert annotation correctness or absence of unknown-source leakage.

Task 03 adds a from-scratch PyTorch U-Net and a version-pinned training/evaluation
workflow. See [Task 03 commands](#task-03-semantic-segmentation) and the complete
[training lesson and viva guide](docs/learning/task_03.md). Its original implementation
had no PyTorch runtime; a baseline was subsequently trained and its final test executed.
**That used test is excluded from Task 04 selection and iterative comparisons.**
Task 04 adds [controlled improvements and scoped region analytics](#task-04-improvements-and-damage-analytics).

The [RC1841 source](https://data.mendeley.com/datasets/yfhwfcrfmk/1)
(DOI: `10.17632/yfhwfcrfmk.1`) describes damage segmentation.
**The actual supplied label archive contains JSON polygons, not raster masks.**
This project preserves the eight labels found in those files and derives masks.
Damage categories are not an independently validated severity target.
See [dataset evidence and limitations](docs/dataset.md) before using the results.

## Project structure

```text
damage-severity-platform/
├── .gitignore
├── pyproject.toml
├── README.md
├── config/
│   ├── task01.toml
│   └── task02.toml
├── src/
│   └── damage_analytics/
│       ├── __init__.py
│       ├── __main__.py
│       ├── cli.py
│       ├── config.py
│       ├── labels.py
│       ├── io.py
│       ├── archives.py
│       ├── discovery.py
│       ├── annotations.py
│       ├── validation.py
│       ├── analysis.py
│       └── visualization.py
│       # Task 02 subpackage is listed below.
├── tests/
│   ├── test_archives.py
│   ├── test_annotations.py
│   ├── test_config.py
│   ├── test_discovery.py
│   └── test_pipeline.py
│   # Task 02 test files are listed below.
├── docs/
│   ├── START_HERE.md
│   ├── architecture.md
│   ├── dataset.md
│   └── learning/
│       ├── task_01.md
│       └── task_02.md
├── data/                          # ignored; supplied/generated locally
│   ├── raw/rc1841/
│   │   ├── imagedata.zip
│   │   └── labeldata.zip
│   └── interim/rc1841/            # created by ingest
│       ├── images/               # 1,841 original JPGs in inspected archives
│       ├── annotations/          # 1,841 original JSON files
│       └── provenance.json
└── outputs/task01/               # ignored; see output table below
```

OS metadata, environment internals, Git internals, and individual dataset filenames
are omitted from the tree. No datasets or generated reports belong in Git.

## Setup and complete Task 01 command sequence

Use Python 3.11 or newer. Run everything from the repository root. These are
instructions for you to execute; dependency installation was not executed during
implementation. `pip install -e` makes the `src/` package importable and supplies
the command-line entrypoint. The `dev` extra includes pytest and Ruff.

```bash
# 1. Create an environment if you do not already have .venv.
python3 -m venv .venv

# 2. Activate it (macOS/Linux, bash or zsh).
source .venv/bin/activate

# 3. Install declared dependencies and the local package.
python -m pip install -e '.[dev]'

# 4. Prepare the archive location.
mkdir -p data/raw/rc1841
```

Place the two original archives in that folder. They are already present in the
current workspace. For a fresh clone with archives elsewhere, the optional copy
commands below prompt for the existing local folder; they do not download data.

```bash
printf 'Folder containing imagedata.zip and labeldata.zip: '
read -r RC1841_ARCHIVE_DIR
cp "$RC1841_ARCHIVE_DIR/imagedata.zip" data/raw/rc1841/
cp "$RC1841_ARCHIVE_DIR/labeldata.zip" data/raw/rc1841/
```

Then run the stages separately, in this order:

```bash
# 5. Check archive safety, inventory members, and record SHA-256 hashes.
python -m damage_analytics inspect

# 6. Extract once. Skip if data/interim/rc1841 already exists from a successful ingest.
python -m damage_analytics ingest

# 7. Validate, derive masks, calculate statistics, and generate plots.
python -m damage_analytics analyze

# 8. Generate original / mask / overlay examples from valid records.
python -m damage_analytics visualize

# 9. Run focused tests and lint checks.
python -m pytest
python -m ruff check src tests
```

Windows PowerShell activation: `.venv\Scripts\Activate.ps1`. The remaining Python
commands are the same; use your shell's file-copy commands for dataset placement.

The current workspace has already completed inspection and ingestion. Repeating
`ingest` intentionally fails rather than merging or overwriting data. To ingest a
different archive version, set new extracted/output directories in the configuration.
To change defaults: `python -m damage_analytics --config config/task01.toml analyze`.
Paths are relative to `project_root` in that file, never your username or machine.

**Exit codes:** `0` means the stage completed successfully; `2` means invalid
inputs, a missing dependency, or validation errors. Analysis writes diagnostic
reports and valid-subset outputs even when validation fails. Inspect `issues.csv`
and `validation.json`; a plot does not mean the whole dataset passed. The supplied
annotations include negative coordinates in `crack146` and `damage794`, so a
nonzero analysis exit is expected under the strict policy. After reviewing that
failure, `visualize` can illustrate the valid subset. Do not chain these steps
with `&&` if you intend to inspect a known validation failure.

## Expected outputs

All paths below are relative to `outputs/task01/` unless stated otherwise.

| Output | Contents |
| --- | --- |
| `archive_inventory.csv`, `archives.json` | ZIP members, sizes, and original archive hashes |
| `data/interim/rc1841/provenance.json` | Archive provenance beside extracted originals; outside outputs |
| `manifest.csv` | Valid image/annotation/derived-mask mappings, dimensions, hashes, overlap counts |
| `issues.csv`, `validation.json` | Errors, warnings, validation status, class definitions, processing rules |
| `duplicates.json` | Exact-byte duplicate groups and decoded-image-pixel duplicate groups |
| `masks/000001.png`, … | Lossless grayscale **class IDs 0–8**, not RGB visualizations |
| `summary.json` | Dataset aggregates and analysis population |
| `class_distribution.csv` | Pooled pixels, pixel fractions, image presence, polygon counts |
| `dimensions.csv`, `per_image.csv` | Dimension frequencies and per-image statistics |
| `plots/class_pixels.png` | Pixel fractions including background |
| `plots/category_frequency.png` | Images with each source annotation label |
| `plots/image_dimensions.png` | Stored width/height scatter plot |
| `sample_gallery.png`, `gallery_samples.json` | Reproducible examples, legend, and selected sample IDs |

Reports describe valid pairs only; duplicates remain counted. Mask numbers follow
sorted discovered-pair order, so use `manifest.csv` instead of guessing filenames.
Unused masks from earlier runs may remain; only current manifest entries are
authoritative. Analysis invalidates earlier plots and galleries before regenerating
outputs. Commands operate on one output directory at a time; concurrent runs sharing
that directory are not supported.

## Implementation verification and next step

During the first Task 01 implementation, no packages or datasets were installed or
downloaded, and dependency-free verification passed 16 tests with 6 skipped. The
environment subsequently acquired the declared dependencies outside that implementation.
Task 01 reports now exist: 1,810 pairs passed its original policies and 31 failed.
The Task 02 verification below includes all Task 01 regression tests with no skips.

Next: resolve/document invalid polygons and label terminology, visually review masks,
and design duplicate-aware, source-group-aware train/validation/test splits. Establish
expert-defined, independently annotated severity targets before planning severity
prediction. See [START_HERE](docs/START_HERE.md), [architecture](docs/architecture.md),
and the complete [Task 01 lesson](docs/learning/task_01.md).

## Task 02 curation and splitting

Task 02 reads the existing archives, extracted originals, and Task 01 reports. It
never writes into them. Configuration lives in `config/task02.toml`, which reuses
Task 01's input paths. No additional runtime dependencies are needed.

Run these commands from the repository root, in order. Installation commands below
are for the user to execute; the implementation did not run installation/downloads.
The current workspace already has `.venv`, archives, extracted data, and Task 01
reports, so skip creation/placement/ingestion steps that are already complete.

```bash
# 1. Create only when .venv does not exist; activate your environment.
if [ ! -d .venv ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate

# 2. Install only when permitted/needed. Not executed during implementation.
python -m pip install -e '.[dev]'

# 3. Both original ZIPs must be in data/raw/rc1841/.
mkdir -p data/raw/rc1841
# If absent, use the local archive copy commands in the Task 01 setup above.

# 4. Inspect, and extract only if a completed extraction does not exist.
python -m damage_analytics inspect
if [ ! -d data/interim/rc1841 ]; then
    python -m damage_analytics ingest
fi

# 5. Produce Task 01 evidence only if it is missing.
# Known source errors make this stage exit 2; review reports and continue to curation.
if [ ! -f outputs/task01/validation.json ]; then
    python -m damage_analytics analyze
fi

# 6. Run Task 02, then independently reverify the published release.
python -m damage_analytics --config config/task02.toml curate
python -m damage_analytics --config config/task02.toml verify-curation

# 7. Examine the machine-readable quality report and release location.
python -m damage_analytics curation-report
python -m json.tool outputs/task02/latest.json
python -m json.tool outputs/task02/quality_report.json

# 8. Run all synthetic tests and lint checks, including Task 01 regression tests.
python -m pytest -q
python -m ruff check src tests
```

To inspect individual tables without guessing the content-addressed release path:

```bash
python - <<'PY'
import csv
import json
from pathlib import Path
root = Path('outputs/task02')
release = root / json.loads((root / 'latest.json').read_text())['release_path']
print('Release:', release)
for name in ('exclusions.csv', 'class_statistics.csv', 'decisions.csv'):
    with (release / name).open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    print(name, 'rows:', len(rows), 'first five:', rows[:5])
for split in ('train', 'validation', 'test'):
    with (release / 'splits' / f'{split}.csv').open(newline='', encoding='utf-8') as stream:
        print(split, 'samples:', len(list(csv.DictReader(stream))))
PY
```

Task 02 additions and generated artifacts:

```text
src/damage_analytics/curation/
├── __init__.py
├── config.py              # Policies and Task 01 path reuse
├── geometry.py            # Simple-ring intersection/area checks
├── cleaning.py            # Per-shape decisions, no hidden repairs
├── images.py              # EXIF evidence and explicit RGB derivatives
├── metadata.py            # Conservative camera/time session candidates
├── masks.py               # Class coverage and pixel invariants
├── similarity.py          # Fingerprints, evidence edges, connected groups
├── samples.py             # Per-sample export and exact-image conflicts
├── splitting.py           # Deterministic grouped multi-label search
├── provenance.py          # Archive/member integrity and release identity
├── reporting.py           # Tables, class statistics and Task 01 audit
├── verification.py        # Saved-image/mask/source/partition verification
└── pipeline.py            # Staging, publication, idempotence and frozen versions

tests/
├── test_curation_geometry.py
├── test_curation_images.py
├── test_curation_metadata.py
├── test_curation_similarity.py
├── test_curation_splits.py
└── test_curation_pipeline.py

outputs/task02/                         # All ignored by Git
├── latest.json                        # Convenience pointer, not a mutable split
├── quality_report.json                # Latest report copy
└── versions/<dataset-version>/         # Sealed; never overwritten
    ├── release.json                   # Config, seed, code/environment/input hashes
    ├── checksums.json                 # Hashes of every other release file
    ├── source_inventory.csv           # Every archived/extracted member fingerprint
    ├── sample_audit.csv               # Readability/mode/format/frame/check status for all IDs
    ├── task01_audit.json               # Every historical issue and its current disposition
    ├── curated_manifest.csv
    ├── exclusions.csv
    ├── decisions.csv
    ├── errors.csv
    ├── similarity_edges.csv
    ├── groups.csv
    ├── statistics.json
    ├── class_statistics.csv
    ├── dimensions.csv
    ├── split_statistics.json
    ├── quality_report.json
    ├── splits/{train,validation,test}.csv
    ├── images/<number>.png             # RGB, declared annotation frame; original resolution
    ├── annotations/<number>.json       # Original labels, explicit cleaned vertices
    └── masks/<number>.png              # Lossless class IDs 0–8
```

Identical inputs, policies, code, environment, and Task 01 evidence select the same
release. Rerunning verifies and reuses it without rewriting its test split. A changed
input/policy/code/environment/evidence snapshot creates a new version; pin the chosen
version and test CSV hash before modeling. Do not regenerate test membership to
improve a model result. Pixel-level consistency checks use the documented Pillow
rasterizer; they do not establish independent expert correctness of the annotations.

The default policy does not clamp negative coordinates or repair non-simple rings.
It permits exact redundant-vertex removal, preserves boundary-edge conventions,
aligns EXIF only when dimensions identify the frame, and logs RGB conversion.
Cross-class overlaps and exact-image annotation conflicts are excluded. Same-class
overlaps are accepted as unions. Identical image+mask duplicates may be collapsed;
near-duplicate candidates are retained but grouped. No rare-class rebalancing occurs.

See [dataset.md](docs/dataset.md#task-02-curation-policy-and-evidence) for interpretation
limits and [task_02.md](docs/learning/task_02.md) for all operations and decisions.

### Verified Task 02 result

Final release: `rc1841-curated-v1-d34771dbb662406a` (28 October 2026).
All 1,841 images and annotations were readable and matched original archive member
bytes. **1,617 pairs retained; 224 excluded.** Exported derivatives, full mask grids,
label fidelity, source hashes, and exact split partitions passed verification.

| Split | Samples | Fraction | Evidence components |
| --- | ---: | ---: | ---: |
| Train | 1,132 | 70.0062% | 898 |
| Validation | 243 | 15.0278% | 192 |
| Test | 242 | 14.9660% | 189 |

All eight foreground classes occur in every split. There are 1,279 curated evidence
components; the largest has 27 retained samples from a 29-image temporal sequence.
Zero duplicate manifest IDs, cross-split evidence edges, or cross-split components
were found. Physical source identity remains unverified, so this is not a global
leakage-free claim. The verdict is **CONDITIONAL_RESEARCH_USE**.

Exclusion reasons overlap: 171 non-simple source/pixel rings, two out-of-bounds
annotations, 26 cross-class overlaps, 24 exact-image annotation conflicts, and four
members of exact-image groups with unusable annotations. These refer to 224 unique
IDs, not their sum. Of 29 evaluated EXIF alignments, 27 survive all checks; 17 of 26
redundant-vertex cleanups survive, and the one CMYK conversion survives. Five retained
samples have accepted same-class overlaps. No identical image+mask pair remained
eligible for deduplication; the exact-image groups had conflicting/unusable annotations.

The curated set has 779,919,568 pixels and 252 multi-label images. Foreground pixel
imbalance is approximately 336.91:1. Concrete crushing (44 images) and structural
deformation (46) are flagged as rare under the configured threshold of 60; neither
is automatically discarded or rebalanced. Exact global/split class counts are in
`class_statistics.csv` and the quality report. Similarity evidence includes 674
near-image candidate edges, 82 near-annotation candidate edges, and 28 temporal
candidate edges across the full audited population, including excluded nodes.

The final synthetic test suite passes **89 tests and 21 subtests**, with no skips;
Ruff passes. No installation or download commands were executed for Task 02. Pin
the final test CSV SHA-256 before modeling:

```text
455e76654739e3e91c71553946cb0863ce355ebc32d02e1584e0080379f9a9f7
```

An earlier processing-only release remains as immutable history; `latest.json` points
to the final version that includes capture-session evidence. Neither release has
been used for model tuning. Expert review, source identification, and category
terminology resolution remain prerequisites for stronger benchmark claims.

## Task 03 semantic segmentation

`config/task03.toml` pins `rc1841-curated-v1-d34771dbb662406a` and the SHA-256 of its
sealed checksum manifest. Task 03 never follows `latest.json`, reruns curation, recreates
splits, or changes Task 01/02 code or data. It uses their existing 1,132/243/242 splits.
The added `training` dependency extra declares PyTorch; torchvision is unnecessary.
No dependency, dataset, or pretrained-weight installation/download was executed.

Run from the repository root. Installation is an instruction for you, not an operation
performed during implementation. Use a Python/OS/PyTorch combination with compatible
wheels; package installation and PyTorch runtime compatibility are unverified here.

```bash
# 1. Create only if necessary, then activate.
if [ ! -d .venv ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate

# 2. Install declared extras when permitted. NOT executed during implementation.
python -m pip install -e '.[dev,training]'

# 3. Verify the exact pinned release and audit resizing before training.
python -m damage_analytics.training verify-data
python -m damage_analytics.training audit-data

# 4. Inspect actual tensor batches and run a tiny synthetic CPU smoke test.
python -m damage_analytics.training inspect-loader
python -m damage_analytics.training smoke

# 5. Run tests, then check tiny-subset memorization before a full experiment.
python -m pytest -q
python -m ruff check src tests
python -m damage_analytics.training overfit --run-id overfit_001

# 6. Train normally; only train and validation are consumed.
python -m damage_analytics.training train --run-id unet_001

# 7. Evaluate the best checkpoint on validation and generate comparisons.
python -m damage_analytics.training validate \
  --checkpoint outputs/task03/experiments/unet_001/best.pt

# 8. Predict one known VALIDATION image, using the checkpoint's saved preprocessing.
python -m damage_analytics.training predict \
  --checkpoint outputs/task03/experiments/unet_001/best.pt \
  --image outputs/task02/versions/rc1841-curated-v1-d34771dbb662406a/images/000016.png \
  --prediction-id validation_example_001
```

Use `--device cpu` before the subcommand to force CPU, for example:
`python -m damage_analytics.training --device cpu train --run-id cpu_001`.
`auto` chooses available Apple MPS, otherwise CPU; unsupported MPS/deterministic
operations fail visibly. There is no hidden runtime fallback or cross-device bitwise guarantee.

Historical Task 03 protocol only: its final test has now already been executed.
**Do not rerun this command for Task 04 models or comparisons.** The original explicit
final-evaluation command is retained here to document Task 03:

```bash
python -m damage_analytics.training final-test \
  --checkpoint outputs/task03/experiments/unet_001/best.pt
```

Final test requires a completed normal run's validation-selected best checkpoint and
refuses synthetic/overfit models. Its result directory cannot be overwritten. This
guard does not prevent deliberate test tuning across different experiments; maintain
the protocol. Ordinary prediction rejects known test-release image paths/hashes.
Integrity verification reads sealed test file bytes and partition metadata only;
train/validation auditing does not decode or evaluate test samples.

### Task 03 architecture and output locations

```text
config/task03.toml
src/damage_analytics/training/
├── __init__.py
├── __main__.py
├── cli.py                  # Separate Task 03 commands; earlier entrypoints untouched
├── config.py               # Hyperparameters, path validation, ignore-index convention
├── manifests.py            # Pinned seals/schemas/partitions, selected-pair verification
├── transforms.py           # Shared crop/letterbox/flip geometry; RGB-only normalization
├── dataset.py              # PyTorch Dataset, collator, seeded DataLoader
├── audit.py                # Train/validation resize fidelity and train-only class weights
├── weights.py              # Capped inverse-square-root training pixel weights
├── model.py                # Configurable lightweight nine-logit GroupNorm U-Net
├── losses.py               # Ignored-padding cross entropy and optional foreground Dice
├── metrics.py              # Pooled confusion counts, IoU/Dice/precision/recall
├── runtime.py              # Seed setup and available MPS/CPU selection
├── checkpoints.py          # Atomic state saves and explicit weights-only loading
├── experiments.py          # New run directories, code/environment/dataset provenance
├── engine.py               # Epoch loops, AdamW, clipping, early stopping, best/last states
├── evaluation.py           # Native-resolution validation and explicit final test
├── inference.py            # Native-size ID predictions, no severity interpretation
├── plots.py                # Loss curves, confusion plot, RGB/GT/prediction comparisons
└── smoke.py                # Three-step synthetic CPU/checkpoint diagnostic

tests/
├── conftest.py             # Tiny sealed synthetic release fixture
├── test_training_data.py
├── test_training_metrics.py
└── test_training_torch.py

docs/learning/task_03.md
outputs/task03/                             # Git-ignored
├── data_verification.json                  # Generated here
├── implementation_checks.json              # Actual checks, skipped tests and blockers
├── data_audit/                             # Generated here
│   ├── resize_audit.json
│   ├── resize_occurrences.csv
│   └── training_class_weights.json
├── loader_inspection.json                  # Requires PyTorch; not generated here
├── synthetic_smoke/                        # Requires PyTorch
│   ├── synthetic.pt
│   └── smoke.json
├── experiments/<run-id>/                    # Requires PyTorch
│   ├── run.json
│   ├── class_weights.json
│   ├── history.csv
│   ├── history.json
│   ├── learning_curves.png
│   ├── best.pt
│   ├── last.pt
│   ├── best_validation_metrics.json
│   └── status.json
├── evaluations/<checkpoint-hash-prefix>/    # Requires PyTorch
│   ├── validation/
│   └── test_final/
│       # Each successful evaluation: metrics.json, per_class.csv,
│       # confusion_matrix.npy/.png, comparisons/*.png
└── inference/<prediction-id>/               # Requires PyTorch
    ├── prediction_ids.png
    ├── comparison.png
    └── prediction.json
```

Run IDs, prediction IDs, smoke output, and evaluation directories refuse overwrites.
Read existing artifacts or choose a new ID; do not delete evidence to select favorable
results. `best.pt` tracks every strict validation foreground-mIoU improvement;
`min_delta` separately controls early-stopping patience. Checkpoints include optimizer
state, but this task does not implement exact interrupted-run resume.

Input images become float32 N×3×H×W tensors; targets are int64 N×H×W. Source IDs remain
0–8. Padding uses **255 only as an ignored target**, not a tenth class. Images preserve
aspect ratio and use bilinear resizing; masks always use nearest-neighbor. Training
may use synchronized flips and configurable source-resolution foreground-aware crops.
Validation/inference use full images. Metrics compare native ground truth with unpadded
predicted IDs nearest-restored to native resolution, avoiding resized-target evaluation
that could hide thin-crack losses. Epoch losses are valid-pixel-weighted batch means;
metrics instead pool confusion counts over the entire split.

### Actual Task 03 verification

- The pinned release seal and all artifact checksums passed; 1,132 train and 243
  validation RGB/mask pairs were decoded with aligned sizes and valid IDs/counts.
- At 256×256, no foreground class vanished completely from a train/validation image.
  Microcrack resize/restore recall is 95.6384% train and 95.0950% validation;
  minor-spalling recall is 97.0153% and 96.3013%. Individual fragments can still disappear.
  These are **mask resampling diagnostics, not model results or an upper performance bound**.
- Training-only weights were calculated from the 1,132 training manifest rows. Native
  counts approximate the augmented distribution; cropping intentionally biases sampling.
- **112 tests and 21 subtests passed; eight PyTorch tests skipped. Ruff passed.**
  DataLoader/model/loss/backprop/checkpoint/training/evaluation/inference execution remains
  unverified because PyTorch is absent. `inspect-loader` and `smoke` exited 2 with a clear
  missing-dependency message; installation was not attempted.
- No model checkpoint, learned metric, test prediction, or test evaluation was generated.
  Task 02's conditional research-use/source-identity limitations still apply. Segmentation
  classes are never reported as structural safety or severity scores.

The bullets above record the initial Task 03 implementation. Task 04 below records
the subsequently available trained baseline and current runtime verification.

## Task 04 improvements and damage analytics

Task 04 preserves Task 01–03 implementations, the original `unet_001` files and frozen
splits. Settings live in `config/task04.toml`; a separate
`python -m damage_analytics.analytics` entrypoint has **no final-test command**.
No dependencies were installed/downloaded and no real-data improvement training was
launched during implementation. Existing torch 2.15.0 runs on CPU here; MPS is unavailable.
Task 04 defaults to strict deterministic CPU. An earlier Task 03 MPS sanity run failed
on an unsupported deterministic operation. To deliberately use MPS, set `device="mps"`
and `deterministic=false` in Task 04's TOML before training **all** comparison runs;
document the nondeterminism and rerun the matched control under that same policy.

### Verified baseline findings

Fresh validation inference on all 243 images exactly reproduces epoch 48's foreground
mIoU **0.23990623390029908** and confusion matrix. Saved checkpoint score, history,
status, split provenance and ground-truth support agree. The full 50-epoch history
shows an improving training loss and noisier validation behavior; this suggests a
generalization/optimization gap, without proving a single cause.

| Category | Validation IoU | Precision | Recall | Main concern |
| --- | ---: | ---: | ---: | --- |
| Microcrack | 0.1691 | 0.2014 | 0.5133 | Thin regions and false positives |
| Structural crack | 0.4075 | 0.4489 | 0.8157 | Overprediction/class confusion |
| Minor spalling | 0.1052 | 0.1079 | 0.8048 | Predicted area is 7.46× annotated area |
| Moderate spalling | 0.2328 | 0.4808 | 0.3110 | Misses, often labeled corrosion |
| Delamination | 0.1525 | 0.5341 | 0.1759 | Frequently labeled corrosion |
| Rebar corrosion | 0.5786 | 0.6660 | 0.8150 | Strongest foreground class, still imperfect |
| Concrete crushing | 0.0093 | 0.6910 | 0.0094 | Almost entirely missed; seven validation images |
| Structural deformation | 0.2642 | 0.2915 | 0.7386 | Many false positives; seven validation images |

The fixed-prefix gallery and targeted cases were inspected. `damage490` overpredicts
minor spalling; `damage801` misses annotated crushing and predicts delamination;
`Image_119_3665189` confuses corrosion/spalling with deformation. Targeted galleries
are explicitly worst-case diagnostic selections, not representative accuracy samples.
Class weighting is a plausible contributor, not an established cause without ablation.

### Controlled manual experiments

| Name | Change from saved baseline | Hypothesis / tradeoff |
| --- | --- | --- |
| `control` | None in model/data/optimizer/seed | Replay on the current backend for fair comparisons |
| `capped_weights` | CE maximum class weight 10 → 1 | Soften rare-pixel false-positive bias; recall may fall |
| `resolution_384` | Input grid 256² → 384² | Preserve detail; nominal input-pixel compute is 2.25× |

All retain CE + foreground Dice, architecture, seed, batch size, epoch cap and selection
rule. The cap changes only that hyperparameter; background and lower existing weights
remain unchanged. Original training used nondeterministic MPS, while current runs use
the configured backend/determinism, so variants should be judged primarily against a
fresh matched control. Comparison rejects incompatible splits, native support,
settings, training backend/library versions or core training code.

The executed **resampling-only** audit improves validation Microcrack recall from
95.0950% to 97.5469% at 384². This supports investigating resolution; it is not model
accuracy, a guaranteed improvement, or a model-performance upper bound. Real-data
training for all three designs remains manual and unexecuted here.

### Damage and supported spalling labels

Native class-ID masks produce exact per-class pixel counts/image percentages,
4/8-connected semantic regions, exclusive-maximum bounding boxes, centroids, fill
fractions, boundary flags and mixed foreground complexes. Counts are image-space
measurements, not physical area, crack width, concrete-only surface coverage or safety.
A reporting-size filter never changes masks or area totals. Background components are
not counted as damage. Connected components are not verified physical instances.

Only source classes **Minor spalling** and **Moderate spalling** can yield a scoped,
category-derived label. Mixed/unsupported regions and low-score heuristic cases are
explicitly handled. Maximum softmax and normalized entropy remain **uncalibrated**;
no calibrated confidence or global severity/safety score is produced. Mask-only input
has unknown model uncertainty. A confident-looking segmentation can still be wrong.

The supervision audit finds:

| Category | Train regions / images / groups | Validation regions / images / groups |
| --- | --- | --- |
| Minor spalling | 121 / 54 / 36 | 30 / 12 / 6 |
| Moderate spalling | 85 / 78 / 67 | 21 / 17 / 12 |

Regions sharing images/groups are correlated. Only six known minor-spalling validation
groups are available, and labels share the segmentation annotation lineage. A separate
**independent severity classifier is not justified or trained**. A future constrained
category-discrimination study would require explicit scope, expert review and stronger
independent evaluation; neither names nor area percentages establish overall severity.

### Task 04 commands

Use existing dependencies; installation is not part of this task. Run from the root:

```bash
source .venv/bin/activate
python -m damage_analytics.analytics audit-baseline
python -m damage_analytics.analytics plan
python -m damage_analytics.analytics resolution-audit
python -m damage_analytics.analytics supervision
python -m pytest -q
python -m ruff check src tests
```

The completed baseline audit is sealed and reused idempotently if its inputs/policies
still match. Existing review/inference IDs refuse overwrites; view the implementation's
generated examples or choose a fresh report ID:

```bash
python -m json.tool outputs/task04/baseline_audit/audit.json
python -m json.tool outputs/task04/severity_supervision/assessment.json
python -m damage_analytics.analytics predict \
  --checkpoint outputs/task03/experiments/unet_001/best.pt \
  --image outputs/task02/versions/rc1841-curated-v1-d34771dbb662406a/images/000016.png \
  --report-id validation_demo_002
python -m damage_analytics.analytics analyze-mask \
  --mask outputs/task04/inference/validation_demo_002/prediction_ids.png \
  --report-id existing_mask_002
```

The following are **manual, potentially lengthy training commands**. They were not run
on real data during implementation. Review the plan/budget first; execute individually:

```bash
python -m damage_analytics.analytics train --experiment control --run-id control_s42
python -m damage_analytics.analytics train --experiment capped_weights --run-id capped_s42
python -m damage_analytics.analytics train --experiment resolution_384 --run-id resolution384_s42

python -m damage_analytics.analytics evaluate --run-id control_s42
python -m damage_analytics.analytics evaluate --run-id capped_s42
python -m damage_analytics.analytics evaluate --run-id resolution384_s42
python -m damage_analytics.analytics compare \
  --run-id control_s42 --run-id capped_s42 --run-id resolution384_s42 \
  --comparison-id controlled_s42
```

Per-class metrics and descriptive paired validation-group bootstrap intervals accompany
comparisons. They do not account for training-seed variance, unknown physical sources,
adaptive validation selection or multiple comparisons. Do not declare generalization
from these comparisons; the already-used test stays excluded. Confirmatory claims
would need genuinely new independent evaluation data.

### Task 04 structure and outputs

```text
config/task04.toml
src/damage_analytics/analytics/
├── __init__.py
├── __main__.py
├── config.py         # Typed policies and baseline/input protection
├── cli.py            # No test-evaluation command; training is explicit
├── baseline.py       # History/checkpoint/report audit and error diagnosis
├── validation.py     # Fresh validation inference and independent confusion counting
├── experiments.py    # Saved-baseline settings, registered variants and manual training
├── comparison.py     # Fairness gates, per-class tables, paired descriptive bootstrap
├── components.py     # Exact run-length connected components without new dependencies
├── regions.py        # Pixel-space region measurements and uncertainty summaries
├── severity.py       # Only supported category-derived spalling labels
├── supervision.py    # Pure/mixed region counts and independent-group limitations
├── inference.py      # Native alignment, score maps and structured reports
└── review.py         # Explicitly targeted worst-case validation galleries
tests/
├── test_analytics_regions.py
├── test_analytics_comparison.py
└── test_analytics_workflow.py
docs/learning/task_04.md
outputs/task04/
├── implementation_report.json    # Actual checks and executed/unexecuted distinction
├── baseline_audit/               # Executed: audit.json, class_errors.csv, checksums.json
│   └── validation/               # Fresh metrics, per-class/per-image confusion,
│                                 # predictions, comparisons and provenance
├── experiment_plan.json         # Designs only; no model improvements claimed
├── plans/*.json
├── resolution_audit/             # Executed: 256² and 384² resampling diagnostics
├── severity_supervision/         # Executed: assessment, candidates, all-complex audit
├── targeted_validation_review/   # Executed: labeled worst-case galleries
├── inference/baseline_validation_000016/ # Executed: native mask, score maps, damage_report.json
├── development_history/         # Earlier implementation snapshots, not extra model trials
├── experiments/<run-id>/         # Expected only after manual real-data training
├── evaluations/<checkpoint-hash>/validation/ # Expected after evaluating those runs
└── comparisons/<comparison-id>/ # Expected after manual comparison
```

The original analytics delivery included tested synthetic integration and real baseline-image
inference. **154 tests and 21 subtests passed at that stage; Ruff passed.** The user has since
completed the 50-epoch CPU `control_s42` run (best validation mIoU **0.2431215908**, epoch 48).
No improvement variant or region-severity classifier has been trained, and no Task 03
final-test score was consumed for Task 04 selection or benchmarking. See the complete
[Task 04 lesson](docs/learning/task_04.md) for mathematics, findings, caveats and viva questions.

### Task 04: training performance

The shared trainer now supports bounded verified decoding caches, persistent workers
with correct epoch propagation, optional channels-last storage, explicit validation
cadence and tagged exploratory runs. It retains the original architecture, loss
mathematics, frozen splits and checkpoints. CPU prediction extraction and device scalar
handling were optimized without dropping native validation pixels. New histories record
training, validation and artifact times; plotting need not run every epoch.

Measured host medians for the same 64 training/validation images, batch size four:

| Execution | Training seconds | Validation seconds |
| --- | ---: | ---: |
| Original CPU, four threads | 7.566 | 3.040 |
| Optimized CPU, four threads | 6.889 | 2.631 |
| Optional channels-last CPU | 6.018 | 2.136 |
| MPS, strict determinism explicitly disabled | 1.431 | 0.683 |

These are short throughput probes, not full-run forecasts or segmentation improvements.
Compute dominates the CPU path. Strict deterministic MPS backward is unsupported for
the unchanged model's bilinear upsampling in this build; CPU determinism remains the
default. MPS is an explicit seeded, nondeterministic alternative. Never mix the old
CPU control and new MPS variants as a matched causal comparison.

```bash
source .venv/bin/activate
python -m damage_analytics.analytics profile --report-id my_cpu --samples 64 --repeats 3
python -m damage_analytics.analytics profile --report-id my_loaders \
  --suite loaders --samples 1132 --repeats 2
python -m damage_analytics.analytics profile --report-id my_mps \
  --device mps --nondeterministic --samples 64 --repeats 3
python -m damage_analytics.analytics profile --report-id my_mps_batches \
  --suite batches --device mps --nondeterministic --samples 64 --repeats 3
python -m damage_analytics.analytics profile-summary \
  --profile-id my_cpu --profile-id my_loaders --profile-id my_mps \
  --profile-id my_mps_batches --report-id my_summary
python -m pytest -q
python -m ruff check src tests
```

Profiles write only under `outputs/task04/performance/`, refuse overwrite and never
consume test samples. They benchmark temporary updates without saving model checkpoints.
The full [performance lesson](docs/learning/task_04_performance.md) documents memory
measurement limits, worker/cache/batch sweeps, every new module, validation scheduling,
and exact commands for matched five-epoch pilots followed by selected full runs.
No pilot or full training run was launched during performance implementation.

The full 1,132-image loader sweep tested workers 0/2/4, persistence and 0/128 MiB
caches. MPS with **two persistent workers** measured **1.122 seconds per 64 training
images**; four workers were only slightly faster with substantially higher process
memory. Batch eight gave little extra throughput while roughly doubling MPS allocation,
so batch four and no cache are recommended initially. Use `--workers 2 --persistent-workers`
with the manual MPS training commands in the lesson. Defaults remain deterministic CPU.

**174 tests and 21 subtests passed, no skips; Ruff passed.** The optimized epoch evaluator
also reproduced the existing control's confusion matrix exactly across all **54,435,143
native validation pixels**. Original checkpoints, control artifacts, model architecture,
paired transforms and dataset seal were verified unchanged. Actual timing tables are in
`outputs/task04/performance/measured_summary/`; this work establishes execution speed,
not a learned-model accuracy improvement.
