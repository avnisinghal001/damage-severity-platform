# Damage Analytics and Severity Prediction Platform

Task 01 provides the dataset foundation for future concrete damage research:
safe ingestion, validation, exploratory analysis, and sample visualization.
It does not train models, infer severity, or implement an API or frontend.

Task 02 adds explicit polygon/image cleaning policies, exact and near-duplicate
evidence, conservative grouping, reproducible multi-label splits, and verified,
immutable dataset releases. Start with the [Task 02 workflow](#task-02-curation-and-splitting)
or the [beginner lesson](docs/learning/task_02.md). A technically valid release does
not prove expert annotation correctness or absence of unknown-source leakage.

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
