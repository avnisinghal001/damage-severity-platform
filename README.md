# Damage Analytics and Severity Prediction Platform

Task 01 provides the dataset foundation for future concrete damage research:
safe ingestion, validation, exploratory analysis, and sample visualization.
It does not train models, infer severity, or implement an API or frontend.

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
│   └── task01.toml
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
├── tests/
│   ├── test_archives.py
│   ├── test_annotations.py
│   ├── test_config.py
│   ├── test_discovery.py
│   └── test_pipeline.py
├── docs/
│   ├── START_HERE.md
│   ├── architecture.md
│   ├── dataset.md
│   └── learning/
│       └── task_01.md
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

No packages or datasets were installed or downloaded. The existing archives were
inspected and safely extracted. Standard-library checks passed: 16 tests passed,
6 image-dependent tests were skipped. Pytest, Ruff, image decoding, mask generation,
statistics, plots, and gallery rendering could not be executed because the required
packages are absent. After installation, run the full command sequence above.

Next: resolve/document invalid polygons and label terminology, visually review masks,
and design duplicate-aware, source-group-aware train/validation/test splits. Establish
expert-defined, independently annotated severity targets before planning severity
prediction. See [START_HERE](docs/START_HERE.md), [architecture](docs/architecture.md),
and the complete [Task 01 lesson](docs/learning/task_01.md).
