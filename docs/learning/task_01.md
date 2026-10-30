# Task 01: understand and prepare the dataset

## 1. What you are building

The future platform may identify concrete damage and study severity. Task 01 builds the trustworthy input to that work: an inventory, validated image/annotation pairs, generated segmentation masks, statistics, and a visual review gallery. There is no model training yet. An accurate-looking model trained on incorrectly interpreted labels would still solve the wrong problem.

Read [dataset.md](../dataset.md) for evidence from the publisher and supplied files. The label archive contains JSON polygons, not colored masks. Its category strings must be preserved. In particular, `Microcrack` cannot be silently renamed from the landing page's terminology, and IDs cannot become severity ranks.

## 2. The important concepts

| Concept | Simple explanation |
| --- | --- |
| Pixel | One location in an image's rectangular grid |
| RGB | Three values describing red, green, and blue display intensity |
| Annotation | Human-provided information describing regions in an image |
| Polygon | A closed boundary described by a list of `[x, y]` points |
| Semantic segmentation | Assigning a category to each pixel |
| Rasterization | Filling polygon regions on a pixel grid to make a mask |
| Class ID | A project-assigned integer naming a category; not an ordered score |
| Mask | One class ID per pixel, stored here as a grayscale PNG |
| Overlay | A photograph blended with mask display colors for human review |
| Manifest | A table connecting each valid image, JSON annotation, and generated mask |
| SHA-256 | A content fingerprint used to record provenance and identify exact duplicates |
| Validation | Explicit checks that data follows the rules we understand |
| EDA | Exploratory data analysis: describing the dataset before modeling |
| Severity target | A separately defined outcome; this dataset foundation does not create one |

For a width of 640 and height of 360, stored pixel locations run from x=0 to 639 and y=0 to 359. NumPy image arrays are indexed `[y, x]`. Pillow image sizes are `(width, height)`. Confusing the two swaps the axes and misaligns masks.

A segmentation mask is not an ordinary RGB picture. A value of `2` means the project's `Structural crack` class, not a darker shade or a severity of two. The gallery converts IDs to display colors so people can see them. Background means unannotated pixels; it is not a verified safety assessment.

## 3. How the repository is organized

`src/` contains reusable Python code. `tests/` contains tiny synthetic examples. `config/` holds settings. `docs/` explains decisions. The original ZIPs live in `data/raw/`; extracted originals live in `data/interim/`; generated artifacts live in `outputs/`. Keeping these separate prevents accidental edits to the only copy of the source data. `.gitignore` keeps data, output, cache, and virtual-environment files out of version control.

`pyproject.toml` describes the package and dependencies. An **editable installation** connects your environment to this repository's source files, so code changes work without reinstalling the package. A **virtual environment** keeps dependencies for this project separate from other projects. Activating it chooses its Python and pip.

Functions have type hints and docstrings. A type hint describes intended inputs and outputs; it does not perform validation by itself. A docstring explains a function's purpose and important policies. Frozen dataclasses group related settings or polygon data and discourage accidental reassignment. Simple functions handle the work, rather than a large hierarchy of classes.

## 4. What each code module does

- `__init__.py` identifies the package and version. It avoids importing heavy libraries.
- `__main__.py` calls the CLI for `python -m damage_analytics`.
- `cli.py` reads your command and configuration path. It coordinates work and returns a useful exit code. It contains no image-processing rules.
- `config.py` reads TOML, resolves paths relative to the config's project root, and rejects overlapping input/output directories. `ArchiveLimits` bounds extraction; `Config` stores the resolved settings.
- `labels.py` stores the eight exact archive labels, project IDs, project display colors, source URL, and the annotation interpretation note.
- `io.py` writes JSON and CSV consistently and hashes files in small chunks. Streaming avoids loading an entire archive just to calculate its fingerprint.
- `archives.py` first inspects ZIP metadata. It rejects traversal such as `../file`, absolute paths, links, collisions, encryption, and excessive expansion. It then extracts into a temporary staging folder, reads through CRC verification, and publishes a completed tree. Raw files are never rewritten.
- `discovery.py` finds supported images and JSON files and matches exact filename stems. A stem is the name without its extension: `10` for `10.jpg`. It reports missing pairs and ambiguous repeated stems; it never pairs by listing position.
- `annotations.py` checks JSON schema, known labels, polygon points, image reference, and dimensions. `read_annotation` returns validated polygons. `rasterize` makes a mask and counts overlap pixels. Later shapes win when regions overlap.
- `validation.py` opens and fully decodes photographs, runs annotation checks, finds duplicates, saves masks, and records valid pairs and errors. It uses stored pixel orientation instead of silently applying EXIF rotations.
- `analysis.py` aggregates per-image records into pixel counts, image-level category frequencies, polygon frequencies, dimension summaries, CSV/JSON reports, and plots.
- `visualization.py` samples manifest records reproducibly, checks selected-file fingerprints, maps IDs to colors, and creates the original/mask/overlay gallery. It blends only foreground pixels, preserving the photograph in background regions.

Pillow reads images and fills polygons. NumPy works with pixel arrays and counts. Matplotlib draws plots without needing a GUI. Most other work uses the standard library. None of these modules fetches data or installs packages.

## 5. Run the complete workflow

Use Python 3.11 or later and open a terminal in the repository root. The commands below are instructions, not commands executed during project implementation.

### A. Create and activate the environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

If `.venv` already exists, activate it and skip creation. On PowerShell use `.venv\Scripts\Activate.ps1` for activation. The declared runtime dependencies are Pillow, NumPy, and Matplotlib; development dependencies include pytest and Ruff.

### B. Place the existing dataset archives

```bash
mkdir -p data/raw/rc1841
```

Put `imagedata.zip` and `labeldata.zip` there, with those exact names. They are already present in the current workspace. If they are in another local directory, use the following optional commands rather than downloading anything:

```bash
printf 'Folder containing imagedata.zip and labeldata.zip: '
read -r RC1841_ARCHIVE_DIR
cp "$RC1841_ARCHIVE_DIR/imagedata.zip" data/raw/rc1841/
cp "$RC1841_ARCHIVE_DIR/labeldata.zip" data/raw/rc1841/
```

### C. Inspect before extraction

```bash
python -m damage_analytics inspect
```

This validates ZIP member metadata and writes `archive_inventory.csv` plus `archives.json` under `outputs/task01/`. Check file counts and extensions before assuming that the label archive contains images. Inspection checks metadata; extraction reads every member and checks its CRC. The archive SHA-256 records which exact input bytes were used, not a guarantee that labels are correct.

### D. Extract once

```bash
python -m damage_analytics ingest
```

This creates `data/interim/rc1841/images/`, `annotations/`, and `provenance.json`. It refuses an existing destination. The current workspace has already completed this step; skip it there. For another dataset version, use new configured folders so archive versions do not get mixed. Never edit the original ZIPs.

### E. Validate and analyze

```bash
python -m damage_analytics analyze
```

This is one coordinated stage: discover pairs, decode images, validate JSON, rasterize valid polygons, identify duplicates, write reports, and draw plots. Start by opening `outputs/task01/validation.json` and `issues.csv`.

Errors include missing pairs, unknown source labels, corrupt images, mismatched dimensions, unsupported annotation formats, and invalid coordinates. Invalid pairs do not enter statistics. Warnings include duplicates, empty annotations, EXIF rotation tags, and overlapping polygons. Warnings remain visible but do not automatically exclude a pair.

An exit code of 2 means a check failed; it does not mean every output is unusable. The supplied source has negative coordinates in `crack146` and `damage794`, so strict validation is expected to fail even while producing valid-subset reports. Review the issue, preserve raw files, and document a later curation decision. Do not call those subset results a fully validated dataset.

### F. Generate the gallery

```bash
python -m damage_analytics visualize
```

Open `outputs/task01/sample_gallery.png`. Each row shows the stored photograph, derived colored mask, and overlay. The legend uses project display colors, not verified publisher RGB values. The configured seed chooses reproducible examples; `gallery_samples.json` records the selected IDs. This stage can run after you have reviewed a validation failure if at least one valid record remains.

Look for polygon alignment, narrow regions that disappear at gallery scale, overlapping categories, unusual source annotations, and empty-looking masks. The gallery is a small sample, not a complete annotation-quality audit.

### G. Run the synthetic tests and style checks

```bash
python -m pytest
python -m ruff check src tests
```

Tests do not require RC1841. They create tiny files in temporary directories, check important behavior, and clean up those fixtures. Archive tests check unsafe names, links, collisions, resource limits, preservation, and overwrite refusal. Annotation tests check labels, geometry, dimension/reference mismatches, overlap ordering, and boundary coordinates. Discovery tests check missing and ambiguous pairs. Pipeline tests check corrupt files, duplicate detection, conserved pixel counts, generated artifacts, deterministic sampling, and stale-source rejection.

For an environment where you are explicitly not installing anything, a subset can also be run with the standard library:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

Image-dependent tests are skipped if their dependencies are missing. This fallback passed 16 tests with 6 skipped during implementation; it is not a substitute for the full pytest and plotting verification after installation.

## 6. Understand the results

`manifest.csv` is the authoritative mapping. It records source-relative paths, output-relative mask paths, dimensions, fingerprints, and overlap information. Never pair files by sorting two directory lists or assume a mask number equals an original filename. Unused mask files may remain after another analysis run; only current manifest rows belong to the current report.

`class_distribution.csv` answers several distinct questions:

- `pixels`: how many pixels receive each class after overlap resolution.
- `pixel_fraction`: class pixels divided by all valid image pixels, including background.
- `images_with_pixels`: how many final masks visibly contain the class.
- `images_with_annotation`: how many JSON files have a polygon for the label.
- `polygons`: how many source polygon shapes carry that label.

If one image has three crack polygons, it adds one to image frequency and three to polygon frequency. If another polygon covers an earlier region completely, the earlier label can still contribute to annotation frequency while having no visible final pixels. These are both useful counts, but they answer different questions.

Pixel fractions pool the entire dataset. A 1000×1000 image has four times the pixel weight of a 500×500 image. A very small crack can occur in many images while using few pixels. Category frequency is multi-label, so image counts across categories need not sum to the number of images. Counts are not physical damage measurements.

`dimensions.csv` counts repeated sizes. `per_image.csv` supports detailed checks.
`duplicates.json` distinguishes exact file bytes from decoded-image pixels. Neither detects every near-duplicate, crop, resize, or image from the same structure.
`summary.json` includes population and severity limitations alongside the aggregates.

The [README output table](../../README.md#expected-outputs) lists every report and plot location. A run with no valid pairs produces diagnostic tables but no plots; gallery creation then fails with a clear message.

## 7. Reproducibility and troubleshooting

Change paths and settings in `config/task01.toml`. Its `project_root` is relative to that config file. Example invocation:

```bash
python -m damage_analytics --config config/task01.toml analyze
```

Different raw, extracted, and output locations must not overlap. Archive limits guard excessive extraction; raising them should follow inspection of trusted inputs. `expected_pairs` is a validation rule; do not reduce it just to hide missing files. The declared dependency ranges are compatible version constraints, not a locked environment. Record your resolved package versions when publishing results.

| Message or symptom | What to do |
| --- | --- |
| Missing module | Activate `.venv` and install the declared package/extras when permitted |
| Missing archive | Check placement and exact ZIP names; no code downloads them |
| Destination exists | Reuse the extracted tree or configure a new destination |
| Unknown label | Inspect the JSON and source documentation; do not invent an alias |
| Dimension mismatch | Compare stored image size and JSON metadata; do not resize silently |
| Out-of-bounds vertex | Preserve the source and record a curation decision |
| Manifest hash changed | Rerun `analyze` before generating the gallery |
| Mask looks black | It stores IDs 0–8; view the gallery instead of interpreting it as RGB |

## 8. What comes next

After installing dependencies, run full validation and tests, review the gallery, resolve invalid geometry and terminology with a recorded policy, and examine duplicate groups. The next engineering task should prepare reproducible, duplicate-aware and source-group-aware segmentation splits and a curated manifest. Do not train until those decisions are defensible. Severity prediction needs its own target definition and independent expert annotation before it becomes a model task.