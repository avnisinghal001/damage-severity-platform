# Task 02: clean, curate, group, and split the dataset

## 1. The outcome and its limits

Task 01 described the supplied data. Task 02 makes a versioned segmentation dataset
with explicit inclusion rules, lossless class-ID masks, reproducible train/validation/test
tables, and evidence showing what was checked. It does not make severity targets,
train a model, augment photographs, or prove that every human annotation is correct.

The release verdict separates two questions:

1. Are the exported files, labels, masks, counts, groups, and partitions internally
   consistent and traceable to the original bytes? This can be tested.
2. Are the categories semantically settled, all physical sources independent, and all
   polygons expert-correct? Available evidence cannot establish all of this.

`CONDITIONAL_RESEARCH_USE` answers the first question positively and lists the remaining
limitations. It does not mean the data is globally leakage-free or suitable for safety
decisions. Read [dataset.md](../dataset.md#task-02-curation-policy-and-evidence) before modeling.

## 2. The important concepts

| Concept | Meaning in this project |
| --- | --- |
| Cleaning | Applying a specific, recorded policy to a source representation |
| Curation | Deciding which complete samples are defensible to include |
| Ground truth | Human annotations we must preserve and interpret carefully |
| Simple polygon ring | A boundary that does not cross, retrace, or touch itself away from adjacent endpoints |
| Degenerate polygon | A shape with insufficient distinct points or effectively zero area |
| EXIF orientation | A camera tag telling a viewer how stored image pixels should be displayed |
| Coordinate frame | The image width, height, and orientation in which polygon points were drawn |
| Exact duplicate | Matching file bytes or matching decoded pixels under a stated signature |
| Near-duplicate candidate | Similar-looking data found by heuristic fingerprints, not proven identity |
| Leakage | Related examples appearing across splits, making evaluation misleading |
| Connected component | Every sample reachable through chains of known relationship evidence |
| Multi-label sample | One image containing more than one damage category |
| Stratification | Trying to distribute category presence sensibly across splits |
| Seed | An explicit random starting state that makes the same search repeatable |
| Idempotence | Repeating the same run preserves the same version and artifact bytes |
| Immutable release | A named dataset snapshot the pipeline verifies and never overwrites |

A mask value of 1 means `Microcrack`, not severity level one. Background is the
complement of annotated regions, not expert-confirmed undamaged concrete. The exact
eight source strings and IDs 0–8 are reused from Task 01's `labels.py`.

## 3. Read the previous evidence first

The completed Task 01 report has 124 issue rows: 31 invalid-pair errors, 29 EXIF
warnings, 36 polygon-overlap warnings, and 28 duplicate warnings. The latter describe
14 image pairs twice: once by bytes and once by decoded pixels, not 28 distinct pairs.

The 31 errors have two causes. Twenty-nine JPEGs store a 4032×3024 array but declare
3024×4032 annotation dimensions and carry EXIF orientation 6. Two annotations have
negative points (`crack146` and `damage794`). Task 01 also has one CMYK photograph.

Task 02 fingerprints the previous reports and embeds their contents and every issue's
affected IDs, root-cause explanation, and final disposition in `task01_audit.json`.
It then audits original data again rather than assuming the old checks were sufficient.
For example, Task 01 did not reject every non-simple polygon or compare competing
ground truth on identical photographs.

## 4. Preserve and verify source data

`provenance.py` compares archive hashes to extraction provenance, reads every archive
member through CRC verification, and compares its bytes to the extracted file. Missing,
changed, unexpected, or linked source files abort release publication. Repairing the
extracted copy silently would hide which original data was processed.

`source_inventory.csv` records all member paths and hashes. Original ZIPs, images,
JSON files, and Task 01 reports are never rewritten by Task 02. Output paths are
separate, validated directories. The same source ID remains available in every
manifest, decision, group, exclusion, and split table.

## 5. Decode images and choose a justified frame

`images.py` verifies and fully decodes each file, compares its content format with its
extension, checks color mode, checks single-frame compatibility, and reads EXIF metadata.
Transparency without a justified compositing background is excluded. Unsupported modes
are not silently scaled to eight bits.

The default EXIF rule allows a transform only when declared JSON dimensions uniquely
match the transformed size and differ from the stored size. A same-size flip or rotation
is ambiguous and excluded. With dimension evidence, rotate the image derivative, keep
the original polygon coordinates, and compare the exported pixels during verification.
Do not rotate both the image and points, which would apply the transformation twice.

Accepted derivatives are RGB PNGs at original annotation resolution. Converting CMYK,
grayscale, or other supported modes is explicit and logged. RGB conversion is a common
input representation, not model normalization or calibrated color reproduction. EXIF/ICC
metadata is not copied to derivatives, preventing a downstream viewer from rotating them again.

## 6. Audit every polygon without silently removing ground truth

`cleaning.py` checks document schema, filename references, dimensions, exact category
strings, shape type, flags, every usable coordinate, geometry, and pixel-frame geometry.
It continues to audit later shapes after finding an invalid one. Any invalid shape
excludes the whole image–annotation pair; keeping the image with the remaining polygons
would falsely teach a model that the removed damage is background.

`geometry.py` uses segment orientation and bounding-box tests to find crossings,
nonadjacent touches, collinear edge overlap, and adjacent retracing. A shoelace-area
check catches zero-area shapes. The checks are repeated after mapping boundary-edge
coordinates to the actual pixel grid. Clockwise and counterclockwise simple rings
are both allowed. This is an explicit simple-ring validator, not a repair engine.

The default policies in `config/task02.toml` distinguish:

- **Representation-only fixes:** remove exact consecutive duplicates or a repeated
  closing vertex. The polygon boundary and every raster pixel must remain identical.
- **Acceptable edge cases:** width/height edge coordinates follow Task 01's documented
  pixel clipping convention; same-label overlap is a semantic union.
- **Unresolved or invalid ground truth:** nonfinite points, negative coordinates,
  self-contact/crossing, degenerate rings, unsupported semantics, and conflicting labels
  exclude the sample rather than triggering a guessed repair.

Out-of-bounds tolerance defaults to zero. Opting into a positive tolerance records each
changed point's before/after coordinates and creates a different version. Never turn on
clamping or topology shortcuts merely to get a larger sample count or better model score.

`decisions.csv` records source ID, zero-based shape index, issue code, action, and detail.
Actions are `fix`, `accept`, or `exclude`. A candidate may receive a justified fix and
still be excluded for another problem; the report gives retained-sample counts so these
cases are not confused with exported corrections. Reasons can overlap.

## 7. Validate masks at pixel level

`masks.py` reuses Task 01's rasterizer, then independently constructs per-class coverage
bit planes. It checks that background equals the uncovered complement and that singly
labeled regions receive exactly their expected IDs. It distinguishes multiple polygons
of one class from multiple classes at the same pixel. Shared inclusive raster boundaries
also count as cross-class conflicts under the conservative default.

Masks are lossless grayscale PNGs with IDs 0–8, not RGB colors. They must have the same
width and height as their associated RGB image. Pixel counts must sum to image area.
The default excludes empty masks and cross-class overlap. `last_wins` is an explicit
alternative policy, not the default scientific interpretation.

`verification.py` reopens every exported mask, compares every pixel with rasterized
cleaned JSON, verifies exact class counts, and compares raw/cleaned raster pixels for
boundary-preserving edits. It also recreates the documented image transformation and
compares decoded RGB pixels. These are processing-consistency checks using Pillow's
documented drawing convention, not independent expert confirmation of annotations.

## 8. Detect duplicates and related-source candidates

`similarity.py` compares original image/JSON hashes, stored and selected-frame pixel
hashes, and canonical ordered polygon geometry. Ring starting point and winding do
not change canonical identity. Geometry retains exact labels and image dimensions;
empty annotations provide no grouping evidence.

If identical photographs have different or unusable annotations, `samples.py` excludes
every member. It does not choose the largest mask, merge labels, or vote. If image and
mask are identical, an optional policy retains the smallest source ID and records all
redundant IDs. This is duplicate handling, not rare-class rebalancing.

Near-image detection combines pHash, dHash, thumbnail RGB difference, and aspect ratio.
Thresholds are centralized and saved. Near-annotation IoU is evaluated within visual
candidates on small nearest-neighbor mask grids. Thumbnail resizing is for comparisons
only; exported images and masks remain full resolution. Similarity is heuristic and
can miss crops, rotations, lighting changes, or distinct views of the same structure.

`metadata.py` also records real camera/time metadata. It links nearby timestamps for
the same make/model/available serial using a configurable 600-second gap. A camera
model is not a unique device ID; `DateTime` can describe editing rather than capture.
These are conservative session candidates, not invented structure identifiers. Missing
or invalid metadata creates no session relationship. The supplied data contains a
related temporal sequence among the 29 EXIF-rotated photographs.

If verified structure IDs become available, provide an optional source-evidence CSV:

```csv
sample_id,source_group,evidence
sample_a,inspection_structure_1,inspection log reference
sample_b,inspection_structure_1,inspection log reference
```

Configure its path with `source_groups_file`. This illustration is not a file created
in the project. Every row needs a known unique sample ID, group, and nonempty evidence.
Filename prefixes are never assumed to identify a structure.

All evidence becomes a connected graph before final exclusions are removed. If A is
related to excluded B and B is related to C, retained A and C must remain together.
`groups.csv` includes excluded IDs, preserving this chain for independent verification.

## 9. Split whole groups and report what balancing can achieve

`splitting.py` searches deterministic assignments of complete components with seed 42
and fractions 70/15/15. Greedy assignment and whole-group coordinate descent balance
sample counts and multi-label image presence. Twenty-four seeded restarts improve
the assignment without consulting a trained model. Class coverage is prioritized.

The search cannot divide a group to achieve an exact percentage. A class in fewer
than three groups cannot appear in all three splits. Having three groups is necessary
but not sufficient for every simultaneous multi-label constraint. The report distinguishes
known infeasibility from a heuristic that did not find full coverage. Rare classes are
retained; no oversampling, automatic class removal, or pixel rebalance is applied.

`class_statistics.csv` reports image presence, source polygon presence/counts, final
pixels, and pooled pixel fractions for the full curated set and each split.
`statistics.json` adds multi-label combinations, pairwise occurrences, dimension
frequencies, rare-class flags, and foreground pixel imbalance. Large photographs
contribute more pixels; this is not a physical damage-area measurement.

## 10. Freeze and verify a reproducible release

`pipeline.py` stages output, verifies all saved artifacts, rechecks inputs, and publishes
`outputs/task02/versions/<dataset-name>-<hash>/`. `release.json` records configuration,
seed, original archive provenance, all-source inventory hash, Task 01 evidence hashes,
optional source-evidence hash, code hashes, and Python/package versions.

`checksums.json` seals every other release file. Rerunning identical inputs and code
verifies and reuses the same release; it does not rewrite the test CSV. A changed
scientific or implementation snapshot creates a new version. `latest.json` is only a
convenience pointer. Pin the final version and test CSV hash before training, and never
search policy/seed versions to improve test performance. Initial class-aware dataset
design is allowed; future model selection belongs to train/validation data.

Two supporting modules keep this workflow simple: `curation/config.py` loads TOML into
frozen typed settings and rejects unsafe paths, unsupported policies, invalid thresholds,
and invalid fractions. `reporting.py` writes stable CSV schemas, class/co-occurrence
statistics, exclusions, decisions, and the historical Task 01 audit. `curation/__init__.py`
identifies the subpackage; the existing `cli.py` routes commands to it while preserving
Task 01 commands. The package uses functions and small data objects rather than a service framework.

`verify-curation` rechecks archives, extracted sources, sealed output hashes, every
saved sample, all evidence components including excluded bridges, and disjoint exhaustive
split membership. A verified partition establishes no **known** evidence leakage; missing
physical source IDs and heuristic false negatives remain limitations.

## 11. Complete command sequence

Run from the repository root. Installation is provided for you to execute only when
permitted; no dependencies or datasets were installed/downloaded during implementation.

```bash
if [ ! -d .venv ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate
python -m pip install -e '.[dev]'
mkdir -p data/raw/rc1841
```

The two ZIPs are already present here. For a fresh workspace, copy existing local files:

```bash
printf 'Folder containing imagedata.zip and labeldata.zip: '
read -r RC1841_ARCHIVE_DIR
cp "$RC1841_ARCHIVE_DIR/imagedata.zip" data/raw/rc1841/
cp "$RC1841_ARCHIVE_DIR/labeldata.zip" data/raw/rc1841/
```

Then inspect and reuse or create the prerequisites:

```bash
python -m damage_analytics inspect
if [ ! -d data/interim/rc1841 ]; then
    python -m damage_analytics ingest
fi
if [ ! -f outputs/task01/validation.json ]; then
    python -m damage_analytics analyze
fi
```

Task 01's known validation errors give exit code 2. Review those reports and continue
to Task 02; do not use `set -e` or chain this known failure to curation with `&&`.

```bash
python -m damage_analytics --config config/task02.toml curate
python -m damage_analytics --config config/task02.toml verify-curation
python -m damage_analytics curation-report
python -m json.tool outputs/task02/latest.json
python -m json.tool outputs/task02/quality_report.json
python -m pytest -q
python -m ruff check src tests
```

The [README](../../README.md#task-02-curation-and-splitting) contains the complete
output tree and exact Python commands for opening individual CSVs through `latest.json`.
Start with the quality verdict, exclusions, decisions, historical audit, split coverage,
and grouping uncertainty. Only after that should a future model task consume the
version-pinned manifest and split CSVs.

## 12. Tests and next work

Task 02 tests use tiny synthetic images, annotations, and ZIPs. They cover malformed and
nonfinite coordinates, crossings/touches/retracing, boundary conventions, empty masks,
exact label IDs, hand-computed mask pixels, orientation ambiguity, RGB conversion,
duplicates/conflicts, temporal evidence, source CSV validation, transitive leakage,
split reproducibility/coverage, unsafe settings, complete release counts, preserved
source bytes, frozen test CSV idempotence, and source/output tampering. Existing Task 01
tests remain part of the suite.

The next scientific work is expert review of excluded rings/conflicts and EXIF alignment,
resolution of category terminology, and acquisition of reliable physical source IDs.
After that, a segmentation loader/baseline can use the pinned release. Severity remains
a separate target-design and expert-annotation task.
