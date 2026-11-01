# Task 07: external datasets and crack classification

Task 07 adds a separate research pipeline. It does not change the existing nine-class
U-Net, API, database, dashboard, frozen RC splits or serving decisions. A crack-negative
prediction must never gate out corrosion-only or spalling-only inputs.

## Evidence and limits

Read `outputs/task07/audit_report.md` first. Its evidence is in `archive_audit.json`
and `rc1841_audit.json`. Archive inventories and representative decoding were run;
full real-data extraction, QA, cross-dataset fingerprint screening and classifier
training were **not** run during implementation. Prepared valid/excluded counts,
split sizes and real classifier metrics are therefore pending, not zero.

* SDNET2018 contains 56,092 patch JPEGs: 8,484 C-folder positives and 47,608
  U-folder negatives. Material/prefix names suggest 230 source buckets but do not
  authenticate original photographs. Negative means no crack under source labels.
* DamSegment classification contains 2,000 images (1,000 per folder class). There
  are 162 repeated stems across JPEG/PNG: extensions are part of sample identity.
* DamSegment detection and segmentation each contain 1,500 images. All 1,500
  CRC/size tuples overlap; three sampled pairs were also SHA-256 identical.
  These are overlapping annotation views, not 3,000 independent photographs.
* The nested `Damage Segmentaion.zip` spelling is preserved. JSON in directories
  named `Pascal VOC` is COCO-like; detection YOLO contains boxes, segmentation
  YOLO contains polygons. Easy/Medium/Hard are source folder tags, not severity.
* The archives provide numeric annotation categories 0/1 without semantic names.
  Native RGB masks use black/red/blue. Local mask IDs are background=0,
  red/source-0=1, blue/source-1=2, checked against polygon evidence. Do not call
  these crack/spalling until an authoritative category map is supplied.
* All 1,617 curated RC images have foreground annotations. Background is polygon
  complement, not a verified intact-concrete or unrelated-image class. Existing
  validation confusion matrices demonstrate false damage pixels; they do not
  measure unrelated-photo rejection or whole-image specificity.

No additional severity supervision, concrete/non-concrete classifier or OOD mechanism
is supplied by this task. Classification scores are uncalibrated. Resizing may weaken
thin cracks; the configurable full-patch letterbox avoids label-invalidating random
crops but cannot guarantee crack preservation. Inspect errors before changing resolution.

## Modules and data flow

All new Python modules live in `src/damage_analytics/task07/`:

| Modules | Responsibility |
|---|---|
| `config`, `cli`, `__main__` | Typed central settings and explicit commands |
| `archives`, `adapters` | Reuse ZIP security; discover the actual nested layouts and stable extension-aware IDs |
| `annotations`, `validation` | Decode images, EXIF/mode/size/hash checks, native JSON/YOLO geometry and mask validation |
| `duplicates`, `splits` | Exact components, bounded perceptual screening, evidence reviews and deterministic group balancing |
| `prepare`, `release` | Staged immutable releases, provenance, exclusions, seals, saved splits and checked reads |
| `audit`, `reports`, `crosscheck` | Local archive/RC audit and cross-dataset fingerprint reports |
| `data`, `models`, `loops`, `engine` | ImageNet normalization, loaders, two torchvision backbones, AdamW and validation selection |
| `metrics`, `checkpoints`, `evaluation`, `plots` | Metrics, restricted checkpoint loading, frozen-threshold evaluation and comparisons |
| `segmentation`, `visualization` | Native three-class mask evaluation and detection annotation galleries |
| `smoke` | Temporary synthetic data, tiny CPU model and two-epoch plumbing check |

```text
ZIPs → protected staged extraction → sample QA/exclusions → exact/source components
     → perceptual review candidates → saved grouped splits → sealed dataset release
     → explicit train → best validation ROC-AUC checkpoint → validation threshold
     → frozen-threshold validation / compatible external validation
     → final-test only after source and screening eligibility checks
```

Raw files are never repaired. Annotated images with unsupported orientation, invalid
geometry, unknown colors or inconsistent targets are excluded with reasons. Closing or
consecutive duplicate polygon vertices are normalized only for validation/raster
comparison; original JSON stays unchanged. Native PNGs are authoritative: boundary
disagreement is measured; category-presence mismatch or class polygon IoU below the
configured minimum causes exclusion. That minimum is a consistency-screen policy,
not a measured model score or proof of annotation truth.

Filename buckets are deliberately marked provisional. Exact byte/decoded duplicates,
reviewed related samples and candidate buckets form connected components shared across
all tasks in a dataset. No component crosses splits. A seeded group-level heuristic
balances task/class/material/difficulty strata where feasible, recording actual coverage
and ratios. Components are never broken merely to reach 70/15/15. No rare-class deletion
or automatic near-duplicate identity assumption occurs.

## Manual commands, in order

Run from the repository root using the existing virtual environment. The installation
and all real-data preparation/training commands below were **not executed here**.
Only an explicit real `train` invocation requests ImageNet weights; evaluation loads
the saved state with `weights=None`. If needed, create `.venv` with
`python3 -m venv .venv` before activation.

```bash
source .venv/bin/activate
python -m pip install -e '.[classification,dev]'

# Existing archives must remain at these exact locations:
ls -lh data/external/sdnet2018/SDNET2018.zip data/external/damsegment/DamSegment.zip
python -m damage_analytics.task07 audit
cat outputs/task07/audit_report.md

# Explicit heavy steps: hashing, extraction, decoding, geometry, duplicate screening.
# Allow several GB of disk space plus temporary nested ZIPs; no runtime estimate claimed.
python -m damage_analytics.task07 prepare --dataset sdnet2018
SDNET_RELEASE=$(python -c 'import json; print(json.load(open("outputs/task07/datasets/sdnet2018/latest.json"))["release_dir"])')
python -m damage_analytics.task07 validate --release "$SDNET_RELEASE" --full
cat "$SDNET_RELEASE/audit.md"

python -m damage_analytics.task07 prepare --dataset damsegment
DAM_RELEASE=$(python -c 'import json; print(json.load(open("outputs/task07/datasets/damsegment/latest.json"))["release_dir"])')
python -m damage_analytics.task07 validate --release "$DAM_RELEASE" --full
cat "$DAM_RELEASE/audit.md"

python -m damage_analytics.task07 cross-check \
  --left "$SDNET_RELEASE" --right "$DAM_RELEASE" --report-id external_overlap
python -m damage_analytics.task07 detection-gallery \
  --release "$DAM_RELEASE" --report-id dam_boxes --count 6
python -m damage_analytics.task07 smoke --run-id local_smoke
python -m pytest -q tests/test_task07_data.py tests/test_task07_training.py
python -m ruff check src/damage_analytics/task07 tests/test_task07_data.py tests/test_task07_training.py
```

`validate --full` is dataset QA: it reads all **new** dataset files, including prospective
test pixels, to verify hashes and annotation integrity. Routine training/evaluation
validates metadata seals and only loads the requested train/validation pixels. It
never evaluates test data implicitly. The previously used RC test is not accessed.

### First experiment and matched baseline

Start with a five-epoch MobileNetV3-Small SDNET pilot. SDNET offers the larger, explicit
binary label corpus; MobileNet reduces initial compute. Conservative filename grouping
is preferable to a random patch split but remains provisional. The flag below acknowledges
that limitation and **does not** enable final testing. Both backbones use identical splits,
seed, budget, image size, augmentation and evaluation rules. Batch size defaults to 32,
workers to 0; tune execution separately before a matched comparison if memory demands it.

```bash
python -m damage_analytics.task07 train --release "$SDNET_RELEASE" \
  --backbone mobilenet_v3_small --run-id sdnet_mobile_pilot --epochs 5 --device mps --allow-provisional
python -m damage_analytics.task07 train --release "$SDNET_RELEASE" \
  --backbone resnet18 --run-id sdnet_resnet_pilot --epochs 5 --device mps --allow-provisional
python -m damage_analytics.task07 evaluate --release "$SDNET_RELEASE" \
  --checkpoint outputs/task07/experiments/sdnet_mobile_pilot/best.pt --device mps
python -m damage_analytics.task07 evaluate --release "$SDNET_RELEASE" \
  --checkpoint outputs/task07/experiments/sdnet_resnet_pilot/best.pt --device mps
python -m damage_analytics.task07 compare \
  --run-id sdnet_mobile_pilot --run-id sdnet_resnet_pilot --report-id sdnet_pilots

# Independent DamSegment classification pilots; no mixing source taxonomies.
python -m damage_analytics.task07 train --release "$DAM_RELEASE" \
  --backbone mobilenet_v3_small --run-id dam_mobile_pilot --epochs 5 --device mps --allow-provisional
python -m damage_analytics.task07 train --release "$DAM_RELEASE" \
  --backbone resnet18 --run-id dam_resnet_pilot --epochs 5 --device mps --allow-provisional
python -m damage_analytics.task07 evaluate --release "$DAM_RELEASE" \
  --checkpoint outputs/task07/experiments/dam_mobile_pilot/best.pt --device mps
python -m damage_analytics.task07 evaluate --release "$DAM_RELEASE" \
  --checkpoint outputs/task07/experiments/dam_resnet_pilot/best.pt --device mps
python -m damage_analytics.task07 compare \
  --run-id dam_mobile_pilot --run-id dam_resnet_pilot --report-id dam_pilots
```

For subsequent matched 30-epoch runs, use new run IDs and omit `--epochs 5` (or set the
same budget explicitly). `config/task07.toml` owns hyperparameters. Selection uses
validation ROC-AUC with early stopping; balanced CE weights use training counts only.
The threshold maximizes validation balanced accuracy by default, once on the selected
checkpoint. F1 tuning is configurable. Threshold-selected validation scores are optimistic
development metrics, not independent performance estimates. PR-AUC is trapezoidal;
average precision is also reported separately, with tied scores handled together.

MPS is optional; replace `--device mps` with `--device cpu` if unavailable or an operation
fails. There is no silent CPU fallback for unsupported MPS operations. Strict determinism
is configurable and can limit MPS support; seeded runs are not a promise of identical
CPU/MPS arithmetic. Full backbone and MPS execution require local verification.

### Cross-domain, segmentation and final evaluation

Review crack definitions, label completeness, source identity and the overlap report
before asserting compatibility. This command requires explicit confirmation, checks
target samples against the source training **and validation** fingerprints, and rejects
exact/near candidates or truncated screening. It never retunes the source threshold:

```bash
python -m damage_analytics.task07 evaluate --release "$DAM_RELEASE" \
  --source-release "$SDNET_RELEASE" \
  --checkpoint outputs/task07/experiments/sdnet_mobile_pilot/best.pt \
  --confirm-compatible-labels --device mps
```

External validation is exploratory domain-shift evidence. A clean hash screen does not
authenticate physical independence or exclude all crops. Do not repeatedly tune on an
external cohort later called an untouched test. Unresolved overlaps require evidence and
a quarantined/new release, followed by retraining where split populations changed.
Optional `cross-check --rc-manifest PATH` compares existing SHA metadata only; it does
not load RC images or authorize reuse of its held-out test.

Native DamSegment segmentation evaluation accepts externally generated predictions,
one `sample_id.png` per segmentation validation row. Use single-channel IDs 0/1/2 or
explicit `--rgb` for the source palette. No segmentation model is trained in Task 07;
the nine-class RC checkpoint cannot supply these predictions without a justified new
native-taxonomy experiment. The folder must contain exactly the requested population.

```bash
# Run only after generating genuine native-taxonomy prediction masks.
python -m damage_analytics.task07 evaluate-segmentation --release "$DAM_RELEASE" \
  --predictions outputs/task07/native_predictions --report-id dam_native_validation
```

Before final testing, copy a generated `source_groups_template.csv` outside the sealed
release and fill the copy with authenticated original-source identity and evidence for
every included sample. Point `source_groups_csv`
in the config to that file. Supply `near_reviews_csv` with columns `left,right,decision,evidence`
for reviewed candidate pairs; decisions are `same_source` or `different_source`. Each pair
must be unique. Use separate evidence files/config paths for each dataset. Unverified
prefixes must not simply be relabeled as authenticated groups. Resolve orphan labels and
cross-split candidates; an exhausted screen needs a higher budget and a new release.

Re-run preparation and full validation, refresh the release variables and **retrain on
the resulting verified release**. Do not edit sealed manifests/checkpoints in place.
Real documentary category evidence is separately required to give DamSegment semantic
names (`damsegment.semantic_names_verified`, `semantic_evidence`, `category_names`).
With a completed best checkpoint trained on a final-eligible release, final evaluation
is explicit; replace the example run ID with that verified run:

```bash
python -m damage_analytics.task07 final-test --release "$SDNET_RELEASE" \
  --checkpoint outputs/task07/experiments/sdnet_mobile_verified/best.pt --device mps
```

Current provisional releases will refuse this. Comparison commands reject test results.
Repeated evaluation of the same checkpoint/cohort refuses to overwrite an existing report;
test access is explicit, not a globally enforceable one-shot authorization mechanism.
Keep the final-test cohort isolated by protocol and report grouping uncertainty even
after the eligibility checks pass.

## Artifacts and verification

```text
outputs/task07/
  archive_audit.json, rc1841_audit.json, audit_report.md, implementation_report.json
  datasets/{dataset}/latest.json
  releases/{dataset}-{input_code_environment_hash}/
    release.json, quality.json, audit.md, manifest.jsonl, checksums.json
    exclusions.csv, grouping.json, near_candidates.json, source_groups_template.csv
    splits/{train,validation,test}.csv
    payload/...                       # original nested dataset files
    masks/{sample_id}.png              # explicit derived native ID masks
  experiments/{run_id}/
    run.json, history.json, history.csv, learning_curves.png
    best.pt, last.pt, threshold.json, validation_predictions.csv
    validation_metrics.json, status.json
  evaluations/{checkpoint_hash}/{release_id}/{validation|test}/
    metrics.json, predictions.csv, confusion_matrix.png
  comparisons/{report_id}.json
  cross_checks/{report_id}.json
  detection_galleries/{report_id}/...
  segmentation_evaluations/{report_id}/...
  smoke/{run_id}/...                   # tiny synthetic artifacts; not model evidence
```

Release identity pins full archive SHA-256, preparation/reused utility code, numerical
library versions, configuration, evidence-file hashes and seed. `checksums.json` seals
files; experiments additionally pin its hash and all split hashes. This detects accidental
mutation; it is not a cryptographic signature from an external trusted authority.
Identical preparation inputs reuse the release, while changes create a new version.
Run/report IDs never overwrite previous experiments. Temporary smoke source data is
deleted after evaluation; smoke checkpoints are fixtures, not reusable research releases.

Focused tests cover ZIP traversal, extension collisions, masks/geometry, split repeatability,
source components, exact/perceptual candidates, evidence/version changes, integrity
mutation, metrics/ties, validation thresholding, checkpoint reload, transforms and the
two-epoch CPU smoke loop. See `outputs/task07/implementation_report.json` for actual
pass/skip counts. Real dataset quality counts, actual backbone training, weight downloads,
MPS behavior and cross-domain generalization remain unverified until the manual steps run.
