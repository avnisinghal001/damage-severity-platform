# Task 10: PEER Φ-Net Task 7 preparation and transfer learning

This isolated package adds four-class **visible component damage-level classification**.
It does not change Tasks 01–09, their datasets, models, API, frontend or dependencies.
It does not pair Φ-Net image labels with RC1841 segmentation labels. Class probabilities
are uncalibrated; this is not whole-building severity, structural capacity or safety assessment.

## Verified archive and readiness

The actual `data/external/phi_net_task7/task7_damage_level.zip` contains six files under
`task7/`: four NumPy arrays, `README.txt` and `license.txt`. There are no original
JPEG/PNG files or source-identity records. All image tensors are C-order float32,
224 × 224 × 3. `X[i]` is paired with one-hot `y[i]` in the same official partition.
Index alignment is verifiable; semantic annotation correctness or an upstream row
permutation cannot be independently established without original image identities.

**Keep the archive's numeric registration, even though it is not ordinal:**

| ID | Original label | Official train | Prepared train | Validation | Quarantined train | Official test |
|---:|---|---:|---:|---:|---:|---:|
| 0 | Heavy damage | 919 | 724 | 181 | 14 | 94 |
| 1 | Minor damage | 869 | 692 | 173 | 4 | 93 |
| 2 | Moderate damage | 799 | 635 | 159 | 5 | 104 |
| 3 | Undamaged state | 1,551 | 1,241 | 310 | 0 | 207 |
| Total | | 4,138 | 3,292 | 823 | 23 | 498 |

All **4,636** rows passed finite-image, exact one-hot, dimensional and numerical
inverse checks. ZIP members were consumed through CRC verification. No exact raw-tensor
or recovered-pixel duplicates, or exact-image label conflicts, were found.
There is no official validation partition; the 80/20 development split uses official
training rows only, after quarantine. No test model performance was inspected.

The archive README provides no counts. The official site's 36,413 figure is for the
larger multi-task resource, not a verified Task 7 count. We therefore do not assert
a published-count discrepancy. The retained license is CC BY-NC-SA 4.0.

**Readiness verdict: technically prepared, scientifically provisional.** The tensors
contain negative channel-mean-subtracted values. Adding float32 Caffe means
`[103.939, 116.779, 123.68]` recovers byte integers for every image, with maximum residual
0.0. A BGR-to-RGB reversal is a plausible decoder, but the supplied README does not
document channel order. Numerical inversion does not prove RGB semantics. Obtain
publisher preprocessing code or a paired original image before calling it verified.
Default training refuses this unresolved encoding and incomplete source provenance.
`--allow-provisional` explicitly permits a research experiment with both limitations
recorded; it never overrides corrupt data, infeasible splits or incomplete screening.

## Leakage investigation

Screening combines exact hashes, optional documented source groups, and grayscale
perceptual/difference hashes plus thumbnail MAE for full images and 75% center/corner
crops. Grayscale screening is invariant to the unresolved channel reversal. Screening
crops are never new labeled samples and are never training augmentations.

The real scan found **119 unresolved candidate pairs**, involving 235 rows:

- 96 pairs within official training; 23 pairs across official train/test.
- 117 multi-row graph components; 20 candidate pairs have different registered labels.
- 27,816 view queries and 36,227,240 hash-tree distance checks; screening did not truncate.
- After grouping, 76 candidate pairs remain within training and 20 within validation.
  None of the detected candidate edges crosses prepared train/validation.
- All 23 training rows connected to official test components are quarantined.
  Every one of the original 498 test rows remains in the official test partition.

These are candidate similarities, not confirmed duplicates. Different labels can
reflect distinct crops, mixed damage or false matches; they do not justify relabeling.
`leakage_component` means a conservative evidence-graph component, **not** an invented
photo, building or earthquake ID. `source_group` stays blank without supplied evidence.
Arbitrary crops, rotations, repeat views or shared structures may evade this limited
screen. Statistical independence and complete absence of leakage are not established.
Conservative grouping may also over-quarantine false matches.

The grouped split uses 24 seeded greedy searches with class/size targets and complete
four-class coverage checks. A class with fewer than two eligible components blocks
preparation instead of silently falling back to image-random splitting. No rare class
is discarded or duplicated to force balance.

`review` writes editable copies of inventories outside the sealed release. Its contact
sheets show full **grayscale training images only**, with matching view names; official
test images are never rendered. Review decisions remain blank. If authentic source
evidence becomes available, supply a CSV of `sample_id,source_group,evidence` with only
documented nonempty rows, change `source_groups_csv` in a copied config, and prepare a
new version. Do not edit a sealed release or infer source IDs from adjacent array rows.

## Code and data flow

```text
config/task10.toml
src/damage_analytics/task10/
  __init__.py, __main__.py    # isolated package / module command
  config.py                  # policies, hyperparameters, immutable source class order
  archives.py                # strict ZIP schema, safe NPY headers, CRC extraction
  encoding.py                # checked inverse; unsupported values rejected, never clipped
  prepare.py                 # row audit, immutable release, provenance and manifests
  leakage.py                 # exact/visual/source evidence and conservative quarantine
  splits.py                  # seeded stratified group assignment of official train only
  release.py                 # seals, population checks, test isolation and row hashes
  reports.py                 # release integrity report
  review.py                  # pair inventory, class CSV, train-only contact sheets
  data.py                    # memory-mapped Dataset and reproducible DataLoaders
  models.py                  # reused ResNet18 initialization with a new four-class head
  learning.py                # multiclass CE train/evaluation passes and predictions
  metrics.py                 # accuracy, fixed-four-class macro-F1, per-class P/R/F1
  checkpoints.py             # atomic CPU checkpoints pinned to the dataset identity
  engine.py                  # optimizer, scheduler, selection, early stopping, validation
  plots.py                   # learning curves
  smoke.py                   # random fixture and tiny CPU end-to-end check
  cli.py                     # explicit commands; no final-test performance branch
tests/test_task10.py
docs/task10.md
```

Data flow: unchanged ZIP → safely extracted arrays → per-row validation/hashes →
evidence graph and test-related quarantine → saved train/validation manifests →
memory-mapped rows → checked candidate RGB inverse → ImageNet normalization →
four-logit ResNet18 → validation metrics and pinned checkpoint.

Existing Task 07 utilities provide full-image preprocessing, seeded loader workers,
ResNet18 initialization, confusion plots and timing. Existing training utilities provide
device selection, provenance and early stopping. Binary probability slicing and binary
metrics are deliberately not used: class 0 is Heavy, not segmentation background.

## Training protocol

The configured experiment fine-tunes ImageNet ResNet18 for at most 25 epochs with AdamW
(`lr=0.0003`, `weight_decay=0.0001`), cosine LR scheduling, batch size 32, seed 42 and
patience 6. Balanced cross-entropy weights are `N_train / (4 * class_count_train)`;
validation/test populations never supply weights. No oversampling is combined with
weighted loss. All four classes are nominal categories; numeric ID distances are not
used as severity distances.

Training preserves the full patch, using horizontal flips and mild brightness/contrast
variation from Task 07. There is no label-breaking random crop. Validation uses only
full-image resizing/letterboxing and ImageNet normalization. Workers reopen read-only
memory maps under spawn rather than copying GB arrays. Every loaded row and label is
checked against preparation hashes. CPU and optional Apple MPS are supported; no
automatic device fallback is hidden. Seeds are recorded; bitwise equality across MPS,
CPU, library versions or hardware is not promised. `deterministic=true` requests strict
PyTorch deterministic operations and can reject unsupported MPS operations.

Every epoch is validated. The best validation macro-F1 checkpoint wins; patience tracks
significant improvement using `min_delta`, without dropping a genuinely best score.
Reports include accuracy, macro-F1, balanced accuracy, class precision/recall/F1,
confusion matrix, probabilities, losses and runtime. Undefined precision/recall is JSON
null; absent-in-both class F1 is 0 in the fixed four-class macro average. Probabilities
are not calibrated. Checkpoints are model-selection snapshots, not resumable optimizer
sessions. No thresholds are tuned and no final-test evaluation command is provided.

## Exact Mac commands, in order

Run from the repository root. Existing dependencies already support this task; none
were installed or changed. In an environment that needs setup, the existing declaration
is `python -m pip install -e '.[classification,dev]'` — run that yourself only when
appropriate, outside any ongoing training environment. No new dependency is required.

```bash
source .venv/bin/activate
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR="$PWD/outputs/task10/.matplotlib"
export XDG_CACHE_HOME="$PWD/outputs/task10/.cache"

# The ZIP is already present; do not unzip or rename its contents manually.
python -m damage_analytics.task10 inspect --report-id mac_archive_001
python -m damage_analytics.task10 prepare

# Pin the release returned by preparation, instead of selecting a changing directory.
export PHI_RELEASE="$(python -c 'import json; print(json.load(open("outputs/task10/latest.json"))["release_dir"])')"
python -m damage_analytics.task10 validate --release "$PHI_RELEASE" --full
python -m damage_analytics.task10 review --release "$PHI_RELEASE" --report-id mac_review_001 --limit 32
cat "$PHI_RELEASE/INTEGRITY.md"
cat "$PHI_RELEASE/quality.json"
cat outputs/task10/reviews/mac_review_001/summary.json

python -m pytest tests/test_task10.py -q
python -m ruff check src/damage_analytics/task10 tests/test_task10.py
python -m damage_analytics.task10 smoke --run-id mac_smoke_001
```

Do the following **later**, after reviewing the documented uncertainty and when other
MPS training is finished. This is explicitly provisional research; it does not resolve
the scientific blockers. Without `--allow-provisional`, current training intentionally
stops. The real training branch requests torchvision's ImageNet ResNet18 weights; if
not already cached, torchvision may download them when **you** run this command.
No weights were downloaded during this implementation. Prefer authenticating the
publisher color preprocessing first, then creating and pinning a new verified release.

```bash
python -m damage_analytics.task10 train \
  --release "$PHI_RELEASE" --run-id phi7_resnet18_s42 \
  --device mps --allow-provisional

python -m damage_analytics.task10 evaluate \
  --release "$PHI_RELEASE" \
  --checkpoint outputs/task10/experiments/phi7_resnet18_s42/best.pt \
  --device mps
```

Use `--device cpu` if MPS is unavailable. Use fresh run/report IDs: existing experiments
and review directories are not overwritten. `prepare` is idempotent for identical
archive, settings, source evidence, preparation code and recorded environment. Changed
data policy/code/evidence produces a new release without moving original official test
rows. Changing training hyperparameters alone does not recreate the dataset split.

## Outputs and checks actually executed

The final local release is `outputs/task10/releases/phi7-aa0dd3b1e24ff2fa/`:

- `payload/task7/`: unchanged original README/license and four arrays (~2.8 GB).
- `archive_inspection.json`, `release.json`, `checksums.json`: provenance and integrity seals.
- `manifest.jsonl`, `manifests/{train,validation,official_test,quarantine}.csv`: row provenance,
  class IDs and fixed memberships.
- `quality.json`, `INTEGRITY.md`, `issues.csv`, `quarantine.csv`: actual QA and decisions.
- `near_candidates.json`, `leakage_edges.json`, `near_review.csv`: unresolved evidence.
- `source_groups_template.csv`: blank source/evidence fields, not invented identities.

`outputs/task10/reviews/initial/` contains enriched review and class-distribution CSVs,
summary JSON and four grayscale contact sheets covering the first 32 training pairs.
`outputs/task10/smoke/synthetic_cpu_001/` records the executed two-epoch tiny synthetic
training, checkpoint reload and validation check. Synthetic metrics are plumbing results,
not a Φ-Net model benchmark. `outputs/task10/verification/` records focused pytest results,
full verification and preservation checks. See `outputs/task10/IMPLEMENTATION.md` for the
final executed-check summary. An earlier development preparation snapshot is retained;
the final release has complete namespaced code fingerprints and identical split rows.

Future real runs write `outputs/task10/experiments/<run-id>/`: `run.json`, `history.*`,
`best.pt`, `last.pt`, `status.json`, `validation_metrics.json`, `validation_predictions.csv`,
`per_class.csv`, `confusion_matrix.png`, `learning_curves.png`.
Explicit validation writes `outputs/task10/evaluations/<checkpoint-sha-prefix>/validation/`.
Raw data and all generated outputs remain ignored by Git. No real training or final-test
model evaluation was executed.

The next step is publisher preprocessing verification and evidence-backed review of
source/crop relationships, followed by one pinned provisional or verified ResNet18 run.
Any later test evaluation needs a separate explicit protocol, performed once after
model selection; uncertainty about physical source overlap must remain in its report.

## Primary references

- [PEER task definitions](https://apps.peer.berkeley.edu/phi-net/detection-tasks/): Task 7 is
  qualitative pixel/object damage level, with four categories.
- [PEER downloads](https://apps.peer.berkeley.edu/phi-net/download/): official Task 7 resource.
- [PEER overview](https://apps.peer.berkeley.edu/phi-net/): the multi-task dataset context.
- [Keras VGG preprocessing](https://keras.io/api/applications/vgg/#preprocess_input-function):
  the generic Caffe convention, **not proof of the Φ-Net archive's channel semantics**.
