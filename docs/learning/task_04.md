# Task 04: improve segmentation carefully and describe damage regions honestly

**Performance continuation:** the user subsequently completed the 50-epoch CPU
`control_s42` run (best validation mIoU 0.2431215908 at epoch 48). The execution counts
below describe the original Task 04 implementation. See the new
[performance lesson](task_04_performance.md) for actual profiles, optimization changes,
explicit MPS reproducibility trade-offs, and commands for the remaining experiments.

## 1. What is implemented and what was actually run

This task continues the trained Task 03 baseline. It implements a validation audit,
three controlled model-run designs, comparison tools, native-resolution region
analytics, uncertainty summaries and a source-label supervision investigation.

Executed: fresh inference for all 243 validation images, numerical baseline checks,
targeted gallery review, train/validation resampling comparisons, a source-region
supervision count, one real validation-image structured inference, synthetic integration
checks and the existing regression suite. **No real-data improvement model or separate
severity classifier was trained.** No dependencies or pretrained weights were installed.

The Task 03 test was already executed before this task. It is not an untouched holdout
for further development. Task 04 has no final-test command and does not consume final-test
scores or predictions for selection. Its metrics are validation development evidence.
Future confirmatory generalization claims need genuinely new independent evaluation data.

## 2. The actual baseline audit

`baseline.py` pins `unet_001/best.pt` by its complete SHA-256, inspects history/status,
authenticates the saved validation report and checks its confusion matrix. `validation.py`
then loads that model, performs fresh validation inference and reconstructs counts
using `numpy.add.at`, separately from Task 03's `bincount` implementation. An explicit
scalar calculation cross-checks foreground mIoU. Native target support must match
the frozen validation manifest, and no test DataLoader is opened.

The baseline trained for 50 epochs; best validation foreground mIoU is at epoch 48:

```text
Saved foreground mIoU:      0.23990623390029908
Fresh CPU foreground mIoU:  0.23990623390029908
Difference:                0.0
Confusion count difference: 0
```

Original training/evaluation used MPS with deterministic mode off. The fresh CPU audit
exactly reproduces aggregate metrics. This is not a new model improvement. Matching
confusion counts also does not prove pointwise equality to historical predictions
that were never all saved as native ID maps. Count-matrix half-L1 difference is only
a lower bound on changed prediction assignments, not an aligned pixel comparison.

### Diagnose errors rather than reading pixel accuracy alone

The baseline has foreground mIoU about 23.99% and pixel accuracy about 85.43%.
Background performance can hide weak damage categories. Key validation findings:

| Category | IoU | Dice | Precision | Recall |
| --- | ---: | ---: | ---: | ---: |
| Microcrack | 0.1691 | 0.2893 | 0.2014 | 0.5133 |
| Structural crack | 0.4075 | 0.5791 | 0.4489 | 0.8157 |
| Minor spalling | 0.1052 | 0.1903 | 0.1079 | 0.8048 |
| Moderate spalling | 0.2328 | 0.3777 | 0.4808 | 0.3110 |
| Delamination | 0.1525 | 0.2646 | 0.5341 | 0.1759 |
| Rebar corrosion | 0.5786 | 0.7330 | 0.6660 | 0.8150 |
| Concrete crushing | 0.0093 | 0.0185 | 0.6910 | 0.0094 |
| Structural deformation | 0.2642 | 0.4180 | 0.2915 | 0.7386 |

Minor-spalling predictions occupy 228,013 validation pixels, compared with 30,578
annotated pixels: about 7.46 times as much area. High recall plus low precision means
many annotated pixels are found, but many other pixels are incorrectly assigned that
category. A high recall alone would give a misleading impression.

Crushing has only 6,269 true-positive pixels out of 670,130 annotated pixels. It is
mostly confused with corrosion, deformation and moderate spalling. High precision
here concerns very few predicted pixels and does not mean the model detects crushing
reliably. Crushing and deformation each have only seven validation images.

The confusion matrix also sends 806,486 moderate-spalling pixels and 1,428,936
delamination pixels to corrosion. This is evidence of errors relative to the annotations,
not proof that one architectural factor caused them. Annotation limitations remain.

`review.py` selects explicit worst cases: `damage490` for minor-spalling false positives,
`damage801` for crushing misses, `damage987` for moderate-spalling misses, and
`Image_119_3665189` for deformation false positives. These deliberately biased galleries
help diagnose errors; they are not representative performance samples.

## 3. Controlled improvement experiments

A useful comparison changes one factor while keeping frozen samples and evaluation
rules fixed. `experiments.py` reconstructs the baseline's saved model/data/optimizer
settings rather than trusting a later-edited Task 03 config. `plan` only writes designs.
Only the explicit `train` subcommand launches real training.

### Control

The matched control keeps the saved architecture, seed, losses, crops, optimizer,
batch size, epoch cap and selection rule. It retrains from scratch on the current
backend. The historical MPS baseline and a current CPU model can diverge during training
even if their initial seed matches, so this replay provides the primary comparison.

Task 04 defaults to strict deterministic CPU because the earlier Task 03 MPS sanity
run recorded an unsupported deterministic `index_put_with_accumulate_mps` operation.
MPS is unavailable in this execution environment. For a deliberate GPU run, change
Task 04 TOML to `device="mps"` and `deterministic=false` before training every model
in the comparison. Record that choice and retrain the control with the same policy.
There is no silent fallback or promise of bitwise GPU reproducibility.

### Capped CE weights

The baseline uses inverse-square-root pixel-frequency weights. Its background weight
is about 0.0963, Microcrack 2.2392 and Minor spalling 3.3290. Crushing is about 0.7446:
it has few images but more pixels than minor spalling. Image frequency and pixel
frequency are different concepts.

`capped_weights` changes only `max_class_weight` from 10 to 1. The weighting algorithm,
CE + Dice combination and all other settings stay fixed. This softens the per-pixel
penalty imbalance that may encourage minority-class false positives. It can also lower
recall. Whether it improves mIoU is unknown until the manual experiment is executed.
There is no claim that weighting caused every observed error.

### Resolution 384

`resolution_384` changes only the input grid from 256×256 to 384×384. It retains native
source images, 512-pixel source crop settings and the same labels. Nominal network-grid
pixels increase by `(384/256)^2 = 2.25`; actual runtime/memory depends on the backend.
No unreported batch-size, architecture or learning-rate changes are made.

The executed resampling audit shows validation Microcrack ground-truth resize/restore
recall increasing from **95.0950% to 97.5469%**. Structural crack changes from 98.4477%
to 99.1786%, and minor spalling from 96.3013% to 97.4426%. These are preprocessing
fidelity measurements, not learned accuracy or an upper bound on model performance.
They justify testing resolution without proving the proposed model improves.

### Fairness checks and comparison statistics

`comparison.py` evaluates completed registered runs' best checkpoints. It checks saved
settings, exact release/split identity, architecture, native support and protocol.
Matched-control comparisons require the same training backend, torch/Pillow/NumPy/Python
versions, deterministic mode, CPU thread count and relevant training-code hashes.
Numeric losses from differently weighted objectives should not be ranked directly.

The main outcomes are foreground mIoU and the full per-class IoU/Dice/precision/recall
table, supported by confusion matrices and qualitative predictions. A higher average
can hide a crushing/deformation regression, so inspect all classes and actual image/group
support. The tool reports deltas; it does not automatically declare a reliable winner.

An optional paired bootstrap samples known validation components with replacement and
recomputes the metric difference. Pairing means both models use exactly the same sampled
groups. Its percentile interval describes variation conditional on those selected models
and known groups. It is not calibrated prediction confidence, a training-seed study,
an adaptive-selection correction or a multiple-comparison significance test. Unknown
physical-source relationships and repeated validation tuning limit interpretation.

## 4. What damage analytics measures

`regions.py` takes an integer 2D semantic mask at original image resolution. For each
class it counts pixels and calculates:

```text
image-area percentage = 100 × class pixels / (image height × image width)
```

The denominator is the whole image, including background and non-concrete content.
Without calibrated scale and scene geometry this is not square millimeters, physical
surface coverage, crack width, structural capacity or a severity threshold. Different
camera distances/perspectives can change percentages for the same physical defect.

`components.py` labels connected foreground runs with union-find. Four-connectivity
joins horizontal/vertical neighbors; eight-connectivity also joins diagonal neighbors.
Adjacent rows merge overlapping runs, so large filled regions do not require a Python
object for every pixel. Tests compare this algorithm with an independent breadth-first
reference on randomized masks.

Each same-class region reports pixel area, percentage, bounding box, centroid, bounding-box
fill fraction and whether it touches the image boundary. Boxes are
`[x0, y0, x1, y1]` with exclusive maxima, matching Python slices. Centroids are means
of zero-based pixel-index coordinates, not physical positions. Touching a boundary
indicates possible truncation, not a structural grade.

A component is a semantic-mask region, not a verified physical damage instance. Noise
can fragment one defect; diagonal contacts can join distinct defects. The reporting
minimum area only filters the class-region list. It never changes predicted masks,
class totals, or the recorded count of omitted region pixels. Background area is counted,
but background components are not enumerated as damage.

The report also forms all-foreground **damage complexes**. One complex can contain
several category regions. This exposes mixed cases instead of assigning one grade to
an adjacent mixture. Class-region rows carry their parent complex ID.

## 5. Supported spalling labels and unknown cases

The [dataset documentation](https://data.mendeley.com/datasets/yfhwfcrfmk/1) explicitly
names Minor spalling and Moderate spalling. The local masks preserve these as IDs 3
and 4. Task 04 uses those strings only within this source-category scope.

`severity.py` never derives a label from area. Its states include:

- `source_category_derived_unvalidated`: one supported category is present, with no
  triggered score-review heuristic. This remains an unvalidated segmentation-derived label.
- `mixed_minor_and_moderate`: both supported categories occur; no single label is assigned.
- `mixed_supported_and_unsupported`: a supported category shares a complex with other damage.
- `unsupported_category`: cracks, delamination, corrosion, crushing or deformation do
  not acquire a new severity rating from this tool.
- `uncertain_category_prediction`: model scores trigger review, or model scores are unavailable.
- `not_applicable_no_foreground`: no foreground is represented; this does not prove safety.

Every image report leaves overall structural severity **not assessed** with no score.
Delamination is not silently renamed a major-severity target. Even a high-scoring minor
spalling prediction can be wrong, as the baseline's poor precision demonstrates.

### Uncertainty is uncalibrated

For a softmax distribution over nine categories, inference records maximum softmax
and normalized entropy:

```text
H = -sum(p[c] × log(p[c])) / log(9)
```

These scalar maps are unpadded and nearest-restored with the same geometry as class IDs.
Region means summarize internal model scores. Configurable mean-score/entropy thresholds
can request human review; they are arbitrary research heuristics, not calibrated
probabilities of correctness, safety alarms or severity thresholds. Region averaging
can hide uncertain boundaries. Mask-only input has no model scores and retains unknown
uncertainty instead of inventing confidence.

## 6. Is a separate region-level severity classifier justified?

`supervision.py` inspects train/validation ground-truth masks only. It considers a pure
foreground complex with only class 3 or only class 4 an eligible source-category region.
Mixed and unsupported cases remain in `complex_audit.csv` with explicit dispositions.

Observed support:

| Source category | Train regions | Train images/groups | Validation regions | Validation images/groups |
| --- | ---: | --- | ---: | --- |
| Minor spalling | 121 | 54 / 36 | 30 | 12 / 6 |
| Moderate spalling | 85 | 78 / 67 | 21 | 17 / 12 |

Thirty region crops do not mean thirty independent validation samples. Several share
the same image or related source group. The configured planning gate requires 30 train
and 10 validation groups per category; minor spalling fails the validation gate. These
numbers are planning heuristics, not a formal power calculation or engineering standard.

More fundamentally, the labels come from the same pixel-category annotation lineage.
They do not independently rate structural severity. A narrowly named Minor-versus-Moderate
RGB category-discrimination study could be investigated after expert audit and stronger
independent evaluation, but it would not add a validated global severity target.
Do not feed a class-ID mask or label-coded feature into a classifier whose target is
that same mask label: this would be trivial target leakage. Ground-truth region proposals
also differ from noisy predicted proposals and require a stated evaluation protocol.

**No separate severity classifier is implemented or trained.** This is a supervision
limitation, not an excuse to invent area-based grades or relabel unsupported categories.

## 7. Reusable structured inference

`inference.py` loads a checkpoint bound to the frozen dataset, uses its saved preprocessing,
and predicts at native EXIF-oriented image dimensions. The structured report contains:

- source image/checkpoint hashes, checkpoint epoch, architecture and preprocessing;
- native class-ID mask location/hash and uncalibrated score maps;
- per-class areas and same-class component statistics;
- mixed foreground complexes and supported category-derived spalling states;
- calibrated-confidence status false, unknown states and review flags;
- physical measurements unavailable and structural severity not assessed.

The real implementation example uses validation image `Image_119_3665189`, 666×1000
pixels. It reports 77 class regions and eight foreground complexes. Its prediction
includes 13,262 minor-spalling pixels and 172,968 deformation pixels. These are exact
measurements of the predicted mask; they are not proof those damages exist. The gallery
shows substantial errors relative to annotations, illustrating why area statistics alone
cannot justify severity claims.

Known test image/mask hashes and final-test paths are rejected. The guard uses sealed
manifest metadata without decoding test pixels. It prevents common accidental reuse,
not deliberate protocol violations or unknown externally transformed test copies.

## 8. Modules and output locations

| Module | Important operation |
| --- | --- |
| `config.py` | Isolate outputs, pin original checkpoint, validate policies and single-factor designs |
| `baseline.py` | Verify training/history/artifact consistency and quantify error transitions |
| `validation.py` | Fresh native validation inference with independent count/metric cross-checks |
| `experiments.py` | Reconstruct saved settings, plan runs, audit resolutions and explicitly train |
| `comparison.py` | Enforce fair comparison, export all classes and paired group uncertainty |
| `components.py` | Exact run-length connected components |
| `regions.py` | Area/geometry, reporting filters, mixed complexes and score summaries |
| `severity.py` | Restrict label semantics to supported source categories |
| `supervision.py` | Audit pure/mixed source-region candidates and group-level support |
| `inference.py` | Original-resolution structured output and known-test firewall |
| `review.py` | Transparent worst-case validation galleries |
| `cli.py`, `__main__.py` | Explicit Task 04 commands, no test-evaluation subcommand |

Everything is saved under `outputs/task04/`. `baseline_audit/` contains sealed audit
results, full confusion/per-image counts, predictions and comparisons. `plans/` and
`experiment_plan.json` are designs, not trained results. `resolution_audit/` contains
resampling-only evidence. `severity_supervision/` contains counts, eligible candidates
and all-complex dispositions. `targeted_validation_review/` and `inference/` contain
executed diagnostic examples. Early implementation snapshots are retained separately
under `development_history/`, not counted as extra model experiments.
`implementation_report.json` summarizes actual verification and explicitly lists no
executed real-data improvement training runs.

After manual training, `experiments/<run-id>/` contains ordinary Task 03-style training
artifacts plus `task04_experiment.json`. Evaluation writes
`evaluations/<checkpoint-hash>/validation/`; comparison writes per-class CSVs and
`comparison.json` under `comparisons/<comparison-id>/`. Different loss scales are
recorded but not used to rank models.

## 9. Exact commands in order

Run from the repository root using existing dependencies. Installation is not part of
Task 04. Baseline audit caching is sealed/idempotent; inference/review IDs refuse overwrites.

```bash
source .venv/bin/activate
python -m damage_analytics.analytics audit-baseline
python -m damage_analytics.analytics plan
python -m damage_analytics.analytics resolution-audit
python -m damage_analytics.analytics supervision
python -m pytest -q
python -m ruff check src tests
python -m json.tool outputs/task04/baseline_audit/audit.json
python -m json.tool outputs/task04/severity_supervision/assessment.json
```

View the already generated `targeted_validation_review/*.png`. On a fresh Task 04 output
root, the following creates that deliberately biased gallery once:

```bash
python -m damage_analytics.analytics review
```

For a new structured report, choose an unused ID:

```bash
python -m damage_analytics.analytics predict \
  --checkpoint outputs/task03/experiments/unet_001/best.pt \
  --image outputs/task02/versions/rc1841-curated-v1-d34771dbb662406a/images/000016.png \
  --report-id validation_demo_002
python -m damage_analytics.analytics analyze-mask \
  --mask outputs/task04/inference/validation_demo_002/prediction_ids.png \
  --report-id existing_mask_002
```

Only after reviewing the plan and compute budget, launch these manual training jobs
individually. They have not been run on real data during implementation:

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

There is deliberately no Task 04 final-test command. Do not use Task 03's old final-test
entrypoint for iterative evaluation of these models. Do not recreate splits after seeing
results. Freeze a validation-selected candidate and obtain new independent data for any
future confirmatory assessment.

## 10. Verification and findings

The regression suite currently passes **154 tests and 21 subtests, with no skips**;
Ruff passes. New checks cover randomized component equality against independent BFS,
diagonal/bridge geometry, exact areas/boxes, reporting filters, invalid masks, mixed and
unsupported labels, unknown/uncalibrated scores, synthetic checkpoint inference,
baseline recomputation, seal tampering, one-factor plans without training, native support,
paired bootstrap reproducibility, incompatible comparisons, metric tampering and test exclusion.
Earlier Task 03 Torch checks now execute with the available runtime.

The implemented pipeline does not establish an improved model score yet. It establishes
a verified baseline, a defensible experiment plan, and transparent analytics with explicit
semantic limits. No test score or global structural severity is used to fill that gap.

## 11. Interview / viva questions

1. **Why independently recompute a checkpoint's metrics?** A saved number can be stale,
   associated with another checkpoint, or derived using a different target/protocol.
2. **Does the same confusion matrix imply identical prediction maps?** No. Errors at
   different pixels can cancel in count totals; it establishes aggregate agreement only.
3. **Why a matched control after an MPS baseline?** Backend/seed/numerical differences
   during training can otherwise be confused with the proposed intervention.
4. **Why cap weights instead of assuming more balancing is better?** Strong weights
   can trade precision for recall; minor spalling already has many false positives.
5. **Does larger input resolution prove a better model?** No. It improves measured
   mask fidelity here but raises cost; learned performance must be measured separately.
6. **Why not rank experiments by loss?** Different objective weights produce different
   scales. Use identical validation metric definitions and inspect every class.
7. **Why are small region counts misleading?** Multiple fragments from one image/group
   are correlated and do not supply independent validation evidence.
8. **Why 4- versus 8-connectivity?** Diagonal contacts can split or merge semantic regions;
   the choice affects counts and must be recorded.
9. **Can percentage damage area define severity?** Not without a separately justified
   physical/annotation target. Camera geometry and non-concrete background affect image area.
10. **What is supported about Minor/Moderate labels?** They are explicit source categories,
    not independently validated overall structural grades.
11. **Is a high softmax value calibrated confidence?** No. It is an internal score and
    can be confidently wrong; entropy thresholds here only request review.
12. **Why refuse a separate severity classifier now?** Independent severity supervision
    is missing, labels share segmentation lineage, and minor validation support is six groups.
13. **Why group bootstrap rather than pixel bootstrap?** Pixels are strongly correlated;
    known groups are a more defensible resampling unit, though true source identity remains incomplete.
14. **Can the used test be reused after improvement?** It would compromise its independent
    role in model development. Task 04 excludes it and makes validation-only claims.
15. **What would justify a stronger severity study?** Explicit expert criteria, suitable
    region labels, reliable source grouping, calibration where relevant, adequate independent
    cases and genuinely new confirmatory evaluation.
