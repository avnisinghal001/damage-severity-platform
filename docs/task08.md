# Task 08: crack reliability and controlled training

Task 08 reuses Task 07 and leaves the API, React, PostgreSQL, RC segmentation and severity
code unchanged. No model weights, existing dataset releases or frozen test populations
are changed. All new artifacts go under `outputs/task08/` and are ignored by Git.

## Findings and scope

Read these executed reports:

* `outputs/task08/integrity/audit_003/integrity_report.md`
* `outputs/task08/analysis/baselines_001/failure_analysis.md`
* `outputs/task08/analysis/baselines_001/visual_observations.md`
* `outputs/task08/implementation_report.json` for actual checks and limitations.

SDNET includes 56,088 records after four conflicting-label duplicate exclusions.
DamSegment includes 2,000 classification, 1,500 detection and 1,411 segmentation
records; 89 segmentations were excluded for degenerate/self-intersecting polygons.
Known exact components do not cross splits, but physical source grouping is unverified.

DamSegment has 299 near pairs, 105 unresolved across splits. The 105 contain 44
classification/detection, 43 classification/segmentation, nine detection/segmentation,
five detection/detection and four segmentation/segmentation pairs. They involve 150
records, including 44 classification records: 14 train, 14 validation and 16 test.
There are no direct classification/classification cross-split candidate pairs at this
screen's settings. Tracing indirect candidates together with existing components found
49 potential cross-split components (153 records), but zero components connecting
classification records across splits. This does not rule out undetected relationships.

Eighty-seven cross-split pairs join 512-pixel classification to 640-pixel annotation
images; 73 pairs have both perceptual distances zero. Re-encoded/resized views are a
plausible reason exact hashes alone did not join these candidates. Identity decisions
remain unresolved. Review CSVs retain blank `decision,evidence` fields; automated code
does not fabricate human judgments. All test thumbnails are redacted. Test membership
and fingerprints are used only for integrity QA, never for metrics or parameter tuning.

The old cross-dataset `screened_comparisons=0` counted radius candidates consumed, not
search distances. Instrumentation now separately records queries, BK node-distance
evaluations, candidate hits and dHash checks. The recheck performed 19,852,037 node
distance evaluations, returned no radius-4 candidates and matched 12 independent
brute-force validation-anchor queries. Crops/rotations/unknown physical origins can
still evade this screen. No leakage-free or statistical-independence claim follows.

Existing validation CSVs independently reproduce the baseline metrics. ResNet18 has
SDNET crack recall 81.41%, versus 70% on DamSegment; ROC-AUC is 0.9294 versus 0.9260.
MobileNet recall is 80.78% versus 64%, and ROC-AUC 0.9217 versus 0.8183. The report
includes precision/F1/PR-AUC, confusion matrices, material metrics and score quantiles.
Recall is a fixed-threshold property; strong ranking does not guarantee transfer of a
source operating point. Domain/scale/label differences are hypotheses, not proved causes.
Prevalence affects precision and PR metrics, but alone does not explain a recall decrease.

No data-release correction was made: a telemetry defect and unreviewed candidates do
not justify moving samples. If later evidence confirms contamination, retain frozen
test membership and quarantine conflicting development components in a new version.
Do not run ordinary Task 07 re-splitting to repair these frozen releases. A previously
exposed test cannot become unbiased merely by changing a release identifier.

## Implementation

| File/package | Responsibility |
|---|---|
| `config/task08.toml` | Fixed release paths/seals, bounded gallery/search settings, matched 25-epoch settings |
| `task08/config.py` | Validate config, verify pinned source/target releases; source-only training metadata reads |
| `task08/integrity.py` | Exclusions, independent split checks, exact/near view counts, pair IDs and search verification |
| `task08/inputs.py` | Authenticate checkpoints, thresholds, exact validation populations, targets and recomputed metrics |
| `task08/diagnostics.py` | ROC/PR points, score distributions, material metrics and SDNET-only threshold sensitivity |
| `task08/galleries.py` | FP/FN/borderline/disagreement sheets; enforced test-pixel redaction |
| `task08/analysis.py`, `reports.py` | Validation evidence, plots, human-readable findings and limitations |
| `task08/comparison.py` | Matched source-validation comparisons; reject unequal budget/seed/loss/settings |
| `task08/cli.py` | Explicit integrity/analyze/train/evaluate/compare/smoke commands; no test command |
| Task 07 `duplicates.py`, `crosscheck.py` | Correct search telemetry without changing match criteria |
| Task 07 `config.py`, `data.py` | Backward-compatible optional orientation augmentation and inference preprocessing guard |
| Task 07 `release.py` | Verify the checkpoint development-ID list covers every frozen source train/validation sample |
| Task 07 `timing.py`, `loops.py`, `engine.py`, `evaluation.py` | Pass runtime, throughput, required history metrics and training wall time |

Old checkpoints default to baseline augmentation when the new setting is absent.
Baseline augmentation remains horizontal flip and mild brightness/contrast, training
only. Optional `orientation` first applies a uniform right-angle rotation, preserving
the complete image without additional rotation interpolation. No random crop can remove
a crack while retaining its image-level positive label. Validation never augments.
ImageNet normalization and full-image bilinear letterboxing are unchanged. Inference now
fails if the checkpoint's declared normalization/resize contract disagrees with the code.

Training retains train-count balanced CE, AdamW, cosine scheduling, validation ROC-AUC
selection and a threshold selected once on the selected SDNET validation checkpoint.
No incorrect loss formula or actual preprocessing mismatch was found in the saved
baselines. The preprocessing check is a new fail-closed guard, not evidence old scores
were wrong. Unweighted `CrossEntropyLoss(weight=None, reduction="sum")` now also works
in the shared loop. Historical runs used explicit weight tensors and were unaffected.
The loop rejects unsupported mean-reduction criteria rather than silently normalizing
twice. Evaluation also refuses incomplete development-ID lists, preventing a malformed
checkpoint from narrowing overlap screening. Both baselines' complete 47,782 development
IDs were verified; these guards do not imply their saved results were incorrect.

Runtime is end-to-end pass wall time: loading, file-hash verification, decoding,
transforms, transfers and compute; training includes backward/optimizer. MPS synchronizes
only at measured pass boundaries beyond the existing finite checks/result transfers.
It is not model-only latency. Training wall time includes checkpointing/plots and excludes
model initialization/downloads. Baseline historical runtime is unavailable, not inferred
from file timestamps. No measured real-data training improvement is claimed.

## Exact commands, in execution order

Run from the repository root. No new dependencies are needed beyond Task 07's
`classification` and `dev` extras. No installation/download was executed in Task 08.
If your environment is missing them, the manual installation command is
`python -m pip install -e '.[classification,dev]'` after activation.

```bash
source .venv/bin/activate

# Lightweight checks; no real training or pretrained downloads.
python -m ruff check src/damage_analytics/task07 src/damage_analytics/task08 tests/test_task08.py
python -m pytest -q tests/test_task07_data.py tests/test_task07_training.py tests/test_task08.py
python -m damage_analytics.task08 smoke --run-id local_smoke

# New IDs are mandatory: reports/runs do not overwrite existing artifacts.
python -m damage_analytics.task08 integrity --report-id local_integrity
python -m damage_analytics.task08 analyze --report-id local_baselines --include-external
cat outputs/task08/integrity/local_integrity/integrity_report.md
cat outputs/task08/analysis/local_baselines/failure_analysis.md
open outputs/task08/integrity/local_integrity/damsegment/contact_sheets
open outputs/task08/analysis/local_baselines/sdnet_resnet_pilot/damsegment/galleries

# Explicit real training: ImageNet weights requested only here (cached if available).
# Fresh ImageNet-initialized runs; do not resume or overwrite the five-epoch baselines.
python -m damage_analytics.task08 train --backbone mobilenet_v3_small \
  --run-id sdnet_mobile_e25_s42 --epochs 25 --seed 42 --device mps --allow-provisional
python -m damage_analytics.task08 train --backbone resnet18 \
  --run-id sdnet_resnet_e25_s42 --epochs 25 --seed 42 --device mps --allow-provisional

# Source validation only for selection and the first comparison.
python -m damage_analytics.task08 evaluate --dataset sdnet2018 --device mps \
  --checkpoint outputs/task08/experiments/sdnet_mobile_e25_s42/best.pt
python -m damage_analytics.task08 evaluate --dataset sdnet2018 --device mps \
  --checkpoint outputs/task08/experiments/sdnet_resnet_e25_s42/best.pt
python -m damage_analytics.task08 compare --report-id matched_e25_s42 \
  --run-id sdnet_mobile_e25_s42 --run-id sdnet_resnet_e25_s42
python -m damage_analytics.task08 analyze --report-id source_e25_s42 \
  --run-dir outputs/task08/experiments/sdnet_mobile_e25_s42 \
  --run-dir outputs/task08/experiments/sdnet_resnet_e25_s42
cat outputs/task08/analysis/source_e25_s42/failure_analysis.md
```

The provisional flag acknowledges unauthenticated source grouping; it does not make
the dataset final-test eligible. Training reads SDNET train/validation only. The ceiling
is 25 epochs; patience is eight validation checks with `min_delta=0.0001`. Each model
may stop earlier. Longer cosine schedules also change learning-rate exposure; improved
metrics cannot be attributed solely to more epochs. MobileNet's previous best was epoch
three, so longer training is not guaranteed to help.

Keep the first comparison to these two runs. If testing orientation augmentation,
predeclare that ablation using source findings, run **both** models with the same budget
and seed using new IDs, then compare on SDNET. Do not select this policy by DamSegment scores:

```bash
# Optional subsequent matched ablation; these are additional manual training runs.
python -m damage_analytics.task08 train --backbone mobilenet_v3_small \
  --run-id sdnet_mobile_orientation_e25_s42 --epochs 25 --seed 42 --device mps \
  --augmentation orientation --allow-provisional
python -m damage_analytics.task08 train --backbone resnet18 \
  --run-id sdnet_resnet_orientation_e25_s42 --epochs 25 --seed 42 --device mps \
  --augmentation orientation --allow-provisional
```

Use the same source evaluation/analysis commands with these new IDs. The strict matched
comparison intentionally rejects baseline versus orientation settings; use separate
source analysis tables and report the declared ablation. Do not claim equal training
budgets when comparing five-epoch and 25-epoch results. If selecting across seeds, use
the same predeclared seeds for both architectures; never choose the best external seed.

After source-based model selection, freeze the winning checkpoint, augmentation,
threshold and protocol. Then perform a single exploratory external assessment. The
example below assumes source validation selected the new ResNet; change the path if
source evidence selected MobileNet instead:

```bash
python -m damage_analytics.task08 evaluate --dataset damsegment --device mps \
  --checkpoint outputs/task08/experiments/sdnet_resnet_e25_s42/best.pt \
  --confirm-compatible-labels
python -m damage_analytics.task08 analyze --report-id frozen_resnet_external \
  --run-dir outputs/task08/experiments/sdnet_resnet_e25_s42 --include-external
```

External evaluation retains the source threshold and checks source development
fingerprints against target validation. Compatibility confirmation is required; source
names alone cannot prove compatible annotation completeness. No DamSegment sensitivity
table is possible: the code rejects that operation. External assessment is not a new
test and should not trigger iterative tuning. No Task 08 command evaluates final tests.

Use `--device cpu` when MPS is unavailable/unsupported; there is no hidden backend
fallback. Seeds and policies are recorded, but MPS/CPU bitwise equivalence is not promised.
Strict determinism can reject MPS kernels. Batch size 32/workers 0 preserve the baseline
execution settings; any execution change must be matched across the comparison cohort.

## Outputs and review protocol

```text
outputs/task08/
  implementation_report.json
  integrity/{report_id}/
    integrity_report.{json,md}, cross_dataset_screen.json, search_verification.json
    {dataset}/quality.json, excluded_records.csv, affected_samples.csv
    {dataset}/review_pairs.csv, contact_sheets/pairs_*.png
    {dataset}/potential_components.csv  # possible indirect links, not source identities
  analysis/{report_id}/
    analysis_report.json, failure_analysis.md, comparison.csv
    {run_id}/{dataset}/metrics.json, class_metrics.csv, prediction_index.csv
    {run_id}/{dataset}/roc_pr_points.csv, roc_pr_scores.png, confusion_matrix.png
    {run_id}/sdnet2018/material_metrics.csv, sdnet_threshold_sensitivity.{csv,png}
    {run_id}/{dataset}/galleries/{false_positives,false_negatives,borderline}.{png,json}
    disagreements/{dataset}/{model_pair}/disagreements.{csv,png,json}, summary.json
  experiments/{run_id}/           # Task 07 checkpoint/history/threshold artifacts
    runtime.json                 # new run timing; history also includes pass throughput
  evaluations/{checkpoint_hash}/{release_id}/validation/
    metrics.json, predictions.csv, confusion_matrix.png
  comparisons/{report_id}.json
  smoke/{run_id}/...              # tiny synthetic CPU fixtures, never model evidence
```

The SDNET material tags remain D/P/W exactly as supplied. Material metrics are patch-level
diagnostics and report provisional component counts, not independent-photo sample sizes.
No group confidence intervals, calibration, general-damage or safety claims are inferred.

For duplicate review, copy `review_pairs.csv` outside its report directory and record
actual documentary/manual evidence. Blank decisions are intentionally unresolved.
Test-related rows can be inspected by a separate data curator under an appropriate
test-isolation protocol, not by opening test photographs during model development.
Likewise, visually ambiguous source labels require expert review before any relabeling.
Neither high model confidence nor a low perceptual distance is itself a human decision.

Tests include no-hit versus no-search telemetry, indexed/brute-force equivalence,
view/partition accounting, test-image redaction, validation population/label integrity,
external-threshold refusal, tied curves, train-only augmentation, parameter updates,
checkpoint/runtime smoke and matched-comparison guards. Actual test outcomes are saved
in the implementation report; no full real training was executed during this task.
