"""Human-readable conclusions, with observations separated from causal hypotheses."""

from pathlib import Path

from damage_analytics.task08.inputs import Run


def percentage(value: float | None) -> str:
    return "undefined" if value is None else f"{100 * value:.2f}%"


def failure_report(path: Path, result: dict, runs: list[Run]) -> None:
    lines = [
        "# Task 08: validation failure analysis",
        "",
        "Existing saved validation predictions were authenticated against pinned releases, "
        "checkpoint SHA-256, source-selected threshold provenance and complete "
        "validation ID populations. "
        "Metrics below were independently recomputed from prediction CSVs. "
        "Checkpoint tensors were "
        "CPU-loaded and checked for finite values; inference was not rerun. No test "
        "pixels or test "
        "predictions were read. Source labels remain supplied annotations, not new "
        "expert adjudications.",
        "",
        "## Measured operating points",
        "",
        "Each model retains its own frozen SDNET validation threshold. DamSegment thresholds were "
        "neither searched nor changed. PR-AUC is trapezoidal; AP is separately "
        "available in JSON/CSV. "
        "Different prevalence makes PR-AUC and precision across datasets directly incomparable.",
        "",
        "| Run / dataset | ROC-AUC | PR-AUC | Crack precision | Crack recall | "
        "Crack F1 | Macro F1 | Balanced acc. | FP / FN |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["results"]:
        metrics, crack = row["metrics"], row["metrics"]["classes"][1]
        matrix = metrics["confusion_matrix"]
        lines.append(
            f"| {row['run_id']} / {row['dataset_name']} | {metrics['roc_auc']:.4f} | "
            f"{metrics['pr_auc']:.4f} | {percentage(crack['precision'])} | "
            f"{percentage(crack['recall'])} | {percentage(crack['f1'])} | "
            f"{percentage(metrics['macro_f1'])} | {percentage(metrics['balanced_accuracy'])} | "
            f"{matrix[0][1]} / {matrix[1][0]} |"
        )
    lines += ["", "## Observations from saved scores", ""]
    for run in runs:
        rows = {r["dataset_name"]: r for r in result["results"] if r["run_id"] == run.root.name}
        source = rows["sdnet2018"]
        lines.append(
            f"* **{run.root.name}**: {source['epochs_trained']} epochs trained; selected epoch "
            f"{source['selected_epoch']}; frozen threshold **{source['threshold']:.6f}**. "
            f"Source validation prevalence {source['crack_prevalence']:.2%}."
        )
        if "damsegment" in rows:
            external = rows["damsegment"]
            source_crack, external_crack = (
                source["metrics"]["classes"][1],
                external["metrics"]["classes"][1],
            )
            lines.append(
                f"  DamSegment prevalence {external['crack_prevalence']:.2%}; crack recall "
                f"changes "
                f"by {100 * (external_crack['recall'] - source_crack['recall']):+.2f} "
                f"percentage points. "
                f"ROC-AUC changes by "
                f"{external['metrics']['roc_auc'] - source['metrics']['roc_auc']:+.4f}. "
                f"Median positive score: source "
                f"{source['scores']['1']['score_quantiles_05_25_50_75_95'][2]:.4f}, external "
                f"{external['scores']['1']['score_quantiles_05_25_50_75_95'][2]:.4f}."
            )
        material = source["material_metrics"]
        if material:
            lines.append(
                "  Source material tags (patch-level, provisional groups): "
                + "; ".join(
                    f"{r['material_tag']}: n={r['samples']}, "
                    f"recall={percentage(r['crack_recall'])}, "
                    f"ROC-AUC={r['roc_auc']:.4f}, groups={r['candidate_components']}"
                    for r in material
                )
                + "."
            )
    lines += [
        "",
        "ROC-AUC describes ranking across thresholds; fixed-threshold recall describes one "
        "operating point. A source-selected threshold can miss more external positives even when "
        "ranking remains strong. Score distributions and error galleries show which "
        "source-labeled "
        "positives fall below that threshold; they do not establish why the network did so.",
        "",
        "## Hypotheses and limits",
        "",
        "* Acquisition, texture, lighting, crack scale, source labeling and "
        "preprocessing scale may "
        "differ. SDNET uses 256-pixel patches versus 512-pixel DamSegment classification images; "
        "both are reduced to 224. No pixelwise crack labels exist for these classification images "
        "to quantify thin-crack loss, and no controlled causal ablation was run.",
        "* Class-weighted CE and a changed class prior do not produce calibrated posterior "
        "probabilities. Domain score shift is plausible; no calibration claim or "
        "external threshold "
        "tuning follows from it. Prevalence alone does not algebraically explain "
        "recall or ROC-AUC changes.",
        "* Selected FP/FN galleries are illustrative extremes, not unbiased prevalence estimates. "
        "Borderline means closest to each model's frozen threshold, not closest to "
        "0.5. Disagreements "
        "also reflect different thresholds; they are not automatically annotation errors.",
        "* Original-photo identities are unverified. DamSegment mixed-view near pairs require "
        "review; see the integrity report for exact IDs. These models trained on SDNET, not the "
        "DamSegment detection/segmentation training views. Such pairs are not automatically "
        "SDNET-to-DamSegment training leakage. Hash screens can miss crops and "
        "physical relatedness.",
        "* DamSegment validation has now been inspected. It is exploratory domain-shift evidence, "
        "not an untouched external test. Keep all frozen test splits sealed. Do not iterate "
        "augmentation, architecture or thresholds against DamSegment results.",
        "",
        "## Controlled next experiments",
        "",
        "Run matched 25-epoch ceilings for both backbones, seed 42, identical "
        "frozen SDNET splits, "
        "baseline full-image preprocessing/augmentation, training-count CE weights, AdamW and "
        "cosine scheduling over the declared budget. Validate each epoch, select by "
        "source ROC-AUC "
        "and stop after eight stagnant validation checks. Choose the threshold once "
        "on the selected "
        "source-validation checkpoint. Preserve five-epoch baselines. Longer "
        "training is a hypothesis, "
        "not an improvement result; the selected MobileNet baseline epoch already "
        "preceded epoch five.",
        "",
        "Optional follow-up: matched right-angle rotation augmentation (`orientation`), with the "
        "same budgets/seeds. It preserves full images and introduces no rotation interpolation. "
        "Keep it separate from the initial budget comparison. No new weighting, resolution or "
        "threshold policy is silently bundled into the main runs.",
        "",
        "New training/evaluation records synchronized pass runtime, throughput and "
        "training wall time. "
        "Historical baseline runtime was not recorded and cannot be recovered "
        "reliably from file times. "
        "Model-only latency, training speedup and real-data improvements are not claimed.",
        "",
        "Crack/no-crack is not general structural-damage detection, concrete/OOD recognition, "
        "severity grading or a safety score. ResNet18 remains a provisional research candidate; "
        "final selection for the new cohort must use the predeclared SDNET validation protocol.",
        "",
        "## Files",
        "",
        "`comparison.csv` and `analysis_report.json`: all numeric evidence/provenance. "
        "Each run/dataset folder contains `class_metrics.csv`, `roc_pr_points.csv`, "
        "`roc_pr_scores.png`, `confusion_matrix.png`, `prediction_index.csv`, and indexed FP/FN/"
        "borderline galleries. SDNET alone has material metrics and threshold sensitivity. "
        "`disagreements/` contains aligned model-pair CSVs/contact sheets.",
    ]
    path.write_text("\n".join(lines) + "\n")


def integrity_report(path: Path, result: dict) -> None:
    lines = [
        "# Task 08: dataset integrity and leakage investigation",
        "",
        "Sealed manifests, exclusions, saved split membership, exact hashes and candidate "
        "fingerprints were checked. All-partition metadata is used only for integrity QA; "
        "test pixels and model test results were not opened. No source identities or human "
        "review decisions were invented. No existing release or frozen split was changed.",
        "",
    ]
    for name, report in result["datasets"].items():
        pairs = report["pair_investigation"]
        lines += [
            f"## {name}",
            "",
            f"Included **{report['included']}**, excluded **{report['excluded']}**. "
            f"Included views: {report['included_by_view']}. "
            f"Exclusion reasons: {report['exclusions_by_reason']}.",
            "",
            f"Exact cross-split decoded components: **{report['exact_cross_split_components']}**. "
            f"Known-component checks: {report['split_checks']['known_component_checks_passed']}; "
            f"authenticated source identity: "
            f"{report['split_checks']['source_identity_verified']}. "
            "This does not establish unknown-source independence.",
            "",
            f"Near pairs: **{pairs['all_near_pairs']}**, unresolved cross-split "
            f"**{pairs['unresolved_cross_split_pairs']}**. "
            f"Affected records: **{pairs['affected_samples']}**, including "
            f"**{pairs['affected_classification_samples']} classification records**. "
            f"View pairs: {pairs['cross_split_view_pairs']}. "
            f"Partitions: {pairs['cross_split_partition_pairs']}.",
            "",
            f"Affected views/partitions: {pairs['affected_by_view_partition']}.",
            "",
            f"Cross-split native dimensions: {pairs['cross_split_dimension_pairs']}. "
            f"Both perceptual distances are zero for "
            f"{pairs['zero_phash_and_dhash_cross_split_pairs']} pairs. "
            "Different-sized views can evade exact decoded hashing; zero perceptual distance "
            "still does not authenticate common physical source identity.",
            "",
            f"Potential candidate graph: {report['potential_candidate_graph']}. "
            "`potential_components.csv` traces indirect connections through existing exact/name "
            "components. Graph diagnostics do not establish source identities or split decisions.",
            "",
            f"`{name}/review_pairs.csv` contains every pair, paths, splits, classes, distances "
            "and blank decision/evidence fields; `affected_samples.csv` identifies each affected "
            "sample and pair IDs. `contact_sheets/` shows all candidates with every test endpoint "
            "redacted. Contact sheets are for review, not automated identity assertions.",
            "",
        ]
    search = result["overlap_search"]
    lines += [
        "## Why the old overlap count was zero",
        "",
        result["reporting_defect"],
        "",
        f"Re-run search: {search['indexed_records']} indexed records, "
        f"{search['phash_search']['queries']} queries, "
        f"**{search['screened_comparisons']:,} node-distance evaluations**, "
        f"{search['phash_candidate_hits_consumed']} radius candidates, "
        f"{search['dhash_comparisons']} dHash checks; exact pairs "
        f"{search['exact_pair_count']}, near candidates {search['near_candidate_count']}. "
        "Full evidence is in `cross_dataset_screen.json`. Brute-force validation anchors "
        f"matched indexed radius results: **{result['search_verification']['all_matched']}**.",
        "",
        "This verifies the implemented radius search, not a guarantee against crops, rotations, "
        "re-encoding or shared unknown source photographs. A zero candidate count alone does "
        "not certify leakage freedom.",
        "",
        "## Disposition",
        "",
        result["decisions"],
        "",
        "No versioned data correction is justified by telemetry alone or by unreviewed near "
        "matches. Review these candidates without exposing test images to model development. "
        "If source evidence later confirms related samples across partitions, create a new "
        "version that quarantines conflicting development components while retaining the "
        "frozen test membership; do not use ordinary re-splitting to move test samples. "
        "Any confirmed test-related contamination requires an explicit evaluation limitation "
        "and potentially a genuinely independent future cohort.",
    ]
    path.write_text("\n".join(lines) + "\n")
