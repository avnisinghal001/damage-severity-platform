"""Concise audit conclusions derived from the actual stored/archive evidence."""

from pathlib import Path


def audit_markdown(path: Path, archives: dict, rc: dict) -> None:
    sd = archives["datasets"]["sdnet2018"]["views"]["classification"]
    dam = archives["datasets"]["damsegment"]
    dv = dam["views"]
    classification, detection, segmentation = (
        dv[k] for k in ("classification", "detection", "segmentation")
    )
    dam_crack = sum(
        n
        for folder, n in classification["image_directories"].items()
        if Path(folder).name == "Crack"
    )
    pair_missing = sum(len(dv[k]["missing_expected_pairs"]) for k in ("detection", "segmentation"))
    crack = sum(
        n
        for folder, n in sd["image_directories"].items()
        if folder.split("/")[-1] in ("CD", "CP", "CW")
    )
    lines = [
        "# Task 07 — dataset and model validity audit",
        "",
        "Evidence: local archive directories, all small DamSegment JSON files for label counts, "
        "representative image/mask decoding, existing RC reports and train/validation metadata. "
        "No full real-dataset extraction, real-model training, downloads or new real-data "
        "inference was run. Synthetic implementation tests are reported separately. "
        "The old RC test images, manifest and evaluation results were not opened.",
        "",
        "## Verified archive facts",
        "",
        f"* SDNET2018: **{sd['images']:,} JPEG patch entries**, {crack:,} C-folder crack labels "
        f"and {sd['images'] - crack:,} U-folder no-crack labels. "
        f"{sd['candidate_source_prefixes']} material/prefix candidate groups. Sampled "
        f"patches are 256² RGB.",
        f"* DamSegment classification: **{classification['images']:,} images**, "
        f"{dam_crack:,} Crack "
        f"and {classification['images'] - dam_crack:,} Non-Crack. Encoding counts: "
        f"{classification['extensions']}; sampled images 512² RGB. "
        f"**{dv['classification']['same_stem_collisions']} repeated stems** require "
        f"extension-aware IDs.",
        f"* DamSegment detection: **{detection['images']:,} image entries**, "
        f"{detection['extensions'].get('.json', 0):,} COCO-like JSON files "
        f"(despite the Pascal VOC directory name), and "
        f"{detection['extensions'].get('.txt', 0):,} YOLO box files. Sampled images are 640².",
        f"* DamSegment segmentation: **{segmentation['images']:,} image/mask candidates**; "
        f"image directories: {segmentation['image_directories']}, with JSON and YOLO polygons. "
        "These folder names are NOT validated severity grades.",
        f"* Both annotation views contain source-ID object counts "
        f"{dv['detection']['annotations']['objects_by_source_id']}, "
        f"and {dv['detection']['annotations']['empty']} empty annotation files each. "
        f"Missing expected pairing paths: **{pair_missing}**. "
        "Full decoding/geometry validation and post-exclusion counts are pending manual "
        "preparation.",
        "* Sampled segmentation PNGs use RGB black/red/blue. Explicit local encoding is "
        "background=0, red/source-ID-0=1, blue/source-ID-1=2. Numeric source IDs remain "
        "available; **semantic crack/spalling names were not supplied in the archives**. "
        "Preparation verifies category presence and reports PNG/polygon raster disagreement.",
        "",
        "## Leakage evidence and limits",
        "",
        f"Detection/segmentation have **{dam['cross_view_crc_size_candidates']} CRC/size matching "
        f"image candidates**; **{len(dam['sha256_verified_sample_pairs'])} sampled pairs "
        f"match SHA-256**. "
        "They are overlapping views, not independent image evidence. Full hashing during "
        f"preparation groups exact views together. Classification/segmentation CRC/size "
        f"candidates: "
        f"{dam['classification_segmentation_crc_size_candidates']}; crops/re-encoding may "
        f"be missed.",
        "",
        "SDNET filename prefixes are plausible source-photo buckets, but no "
        "authoritative original-photo "
        "mapping is supplied. DamSegment C/N enumeration is not authenticated photo identity. "
        "Conservative filename buckets and exact-image components are explicitly **provisional**, "
        "never invented physical-source IDs. Perceptual matches remain review items; "
        "search budgets "
        "are recorded and truncation blocks final eligibility. User evidence CSVs can "
        "establish groups "
        "and adjudicate candidate pairs in a new immutable release. Cross-dataset "
        "screening is available "
        "after full preparation; it was not claimed executed here.",
        "",
        "## RC1841 validity",
        "",
        f"The existing release has **{rc['samples']} images**, all "
        f"**{rc['annotated_damage_images']} "
        f"with foreground annotation**, **{rc['zero_foreground_images']} "
        f"annotation-negative images**. "
        f"Train/validation manifests contain {rc['subsets']['train']['samples']}/"
        f"{rc['subsets']['validation']['samples']} records, with "
        f"{rc['subsets']['train']['zero_foreground']}/"
        f"{rc['subsets']['validation']['zero_foreground']} empty masks. "
        f"Background occupies {rc['classes'][0]['pixel_fraction']:.2%} of aggregate image "
        f"pixels but is polygon complement, "
        "not an expert intact-concrete or non-concrete label. Technical QA does not "
        "establish annotation completeness.",
        "",
        "| RC category | Images in aggregate release | Pixels |",
        "|---|---:|---:|",
    ]
    for row in rc["classes"]:
        lines.append(f"| {row['source_label']} | {row['images']} | {row['pixels']:,} |")
    lines += [
        "",
        "Existing native validation (243 images), **not rerun inference**:",
        "",
        "| Model | Foreground mIoU | Annotated-background → damage pixels | Fraction of "
        "GT background |",
        "|---|---:|---:|---:|",
    ]
    for row in rc["validation"]:
        lines.append(
            f"| {row['run']} | {row['foreground_miou']:.4%} | "
            f"{row['background_to_damage_pixels']:,} | "
            f"{row['background_pixel_false_damage_fraction']:.2%} |"
        )
    lines += [
        "",
        "The small five-epoch pilot results are not comparable training budgets to 50-epoch "
        "baselines. Higher background/pixel accuracy does not establish useful rare-class recall. "
        "In the completed control, crushing recall is about 2.03%, microcrack precision 17.41%, "
        "and minor-spalling precision 14.25%. Class imbalance and class confusion are material.",
        "",
        "Demonstrated: annotated-background false damage and source-class confusions. "
        "Hypotheses for unrelated-photo failures: missing explicit out-of-domain supervision, "
        "foreground-biased sampling/weighting, appearance/domain shift and uncalibrated scores. "
        "The nine-class argmax includes background; it is NOT forced to predict foreground, "
        "but it has no learned concrete/OOD rejection state. Negative full-image specificity "
        "and unrelated-photo error rates cannot be measured from these all-positive RC splits.",
        "",
        "## Supported targets and next approach",
        "",
        "* **Binary crack recognition:** SDNET and DamSegment classification support their "
        "supplied crack/no-crack distinction, subject to source grouping and annotation quality. "
        "No-crack can contain other defects. Do not use this classifier to discard spalling-only "
        "or corrosion-only images before the RC segmenter.",
        "* **Eight-class RC segmentation:** conditional research use; rare categories have few "
        "images, weak measured recall and unresolved source independence/completeness limits.",
        "* **Concrete/non-concrete and generally damaged/intact:** not established by "
        "current labels. "
        "Add expert material/context tags, genuinely intact concrete, "
        "unrelated/hard-negative scenes, "
        "and mixed-context regions including eligible exposed rebar. Crack negatives "
        "alone do not fill this gap.",
        "* **Minor/moderate/severe image-level severity:** unsupported. Minor/Moderate spalling "
        "are source segmentation categories; there is no severe image-grade target or validated "
        "rubric/independent outcome supervision. Obtain mechanism-specific expert grades and "
        "adjudication before a severity experiment; area is not a safety score.",
        "",
        "First run a **five-epoch MobileNetV3-Small SDNET pilot**, with conservative "
        "material/prefix "
        "grouping, ImageNet initialization, whole-patch augmentations and train-only weighting. "
        "It is explicitly exploratory (`--allow-provisional`), not an unbiased benchmark. "
        "Select by validation ROC-AUC; choose a threshold once on the selected "
        "validation checkpoint. "
        "Compare ResNet18 using identical saved splits/budget and report class/domain results. "
        "Before final testing, authenticate source grouping, resolve cross-split near matches and "
        "pre-register the protocol. Keep the previously used RC test sealed; confirmation needs "
        "genuinely independent data, never more tuning on that test.",
        "",
        "There is no evidence-derived universal training sample minimum. The minimum missing "
        "label types are verified intact concrete, non-concrete/hard negatives and independent "
        "severity targets if that task is pursued. Size new cohorts by source diversity "
        "and learning "
        "curves. As a conditional statistical design example, 59 independent negative images with "
        "zero false alarms give a one-sided 95% binomial upper bound below 5% per cohort "
        "(299 for 1%); these are prospective evaluation requirements, not observed model results "
        "or guarantees of sufficient training/subgroup coverage.",
        "",
        "Machine-readable evidence: archive_audit.json and rc1841_audit.json. "
        "See docs/task07.md for exact commands, blockers and implementation/test scope.",
    ]
    path.write_text("\n".join(lines) + "\n")
