"""Evidence-based integrity summaries; model performance is never part of preparation."""

from pathlib import Path


def integrity_report(path: Path, archive: dict, quality: dict) -> None:
    lines = [
        "# PEER Phi-Net Task 7 integrity and preparation",
        "",
        "## Verified local archive",
        "",
        f"Archive SHA-256: {archive['archive_sha256']}. It contains four float32 NumPy arrays, "
        "README and license, not original image files. Image/label pairing is by official "
        "partition and row index. No source photograph, component, building, capture, crop-parent "
        "or original-filename mapping is supplied. Index identifiers are provenance, not "
        "source identity.",
        "",
        "| Source ID | Preserved label | Original train rows | Official test rows |",
        "|---|---|---:|---:|",
    ]
    for item in archive["class_mapping"]:
        i = str(item["class_id"])
        lines.append(
            f"| {i} | {item['source_label']} ({item['source_code']}) | "
            f"{quality['official_partitions']['train']['class_counts'].get(i, 0)} | "
            f"{quality['official_partitions']['test']['class_counts'].get(i, 0)} |"
        )
    lines += [
        "",
        f"Array headers: {archive['headers']}. "
        f"Invalid original rows by partition: "
        f"{ {s: v['invalid_rows'] for s, v in quality['official_partitions'].items()} }.",
        "",
        "Each valid label is exactly one-hot over four columns; no soft-label argmax "
        "guessing occurs. "
        "IDs are not ordered severity scores: ID 0 is Heavy, ID 3 is Undamaged. "
        "Test class counts are integrity statistics, not model performance.",
        "",
        archive["count_reference_note"],
        "No per-task published-count discrepancy is asserted without "
        "a verified reference. Local header, body length and image/label row counts are "
        "checked independently.",
        "",
        "## Image representation and unresolved preprocessing",
        "",
        "Stored tensors have negative, mean-subtracted float values. Adding the Caffe offsets "
        "[103.939,116.779,123.68] reconstructs an integer byte grid for all supported rows. "
        f"Maximum inverse integer residual: {quality['maximum_inverse_integer_error']}. "
        "No clipping of unsupported pixels occurs. Proposed RGB decoding reverses those recovered "
        "channels, as in inverse Caffe/BGR preprocessing. This is numerical inference; the "
        "archive "
        "README does not authenticate channel semantics. A plausible image is not proof of "
        "color order.",
        "",
        f"Encoding documentary verification: {quality['encoding_verified']}; "
        f"evidence: {quality['encoding_evidence'] or 'not supplied'}. "
        "Default training refuses unresolved encoding/source provenance. Explicit "
        "provisional training "
        "must acknowledge the assumptions. Obtain publisher preprocessing code or a paired "
        "original "
        "image before presenting the decoding as verified.",
        "",
        "## Leakage controls and frozen test",
        "",
        f"Screening summary: {quality['screening']}.",
        "",
        "Exact array/recovered-pixel hashes, optional supplied source evidence and "
        "full/center/corner "
        "grayscale crop fingerprints form conservative leakage components. Grayscale is "
        "invariant to "
        "the unresolved RGB/BGR reversal. Near matches remain candidates, never human "
        "review decisions. "
        "Low-information views are not used as identity evidence; exact hashing still "
        "applies. The "
        "fixed crop screen cannot rule out arbitrary crops, rotations or shared physical sources.",
        "",
        "Every original test row stays in the original test partition, even if invalid. All "
        "development "
        "members connected to it are conservatively quarantined. Exact conflicting-label "
        "training rows "
        "are quarantined, never relabeled. Remaining official training components are assigned to "
        "train/validation with a seeded stratified group heuristic. Unknown source groups "
        "stay blank.",
        "",
        f"Prepared fold sizes: {quality['prepared_split_counts']}. "
        f"Prepared class counts: {quality['prepared_class_counts']}. "
        f"Dispositions: {quality['dispositions']}. "
        f"Official test membership preserved: {quality['official_test_preserved']}.",
        "",
        "Component grouping can over-quarantine false-positive visual candidates. That "
        "tradeoff is "
        "explicit; there is no claim of authenticated source identity or complete leakage "
        "freedom. "
        "Review evidence is exported, not fabricated. A changed policy/evidence file "
        "creates a new "
        "immutable release with the same original official test indices.",
        "",
        "## Readiness",
        "",
        f"Technical training readiness: {quality['technical_training_ready']}. "
        f"Blockers: {quality['blockers']}. Complete supplied source evidence: "
        f"{quality['provided_source_evidence_complete']}.",
        "",
        quality["scientific_status"],
        "",
        "No real model was trained or evaluated on test data. This dataset is not paired with "
        "RC1841 masks. Its target is visible damage level in pixel/object/component imagery, "
        "not whole-building safety, load capacity, physical damage area or an engineering "
        "decision.",
        "",
        "## Provenance and source references",
        "",
        "Archive README supplies exact numeric registration; original README/license/NPY bytes "
        "are retained in payload/. Every row retains its partition/index and hashes. See "
        "manifest.jsonl, "
        "manifests/, issues.csv, quarantine.csv, near_review.csv, leakage_edges.json and "
        "quality.json.",
        "",
        "[PEER task definitions](https://apps.peer.berkeley.edu/phi-net/detection-tasks/) "
        "describe qualitative component/pixel/object damage levels. "
        "[Official download page](https://apps.peer.berkeley.edu/phi-net/download/) "
        "identifies Task 7. "
        "[Keras VGG "
        "preprocessing](https://keras.io/api/applications/vgg/#preprocess_input-function) "
        "explains the generic Caffe convention; it does not establish that this archive used it.",
    ]
    path.write_text("\n".join(lines) + "\n")
