# RC1841: evidence, annotations, and limitations

## Provenance

Dataset: **Reinforced concrete structure segmentation dataset 1841**, version 1,
contributed by Jiehui Wang. The [publisher landing page](https://data.mendeley.com/datasets/yfhwfcrfmk/1)
identifies DOI `10.17632/yfhwfcrfmk.1`, publication date 2 February 2026, and a
CC BY 4.0 license. Consult that page for attribution and its requested related-paper
citations when publishing results. The project does not redistribute the archives.

The documentation and the local archives were inspected on 28 October 2026.
The publisher describes eight categories with named display colors but does not
provide exact RGB triplets on that page. The supplied archive uses polygon JSON
annotations and somewhat different category names. Both pieces of evidence matter;
neither establishes an independent ordinal severity target.

## Actual local archive structure

```text
data/raw/rc1841/
├── imagedata.zip        # 1,841 JPG members; 149,077,716 uncompressed bytes
└── labeldata.zip        # 1,841 JSON members; 14,139,885 uncompressed bytes

data/interim/rc1841/
├── images/
│   ├── 10.jpg
│   ├── corrosion_103.jpg
│   └── ...
├── annotations/
│   ├── 10.json
│   ├── corrosion_103.json
│   └── ...
└── provenance.json
```

Inspection found 1,841 unique matching stems, no discovery mismatches, and
2,764 polygon shapes across the source annotations. Archive hashes are recorded
by `inspect` in `outputs/task01/archives.json` and copied beside extracted data
in `provenance.json`. Counts here describe the supplied archives, not a claim that
every image has passed decoding and annotation validation.

Images and JSON are paired by exact filename stem, not ZIP order. Wrapper folders
are supported. Repeated stems are errors, even in separate folders. The JSON's
`imagePath` basename must additionally match the paired image filename.

## Source labels versus publisher terminology

The following source spellings and counts were obtained directly from every JSON
in the supplied archive, before excluding invalid geometry. Project IDs remain
stable in `labels.py`; they are not published dataset IDs or a severity ranking.

| Project ID | Verbatim archive label | Source polygons | Relationship to page wording |
| --- | --- | ---: | --- |
| 0 | Background | Not annotated as polygons | Complement of all drawn regions |
| 1 | Microcrack | 548 | Page uses non-structural crack; equivalence is unverified |
| 2 | Structural crack | 923 | Name corresponds to the page category |
| 3 | Minor spalling | 207 | Name corresponds to the page category |
| 4 | Moderate spalling | 146 | Name corresponds to the page category |
| 5 | Delamination | 201 | Page groups major spalling/delamination; exact scope is unresolved |
| 6 | Rebar corrosion | 629 | Page describes exposed corroded reinforcement with spalling |
| 7 | Concrete crushing | 51 | Name corresponds to the page category |
| 8 | Structural deformation | 59 | Name corresponds to the page category |

The project does not rename `Microcrack` to non-structural crack, merge classes,
or turn minor/moderate/category names into a severity score. Class IDs have no
ordinal meaning. A future severity task requires a defensible target definition,
independent expert labels, and validation appropriate to that target.

## Annotation schema

The source follows a LabelMe-style format. This is a simplified illustration:

```json
{
  "imagePath": "sample.jpg",
  "imageWidth": 640,
  "imageHeight": 360,
  "shapes": [
    {
      "label": "Rebar corrosion",
      "shape_type": "polygon",
      "points": [[20.0, 30.0], [80.0, 30.0], [80.0, 70.0]]
    }
  ]
}
```

`imageWidth` and `imageHeight` must equal decoded stored-image dimensions. Points
are `[x, y]`: x increases to the right, y increases downwards. Coordinates must
be finite numbers, and polygons need at least three distinct points and nonzero
signed area. Only `polygon` shapes with the eight exact known labels are supported.
`group_id`, text, version, and flags do not create additional segmentation classes.
Embedded `imageData` is never decoded; `imagePath` is never opened or followed.

Boundary coordinates exactly equal to width/height are treated as edge coordinates
and clipped to the last pixel. Negative or larger coordinates fail. The parser
does not repair geometry or provide a complete self-intersection/topology analysis;
visual inspection remains necessary.

## Derived masks and colors

Each valid pair generates an 8-bit grayscale PNG. Pixel values are integer class
IDs `0..8`; zero is the region not covered by any accepted polygon. It means
unannotated/background, not independently verified undamaged concrete. Do not open
these masks as photographs and interpret the darkness as damage intensity.

Pillow fills polygons at the original resolution, with inclusive boundary pixels.
Fractional coordinates follow Pillow's rasterizer, so edge coverage can differ from
other tools. When regions overlap, the later JSON shape overwrites earlier labels.
The number of multiply covered pixels is reported, including same-label overlaps.
Thus final pixel presence and original polygon presence need not be identical.
Polygon holes and general instance segmentation are not modeled in Task 01.

**RGB colors are only a project display palette.** The gallery maps IDs to distinct
colors defined in `labels.py` and records a legend. These are not verified publisher
RGB values. The source has no raster masks to check for unexpected colors; reports
explicitly mark that check as not applicable. Unexpected source label strings and
unexpected generated class IDs are rejected. Source raster annotation files, if
encountered in another archive version, are reported as unsupported rather than
silently decoded using guessed color names.

## Validation findings and reporting policy

Annotation-only checks against declared JSON dimensions found negative coordinates:

- `crack146`: an x coordinate approximately `-0.339`.
- `damage794`: an x coordinate approximately `-38.5`.

These findings were initially checked without installing dependencies. Subsequent
Task 01 analysis found 1,810 valid pairs and 31 exclusions under its stored-frame
policy; the additional 29 are EXIF dimension mismatches addressed explicitly in
Task 02 below. `analyze` excludes invalid pairs, writes reports, and exits with
status 2. Do not silently clamp these
negative coordinates or edit raw data; resolve them with a documented curation
policy or a separately versioned correction process in a later task.

Exact-byte duplicate checks include extracted files within each role. Decoded RGB
pixel hashes also detect paired photographs with identical pixels but different
file metadata/encoding. Perceptually similar photographs, resized copies, crop
families, or shared source structures are not automatically identified. Duplicate
warnings do not delete examples; all valid duplicates remain in EDA counts.

Statistics and dimensions use valid pairs only. Pixel fractions pool all valid
pixels, so high-resolution images contribute more. `images_with_annotation` counts
images with source polygons; `images_with_pixels` counts images where a class remains
visible after overlap resolution. An image can contribute to multiple categories.
No severity frequencies are calculated.

## Limits on interpretation

- Source descriptions are not a substitute for a detailed annotation manual.
- Annotations and class terminology can be inconsistent; masks need visual review.
- Stored EXIF orientation is not applied; non-default values produce a warning.
- Images vary in resolution and capture conditions. Pixel area is not physical area
  without calibrated scale information.
- Background, category names, or pixel fractions cannot establish structural safety.
- Physical source/structure independence is not verified. Task 02 supplies splits
  isolated by known evidence; unknown-source leakage remains unresolved.
- Task 01 has no trained model, severity estimator, dataset download, or augmentation.

## Task 02 curation policy and evidence

Task 02 audits the actual decoded dataset and the completed Task 01 reports. Task 01's
31 invalid pairs comprise 29 stored/declared dimension mismatches and two samples with
negative coordinates. All 29 mismatches have EXIF orientation 6, stored dimensions
4032×3024 and declared dimensions 3024×4032. Applying the EXIF transform to a derivative
matches the declared coordinate frame; annotation vertices are not rotated or scaled.
This evidence supports the configured transform but still warrants expert visual review.

The supplied photographs comprise 1,840 RGB images and one CMYK image. Task 02 fully
decodes inputs and exports lossless RGB PNG derivatives. CMYK conversion is logged;
it is not a color-calibration claim. Missing files, unreadable files, ambiguous stems,
unknown labels, unsupported shapes/flags, unresolvable dimensions, transparent pixels
without a compositing convention, and ambiguous EXIF frames are not silently accepted.

The default cleaning policy is conservative:

| Finding | Default disposition and reason |
| --- | --- |
| Exact consecutive duplicate/repeated closing vertex | Remove explicitly; boundary and all raster pixels must remain unchanged |
| Coordinate exactly at width/height | Accept source edge convention; map to last pixel as in Task 01 |
| Negative/oversized coordinate | Exclude the entire sample; tolerance clamp defaults to zero |
| Nonfinite/non-numeric point or malformed polygon | Exclude the sample; no type coercion |
| Non-simple ring, retraced edge, nonadjacent edge contact, zero-area ring | Exclude; no buffer, convex hull, guessed topology repair, or polygon dropping |
| Geometry degenerating after edge-to-pixel mapping | Exclude; the actual pixel-frame ring must also be valid |
| Same-label polygon overlap | Accept as a semantic union and report overlapping pixels |
| Cross-label raster overlap | Exclude; inclusive shared raster boundaries count conservatively as conflicts |
| Empty annotation/mask | Exclude by default; no verified undamaged-negative meaning is assumed |
| Exact image with disagreeing/unusable annotations | Exclude every member; no majority vote or arbitrary representative |
| Exact image and identical mask | Keep the lexicographically smallest source ID by default; record redundant IDs |
| Heuristic near duplicate | Retain, record evidence, place the full connected group in one split |

All shape indices and affected IDs appear in `decisions.csv`. One invalid shape excludes
the entire sample so ground truth is not silently made incomplete. Shape checks continue
after finding an invalid shape. Mask checks are not performed on uninterpretable geometry;
`sample_audit.csv` explicitly marks check coverage. A `fix` decision describes an evaluated
policy action; `retained_samples` in the quality report distinguishes fixes actually exported
from candidates subsequently excluded for another reason. Exclusion reasons can overlap.

You may opt into a nonzero coordinate-clamp tolerance or last-shape precedence in the
configuration, but these are scientific policy changes, recorded in a new dataset version.
They should follow annotation review, not attempts to improve validation/test metrics.

### Duplicate and source evidence

Exact comparisons cover original image/JSON bytes, stored and selected-frame RGB pixels,
and canonical ordered polygon geometry with labels and image size. Ring start and winding
do not affect geometry identity; polygon order is preserved because of possible precedence.
Empty annotations do not establish grouping evidence. Exact annotation geometry is grouped
conservatively, even though identical outlines alone do not prove a common physical source.

Near-image candidates must satisfy all configured conditions: 63-AC-bit pHash distance ≤6,
64-bit dHash distance ≤8, RGB thumbnail mean absolute error ≤0.12, and absolute log-aspect
difference ≤0.15. These are transparent heuristic thresholds, not calibrated match probabilities.
Near-annotation candidates are checked within visual candidates by mean foreground-class
IoU ≥0.98 on nearest-neighbor 32×32 mask thumbnails. All candidate edges are retained in
`similarity_edges.csv`. A missed candidate is possible, particularly for crops, rotation,
large lighting changes, or different views of one structure.

No physical structure/source identifiers are supplied in the inspected data. Filename
families such as `damage`, `crack`, or `corrosion_` are not proven source IDs and are not
invented as such. Optional `source_groups_file` configuration accepts a CSV with columns
`sample_id,source_group,evidence`; entries must reference known unique IDs and include
nonempty evidence. These edges join the same conservative connected components.
Excluded nodes remain in the grouping graph so they cannot break known relationships.

Thirty-three photographs also carry camera/time metadata. The 29 EXIF-rotated images
include a close timestamp sequence with the same camera make/model. Task 02 links
session candidates using a configurable 600-second maximum adjacent timestamp gap.
DateTimeOriginal is preferred; DateTime is a weaker fallback that may indicate editing.
Without serial numbers, camera make/model is not device-unique. This conservative
temporal grouping reduces known related-sample risk while leaving physical identity
uncertain; invalid/missing metadata creates no group.

### Splits, statistics and readiness

Fractions default to 70/15/15 with seed 42. A deterministic whole-group greedy search
with 24 restarts and coordinate descent balances image counts and multi-label image
presence. Class coverage is prioritized; no oversampling, class dropping, augmentation,
or model feedback is involved. Pixel-level distributions are reported rather than
silently optimized. Fewer than three groups carrying a class makes all-three coverage
infeasible; other unmet coverage is reported as a heuristic limitation, not proved impossible.

The test split is created once for a named content-addressed version and checksummed.
Reuse that version and do not tune policies, grouping thresholds, or model choices using
test performance. Code/config/input/environment changes create a distinct version and
must not be used as an excuse to search for a favorable test split.

Class distributions are calculated for the curated population and separately for each
split, including background. Reports include multi-label combinations, pairwise occurrences,
rare-class flags, excluded IDs, dimensions, source modes, and foreground pixel imbalance.
Rarity is descriptive; no category is discarded because it is rare. Background remains
unannotated complement rather than verified healthy concrete. No severity labels are added.

The quality report distinguishes verified technical consistency from scientific validity.
`CONDITIONAL_RESEARCH_USE` means accepted artifacts and known grouping/partition checks passed.
It does not mean all annotations are expert-correct, every physical source is independent,
or the dataset is globally leakage-free. Read the generated report and remaining limitations
before making benchmark or generalization claims.

### Verified release snapshot: 28 October 2026

Version `rc1841-curated-v1-d34771dbb662406a` retains **1,617** of **1,841** original pairs;
**224** unique samples are excluded. All source files were readable and all 3,682
extracted member files matched the unchanged archives. Of 29 EXIF alignment candidates,
27 survive geometry checks; `crack22919` and `crack22920` do not. `corrosion_103` is the
single CMYK conversion. Seventeen of 26 redundant-vertex cleanup candidates survive all
checks. Five retained samples have accepted same-label overlap.

Nonexclusive exclusion counts are 171 non-simple source/pixel rings, two out-of-bounds
annotations, 26 cross-class overlaps, 24 exact-image annotation-conflict members, and
four members of exact-image groups containing unusable ground truth. All non-simple
rings are also flagged by the pixel-ring check, and some duplicate/out-of-bounds
findings share IDs; do not add these counts to estimate unique exclusions.

For disjoint accounting: 171 non-simple-ring samples, one additional out-of-bounds
sample, 26 cross-class-overlap samples, 24 exact-image annotation conflicts, and two
additional partners excluded because an identical image's annotation is unusable sum
to 224. Every affected source ID and complete reason set is in `exclusions.csv`.

| Source label | Curated images | Pixels | Pixel fraction | Train / validation / test images |
| --- | ---: | ---: | ---: | --- |
| Background | 1,617 | 640,762,443 | 82.157503% | 1,132 / 243 / 242 |
| Microcrack | 267 | 1,227,168 | 0.157345% | 187 / 40 / 40 |
| Structural crack | 730 | 14,304,195 | 1.834060% | 511 / 109 / 110 |
| Minor spalling | 78 | 249,272 | 0.031961% | 54 / 12 / 12 |
| Moderate spalling | 112 | 8,533,026 | 1.094091% | 78 / 17 / 17 |
| Delamination | 152 | 16,651,509 | 2.135029% | 106 / 23 / 23 |
| Rebar corrosion | 486 | 83,981,983 | 10.768031% | 340 / 73 / 73 |
| Concrete crushing | 44 | 7,038,306 | 0.902440% | 30 / 7 / 7 |
| Structural deformation | 46 | 7,171,666 | 0.919539% | 32 / 7 / 7 |

There are 779,919,568 total pixels, 252 multi-label images, and approximately 336.91:1
max/min foreground pixel imbalance. Counts are on the final curated population, not
the raw polygon table above. Restoring high-resolution EXIF-aligned images changes
the pooled pixel denominator substantially relative to Task 01's 1,810-image population.

Train/validation/test contain 1,132/243/242 samples and 898/192/189 known-evidence
components, respectively. All eight foreground classes occur in every split, with no
unmet coverage constraints in this run. Across curated samples there are 1,279 components;
the largest contains 27 retained members of the temporal sequence. No component,
direct known-evidence edge, or duplicate manifest ID crosses splits.

The full audited population produces 674 near-image candidate edges, 82 near-annotation
candidate edges, 28 temporal edges, and 14 exact-image pairs detected by each of three
fingerprints. These are overlapping evidence counts, not independent sample counts.
No exact JSON-byte or eligible canonical-geometry duplicates were found. There are no
supplied verified physical source IDs, so unknown-source leakage remains unverified.

All 1,617 saved samples pass complete pixel/hash/label/partition checks. The separate
`verify-curation` command also passes. The test suite passes 89 tests and 21 subtests
with no skips; Ruff passes. The verdict remains **CONDITIONAL_RESEARCH_USE**, with
expert annotation review, terminology resolution, and physical source identification
outstanding. No severity target or model was created.
