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

These findings were checked without installing dependencies. Full image decoding
and dimension comparison remain pending. `analyze` will exclude invalid pairs,
write diagnostic reports, and exit with status 2. Do not silently clamp these
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
- No verified source/structure grouping or clean split is supplied by this project.
  Resolve duplicates and grouping before claiming generalization.
- Task 01 has no trained model, severity estimator, dataset download, or augmentation.
