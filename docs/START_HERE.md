# Start here

We are building a tool to understand visible damage in reinforced concrete images.
The first task is to understand the data reliably. A future model can only learn
what its annotations actually represent.

An **image** is the original photograph. An **annotation** says which parts of the
photograph belong to a damage category. In this dataset, the annotations are JSON
files containing polygons: lists of points around a region. A **mask** is an image
we generate from those polygons, with one category number for each pixel.
An **overlay** places display colors on the photograph so we can check alignment.

The supplied files are not severity ratings. For example, a polygon labeled
`Structural crack` describes a source category; it does not give us a validated
building safety score. We keep the source names and avoid creating new targets.

Task 01 does four jobs:

1. Inspect and extract the two ZIP files without modifying the originals.
2. Pair photographs with JSON files and report missing, corrupt, or invalid data.
3. Convert valid polygons to masks and count images, dimensions, and category pixels.
4. Draw plots and a gallery so a person can examine the results.

Start with the [README setup commands](../README.md#setup-and-complete-task-01-command-sequence).
Keep the original archives in `data/raw/rc1841/`. Extracted source files go under
`data/interim/rc1841/`. Generated reports go under `outputs/task01/`. Git ignores
both data and outputs because they are large, local artifacts.

Read `validation.json` and `issues.csv` before reading charts. A failed validation
can still produce useful reports for the valid subset. In the supplied archives,
two annotations contain negative coordinates; the code reports these rather than
silently adjusting them. This is a data-quality finding, not necessarily a bug.

The [Task 01 lesson](learning/task_01.md) explains the concepts, every module, each
command, how to interpret the reports, and what to do when a check fails.
