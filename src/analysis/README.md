# `src/analysis/`

Turns the per-model `*_post_process.csv` evaluation results in `test_results/` into the benchmark's
summary figure and table, grouped into four quadrants along two axes:

- **Narrative type**: classic puzzles (Blue-Eyed Islanders, Muddy Children, Wise Men) vs. the four
  new story variations (Health Screening, Olympic Games, Safety Inspection, Singing Contest).
- **Inference type**: symmetric vs. asymmetric (see the
  [dataset generation README](../dataset_generation/README.md#the-two-observation-settings) for what
  distinguishes them).

```
            Classic narratives   New narratives
Symmetric        Q1                   Q3
Asymmetric       Q2                   Q4
```

## `quadrant_files.py`

`QUARTERS`: the file manifest mapping each quadrant (`"Q1"`&ndash;`"Q4"`) to the list of
`*_post_process.csv` result paths (one per puzzle/model combination) that belong to it, relative to
the repository root. This is the single place to edit if you add a new model's results, a new
narrative, or move files around - both scripts below import `QUARTERS` and read the paths it lists.

## `compute_quadrant_accuracy.py`

Reads every CSV in `QUARTERS`, computes each model's accuracy per puzzle from the `verdict` column
(`PASSED` &rarr; correct, `FAILED` &rarr; incorrect), groups by `(model, version)`, and renders one
2&times;2 accuracy heatmap panel per model (rows: symmetric/asymmetric, columns: classic/new
narratives), combined into a single figure with a shared colorbar. Each cell also reports the
population variance of the per-instance PASSED/FAILED outcomes pooled across every instance in that
quadrant, shown as "accuracy% (&plusmn;variance)". Panels are also outlined by a qualitative
"reasoning strategy" grouping (`STRATEGY_MAPPING`) describing how each model's accuracy tends to
shift between quadrants in the asymmetric inference case.

```bash
python -m src.analysis.compute_quadrant_accuracy
```

Only models with data in **all four** quadrants are plotted. Output: `Graphs/combined_models_accuracy.png`
(directory created if missing). The version embedded in this repo's documentation lives at
[`figures/combined_models_accuracy.png`](../../figures/combined_models_accuracy.png).

## `compute_model_significance.py`

Reads the same `QUARTERS` manifest and pools every per-instance PASSED/FAILED outcome across all
four quadrants for each of the 5 main-analysis models, then runs a pairwise two-proportion z-test
on overall accuracy between every pair of models (instances treated as independent Bernoulli
trials).

```bash
python -m src.analysis.compute_model_significance
```

Only models with data in **all four** quadrants are compared. Prints a symmetric p-value matrix as
a LaTeX table (paste directly into Overleaf) and writes it to `Graphs/model_significance_pvalues.csv`
(directory created if missing).

## `compute_quadrant_macro_f1.py`

Reads the same `QUARTERS` manifest and computes macro-F1 per model per quadrant: for each puzzle
file, F1 is computed per class (`Yes` / `No` / `I don't know`, restricted to classes actually present
in that file's ground truth, since a queried round is only ever asked once knowledge is possible)
using `model response` vs. `correct response`, with unparseable responses (rejected by
`RESPONSE_FORMAT`) counted as wrong; per-puzzle F1 scores are then averaged within each quadrant.

```bash
python -m src.analysis.compute_quadrant_macro_f1               # human-readable table + LaTeX table
python -m src.analysis.compute_quadrant_macro_f1 --no_latex    # skip the LaTeX output
```

Both outputs are printed to stdout.
All three scripts only depend on the paths listed in `QUARTERS` and the standard result-CSV schema
(`model`, `version`, `model response`, `correct response`, `verdict` &mdash; see
`src.constants.COLUMNS`). To analyze a different set of runs, add/replace the relevant `*_post_process.csv`
paths under the appropriate quadrant key in `quadrant_files.py`; no other code changes are needed.
