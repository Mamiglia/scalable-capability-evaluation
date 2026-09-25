# Handoff: quick previews of the three results plots

**Goal.** Render every option combination of the three results figures the user asked for, as
PNGs in `analysis/preview/`, so the user can point at the ones to keep. This is a throwaway
preview pipeline, separate from the results runner (`scripts/analyses/results.py`); nothing
here changes `src/analysis/`, the tests, or any data.

**Read first.** `docs/superpowers/specs/2026-09-15-results-analysis-design.md` (the
Conditions, Metrics and Outputs sections) and the two modules the preview reuses:
`src/analysis/stats.py` (`task_metrics`, `one_per_target`, `BASELINE`, `BEST`, `METRICS`) and
`src/analysis/plots.py` (`forecast_figure`, `bar_figure`, `semester_table`, `CONDITIONS`,
`LABELS`). Both are on disk (untracked until their plan tasks commit them); import them as they
are and do not edit them. Another agent is implementing `src/analysis/judges.py` and
`calibration.py` in this checkout right now: do not touch anything under `src/analysis/` or
`tests/`.

## Hard rules

- Never download anything: no `git lfs pull`, no `git submodule update`, no
  `huggingface_hub`. A previous agent crashed this laptop by pulling 37 GB. Everything the
  preview needs is a CSV on disk.
- Run from the repository root with `PYTHONPATH=src uv run python ...`. Loading tasks prints
  `[WARNING] ... excluding historical models` lines; expected.
- Figures follow the Nature-panel spec already set in `plots.py` (8pt Arial, 1pt lines, no
  top/right spines, legend outside, no titles on the canvas); panels ≤ about 2 in tall; check
  every PNG for overlapping tick labels before listing it.
- Names describe what a thing is (`bars_delta_best-judge.png`), never option codes.
- Put the script at `scripts/analyses/preview_plots.py` as a `# %%` cell script (settings
  cell, load cell, one cell per figure family, index cell). Commit nothing; leave the script
  and outputs on disk for the user.

## Input table

A forecasts table with columns `domain, semester, protocol, variant, judge, condition,
judge_loo_mse, model, theta_hat, theta, missing` (see the spec's Outputs). Use, in this order
of preference:

1. `analysis/results/forecasts.csv`, written by the results runner once Tasks 6–10 land: the
   new conditions (`intercept`, `accuracy_hard`, `glm_regex`, `embedding` × {full, answer},
   `direct` × {raw, item_z}, `pairwise` × {linear, calibrated}), all four domains.
2. Until then, `analysis/results-smoke/forecasts.csv` (science only, the earlier protocols),
   adapted like this so the plotting code accepts it:

```python
f = pd.read_csv("analysis/results-smoke/forecasts.csv")
f = f[f.protocol.isin(["reference_mean", "accuracy_all", "glm_regex", "glm_length", "pairwise", "direct"])]
f = f.replace({"protocol": {"reference_mean": "intercept", "accuracy_all": "accuracy_hard", "glm_length": "embedding"}})
# no reference LOO error in the old table: rank judges by name so "best judge on references" picks one deterministically
f = f.assign(variant="-", judge_loo_mse=np.where(f.judge == "-", np.nan, f.judge.rank(method="dense")))
```

State in the index which input was used. The smoke table's values are placeholders (old
Bradley-Terry pairwise, old direct OLS, length GLM standing in for the embedding); only the
layouts matter until the new table exists.

Toggles for the new table: `VARIANTS = {"pairwise": "linear", "direct": "raw", "embedding": "full"}`
for the default previews, plus one extra set per alternative variant (`calibrated`, `item_z`,
`answer`) for the bar plot and tables only.

## Figure 1: true θ vs forecast dots (`plots.forecast_figure`)

Layout as the user's reference figure: one panel per domain, semesters along x, targets within
a semester sorted by θ\*, black = true θ, orange = forecast, dotted connectors.

Render every combination of:

- condition: `glm_regex`, `embedding`, `direct`, `pairwise` (judge rows = best judge on
  references, which `forecast_figure` already does);
- semesters: all, and the latest only (`semesters=["2026-I"]`);
- domains: every domain as panels (the function's default); each single domain (filter the
  table to that domain before calling); and one "pooled" panel per semester where every
  domain's targets are concatenated and sorted by θ\* together (build this by setting
  `domain="all"` on a copy of the table before calling, so the function draws one panel).

File names: `forecasts_<condition>_<all|2026-I>_<domains|science|math|coding|cyber|pooled>.png`.

## Figure 2: bars against the intercept (`plots.bar_figure`)

Four bars (length+regex, embedding, direct judge, pairwise judge) at the mean over
(domain, semester) of the metric; jittered circles for the individual (domain, semester)
values; the intercept as a dashed line.

Render every combination of:

- metric: `mse`, `delta` (MSE − intercept MSE on the same targets), `z` (paired bootstrap, see
  the spec) — all three come from `stats.task_metrics(f)`;
- a fourth metric the user wants to compare, `z_baseline`: (condition MSE − intercept MSE) /
  sd of the intercept's own bootstrap MSE distribution over resampled targets (2,000 draws,
  seed 0), i.e. the condition's distance from the baseline in units of the baseline's
  sampling spread, not paired. Compute it in the preview script from `stats.one_per_target(f)`
  with the same grouping as `task_metrics` (per domain × semester), add it as a column named
  `z_baseline`, and pass `metric="z_baseline"` — `bar_figure` asserts `metric in METRICS`, so
  for this one draw the bars with a local copy of `bar_figure`'s body (same styling) rather
  than editing `plots.py`;
- judge rows: best judge on references (what `task_metrics` gives), and mean over judges
  (compute per-judge metrics by running `task_metrics` on the table with each judge's rows
  alone, then average the metric per (domain, semester, protocol); local helper);
- points coloured by domain (one colour per domain, legend outside) versus all black.

File names: `bars_<metric>_<best-judge|mean-judges>_<black|by-domain>.png`.

## Figure 3: semester × condition tables (`plots.semester_table`)

Rows = semesters plus "all", columns = intercept, length+regex, embedding, direct judge,
pairwise judge.

Render every combination of metric (`mse`, `delta`, `z`, `z_baseline`) × scope (pooled over
domains, science, math, coding, cyber). Save each as CSV and as a PNG rendering of the table
(matplotlib `ax.table`, 8pt, no frame, values rounded to 2 decimals, NaN as "–") so it can be
looked at next to the figures.

File names: `table_<metric>_<pooled|science|math|coding|cyber>.{csv,png}`.

## Index

Write `analysis/preview/index.md`: which input table was used, the variant toggles, then one
line per file with its options spelled out. Finish by listing any combination that could not be
rendered and why (e.g. a domain missing from the input table).

## Done when

Every combination above exists as a PNG (and CSV for tables), every PNG was opened and has no
overlapping labels or clipped content, and `index.md` lists them. Report the output directory,
the input table used, and anything skipped.
