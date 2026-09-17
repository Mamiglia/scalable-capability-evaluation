# Results analysis: the forecasting task across protocols

**Status:** Revised 2026-09-15 (late) after the condition set was fixed. Implementation follows the
plan written from this spec; Tasks 1–3 of the earlier plan (loading, matching) stand.

**Goal:** One table and two figures for the results section: every capability-estimation
protocol's forecast error on the semester forecasting task, per domain and semester, against
the easy-item prediction baseline and an accuracy oracle, using only what a protocol may legitimately see.

## The forecasting task

Fixed by the reference freeze and not changed here. For each domain (science, math, coding,
cyber) and each semester task (2024-I ... 2026-I), the previous semester's 8 models are the
references with known θ\*, the 8 models of the semester are the targets. A protocol sees:

1. the correct repeat-0 transcripts of every reference and target on the frozen train set
   (100 saturated EASY items, `<semester>_train` in `items.csv`);
2. the references' θ\*;
3. for judge protocols, the judge's verdicts or ratings on those transcripts, including the
   reference-vs-reference pairs and the references' ratings.

It never sees a target's θ\*, its HARD responses, or its failures on the train items. Every
fit, calibration and normalisation statistic is computed on the references only.
The metric is the mean squared error of θ̂ against θ\* over the 8 targets, per semester and
pooled over semesters (one scale, so pooling is valid), and pooled over domains.

The frozen tables carry no task-specific θ\* for models older than the references, so the
historical cohort is empty and every fit uses the 8 references. The analysis reports this
rather than substituting labels.

## Conditions

| Name | Input | Fit on the references | Forecast for a target |
| --- | --- | --- | --- |
| easy-item prediction (`easy_items`, the baseline) | the references' accuracy on the saturated train items, which is 1 for every model, so the regression of θ\* on it reduces to its intercept: the references' mean θ\* | mean θ\* | that constant |
| pairwise judge (`pairwise`) | verdicts on every (target, reference) and (reference, reference) pair on shared train items, both orders | conversion A `linear`: per bench, logit of pair win rate = k_b (θ_t − θ_r) + noise, k_b and τ_b on the reference pairs (JudgeLinear). Conversion B `calibrated`: Bradley-Terry with per-bench discrimination and position bias on the reference pairs, then shrinkage by the judge's leave-one-out error (JudgeCalibrated). Toggle `PAIRWISE_CONVERSION`, both reported. | A: inverse-variance weighted least squares under the references' N(mean, var) prior. B: 1-D maximum likelihood, shrunk to the frontier prior. |
| direct judge (`direct`) | 1–10 ratings of every correct train transcript, subject reasoning shown | OLS θ\* ~ mean rating over the model's transcripts. Toggle `DIRECT_NORMALISATION`: `raw` ratings, or `item_z`: each rating z-scored per (bench, item) with mean and sd taken over the references' ratings of that item (sd 0 → z 0), to remove item difficulty from the ratings. Both reported. | the line applied to the target's mean (normalised) rating |
| surface-feature ridge (`glm_regex`) | 12 features per transcript: 3 length and 9 regex term rates, averaged per model over its correct train transcripts (the existing `glm_regex` protocol) | standardised ridge, penalty by leave-one-out over the references | ridge prediction |
| embedding ridge (`embedding`) | Qwen3-Embedding-8B vectors from the HF dataset LASR-G5/transcript-embeddings: per model, the mean of its correct train transcripts' L2-normalised vectors. Toggle `EMBEDDING_VARIANT`: `full` (reasoning included, what a judge shown reasoning sees) or `answer` (reasoning stripped). | ridge, penalty by leave-one-out over the references | ridge prediction |
| oracle (`accuracy_hard`, a reference line, not a condition) | mean accuracy over every repeat on this task's HARD items, the responses θ\* was fit on | OLS θ\* ~ accuracy | the line applied to the target |

No plain Bradley-Terry, no length-only or RST tier, no locally computed embedding: those are
challenge or appendix material, not core conditions.

### Judges

Judges are the semester's top-3 references. The core condition is subject reasoning shown
where it was judged, else a reasoning-shown run on a smaller train set (`-<k>pq`), else reasoning
hidden (2024 subjects have no reasoning). Judge protocols are summarised two ways, both
computable without target labels:

- **best judge on references**: the judge whose leave-one-out error on the references is
  lowest that semester. Pairwise: each reference re-estimated from the other references'
  pairs with the conversion calibrated without it. Direct: each reference forecast from the
  other references' calibration (and normalisation statistics) without it.
- **mean over judges**: the mean of the three judges' MSE, with the per-judge values kept as
  points in the figures (the replication unit is the judge).

Judging is never launched by the analysis. Judge runs are read from `logs/`; what a campaign
still lacks is counted and reported in a `missing` column. The cyber pairwise campaign is
finishing and its runs join the table when they land in `logs/`.

## Data locations

Large inputs live on the T7 volume; the repository holds the frozen tables and pointers.

- Transcript logs: `/Volumes/T7/small-judge-remote/data`, a pointer clone of LASR-G5/benchmarks
  at the pinned commit with only the 455 frozen reference/target logs pulled. `SMALL_JUDGE_DATA`
  points `reference.tables.DATA` there; `data/meta` stays in the repository checkout.
- Embeddings: `/Volumes/T7/small-judge-remote/transcript-embeddings`, the 910 `.npz` files of
  those logs (`<log path minus .eval>.<full|answer>.npz`, arrays `ids, epochs, correct,
  embedding[n, 4096] float16`). `SMALL_JUDGE_EMBEDDINGS` points the reader there.
- Judge logs: `logs/` on the laptop (`judge/`, `direct/`, `direct_reasoning-hidden/`; `legacy*/`
  kept as pointers). LFS pulls are always scoped; never a bare `git lfs pull`.

## Module layout

Reusable steps live in `src/analysis/`, one file per step, importable by challenge analyses;
the runner that chains them lives in `scripts/analyses/`. Nothing in `src/` imports `scripts/`.

```
src/analysis/
  loading.py       (done) semester tasks; verdict and rating tables from logs/
  matching.py      (done) restrict judge tables to the frozen train items, dedupe, count unjudged
  baselines.py     easy-item prediction; accuracy oracle on HARD items
  embeddings.py    embedding ridge reading the .npz files (TabularRegression subclass)
  judges.py        pairwise conversions (linear | calibrated) and direct calibration
                   (raw | item_z) over matched tables, plus the reference leave-one-out error
  calibration.py   every signal -> θ̂: forecasts table (one row per target x protocol x judge
                   x condition x variant) with each judge run's reference LOO error
  stats.py         MSE with a bootstrap interval over targets; pooled rows; judge summaries
                   (best judge on references, mean over judges); skill relative to the
                   easy-item prediction and the oracle
  plots.py         forecast dot plot, bar plot, semester tables
  challenges/      one module per extra analysis (truncation, hidden reasoning, RST tier, ...)
scripts/analyses/results.py   the runner
```

The runner is a cell script: `# %%` separators, one cell per step, each cell runnable on its
own in PyCharm or VS Code after the cells above it (state lives in module-level variables, no
`main()`), and the whole file runs top to bottom as a script. Options (domains, semesters,
output dir, the three toggles) are constants in the first cell.

Every function carries a docstring with what it computes, its inputs (types, index and
columns of tables) and its output (shape, index, columns, NaN meaning); non-obvious steps
are commented inline.

## Metrics

Every condition is scored per (domain, semester) on the 8 targets, in three versions:

- **raw MSE**: mean squared error of θ̂ against θ\*.
- **delta**: MSE minus the easy-item prediction's MSE on the same targets (negative = better than the
  baseline).
- **z**: the delta divided by the standard deviation of its paired bootstrap distribution
  (targets resampled with replacement, 2,000 draws; the same draw scores both the condition
  and the easy-item prediction, so the pairing is kept). A z of −2 means the improvement over the
  easy-item prediction is two bootstrap standard deviations wide. NaN when the paired errors do not
  vary across targets.

Pooled rows (semester "all" per domain, domain "all") use the same definitions over the
pooled targets. Judge protocols enter the per-task metrics as the best judge on references
(one judge per task); the mean over judges is reported in the score tables only, because its
rows are not one forecast per target.

## Outputs

`analysis/results/` (gitignored):

- `forecasts.csv`: domain, semester, protocol, variant, judge, condition, judge_loo_mse,
  model, θ̂, θ\*, missing. `variant` is the conversion or normalisation (`linear`,
  `calibrated`, `raw`, `item_z`, `full`, `answer`; `-` otherwise).
- `scores.csv`: n, MSE, bootstrap 95% interval, missing, per domain × semester × protocol ×
  variant × judge × condition; plus semester "all" per domain and domain "all" per protocol;
  judge summary rows (best judge on references, mean over judges) with condition "core".
- `metrics.csv`: per (domain, semester) and pooled: mse, delta, z for every condition
  (variants included, judge = best on references).
- `tables/<metric>_<domain|pooled>.csv`: semester × the five conditions (easy-item prediction,
  length+regex ridge, embedding ridge, direct judge, pairwise judge) for the chosen
  variants; pooled = mean over the domains that have that semester, with an "all" row.
  Every combination of metric and domain is written so the best can be picked by hand.
- `forecasts_<condition>.svg`: one panel per domain, semesters along x; within a semester the
  targets sorted by θ\*, black = true θ, orange = forecast, dotted connectors (the layout of
  the reference figure). Drawn for the condition with the lowest pooled MSE, plus a
  one-semester version (`forecasts_<condition>_<semester>.svg`) of the latest semester.
- `bars_<metric>.svg`: four bars (length+regex ridge, embedding ridge, direct judge,
  pairwise judge) at the mean of the per-(domain, semester) metric, jittered circles for
  the individual (domain, semester) values, the easy-item prediction as a dashed line (its raw MSE, or 0
  for delta and z).

Figures follow the Nature-panel spec (8pt Arial, 1pt lines, no boxes, legend outside).

## Testing

One compact `tests/test_analysis.py`: easy-item prediction and oracle on the toy task from
`test_protocols.py`; direct calibration raw vs item_z on a hand-built rating table; the
reference LOO error on a toy verdict table; pooled scores, judge summaries and skill on a
hand-built forecasts frame. The runner is checked end to end on science.

## Out of scope

Launching judging, refitting θ\*, the challenge analyses themselves (the package exists;
each is its own plan), the historical cohort, the RST structure tier, plain Bradley-Terry.

## Naming

The baseline is called **easy-item prediction** (`easy_items`) in every table, figure, column
and docstring: it is the prediction one can make from the models' accuracy on the easy
(saturated) items. "Intercept" describes the arithmetic, not the condition, and is not used.
