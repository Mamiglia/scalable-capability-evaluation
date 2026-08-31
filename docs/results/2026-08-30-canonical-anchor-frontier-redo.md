# Canonical-anchor frontier redo: noncanonical in-sample result

**Status: noncanonical side analysis; current main result for this frontier redo.** The sole
authoritative output is [MAIN_OUTPUT.json](../../judge_results/frontier/eci-semiannual-canonical-anchors/MAIN_OUTPUT.json),
which designates [analysis/rank_regression/](../../judge_results/frontier/eci-semiannual-canonical-anchors/analysis/rank_regression/).
All older anchor/ECI figure roots are discarded/non-main; the archived notice lists each one in
[the preceding side-analysis document](2026-08-28-frontier-eci-side-analysis.md).

## Interpretation limit

This is intentionally an **in-sample descriptive fit**, not a forecast. Each target's Q2/Q3/Q4
truth intentionally helps fit the curve that produces its displayed value. Therefore its RMSEs,
residuals, and figures are neither held-out performance nor evidence that rankings generalize to
future models or other data. Do not call this a forecast or claim frontier ability/actual gaps.

## Protocol

For each judge × benchmark, one Bradley–Terry ranking is fitted from all 26,780 frozen-Q1,
exact-epoch, both-correct verdict rows, capped deterministically at eight item IDs per unordered
pair. Every model joined to that ranking and to the Q2, Q3, or Q4 hard outcomes enters the same
in-sample curve—including all frontier targets. There are no ECI inputs, anchor subsets,
target exclusions, temporal cutoffs, or rolling fits.

The 30 blocks compare two free-intercept, two-parameter candidates:

- linear OLS: `accuracy = intercept + slope × BT ability`;
- sigmoid: `accuracy = logistic(intercept + slope × BT ability)`.

The selected method is sigmoid if a linear training value leaves `[0, 1]`, or if its sigmoid
training RMSE is strictly lower; otherwise it is linear. Coefficients, both RMSEs, equations, and
exact joined training IDs are in [fits.csv](../../judge_results/frontier/eci-semiannual-canonical-anchors/analysis/rank_regression/fits.csv).
The exact display/export label is `Rank–accuracy fit (in-sample)`.

## Results

All 10 rankings and 30/30 Q2/Q3/Q4 fits succeeded. They join 702 training rows; 23 fits select
sigmoid and 7 linear. Linear candidate RMSE spans 0.0896–0.2729 (median 0.1717), and sigmoid
candidate RMSE spans 0.0743–0.2775 (median 0.1686). The following are mean **selected training
RMSEs**, not predictive error:

| Benchmark | Quartile | Mean selected RMSE | Selected methods |
|---|---:|---:|---|
| GPQA | Q2 | 0.1296 | 5 sigmoid |
| GPQA | Q3 | 0.2174 | 3 linear, 2 sigmoid |
| GPQA | Q4 | 0.1582 | 3 linear, 2 sigmoid |
| SWE-mini | Q2 | 0.2128 | 5 sigmoid |
| SWE-mini | Q3 | 0.2063 | 1 linear, 4 sigmoid |
| SWE-mini | Q4 | 0.0904 | 5 sigmoid |

| Benchmark | Judge | Q2 method / RMSE | Q3 method / RMSE | Q4 method / RMSE |
|---|---|---|---|---|
| GPQA | DeepSeek V4 Flash | sigmoid / 0.1363 | linear / 0.2418 | linear / 0.1678 |
| GPQA | Gemini 2.5 Pro | sigmoid / 0.0925 | sigmoid / 0.1989 | sigmoid / 0.1545 |
| GPQA | GPT-4o 2024-11-20 | sigmoid / 0.1838 | linear / 0.2690 | linear / 0.1737 |
| GPQA | GPT-5.2 | sigmoid / 0.1055 | linear / 0.1614 | sigmoid / 0.1431 |
| GPQA | o4-mini | sigmoid / 0.1301 | sigmoid / 0.2160 | linear / 0.1519 |
| SWE-mini | DeepSeek V4 Flash | sigmoid / 0.1565 | sigmoid / 0.1693 | sigmoid / 0.0743 |
| SWE-mini | Gemini 2.5 Pro | sigmoid / 0.2646 | sigmoid / 0.2040 | sigmoid / 0.0837 |
| SWE-mini | GPT-4o 2024-11-20 | sigmoid / 0.2775 | sigmoid / 0.2448 | sigmoid / 0.0988 |
| SWE-mini | GPT-5.2 | sigmoid / 0.1721 | linear / 0.2204 | sigmoid / 0.1102 |
| SWE-mini | o4-mini | sigmoid / 0.1934 | sigmoid / 0.1932 | sigmoid / 0.0848 |

All 240/240 frontier judge × benchmark × quartile × target rows have both actual and fitted
values. There are no frontier BT-ability gaps under this all-Q1 design. Gemini 3 Pro SWE-mini
uses the exact recovered source without imputation: 46/50 frozen items, with Q1/Q2/Q3/Q4 counts
of 12/13/12/9; the Q2/Q3/Q4 accuracies are 0.846/0.417/0.111.

## Valid artifacts and QA

- [summary.json](../../judge_results/frontier/eci-semiannual-canonical-anchors/analysis/rank_regression/summary.json),
  [checkpoint-main.json](../../judge_results/frontier/eci-semiannual-canonical-anchors/analysis/rank_regression/checkpoint-main.json),
  [ranking blocks](../../judge_results/frontier/eci-semiannual-canonical-anchors/analysis/rank_regression/ranking_blocks.csv),
  [BT abilities](../../judge_results/frontier/eci-semiannual-canonical-anchors/analysis/rank_regression/abilities.csv),
  [joined training points](../../judge_results/frontier/eci-semiannual-canonical-anchors/analysis/rank_regression/training_fitted.csv),
  and [frontier rows](../../judge_results/frontier/eci-semiannual-canonical-anchors/analysis/rank_regression/frontier_fitted.csv).
- [Figures](../../judge_results/frontier/eci-semiannual-canonical-anchors/analysis/rank_regression/figures/):
  60 PNG + 60 PDF + 60 editable-text SVG (180 files total), with 20 PNG panels for each of Q2,
  Q3, and Q4. Original-resolution visual QA reviewed both benchmarks, all quartiles, all five
  judges, and line/points variants; it found readable, unclipped, consistent panels.
- Task 5's corrected main analysis passed 211 tests and independent review. Task 6 then ran
  `uv run pytest tests/frontier_eci tests/test_ranking.py -q` → 98 passed; the requested
  `tests/test_frontier_forecast.py` is absent from this checkout. The Task 6 artifact audit also
  recomputed all 30 fits, joins, free-intercept OLS coefficients, and method choices, and checked
  full Q2/Q3/Q4/frontier coverage, the exact label, cap, and editable SVGs.

## Provenance and cost

The frozen Q1 source is 26,780 exact-epoch, both-correct rows with the maximum cap of eight
per unordered pair. The old 79/80 prior forecast-coverage result, anchored calibrators, and
matched ECI analyses are retained provenance only; they are not a property or a limitation of this
complete in-sample rank-regression grid.

The [provider summary](../../judge_results/frontier/eci-semiannual-canonical-anchors/provenance/api_summary.csv)
and [call-level audit](../../judge_results/frontier/eci-semiannual-canonical-anchors/provenance/api_calls.parquet)
contain **3,220 audited paid response records**. Of these, 3,219 rows belong to latest-successful
configs; one belongs to the retained raw local SIGABRT attempt, which was later recovered. Main
verdict ingestion uses latest-success/deduplicated rows. There was one local raw failure event;
`provider_mismatches=0` and zero missing paid-provider fields are narrower provenance-audit
findings, not a claim of zero provider issues. The endpoint snapshots are
[before the original top-up](../../judge_results/frontier/eci-semiannual-canonical-anchors/provenance/endpoints-20260831T001625Z.json)
and [before the Gemini SWE addition](../../judge_results/frontier/eci-semiannual-canonical-anchors/provenance/endpoints-20260831T032308Z.json).
The original top-up [final run summary](../../judge_results/frontier/eci-semiannual-canonical-anchors/run-runpod/final-summary.json),
[consolidated ledger](../../judge_results/frontier/eci-semiannual-canonical-anchors/run-runpod/ledger.jsonl),
and [monitor](../../judge_results/frontier/eci-semiannual-canonical-anchors/run-runpod/monitor.jsonl)
retain the execution audit.

**Cost totals (approximate, not invoice allocations):** new-key API is **$65.5190**
($51.589830740 original top-up + $13.929153 Gemini SWE addition); Task 4 RunPod was estimated at
$6.8993 and the Task 5 balance-based change was about $22.4999 (from $46.143585 to $23.643660,
including volume billing), for estimated new RunPod of **$29.40**. Estimated new-work combined is
therefore **$94.92**. Historical API spend of about **$133.40** is separate; including it produces
an approximate **$228.32**, still not an invoice allocation.

No scientifically unidentified cell remains in the main rank-regression grid. The four absent
upstream Gemini SWE items remain absent rather than imputed, with the denominators above made
explicit.
