# Frontier rank regression: combined Q2–Q4 result

**Status: noncanonical side analysis.** The authoritative output is
[`MAIN_OUTPUT.json`](../../judge_results/frontier/eci-semiannual-canonical-anchors/MAIN_OUTPUT.json),
which points to
[`analysis/rank_regression/`](../../judge_results/frontier/eci-semiannual-canonical-anchors/analysis/rank_regression/).
All earlier separate-Q2/Q3/Q4, anchor-calibration, and ECI outputs are discarded provenance.

## Question and interpretation

For each judge and benchmark, we infer one Bradley–Terry ranking from successful Q1 transcripts,
then ask how closely that ranking aligns with **combined accuracy over all available frozen Q2–Q4
items**. This is deliberately an **in-sample descriptive fit**, not a forecast: every ranked model
with observed combined hard accuracy, including the frontier targets, helps fit the displayed
curve. The figures therefore diagnose rank–accuracy alignment and systematic residuals; they do
not demonstrate generalization to unseen models.

## Method

- Q1 input: 26,780 exact-epoch, both-correct verdict rows, capped deterministically at eight item
  IDs per unordered model pair.
- Hard target: one accuracy over the union of frozen Q2, Q3, and Q4 items.
- Fit: compare a free-intercept linear OLS curve with a two-parameter bounded sigmoid. Select the
  sigmoid if the linear fit leaves `[0, 1]` or the sigmoid has lower training RMSE; otherwise use
  linear.
- Uncertainty: 1,000 two-source item bootstraps. Each draw independently resamples Q1 item IDs and
  combined Q2–Q4 item IDs, then refits the ranking and regression. Line figures show 95% shaded
  bands; point figures show uncapped 95% error bars.
- Exact coefficients, candidate RMSEs, selected method, bootstrap counts, and training IDs are in
  [`fits.csv`](../../judge_results/frontier/eci-semiannual-canonical-anchors/analysis/rank_regression/fits.csv).

## Results

All 10 judge × benchmark fits succeeded and all 80 frontier rows have actual values, fitted values,
and bootstrap intervals. Six fits select sigmoid and four linear.

| Benchmark | Judge | Selected | Linear RMSE | Sigmoid RMSE | Training models |
|---|---|---:|---:|---:|---:|
| GPQA | DeepSeek V4 Flash | sigmoid | 0.1668 | 0.1666 | 24 |
| GPQA | Gemini 2.5 Pro | sigmoid | 0.1317 | 0.1299 | 20 |
| GPQA | GPT-4o 2024-11-20 | linear | 0.1903 | 0.1929 | 24 |
| GPQA | GPT-5.2 | linear | 0.1055 | 0.1068 | 19 |
| GPQA | o4-mini | linear | 0.1479 | 0.1480 | 28 |
| SWE-mini | DeepSeek V4 Flash | sigmoid | 0.1144 | 0.1231 | 28 |
| SWE-mini | Gemini 2.5 Pro | sigmoid | 0.1622 | 0.1673 | 19 |
| SWE-mini | GPT-4o 2024-11-20 | sigmoid | 0.1834 | 0.1909 | 23 |
| SWE-mini | GPT-5.2 | linear | 0.1620 | 0.1670 | 18 |
| SWE-mini | o4-mini | sigmoid | 0.1549 | 0.1413 | 31 |

The median frontier 95% interval width is 0.183 (maximum 0.366). Every frontier interval has at
least 80% of the requested bootstrap draws; no frontier interval is marked unstable.

Gemini 3 Pro SWE-mini uses the exact recovered source without imputation. It contains 46/50 frozen
items: Q1/Q2/Q3/Q4 counts 12/13/12/9. Its combined hard target therefore uses the 34 available
Q2–Q4 items.

## Figures and tables

- [All combined Q2–Q4 figures](../../judge_results/frontier/eci-semiannual-canonical-anchors/analysis/rank_regression/figures/):
  20 PNG, 20 PDF, and 20 editable-text SVG files (band and points variants for five judges × two
  benchmarks).
- [BT abilities](../../judge_results/frontier/eci-semiannual-canonical-anchors/analysis/rank_regression/abilities.csv)
- [Joined training points](../../judge_results/frontier/eci-semiannual-canonical-anchors/analysis/rank_regression/training_fitted.csv)
- [Frontier fitted rows and intervals](../../judge_results/frontier/eci-semiannual-canonical-anchors/analysis/rank_regression/frontier_fitted.csv)
- [Summary](../../judge_results/frontier/eci-semiannual-canonical-anchors/analysis/rank_regression/summary.json)

The figures use model release date on x, combined Q2–Q4 accuracy on y, a dashed judge-release line,
black actuals, blue fitted values, dotted residuals, and 95% item-bootstrap uncertainty.

## Provenance and cost

The final paid corpus contains 3,220 audited responses: 3,219 latest-success rows plus one retained
response from a recovered local SIGABRT attempt. Missing paid-provider provenance and provider
mismatches are both zero. New-key API cost is approximately $65.519; historical API spend is
approximately $133.40. RunPod spending is balance-derived rather than an invoice allocation; see
the linked lifecycle and run summaries in the main manifest.
