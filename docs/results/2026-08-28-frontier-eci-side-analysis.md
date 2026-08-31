# Discarded ECI frontier plots (archived side analysis)

**Status: discarded / non-main.** This document preserves the location and provenance of an
older noncanonical side analysis. It is not a result to interpret, compare, or cite as the
current frontier analysis.

The current main analysis for this work is the deliberately in-sample
[rank-regression root](../../judge_results/frontier/eci-semiannual-canonical-anchors/analysis/rank_regression/),
as named by the authoritative [MAIN_OUTPUT.json](../../judge_results/frontier/eci-semiannual-canonical-anchors/MAIN_OUTPUT.json).
Its exact label is `Rank–accuracy fit (in-sample)`.

Every older output below—tables, metrics, coverage exports, and figures—is discarded/non-main.
The machine-readable rules in `MAIN_OUTPUT.json` classify `analysis/**` except
`analysis/rank_regression/**`, all of `capability_baseline/**`, and every older `analysis*` root:

- [`judge_results/frontier/eci-semiannual/analysis/`](../../judge_results/frontier/eci-semiannual/analysis/)
  and [`analysis-swe-partial/`](../../judge_results/frontier/eci-semiannual/analysis-swe-partial/),
  including their `figures/frontier_style/` subdirectories;
- [`judge_results/frontier/eci-semiannual/analysis/capability_baseline/`](../../judge_results/frontier/eci-semiannual/analysis/capability_baseline/);
- [`judge_results/frontier/eci-semiannual/calibration-audit/`](../../judge_results/frontier/eci-semiannual/calibration-audit/);
- [`analysis/`](../../judge_results/frontier/eci-semiannual-canonical-anchors/analysis/), except its
  authoritative `rank_regression/` child, and the entire
  [`capability_baseline/`](../../judge_results/frontier/eci-semiannual-canonical-anchors/capability_baseline/).

The old anchor-only, release-time, 79/80-coverage, and Epoch-ECI comparisons remain retained
provenance only. They do not qualify the main curve, which uses neither anchors nor ECI and has
no target exclusions or rolling cutoff. See the final
[write-up](2026-08-30-canonical-anchor-frontier-redo.md) for scope, limitations, and valid paths.
