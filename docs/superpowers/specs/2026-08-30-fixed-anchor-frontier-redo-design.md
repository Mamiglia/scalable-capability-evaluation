# Fixed-anchor frontier judging redo

**Status:** Approved design for a noncanonical side analysis. It does not modify the canonical
pool, canonical verdicts, or frozen difficulty estimates.

## Purpose

Redo the quarterly-frontier analysis with a stable historical calibration panel so that missing
early predictions and changing target-specific fits do not drive the result. The quarterly answer
model frontier remains fixed across GPQA and SWE-mini. Judges see only Q1 items that both answer
models solved correctly and predict accuracy separately on frozen Q2, Q3, and Q4.

The output must distinguish two questions:

1. Does a judge recover a capability ordering from Q1 successful transcripts?
2. Does that ordering transfer to calibrated absolute accuracy on unseen harder quartiles?

A visually smooth release-time trend is not sufficient evidence for either claim, so a matched
Epoch-ECI baseline is required.

## Isolation and provenance

This redo uses new side-analysis-only roots:

- `configs/frontier_eci_fixed_anchors/`
- `logs/frontier-eci-fixed-anchors/`
- `data/<benchmark>/frontier-eci-fixed-anchors/`
- `judge_results/frontier/eci-semiannual-fixed-anchors/`

The previous `judge_results/frontier/eci-semiannual/analysis*` figures are retained only as
discarded provenance and are not overwritten. Frozen item quartiles from
`configs/frontier_eci/item_quartiles.csv` are copied by reference and never recomputed.

Source transcripts and exact existing verdicts are reused only when benchmark, source model,
judge checkpoint, prompt, Q1 item, presentation order, and reasoning-visibility settings match.
Every imported or generated row records its source.

## Fixed frontier and judges

The answer-model frontier is unchanged:

| Quarter | Answer model |
|---|---|
| 2024Q4 | GPT-4o 2024-11-20 |
| 2025Q1 | Claude 3.7 Sonnet |
| 2025Q2 | o3 medium |
| 2025Q3 | GPT-5 medium |
| 2025Q4 | Gemini 3 Pro Preview |
| 2026Q1 | Claude Opus 4.6 |
| 2026Q2 | Claude Opus 4.7 |
| 2026Q3 | Kimi K3 |

`high` and `xhigh` answer variants remain excluded. The judges remain GPT-4o 2024-11-20,
Gemini 2.5 Pro, GPT-5.2, o4-mini, and DeepSeek V4 Flash 0731.

## Common historical anchor panel

The same four answer models calibrate both benchmarks and every judge:

| Anchor | Release | Setting |
|---|---:|---|
| Claude 3.5 Sonnet 0620 | 2024-06-20 | default |
| Claude 3.5 Sonnet 1022 | 2024-10-22 | default |
| o3-mini | 2025-01-31 | medium |
| Gemini 2.0 Flash 001 | 2025-02-05 | default |

All source logs come from public Epoch benchmark artifacts. GPQA uses the Epoch multiple-choice
protocol; SWE uses one internally consistent historical Epoch SWE scaffold. A source log from a
different agent scaffold is not substituted merely to fill a cell.

Each judge evaluates a counterbalanced star joining every anchor to GPT-4o 2024-11-20. Existing
compatible star edges are reused. Only missing judge/anchor/benchmark/order cells are run. The
existing frontier and pool edges connect later targets to that star, minimizing new API calls.

## Temporal eligibility

For target answer model `m` and judge `j`, a pairwise verdict is eligible only when both answer
models in that verdict were released no later than:

`max(release(m), release(j))`.

Equivalently, a comparison model must predate the target or predate the judge. Models newer than
both are excluded. The same rule applies to anchors used by the calibration fit.

All four anchors are eligible for every target except GPT-4o judged by GPT-4o. Only two common
anchors predate that exact checkpoint, so that one cell is reported as unidentified rather than
filled with a two-point extrapolation. Later judges can predict GPT-4o because all four anchors
predate those judges.

## Ranking and calibration

For each benchmark, judge, and target:

1. Fit Bradley-Terry abilities from eligible Q1 verdicts only.
2. Exclude the target's Q2/Q3/Q4 outcomes from every fit.
3. For each hard quartile separately, fit only the eligible members of the four-model fixed panel.
4. Predict the target from its Q1 Bradley-Terry ability.

The headline calibration is a binomial logistic regression with an explicit free intercept:

`logit(p_hard) = intercept + slope * BT_ability`.

The fit uses anchor success counts and quartile sizes, not unweighted percentages. There is no
regularization, forced origin, fixed intercept, or target-dependent ceiling. The main fit is not
allowed to add later pool models or other frontier models as calibration anchors.

### Fit checks

Every fitted judge/benchmark/quartile block must export:

- intercept, slope, convergence status, anchor count, and target position relative to the anchor
  ability range;
- observed and fitted accuracy for every anchor;
- raw and weighted anchor residuals;
- held-out target residuals in a separate table.

With a free intercept, the binomial score equation implies that the weighted anchor residuals sum
to approximately zero at a converged interior optimum. A block fails validation if they do not,
or if every nonzero anchor residual has the same sign. Failed blocks are shown as missing and are
not silently replaced.

Systematically negative held-out target residuals are possible even when the anchor residual check
passes: that would be evidence that Q1 judged ability does not transfer to frontier hard-item
accuracy, not evidence of a missing intercept. No parameter may be tuned using target truth to
make the release-time plot look closer.

For interpretability, a clipped affine fit with a free intercept is exported as a sensitivity
analysis. It is not selected post hoc according to held-out target error.

## Uncertainty

Intervals bootstrap both sources of finite-item uncertainty:

- resample Q1 item IDs before refitting Bradley-Terry abilities;
- independently resample hard-quartile item IDs before refitting the anchor calibration.

The fixed quartile assignment itself is never resampled or recomputed. The bootstrap exports the
number of successful refits and visibly marks intervals based on too few successful refits.

## Matched ECI baseline

The dumb baseline replaces Bradley-Terry ability with the target and anchor Epoch ECI values, then
uses the same fixed anchors, temporal eligibility, binomial-logit calibration, quartiles, target
exclusion, uncertainty procedure, and release-time plot grammar. This makes the judge and generic
capability-trend plots directly comparable. It does not use a target's true hard accuracy as a
training point.

## Outputs and figure contract

For GPQA and SWE, export predictions, anchor fits, residuals, coverage, position bias, reverse-order
consistency, length preference, API provenance, and cost. Report MAE, signed bias, and Spearman by
judge and quartile; report inside-range and extrapolated targets separately.

Each primary plot has:

- answer-model release date on the x-axis;
- actual hard-quartile accuracy on the y-axis;
- a vertical dashed line at the judge release date;
- black actual points, blue held-out predictions, vertical residual connectors, and uncertainty
  bands only where predictions exist;
- one judge, benchmark, and target quartile per panel;
- a short plain-text fit description in the caption or adjacent table, never an unrendered equation.

Figures follow the repository's Nature-manuscript style, are saved as PDF and 300 dpi PNG, and are
visually inspected at final size. The ECI baseline uses the same visual grammar in a separate
directory.

## Execution, cost, and completion criteria

Large source-log downloads, trimming, judging, and full analysis run on SWC HPC. The laptop is
limited to source inspection, small tests, file edits, and final figure inspection. API calls are
resumable and run only after exact reuse coverage is computed.

OpenRouter routes are hard-pinned where possible, fallbacks are disabled, data collection is
denied, required parameters are enforced when compatible, endpoint metadata is snapshotted, and
the served provider is recorded per response. Errors remain errors rather than shrinking the
sample silently.

Expected incremental API cost is $15-$30, keeping estimated total project API spend below $200.
Stop and report before exceeding that total unless the user gives new authorization.

The redo is complete only when:

1. the common anchor panel is present on both benchmarks for all five judges;
2. every temporally identifiable frontier cell has a prediction or an explicit validated failure;
3. anchor-intercept validation passes for every reported headline fit;
4. GPQA and SWE headline and ECI-baseline plots follow the figure contract;
5. tests, provenance, cost accounting, and the side-analysis write-up are committed and pushed.
