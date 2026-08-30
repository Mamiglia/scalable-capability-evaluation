# Canonical-anchor frontier judging redo

**Status:** Revised design awaiting written review. This is a noncanonical side analysis. It does
not modify the canonical pool, canonical verdicts, or frozen difficulty estimates.

## Purpose

Redo the quarterly-frontier analysis using the existing canonical pool as the calibration set.
Missing early predictions must be fixed by improving the eligible pair graph, with historical
models added only when the canonical pool cannot identify an early block. The quarterly answer
model frontier remains fixed across GPQA and SWE-mini. Judges see only Q1 items that both answer
models solved correctly and predict accuracy separately on frozen Q2, Q3, and Q4.

The output must distinguish two questions:

1. Does a judge recover a capability ordering from Q1 successful transcripts?
2. Does that ordering transfer to calibrated absolute accuracy on unseen harder quartiles?

A visually smooth release-time trend is not sufficient evidence for either claim, so a matched
Epoch-ECI baseline is required.

## Isolation and provenance

This redo uses new side-analysis-only roots:

- `configs/frontier_eci_canonical_anchors/`
- `logs/frontier-eci-canonical-anchors/`
- `data/<benchmark>/frontier-eci-canonical-anchors/`
- `judge_results/frontier/eci-semiannual-canonical-anchors/`

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

## Canonical anchor universe

The existing canonical pool is the main calibration set. A canonical model is usable as an anchor
for a benchmark, judge, target, and hard quartile when it has:

1. frozen canonical hard-quartile outcomes on that benchmark;
2. a path to the target in that judge's eligible Q1 comparison graph; and
3. a release date satisfying the temporal rule below.

The target itself is always excluded. All otherwise eligible canonical models are retained; a new
four-model panel does not replace or discard the existing pool.

Before paid calls, export a coverage audit for every benchmark/judge/target block: eligible
canonical models, connected models, ability range, existing pair counts, and the exact reason for
every exclusion. New historical source models are allowed only when a block has fewer than four
connected canonical anchors after missing canonical-pool comparisons have been considered. Such
models are labeled `historical_backfill` and never enter blocks where they violate the temporal
rule. Logs from a different benchmark scaffold are not substituted merely to fill a cell.

## Pair and item allocation

The paid top-up prioritizes breadth of answer-model pairs over many repeated items on a few edges.
For each underidentified block:

1. Reuse every compatible existing canonical verdict.
2. Add missing target-to-anchor or graph-bridging pairs, prioritizing anchors that widen the known
   hard-accuracy and Q1-ability ranges.
3. Freeze a deterministic sample of up to eight jointly-correct Q1 item IDs per new unordered
   answer-model pair; use all available IDs when fewer than eight exist.
4. Judge every selected item in both presentation orders and use the same selected item IDs for
   all judges when their coverage permits.
5. Add another pair before adding more items to a completed pair. Extra items are allowed only
   when fewer than four reliable anchors remain or a fitted block is disconnected.

Existing dense pairs remain in the provenance tables. To stop them dominating merely because they
were judged more often, the primary Bradley-Terry objective caps each unordered answer-model pair's
total weight at eight counterbalanced item IDs. A sensitivity fit uses all rows without this cap.
The cap and deterministic item manifest are fixed before inspecting target errors.

## Temporal eligibility

For target answer model `m` and judge `j`, a pairwise verdict is eligible only when both answer
models in that verdict were released no later than:

`max(release(m), release(j))`.

Equivalently, a comparison model must predate the target or predate the judge. Models newer than
both are excluded. The same rule applies to anchors used by the calibration fit.

The coverage audit, rather than an assumed common panel, determines which early cells are
identifiable. A block with fewer than four eligible connected anchors is reported as unidentified
unless protocol-compatible historical backfill can repair it within the approved cost limit.

## Ranking and calibration

For each benchmark, judge, and target:

1. Fit Bradley-Terry abilities from eligible Q1 verdicts only.
2. Exclude the target's Q2/Q3/Q4 outcomes from every fit.
3. For each hard quartile separately, fit the eligible, connected canonical anchors and any
   explicitly labeled historical backfill.
4. Predict the target from its Q1 Bradley-Terry ability.

The headline calibration is a binomial logistic regression with an explicit free intercept:

`logit(p_hard) = intercept + slope * BT_ability`.

The fit uses anchor success counts and quartile sizes, not unweighted percentages. There is no
regularization, forced origin, fixed intercept, or target-dependent ceiling. Frontier targets do
not become calibration anchors in the headline fit; the canonical pool remains the training set.

### Fit checks

Every fitted judge/benchmark/quartile block must export:

- intercept, slope, convergence status, anchor identities and count, and target position relative
  to the anchor ability range;
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
uses the same eligible canonical anchors, temporal rule, binomial-logit calibration, quartiles,
target exclusion, uncertainty procedure, and release-time plot grammar. Historical backfill enters
the baseline only in the same blocks where it enters the judge fit. This makes the judge and
generic capability-trend plots directly comparable. It does not use a target's true hard accuracy
as a training point.

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

1. every block has an exported canonical-anchor coverage audit and paid calls match its frozen
   pair/item manifest;
2. every temporally identifiable frontier cell has a prediction or an explicit validated failure;
3. anchor-intercept validation passes for every reported headline fit;
4. GPQA and SWE headline and ECI-baseline plots follow the figure contract;
5. tests, provenance, cost accounting, and the side-analysis write-up are committed and pushed.
