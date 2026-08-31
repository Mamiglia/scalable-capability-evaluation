# Handoff — judge-verdict hierarchical refit

Date: 2026-08-31

## Why a refit is required

The completed raw/PC L1 and L2 runs used the intended crossed coefficient
structure, but the relative shrinkage for judge, benchmark,
judge-by-benchmark, and item terms was fixed at `3/3/10/10`. Only the global
regularisation strength `C` was selected from the outer-training data. The
saved fits, lesions, figures, and report are therefore provisional.

The corrected model retains regularisation on every coefficient block. Its
group-specific terms receive additional relative penalties estimated within
each outer-training fold. Held-out outer rows never participate in this
selection.

## Work already completed and retained

- The readable design has 112 deterministic signed composites from 137 source
  features, including the approved overlapping block/family lesions.
- Pair features are left-minus-right and scaled within benchmark on
  representation-training rows.
- One global, outcome-blind 112-component PC basis is shared across all judges,
  benchmarks, folds, and lesions. Primary PC fits retain the first 30
  components (71.2093% of source-feature variance).
- Evaluation uses four model folds by four item folds. Inner folds are grouped
  by benchmark/item.
- Full, Singular, and Unique specifications use the same feature-building and
  fitting routes. Lesions refit coefficients but do not reselect
  hyperparameters.
- Existing artifacts are immutable and remain available for provenance; they
  must not be presented as final corrected results.

## Binding corrected fit

For each outer fold and each raw/PC, L1/L2 variant:

1. Build the complete crossed design: population, judge, benchmark, and
   judge-by-benchmark slopes; global, judge, benchmark, cell, and nested-item
   intercepts.
2. Select the global `C` and four positive relative penalty ratios from only
   the outer-training rows using grouped inner-CV log loss. A small coordinate
   search over a fixed ratio grid is sufficient; no Bayesian sampler or new
   dependency is required.
3. Apply penalty-family-correct column scaling. For relative penalty `r`, use
   `1 / r` under L1 and `1 / sqrt(r)` under L2.
4. Fit the outer-training full model and save `C`, the four ratios, tuning
   scores, coefficients, predictions, and convergence diagnostics.
5. Freeze `C` and all four ratios for the intercept-only, Singular, Unique,
   block, and family refits in that outer fold. Do not re-estimate any
   hyperparameter during lesioning.

The global intercept remains in the penalised design, matching the approved
regularise-everything specification.

## PC score scaling

The global PCA loadings and variance explained remain unchanged. Before any
regularised regression, divide each retained PC score by the square root of
its saved explained variance. This gives every PC unit variance in the frozen
basis, preserves orthogonality and left/right antisymmetry, and makes L1/L2
penalties comparable per one-SD effect.

## Capability companion

Capability is not part of the primary model or any lesion inventory. After
each corrected full fit, run one companion full-model refit with a single
additional feature:

1. Reuse the existing leave-item-out `accuracy_left` and `accuracy_right`
   scores for the item's benchmark.
2. Fit benchmark-specific score means/SDs on unique outer-training
   model/item values and apply them to both train and test sides.
3. Subtract right from left, then divide by the outer-training SD of the
   resulting delta across unique transcript pairs. Do not recenter the delta,
   preserving reversal antisymmetry.
4. Reuse the primary full model's frozen `C` and pooling ratios. Do not retune
   and do not run capability lesions.

Report the companion's change in held-out accuracy and log loss, its paired
delta-log-loss contribution, and whether the leading raw/PC effects change.

## Reporting changes

Report both:

- repeat-limited accuracy headroom recovered,
  `(accuracy - 0.5) / (repeat agreement - 0.5)`; and
- held-out deviance explained relative to the intercept-only model,
  `1 - full log loss / intercept-only log loss`.

Do not call accuracy-headroom recovery “variance explained”.

Retain the current overall figures and generate equivalent model-dependent
result packs for:

- each benchmark;
- each judge; and
- every observed judge-by-benchmark cell.

Each pack contains performance, raw-feature effects, raw-group effects, PC
effects, coefficient summaries, and prediction-failure plots where the
required rows exist. The global PC loading plots are not duplicated because
their basis is identical by design. Store the additional outputs under
`figures/by-benchmark/`, `figures/by-judge/`, and `figures/by-cell/`, with a
compact generated index rather than embedding every figure in the main report.

All figures retain the standing compact Nature style: 8 pt text, 300-DPI PNG,
editable-text SVG, shared axes for comparable panels, no in-panel titles, and
no clipped or overlapping labels.

## Explicitly deferred

- Do not change the outer-fold aggregation/weighting in this refit.
- Do not add capability to the primary model.
- Do not add figure tests.
- Keep the judge-verdict test module small (roughly 10–20 tests total), adding
  only focused coverage for fitted pooling ratios, penalty-specific scaling,
  PC-score standardisation, and capability-difference construction.

## Required reruns

Run locally only on tiny synthetic/smoke inputs. Full work runs on SWC and
writes new immutable result directories:

1. raw L1 full fits, references, feature lesions, and group lesions;
2. PC-30 L1 full fits, references, and PC lesions;
3. raw and PC-30 L2 full fits and lesions as sensitivity analyses;
4. raw and PC-30 capability-added companion full fits only;
5. aggregation, both recovery metrics, judge/benchmark/cell tables, all plot
   packs, and the revised results report.

Never overwrite or delete the existing Ceph artifacts. Stage code and outputs
to new paths, validate manifests/hashes, and commit code/docs before submitting
the full jobs.
