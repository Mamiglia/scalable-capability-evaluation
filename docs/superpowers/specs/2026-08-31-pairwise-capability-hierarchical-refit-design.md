# Pairwise Capability Hierarchical Refit Design

Date: 2026-08-31
Status: Approved

## Scope

Repair the existing overall-accuracy pairwise capability wrappers so benchmark
variation is learned from training data rather than fixed by configuration.
This design supersedes only the fitting and ablation semantics in the prior
pairwise capability designs. It retains the pinned 112-composite matrix,
overall-accuracy target, global PCA basis, strict model-by-item folds, pair
deduplication, and reporting infrastructure recorded in the authoritative
hierarchical-fit invalidation handoff.

The capability workstream does not change the judge hierarchy or judge
artifacts. It extends the established shared sparse-task interface without
changing the judge task's four pooling families or fitted behavior.

## Model

For signed left-minus-right feature vector `x` in benchmark `b`, fit

```text
logit P(left model is stronger) = x' population + x' benchmark_deviation[b]
```

There is no intercept or item term. Reversing a pair negates `x`, so the fitted
probability changes exactly from `p` to `1 - p`. The no-feature reference is
fixed at probability 0.5.

One positive relative penalty ratio is shared by every benchmark-deviation
coefficient in a fit. For ratio `r`, benchmark-deviation design columns use
`1 / r` under L1 and `1 / sqrt(r)` under L2. Population coefficients retain
unit relative penalty.

## Per-fold fitting

Fit each of the 16 held-out model-by-item outer folds independently for each
of raw L1, raw L2, PC L1, and PC L2.

Within each outer fold:

1. Prepare raw composites or the first 30 globally shared PC scores. Divide PC
   scores by the square root of their saved explained variance.
2. Fit the population and benchmark coefficient-family scales from the
   outer-training likelihood using the judge refit's NumPyro SVI procedure.
   Convert them to the single benchmark-deviation relative penalty ratio using
   `(population_scale / benchmark_scale)` under L1 and its square under L2.
3. Freeze that ratio while selecting `C` by grouped inner-CV log loss on the
   outer-training rows.
4. Require the selected inner fits and the final outer-training fit to
   converge.
5. Save the selected `C`, L1 ratio, benchmark-deviation ratio, complete tuning
   trace, coefficients, predictions, and convergence diagnostics.

The selected relative ratio is global within that fold but may differ across
outer folds. Its fold-to-fold distribution is a required stability diagnostic.
No held-out outer row participates in hyperparameter selection.

## Existing wrapper changes

Keep `capability_predictions_glm/pairwise_wrapper.py` and
`capability_predictions_glm/pairwise_wrapper_lesions.py` as the entry points.
Do not add another fitting pipeline.

Make the shared sparse task declare its scale estimator and relative penalty
parameters. The judge task continues to use its existing crossed-hierarchy
estimator and four parameters; the capability task uses a no-intercept
population-plus-benchmark estimator and declares only
`benchmark_deviation_penalty`. This is an interface extension only: judge
design, estimator, tuning, diagnostics, and outputs remain unchanged.

The capability design reads the fitted ratio supplied by the full-fit path.
Coefficient reconstruction continues to save population, benchmark deviation,
and effective benchmark slopes on the natural feature scale.

## Ablations

For every outer fold, freeze the full fit's `C`, L1 ratio, and fitted benchmark
ratio for all singular, drop-one, block, and family refits. Never retune after
changing the feature set. Keep the exact probability-0.5 null reference and
the existing accuracy/log-loss delta semantics.

The lesion preparation step must reject a full-fit artifact if the fitted
benchmark ratio is missing, non-positive, non-finite, inconsistent with its
manifest, or not backed by a selected converged tuning record. Old fixed-ratio
fit and lesion roots remain audit-only and are never overwritten.

## Verification

Add focused checks for:

- task-specific pooling parameters leaving judge behavior unchanged;
- L1 and L2 capability column scaling from the fitted ratio;
- exact pair-reversal antisymmetry;
- a synthetic capability hierarchy selecting a ratio from data rather than
  reading the configuration default;
- complete full-fit diagnostics and provenance; and
- lesion refits reusing the full fold's frozen ratio without retuning.

Per Morio's execution constraint, run all tests—including synthetic and smoke
fits—through Slurm on SWC. The laptop is used only for source editing and
read-only inspection.

## Execution and outputs

After tests pass on SWC, run one outer-fold smoke fit and inspect its tuning
trace, selected ratio, coefficients, convergence, predictions, and reversal
check. Then submit all 64 full-fit fold tasks and dependent capability lesion
stages to new immutable Ceph roots containing the code and input identifiers.

Every stage validates pinned input hashes, exact fold/specification inventory,
and predecessor manifests. Use resumable, non-overwriting Slurm jobs. Never
delete or modify existing Ceph data artifacts.

Only regenerated artifacts may feed capability conclusions. Judge-capability
comparisons wait for the canonical corrected judge outputs and consume them
without modifying the judge pipeline.

## Judge-capability reporting

Make the comparison a primary report section. Reuse the existing aligned
comparison machinery to publish one compact overview table joining, by variant
and judge, capability stronger-model accuracy, judge stronger-model accuracy,
their difference, overall coefficient cosine, Pearson/Spearman agreement, and
judge ECI where available. Follow it with benchmark-specific coefficient and
cosine summaries, raw/PC original-leaf comparisons, cosine drivers, and the
descriptive relationships with judge ECI and empirical judge accuracy.

Do not compare PC indices directly. Back-project through the frozen global
loadings and benchmark-specific composite leaf weights before original-feature
comparisons. Keep predictive performance, singular contribution, unique
contribution, coefficient direction, PC definition, and cross-task similarity
as separate quantities.
