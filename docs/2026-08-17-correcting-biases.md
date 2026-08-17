# Can biased judging verdicts recover corrected capability estimates?

**Status:** design draft, 2026-08-17

## Research question

Can we use pairwise judgments on the easy portion of a benchmark to estimate a
model's capability on its hard portion, after correcting for systematic judge
biases such as transcript length, position, hedging, and pronoun use?

The target is a benchmark-specific capability estimate for each candidate
model. A corrected capability-gap curve is a secondary output. The target is
not an item-level score.

## Motivation

Judges identify the stronger model above chance on average, especially across
large capability gaps. That average hides stable comparison-specific errors.
DeepSeek V4 Flash, for example, is often ranked below weaker but more verbose
models. Reasoning-effort variants can also be systematically misranked.

The existing hierarchical GLMs show that judges prefer longer transcripts and
penalise features such as hedging and personal pronouns. These effects can be
large enough to oppose the capability signal. The question is whether removing
their estimated contribution improves out-of-sample capability estimation.

This is not done by subtracting a bias-only probability from a binary verdict.
Such a residual has no capability units. Instead, measured biases become
nuisance terms in an anchored pairwise model whose remaining latent quantity is
candidate capability.

## Scope

The primary analysis is GPQA. Pool all compatible, natural, both-correct
pairwise judgments from:

- `gpqa-pilot-rd2`;
- the GPQA data-expansion rounds;
- base and fine-tuned judge evaluation rounds where the same natural pairwise
  task was used; and
- the 2026-08-17 Hugging Face judging round.

Dataset boundaries remain as provenance fields; they are not train/test
boundaries. Repeated collections of the same model pair remain separate rounds.
Only exact duplicate artifacts are deduplicated.

Deliberate style manipulations and presentation attacks are not mixed into the
primary natural-data fit. They are useful validation data for the fitted bias
coefficients, but come from a different transcript-generating process.

MBPP is a separate replication. GPQA and MBPP do not share a latent capability
scale because the target is benchmark-specific.

## Input resolution and provenance

Every order-level judgment must retain:

- benchmark, source dataset, round, and prompt condition;
- exact judge endpoint, including base versus fine-tuned status;
- exact left and right candidate configurations;
- item ID and presentation order;
- the binary left-choice verdict; and
- the two original transcripts used in that judgment.

For the Hugging Face round, resolve candidate responses from the recorded
`major_path` and `minor_path` first. If a referenced local eval is absent because
the Epoch collection supplied it, resolve the canonical
`(benchmark, model, item)` response from the Epoch table. Reject ambiguous or
unresolved records. Aggregate `logs.json` metrics are not substitutes for
per-item verdicts or transcripts.

Record the serving route attached to each judge run where available. The Hugging
Face OpenRouter runs used provider preferences rather than hard endpoint pins,
so they are treated as fixed observed data and their routing limitation is
reported rather than ignored.

## Frozen item difficulty and outcome

Fit the existing fixed-floor IRT model using a designated reference response
pool that excludes every model later evaluated as a candidate. Freeze the item
parameters before constructing any candidate-model folds.

The primary item partition is the easiest half versus the hardest half by
reference-model IRT difficulty:

- easy items supply pairwise judging evidence;
- hard items supply the hidden evaluation target.

The existing easiest-quartile to Q2/Q3/Q4 analysis is a sensitivity check. The
ordinary 2PL split remains a model-form sensitivity.

For calibration models, estimate benchmark ability conditional on the frozen
item parameters using their full response vectors. These values define the
anchor scale. For a held-out model, neither its hard responses nor a capability
estimate derived from them may enter fitting. Its easy-item accuracy is used
only as a baseline, not as an input to the corrected estimator.

## Predictors

Express every pairwise feature as left minus right so it changes sign when the
presentation order is reversed. The primary nuisance predictors are:

- log transcript-length difference;
- personal-pronoun difference;
- hedging difference;
- backtracking difference;
- uncertainty difference;
- confidence-language difference;
- question-use difference.

Presentation position is represented by the judge intercept in the left-choice
model. Source and prompt condition are included as recorded design effects.
Judge coefficients are partially pooled, while each exact judge endpoint keeps
its own deviations. A base judge and its fine-tuned version are distinct judges.

Do not add embeddings, model-name features, or new style taxonomies to the
primary model. They make the correction harder to interpret and are not needed
to test the stated hypothesis.

## Statistical model

For order-level judgment \(n\), let \(y_n=1\) mean that judge \(j\) selected the
left transcript. Let \(\theta_L-\theta_R\) be the benchmark-capability difference
and \(x_L-x_R\) the measured nuisance-feature differences:

\[
\operatorname{logit} P(y_n=1) =
\alpha_j + u_{j,\mathrm{source}[n]} +
v_{j,\mathrm{condition}[n]} +
d_j(\theta_L-\theta_R) +
\beta_j^\top(x_L-x_R).
\]

Here:

- \(\alpha_j\) is judge-specific position bias;
- \(u_{j,s}\) permits the same judge's intercept to shift across collection
  sources;
- \(v_{j,c}\) permits the intercept to shift across prompt conditions;
- \(d_j\) is the judge's sensitivity to capability differences; and
- \(\beta_j\) contains judge-specific nuisance-feature effects.

The reference anchors identify the sign and scale of \(\theta\). Judges with
negative or near-zero \(d_j\) are not selected out using outcome accuracy; the
model learns that they provide reversed or little capability information.
Eligibility depends only on calibration and graph coverage.

The primary model keeps one capability sensitivity per judge. Add item-varying
discrimination only if posterior predictive checks show that the simpler model
systematically fails.

## Exact-model cross-fitting

Use five deterministic folds over exact candidate model/configuration IDs. Do
not group folds by model family. Related configurations may remain in the
calibration set.

For fold \(k\):

1. Globally mark its candidate models as held out across every source dataset.
2. Fit the hierarchical judge-bias model using only comparisons between
   non-held-out calibration models.
3. Freeze the posterior over \(\alpha\), \(u\), \(d\), and \(\beta\).
4. Reveal easy-item judgments involving at least one held-out model.
5. Infer held-out \(\theta\) values relative to fixed calibration anchors. Both
   heldout-versus-anchor and heldout-versus-heldout comparisons may contribute.
6. Map inferred \(\theta\) to hard-item accuracy using calibration models from
   that fold only.
7. Reveal held-out hard-item outcomes and score the predictions.

The same model may appear as a judge while its candidate configuration is held
out; the holdout applies to the candidate role. Self-comparisons remain controls
and do not estimate capability.

Judges do not need to cover identical candidate pools. Missing judge-candidate
cells are handled by the joint model. A held-out target is reportable only when
its easy-item comparison component connects to at least two fixed calibration
anchors. Report excluded targets and disconnected components rather than
silently dropping them.

Target-model judgments may update target capabilities only. They must not feed
back into the frozen judge-bias coefficients. This modular separation is the
leakage boundary that makes the estimates genuinely out of fold.

## Comparators

Evaluate every method on the same held-out models and folds:

1. **Easy accuracy:** empirical accuracy on the easy item bank.
2. **Raw judge Elo:** the existing uncorrected Bradley-Terry estimate from easy
   judgments.
3. **Uncorrected hierarchical model:** the same anchored model with position
   and capability terms but without transcript-feature corrections.
4. **Bias-corrected hierarchical model:** the full model above.
5. **Training-fold mean:** a null exact-score prediction.

Any mapping from a predictor to hard accuracy is fitted on calibration models
only and then applied unchanged to the held-out fold.

## Endpoints

### Primary: model ordering

Report Kendall's \(\tau_a\) between each out-of-fold predictor and hard-item
accuracy. Also report pair-ordering accuracy on held-out model pairs. Ties stay
in the denominator.

The primary comparison is corrected versus raw judge Elo on the same models.
Easy-item accuracy is the substantive benchmark baseline.

### Secondary: exact capability estimates

For cross-fitted predicted hard accuracy, report:

- mean absolute error;
- root mean squared error;
- calibration slope and intercept; and
- interval coverage for the corrected model.

### Corrected capability-gap curve

For every pair of out-of-fold target estimates, compute the true hard-accuracy
gap and whether the estimated ordering is correct. Plot ordering accuracy
against true hard-item gap for:

- raw judge Elo;
- the uncorrected hierarchical model; and
- the bias-corrected hierarchical model.

Also plot estimated versus true hard-item gaps. The curve must use only
out-of-fold estimates; fitted values for calibration models do not enter it.

As a diagnostic, split pairs by whether measured superficial features favour
the actually weaker model. Bias correction should help most in this
"bias-adverse" subset, which contains DeepSeek-like failures.

## Uncertainty and reporting

Propagate calibration uncertainty by carrying posterior draws of the frozen
judge coefficients into target-capability inference. Report candidate-level
posterior intervals rather than only point estimates.

For aggregate method comparisons, resample held-out candidate models rather
than treating thousands of verdict rows as independent observations. Keep the
judge-level results as a heterogeneity diagnostic, but use the pooled estimator
as the operational prediction.

Every result table records fold, source coverage, judges, anchors, easy items,
comparison edges, and connected component. Report results both pooled across
sources and separately by source to expose collection-specific failures.

## Success criteria

The main claim is supported if, out of fold:

1. corrected capability estimates predict hard-item model ordering above
   chance;
2. corrected estimates improve Kendall's \(\tau_a\) over raw judge Elo on the
   matched model set; and
3. the improvement is not explained solely by easy-item accuracy.

Lower MAE/RMSE is a secondary claim. A correction that improves ranking but not
absolute calibration is still useful, but should be described as corrected
ordering rather than corrected capability estimation.

## Required diagnostics

- coefficient recovery and posterior predictive checks on calibration rows;
- raw versus corrected performance by judge and source;
- performance with each nuisance-feature group removed;
- performance on neutral versus bias-adverse comparisons;
- coverage under the two-anchor requirement;
- fixed-floor 3PL versus ordinary 2PL item splits; and
- GPQA primary analysis versus the separate MBPP replication.

## Limitations

- Correction only removes measured biases. The residual capability term may
  still contain model identity, unmeasured style, or other shortcuts.
- Feature effects estimated from observational comparisons need not be causal.
  The manipulation datasets test this assumption but do not make every
  coefficient causal.
- Related models may appear across exact-model folds. That is part of the
  chosen model-level estimand; family-level extrapolation is not claimed.
- Sparse comparison graphs can make some held-out models unidentifiable.
- Provider and endpoint changes across collections can appear as judge drift.
- The experiment remains retrospective even though its prediction boundary is
  enforced prospectively within each fold.
