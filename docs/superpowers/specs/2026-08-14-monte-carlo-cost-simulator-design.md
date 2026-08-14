# Monte Carlo Cost Simulator Design

**Status:** Discussion approved; Claude Code reviewed; awaiting written-spec
review, 2026-08-14

**Source draft:** `docs/plans/2026-08-14-MC-cost-simulation.md`

## Purpose

Estimate how pairwise-judging results, uncertainty, and collection cost change
as the experiment scales the number of judges, evaluated models, comparisons,
items, and repetitions. The simulator must use the observed GPQA pilot and
data-expansion data rather than asking users to invent distributions for every
parameter.

The simulator answers two kinds of question:

1. What results should we expect from an automatically sampled design with
   `J` judges, `M` anchor models, and `N` comparison models per anchor?
2. What should we expect from a specific list of
   `(judge, model_1, model_2)` comparisons?

All estimates are posterior predictive distributions, not single best-case
numbers.

Scaling the number of judges drawn from the fitted population is directly
estimable. Predicting that a particular unseen judge will behave differently
because it is newer or more capable is a separate, weaker claim. The simulator
enables that covariate-conditioned extrapolation only if leave-one-judge-out
validation beats a pooled zero-covariate judge baseline; otherwise unseen
judges are labelled pooled-population forecasts.

## Scope

The first implementation covers pairwise judgments using the existing GPQA
pilot and expansion data, model capability and release date, IRT item
difficulty, transcript lengths, presentation order, and model family. It
produces command-line results before adding controls to the existing
dashboard.

The first version does not generate synthetic transcript text, train a neural
text scorer, automatically discover models from OpenRouter or Hugging Face, or
maintain a live pricing service. Those features are unnecessary for estimating
the requested scaling curves. Model metadata and prices remain editable input
data.

## Terminology

- **Judge:** the model making the pairwise choice.
- **Anchor (`M1`):** the first model in an automatically sampled comparison.
  “Anchor” describes its sampling role; it does not assert that it is better
  than `M2`.
- **Comparator (`M2`):** the second model in an automatically sampled
  comparison.
- **Capability:** benchmark-specific IRT ability `theta`. Raw benchmark
  accuracy is an input measurement, not an interchangeable capability scale.
- **Release date:** the model's public release date, represented numerically
  only after preserving the original date in the model catalogue.
- **Observed model or judge:** an identity represented in the fitted judgment
  data.
- **Unseen model or judge:** an eligible identity with capability and release
  metadata but no identity-specific judgment effect in the fitted data.

`J` denotes a count in automatic mode. A manual row uses `judge` rather than
`J` to avoid confusing an identity with that count.

## Inputs

### Model catalogue

Use one editable table keyed by canonical model ID. Each row contains:

- release date;
- IRT ability `theta`, or GPQA accuracy from which `theta` can be derived, plus
  the measurement source;
- model family;
- reasoning class (`reasoning` or `non_reasoning`) for the evaluated model
  configuration;
- whether the model is available for automatic judge or candidate sampling;
- whether observed response and judgment data exist;
- optional input and output token prices. In the current Inspect usage schema,
  reasoning tokens are a reported subset of billed output tokens rather than a
  third billable total.

The canonical set in `docs/canonical-models.md` seeds the catalogue but does
not hard-code the future pool. A model without capability or release date is
excluded from automatic sampling. It may appear in a manual row, but the run
must report which required prediction features are missing and must not invent
them silently.

Capability must use one IRT `theta` scale within a simulation. Models with
response data receive their fitted posterior `theta`. For a model with GPQA
accuracy but no fitted response row, derive `theta` by numerically inverting
the fitted benchmark response curve
`mean_i[c_i + (1-c_i) * sigmoid(a_i * (theta-b_i))]`; propagate uncertainty
from accuracy and the item-parameter posterior. Raw accuracy must never be
inserted directly into a `theta` difference. Models with neither a fitted
`theta` nor a compatible benchmark accuracy are ineligible for automatic
sampling and receive no manual prediction.

### Empirical data

Reuse, rather than redigest, the existing prepared sources where possible:

- item-addressable pairwise judgments from the GPQA pilot and data-expansion
  exports;
- transcript lengths and judge token accounting already extracted by
  `scripts/hibayes/prepare_gpqa.py`;
- candidate input/output usage from the answer-generation `.eval` logs, using
  the generic archive/usage-reading pattern in `scripts/cost/eval_cost.py`;
- model-item correctness and fitted item difficulty from `src/irt`;
- canonical model aliases and release metadata used by the existing analyses.

Observed model-item rows supply empirical correctness and length distributions.
For unseen models, correctness is generated from the fitted IRT response model
using `theta` and item difficulty. Length is drawn from the observed
`theta`/date/item-difficulty/correctness-conditioned population, including
whether visible reasoning is present, with residual uncertainty. No transcript
text is generated.

Token behaviour is fitted separately for candidate generation and judging.
Model input and total billed output as `log1p(tokens)`, with a reasoning-class
population mean, a partially pooled model-configuration effect, and available
item-difficulty/correctness covariates. The judge-input model also includes the
drawn left/right transcript lengths. Non-reasoning configurations have a
structural zero for reasoning tokens. For reasoning configurations, model the
reasoning share of total output and derive ordinary final-output tokens as
`max(output_tokens - reasoning_tokens, 0)`.

This matches the current Inspect convention used by
`scripts/cost/eval_cost.py`: `output_tokens` is the billed completion total and
`reasoning_tokens` is a subset. Ordinary plus reasoning tokens must reconstruct
total output; reasoning is never added to total output a second time.

An observed model configuration with token logs uses its fitted model effect.
An unseen configuration, including a known model used in a role for which no
token logs exist, draws one model effect from its reasoning-class population
per Monte Carlo trial and one request residual per eval. Thus token predictions
vary both between plausible unseen models and between requests. Reasoning class
is the minimum grouping; add model-family or provider effects only if held-out
token prediction demonstrates a need.

### Simulation configuration

Every run records:

- automatic or manual mode;
- random seed and posterior-draw count;
- attempted-item count and presentation repetitions/order policy;
- accuracy target, defaulting to the higher-`theta` candidate;
- model-pool filters and exact selected identities where applicable;
- reusable-response versus newly-generated-response cost assumption;
- reasoning class for any manual model configuration not already in the
  catalogue;
- catalogue, fitted-model, and data version identifiers.

## Comparison selection

### Automatic mode

1. Select or sample `J` judges from the judge pool.
2. Sample one shared set of `M` anchors for those judges. Every anchor must be
   newer than every selected judge. By default it must also have greater
   capability than every selected judge; this strength filter is explicit in
   the saved configuration.
3. For each anchor, sample `N` distinct comparators from all models released
   before that anchor. Release date is the only comparator restriction: a
   comparator may be weaker or stronger than the anchor or judge.
4. Apply the same `M * N` comparison set to all `J` judges. This isolates the
   effect of judge count from changes in the comparison graph.
5. Sample the configured number of attempted items and presentation
   repetitions. A judgment is usable only when it satisfies the real
   collection eligibility rule, including both candidate responses being
   correct.

Sampling is without replacement within each anchor's `N` comparators. The same
model may be a comparator for different anchors. A requested count that exceeds
the eligible pool is an error with the eligible count and failed restriction;
the simulator never lowers `J`, `M`, or `N` silently.

The first version uses a fixed attempted-item budget: it generates or reuses
the requested model-item responses, then drops ineligible candidate pairs. It
does not sample until a target number of usable judgments is reached. The
resulting usable-item and judgment counts are random, capability-dependent
outputs.

### Manual mode

The user supplies explicit `(judge, M1, M2)` rows. Manual rows use the same
prediction and Monte Carlo engine as automatic rows, but automatic newness and
strength restrictions do not block them. The output warns when a row violates
those rules or leaves the calibrated data range.

Duplicate manual rows are retained because they can represent intended
replication. The saved run makes duplicates visible.

## Statistical model

Fit the existing left/right representation. For observation `y = 1` when a
judge chooses the displayed left candidate `L` over the displayed right
candidate `R`, use a hierarchical logistic model:

`y ~ Bernoulli(sigmoid(eta))`.

The initial linear predictor contains:

- signed capability gap
  `g = theta(L) - theta(R)`;
- oversight level
  `o = mean(theta(L), theta(R)) - theta(judge)`;
- signed candidate release-date difference;
- mean candidate release date relative to the judge;
- IRT item difficulty;
- `g * o`, measuring whether gap sensitivity changes when candidates are
  ahead of the judge;
- `g * item_difficulty`, measuring whether gap sensitivity changes on harder
  items;
- log left-minus-right transcript-length difference, same-family-by-gap,
  self-comparison, and signed reasoning-visibility controls;
- item effects and signed candidate-model residual difference `v_L - v_R`.

Continuous predictors are centered and scaled using training-data statistics,
which are stored with the fitted model. Presentation swapping negates all
left-minus-right candidate features and interactions containing them while
leaving the left-position intercept, oversight level, and item terms unchanged.
For a requested `(judge, M1, M2)` row, `P(choose M1)` is `P(choose left)` when
M1 is displayed left and `1 - P(choose left)` when M1 is displayed right.

The pooling structure is fixed rather than inferred from a generic list of
predictors:

| Term | Structure |
| --- | --- |
| Left-position intercept and signed-capability-gap slope | Judge-specific, partially pooled; population mean may depend on judge `theta` and release date after the validation gate below |
| Length-difference slope | Global mean plus zero-mean partially pooled judge deviations |
| Oversight, release-date, item-difficulty, both approved interactions, same-family-by-gap, self, and reasoning visibility | Regularized global fixed effects |
| Item and candidate identities | Zero-mean random residuals |

The judge intercept and capability-gap slope population mean is:

`[judge_position_intercept, judge_gap_slope] ~ MVN(A * [1, theta, release_date], Sigma)`.

Observed judges use their identity-specific posterior. If leave-one-judge-out
prediction supports the capability/date population regression, an unseen judge
receives one draw from this conditional distribution per Monte Carlo trial. If
it does not, `A` contains only an intercept and the unseen judge is drawn from
the pooled population without a claimed capability/date adjustment.
That draw is reused for all of the judge's decisions in the trial, preserving
within-judge correlation. If the data do not support capability/date as
predictors of judge behaviour, regularizing priors shrink those coefficients
toward zero and uncertainty remains wide.

`theta` and release fixed effects give the conditional mean for unseen
candidate models. Observed candidates additionally use their fitted residuals;
unseen candidates receive one draw from the fitted candidate-residual
distribution per trial. They are therefore uncertain population predictions,
not “average model” substitutions.

The existing Bernoulli hierarchy in `scripts/hibayes/model.py` is the starting
point. Add only the approved release, oversight, difficulty-interaction, and
model-residual terms. Release-date terms are omitted by default when
capability/date collinearity is severe and retained only when held-out
predictive performance separates them from the capability terms. Other
optional terms follow the same rule: when models are indistinguishable within
cross-validation uncertainty, keep the simpler one.

## Monte Carlo procedure

For every trial:

1. Draw one posterior parameter sample. One trial represents total posterior
   predictive uncertainty; the first version does not attempt a nested
   variance decomposition.
2. Select automatic comparisons or load the explicit manual rows.
3. Draw the attempted-item set. Use observed model-item outcomes when present; otherwise
   draw correctness from the IRT model. Apply the same item-eligibility rule as
   the real collection protocol, including the requirement that both candidate
   responses be correct, and retain attempted and usable counts separately.
4. Draw missing transcript lengths and all unseen judge/model residuals once.
5. Draw token counts for every planned new candidate-generation and judge call
   from the role-specific posterior-predictive distributions. Reused candidate
   responses create no new generation-token cost; their observed transcript
   lengths still contribute to judge input. Validation replays retain observed
   usage.
6. Assign the configured presentation order and compute `P(judge chooses
   left)` for each planned judgment.
7. Draw the left/right judgment as a Bernoulli outcome, then map it back to an
   M1/M2 vote.
8. Counterbalance/average presentation orders using the real analysis rule,
   then compute the requested accuracy, ranking, uncertainty, and cost outputs.

Posterior, judge, candidate, and item effects are shared wherever the real
experiment shares them. Treating every judgment as an independent draw from a
single average probability would understate uncertainty and is not allowed.

## Outputs

### Scientific outputs

- Judgment accuracy versus signed and absolute capability gap, with posterior
  intervals and judge/pool filters.
- Scaling curves over judge count, anchor count, comparators per anchor,
  attempted-item count, and repetitions.
- Judge- and candidate-ranking recovery/stability, reported with Kendall's
  tau-b where the comparison graph supports a common ranking.
- The easy-item judge-ranking versus hard-item performance endpoint specified
  in `docs/plans/2026-08-14-irt-based-prediction.md`.
- For every manual row: posterior mean `P(judge chooses M1)`, interval, expected
  judgment accuracy under the configured target, and support warnings.
- Total posterior-predictive uncertainty across fitted-parameter, latent-unit,
  item-selection, and finite-judgment variation.

Every forecasted quantity reports the median and 5th/95th posterior-predictive
percentiles by default. This includes manual-row choice probabilities, scaling
curves, usable-item counts, token counts, and costs. Exact configuration values
and deterministic upper bounds remain exact rather than receiving artificial
error bars.

The simulator does not replace the empirical resampling analysis specified in
the IRT plan. For the easy-to-hard endpoint it generates easy-item votes, then
passes those votes through the existing `fit_bt`, graph-component, and LOMO
calibration helpers. NUTS is fitted once, outside the simulation loop. The
empirical IRT-plan Monte Carlo remains the observed-data reference.

“Accuracy” requires an explicit target. For capability-discrimination plots,
the target is choosing the candidate with higher `theta`; ties are reported
separately rather than forced into correct/incorrect. Raw `P(choose M1)` is
always retained so alternative targets can be computed without refitting.

For forward simulations, candidate-ranking “recovery” means consistency with
the `theta` ordering used to generate the data; it is not independent evidence
that this ordering is correct. Observed-data recovery against the complete
collection is reported separately.

### Cost outputs

Report judgment calls and candidate-generation calls, followed by a per-model
and total token/cost breakdown. For both candidate generation and judging,
report input and total billed output, with output decomposed into ordinary
final-output and reasoning-token subsets. Every predicted token and currency
subtotal includes its posterior-predictive interval.

Candidate generation is zero when configured to reuse collected responses.
Otherwise it is counted once per unique attempted model-item response,
including responses later removed by the both-correct filter, not once per
pairwise judgment.

Token counts use observed distributions when available and conditional draws
for unseen models. Judge reasoning tokens stay separate from the final-text
features used by the statistical model. Ordinary-output and reasoning counts
are reported separately for interpretation, but their sum is the single billed
`output_tokens` total priced at `price_out`; reasoning is not billed again.
Missing prices suppress only the currency subtotal for affected models;
request and token totals remain valid and the missing prices are named.

For automatic mode, the maximum judgment-call count before the both-correct
filter is:

`J * M * N * attempted_item_count * presentation_repetitions`.

Manual mode uses the number of retained rows in place of `J * M * N`. Actual
judgment calls use the simulated usable-item count and are therefore no larger
than this maximum.

### Reliability output

Every result records both univariate feature-range checks and joint support.
Joint support uses standardized nearest-neighbour distance to the training
design, with the warning threshold set from the training rows' leave-one-out
distance distribution. Model fitting also reports predictor correlations and
the design-matrix condition number. This catches the capability/date corner
occupied by newer-and-stronger automatic designs without adding a fragile
high-dimensional convex-hull rule.

Extrapolation is permitted for intentional future-model scenarios but is
visibly flagged. Runs with missing required features or disconnected graphs
must produce an explicit failure/partial-result reason rather than a plausible
number.

## Dashboard

After the command-line fit and simulator pass validation, extend the existing
dashboard with two input tabs:

- **Automatic:** judge pool/selection, `J`, anchor pool, `M`, `N`, attempted
  item count, repetitions, strength filter, and seed.
- **Manual:** an editable table of `(judge, M1, M2)` rows using catalogue-backed
  model selectors.

Both tabs show the same results views: design summary and warnings, cost,
capability-gap curve, scaling curve, ranking recovery, and manual-row table
where applicable. The cost view includes a per-model token table, reasoning
class, observed-versus-predicted status, and 5th/median/95th percentiles. The
dashboard edits selections and filters; durable metadata changes are made in
the catalogue so runs remain reproducible.

The dashboard does not fit the hierarchical model interactively. It loads a
versioned fitted artifact and runs posterior prediction. Refit remains an
explicit offline command.

## Validation

### Predictive validation

- Leave one judge out to test the conditional population prediction used for
  unseen judges.
- Compare the capability/date-conditioned unseen-judge prediction with a
  pooled zero-covariate judge baseline. Enable and label judge-type
  extrapolation only when it improves held-out prediction beyond
  cross-validation uncertainty.
- Leave one candidate model out to test unseen-model predictions.
- Use held-out log loss, Brier score, and calibration plots for judgment
  probabilities.
- Run posterior predictive checks by capability gap, oversight level, item
  difficulty, release-date difference, length difference, and presentation
  order.
- Check length-difference calibration within item-difficulty, correctness, and
  visible-reasoning strata, not only in aggregate.
- Leave one model configuration out of each token model and report held-out
  log-scale error plus empirical coverage of the 90% posterior-predictive
  interval, separately for reasoning and non-reasoning configurations and for
  candidate versus judge roles.
- Compare against the simplest capability-gap-only hierarchical baseline.
  Retain extra interactions only when they improve held-out prediction beyond
  cross-validation uncertainty.

### Simulation validation

- Reproduce observed aggregate choice rates, judge differences, position
  effects, and capability-gap curves from posterior predictive draws.
- Downsample the observed experiment across `J`, `M`, `N`, and attempted-item
  count and compare simulated uncertainty/recovery with empirical resampling.
- Treat the complete observed dataset or its fitted posterior as the reference
  for ranking-recovery tests, naming which one is used in every output.
- Verify fixed-seed reproducibility and the specified sign changes under
  presentation swapping, including mapping left/right draws back to M1/M2.
- Test insufficient-pool, missing-metadata, missing-price, extrapolation, and
  disconnected-ranking paths.

Validation reports all diagnostics; it does not hide failed slices behind an
overall average. A more complex statistical specification is rejected when it
does not outperform the simpler baseline out of sample.

## Minimal component boundaries

1. **Preparation:** extend the existing GPQA hierarchical-model preparation
   and reuse `src/irt` outputs to produce validated tables and scaling
   metadata. `scripts/hibayes/prepare_gpqa.py` supplies judge usage and
   transcript lengths; a small reader following `scripts/cost/eval_cost.py`
   supplies candidate-generation usage from the answer-generation `.eval`
   logs.
2. **Fit:** extend `scripts/hibayes/model.py` and save versioned posterior and
   predictor-scaling artifacts.
3. **Simulate:** one core module accepts a saved fit plus automatic or manual
   configuration and returns tidy trial-level votes and summary tables. The
   easy-to-hard endpoint passes those votes to the existing Bradley-Terry,
   component, and LOMO helpers rather than adding another model framework.
4. **Validate:** command-line analyses perform held-out and recovery checks
   using those same tables.
5. **Display:** the existing dashboard reads simulator tables/artifacts and
   adds controls only after validation.

Do not introduce a second model framework, live service, job queue, database,
or custom plugin system. The existing NumPyro/Pandas/IRT/dashboard stack is
sufficient.

## Delivery order

1. Finalize the model catalogue and prepared training table, with validation.
2. Extend and validate the hierarchical judgment model.
3. Implement the deterministic-seed simulation core and command-line entry
   point for both modes.
4. Run predictive and simulation-recovery validation; simplify the model if
   extra terms do not generalize.
5. Fit/validate token prediction, then add cost summaries and the dashboard
   controls/views.

The implementation plan must name exact files and checks for each step. No
dashboard work begins before the fitted model and command-line simulation pass
their held-out checks.
