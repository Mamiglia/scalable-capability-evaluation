# The judge-capability GLM pipeline

## What this document is

This specifies the analysis pipeline end to end. It starts with model rollouts on a benchmark and ends with a per-feature verdict about judge behaviour. It states what each stage does, why it does it that way, and what the output licenses you to claim.

The reader is assumed to know regression and Bayesian inference. No knowledge of this project is assumed.

## The question the pipeline answers

An AI judge reads two transcripts and picks the one it believes came from the more capable model. We want to know whether that judgement rests on the quality of the answer or on the appearance of the transcript.

The pipeline separates those two things. It estimates two quantities for every measurable feature of a transcript:

1. How strongly the judge's choice moves with that feature.
2. Whether that model-level feature is associated with held-out GPQA accuracy.

A feature that the judge follows but that has no detected association with held-out accuracy is a measured bias. That comparison is the product of the pipeline.

## The estimand

State the target quantity before the method, because the method only makes sense against it.

Let a comparison be one judge, one question item, and two transcripts. One transcript comes from the *major* model and one from the *minor* model. Major is the model with the higher aggregate benchmark accuracy.

For feature `f`, let `d_f` be the feature value of the major transcript minus that of the minor transcript. The pipeline divides `d_f` by its standard deviation over contested comparisons. For the primary analysis, it then subtracts the mean for the model pair. Call this standardized within-pair deviation `w_f`. The pipeline estimates `beta_f` in

    logit P(judge picks the major transcript) = alpha_judge + gamma_judge * position + beta_f * w_f + u_item

`beta_f` is the change in log-odds of picking the major model for a one-standard-deviation increase in the raw feature difference. The increase is relative to the usual difference for that model pair. It is not a claim about how the feature arises or about causation.

Three properties of this estimand matter:

- It is signed against the major model, not the left slot. The same pair shown in reverse order produces the same `d_f`, so the estimate does not flip with presentation order.
- It is conditional on the judge and the item, because both enter as separate terms. A feature coefficient cannot absorb the fact that one judge is lenient or one question is hard.
- It is a population effect with per-judge deviations pooled around it. The reported number is the population mean unless stated otherwise.

This repository preserves that estimand and the completed fits. It does not implement the alternative controlled-pair design from the separate `gpqa_tfa` analysis. That design would add leave-one-model-out capability estimates, date matching or date controls, a model-pair effect, and clustered uncertainty. Those changes require new prepared data and new fits. They are future experimental work, not a code-integration step.

## Stage 1: Rollouts

Each candidate model answers every item of the benchmark. The benchmark is GPQA Diamond, a set of graduate-level multiple-choice science questions. Each rollout is stored as an eval log with the model's full response and a per-item score.

A model's *capability* is its aggregate accuracy over that benchmark. This is one scalar per model. Its limitations are discussed under Threats to validity.

## Stage 2: Pair construction

`src/judge/dataset.py` builds the comparisons. Four decisions define the design.

**Both models must have answered the item correctly.** `require_both_correct` defaults to `True`. It was also `True` in the production run. That was verified by reading the run metadata, not by trusting the default. Only samples the model scored correctly are eligible, and a pair is formed only where both models have an eligible sample for the same item.

This is the central design choice. Because both answers are right, the outcome carries no information about which model is stronger. The judge cannot succeed by grading the answer. Any signal it finds must come from how the answer is written. This is what makes the method a candidate for benchmarks that have saturated.

**Both presentation orders are run.** `include_reverse` defaults to `True`. Every pair appears with each model in the first slot. Position effects are large and differ in direction between judges, so pooling both orders is what makes an aggregate estimate interpretable. Do not analyse a single order.

**Some conditions equalise transcript length.** `truncate_pair` cuts both transcripts to the same number of characters. The count is a fraction of the shorter transcript. It also removes the trailing answer line. Conditions at 10%, 50% and 90% were run alongside the untruncated condition. This combined intervention tests length equalisation and answer removal together. It does not identify either effect alone.

**Reasoning blocks are excluded.** `include_reasoning` was `False` in the production run. The judge saw the answer text, not the chain of thought.

The prompt shows the question, the multiple-choice options as each model saw them, and the two transcripts. It asks which transcript came from the more capable model.

## Stage 3: The judge run

Ten judges each process the full comparison set. Each returns a forced choice and a confidence value. The judges are small and mid-sized models. No frontier model was used as a judge, which bounds what the results say about the method's ceiling.

## Stage 4: Feature extraction

`scripts/hibayes/transcript_features.py` computes 138 features from each transcript, grouped into 8 families:

| Family | Features | What it captures |
|---|---|---|
| function_word | 40 | rates of individual function words |
| formatting | 33 | markdown, headings, lists, code blocks |
| reasoning_lexicon | 22 | hedging, verification and conclusion words |
| process | 20 | corrections, alternatives, plan position |
| length_structure | 9 | characters, words, lines, paragraphs |
| lexical_diversity | 6 | type-token ratio and related measures |
| punctuation_casing | 4 | digits, capitals, symbol rates |
| answer_behaviour | 4 | option tokens and answer masking |

For each comparison the pipeline stores the difference `major minus minor` for every feature. Signing against the major model rather than the left slot is what keeps the value stable when the pair is reversed.

`scripts/hibayes/gpqa_features.py` contains the minimum local feature contract from the separate `gpqa_tfa` analysis. The file records the source commit and source-file hashes. It includes only the lexical and process features, the `answer_masked` view, and the visibility rule that this GLM uses. The pipeline does not import code from a sibling checkout.

The optional self-check compares 137 recomputable features and the rebuilt `answer_masked` view with frozen per-record reference files. The 138th feature, `reasoning_to_response_ratio`, has no upstream equivalent for a judge transcript because the judge sees one concatenated text. The pipeline stores it as missing and omits it from fitting. Run the self-check after any change to the local feature contract.

## Stage 5: Data preparation

`scripts/hibayes/prepare_gpqa.py` assembles the analysis table. Two derived quantities need explanation.

**The outcome.** `is_correct` is 1 when the judge picked the major model. Major is defined by aggregate accuracy. So "correct" means the judge agreed with the benchmark ranking of the two models. It does not mean the judge picked the right answer.

**The within and between decomposition.** Each feature difference is scaled, then split into two parts:

    between = mean of the scaled difference over all items for that model pair
    within  = the scaled difference minus that pair mean

The between part is a property of the model pair. The within part is how far one particular item deviates from what that pair usually does.

This split is the pipeline's main defence against a confound. A judge that has learned to recognise which models are involved could score well using the between part alone, without reading the transcript. The within part removes a fixed pair-level difference because it varies inside a pair. It can still contain item-specific model signatures. The within specification is therefore the primary one, and a result that holds only between pairs is much weaker.

The analysis sample is complete cases across all predictors, so every fit uses the same rows and coefficients are comparable across families.

## Stage 6: Specification generation

`scripts/hibayes/fit_features.py` writes one model configuration per specification. For each of the 8 families it produces three kinds:

- **marginal**: one feature, plus the position term. One fit per feature. This is the clean per-feature estimate.
- **joint**: every feature of the family together, plus position. Shows the feature's effect with its family siblings held fixed.
- **adjusted**: the joint block plus the capability gap. A coefficient here is the response that remains after real quality is accounted for.

That is 152 fits per comparison set: 8 joint, 8 adjusted, 136 marginal.

**Collinearity is checked before fitting, not after.** For each joint block the pipeline computes a condition number. Where a block is a single construct measured several ways, its coefficients are not separately interpretable. Blocks above the threshold of 15 are refitted on the principal components that span 90% of the block's variance, and the substitution is recorded. Some blocks in the existing sweep were numerically singular, with condition numbers above 10^15. Component scores have no counterpart in the capability analysis, so they are reported in their own category rather than treated as failures.

The four comparison sets are `within`, `between`, `crossfam`, and `samefam`. `crossfam` restricts the within analysis to different model-name families. `samefam` restricts it to the same model-name family and contains five pairs. A model-name family is only the leading alphabetic token of the model name; it is not a verified developer identity. Each set needs its own output directory, because run names encode the feature family and specification but not the comparison set.

## Stage 7: The model

`scripts/hibayes/model.py` defines a Bernoulli GLM with a logit link. The linear predictor combines:

- a per-judge intercept, drawn around a population mean
- a per-judge position effect, drawn around a population mean
- per-judge predictor effects, pooled around a population mean per predictor
- per-judge transcript-condition effects, with the untruncated condition as the reference
- a random effect per question item

Priors are weakly informative: `Normal(0, 1.5)` on intercept-scale terms, `Normal(0, 0.5)` on effects, and `HalfNormal(0.5)` on the hierarchical scales. On the log-odds scale a prior standard deviation of 0.5 puts most mass below a doubling of the odds. That is deliberately conservative for standardised predictors.

All hierarchical terms use a non-centred parameterisation. This parameterisation generally improves sampling when a group has little data. The historical fits had no divergences, but this analysis did not compare centred and non-centred versions.

Sampling uses Hamiltonian Monte Carlo with 4 chains, 2,000 retained draws per chain, `target_accept` 0.95, and seed 0. The two headline fits use 1,000 warmup iterations. The feature fits use 2,000 warmup iterations.

## Stage 8: Diagnostics

Each generated configuration requests these checks. The completed historical sweep was also audited across all fit directories. The collector verifies configuration and source provenance where available, but it does not enforce the diagnostic thresholds. Apply these thresholds before reporting a coefficient:

1. **Divergences must be zero.** A divergence means the sampler could not traverse the posterior geometry, and the draws cannot be trusted. This is the check that most reliably invalidates a fit.
2. **R-hat must be below 1.01.** Values are stored rounded to two decimals, so a reported 1.01 cannot be distinguished from 1.005. Treat a boundary value as a prompt to raise the draw count, not as a failure. Doubling the draws moved both headline models from 1.010 to 1.000 with no other change.
3. **Effective sample size should reach 1,000.** Below about 400 the posterior summaries are unstable.
4. **BFMI should exceed about 0.3.**
5. **Pareto k should stay below 0.7.** Where it does not, the leave-one-out estimate for that fit is unreliable. This affects accuracy estimates and model comparison. It does not affect the coefficients.

Record which fits fail which check. Do not discard them silently.

## Stage 9: Verdicts

Two tests decide what a feature is doing.

**Does the judge follow it?** The pipeline approximates each coefficient posterior with a normal distribution that has the reported mean and 95% interval width. It converts that approximation to a two-sided tail probability. Benjamini-Hochberg correction is applied within each family and again across every feature of a specification. A feature passes at the global `q < 0.05` threshold.

A credible interval alone is not treated as a finding. At roughly 91,000 rows the intervals are narrow enough that trivial effects exclude zero, so an interval-based rule would report almost everything.

**Does it track capability?** A separate model-level analysis splits the GPQA questions in half. It measures each feature on one half and aggregate accuracy on the other. It then estimates their association with model-family-by-release-year fixed effects, a within-cell release-date term, and standard errors clustered by model family. Benjamini-Hochberg correction is applied across the 138 hand-built features. `scripts/hibayes/capability_features.csv` freezes those 138 rows and the seven fields that this pipeline uses. `transcript_features.py` checks the file hash and exact feature set before loading it. The pipeline uses that analysis's discovery flag.

Crossing the two tests gives four verdicts:

| | judge does not follow | judge follows |
|---|---|---|
| **tracks held-out accuracy** | missed signal | working as intended, or inverted if the signs disagree |
| **no detected capability signal** | correctly ignored | judge bias |

**The verdict comes from two significance tests.** It does not come from the size of the two coefficients. A feature can have a large apparent association with held-out accuracy and still count as having no detected capability signal. That happens when the estimate fails its own test. A figure that places features by coefficient magnitude and then labels the quadrants will mislabel points. Show the two tests directly.

## Stage 10: Evaluation-method analysis

The stages above describe per-comparison behaviour. That is the wrong unit for judging the method as an evaluation tool, because no one ranks two models from a single comparison. `scripts/hibayes/evaluation_power.py` computes the unit that matters.

**Create one vote per question item before pooling a model pair.** The logs repeat some items across judges, presentation orders, and transcript conditions. The pipeline first averages those rows. It records a vote of 1 above 0.5, 0 below 0.5, and 0.5 for a tie. It then pools the item votes for each model pair. This prevents repeated views of one question from counting as independent evidence.

**Compute a power curve.** Draw `k` question-item votes with replacement for each pair, take the majority, and repeat. Sweep `k` and record how often the pooled verdict is right. The output includes a 95% item-bootstrap interval. It marks values above the smallest observed item count in a gap group as extrapolations.

- If accuracy rises towards 1 as `k` grows, the errors are noise and more comparisons fix them.
- If accuracy plateaus below 1, the errors are systematic and more comparisons never fix them.

The script weights model pairs equally for pair-level claims. Analyses that compare panels use the same model-pair set on each side.

## Status of the existing evaluation outputs

The code cleanup did not regenerate data, posterior samples, tables, JSON files, or figures. The existing `evaluation_power.json` pools repeated rows into question-item votes for the pair-level and power-curve analyses. However, that artifact predates the final reviewed code. Its optional feature-diversity and ensemble analyses used control coefficients as if they were transcript features. Do not cite those optional analyses. Regenerate the complete evaluation into a new directory before treating all sections as one coherent result set.

The existing posterior fits remain the inputs to the feature analysis. The cleanup retains their model formula, priors, posterior variable names, and primary estimand. Localizing the feature contract and capability table did not regenerate prepared data or posterior fits. The communicator accepts both stored layouts of the transcript-condition contrast so that old posterior files do not require regeneration.

Run the final evaluation stage through HTCondor before reporting a complete set of evaluation values. The run must write to a new output location if the previous artifacts must remain available.

## Threats to validity

State these with any result from this pipeline.

1. **One benchmark, one domain.** Everything is GPQA Diamond multiple-choice science. Nothing here supports a claim about other domains.
2. **Capability is a single scalar from the benchmark being judged.** Two models 0.03 apart may not differ meaningfully. Part of the small-gap failure may be that the label is noise rather than that the judge is wrong. An independent capability measure would separate these.
3. **The length control changes two things at once.** Truncation equalises length and removes the final answer. The improvement cannot be attributed to length alone.
4. **Saturation and small gaps are confounded by construction.** Every near-saturated pair here has a gap below 0.035. No high-capability pair has a large gap. The within-band comparison in Stage 10 is suggestive, not decisive. One pairing of a strong model against a much weaker modern model would resolve it.
5. **All judges are small or mid-sized.** Whether a frontier judge clears the small-gap plateau is untested.
6. **`samefam` rests on five model pairs.** Its leave-one-out estimates are the least reliable in the set.
7. **Aggregate accuracy defines which model is "major".** Where the gap is near zero that assignment is close to arbitrary. Accuracy against it is correspondingly hard to interpret.
8. **The current GLM has no model-pair effect.** The within-pair feature decomposition limits model-recognition confounding. The likelihood does not model residual dependence within a model pair. Adding that term changes the model and requires a separate refit.

## Reproduction

Run the stages in order. Each command is one step.

1. Optionally run `pipeline.py check-features` against the frozen `/fast` reference files and confirm that it reports parity. This check does not need a `gpqa_tfa` source checkout.
2. Prepare the data with `pipeline.py prepare` and explicit input and output paths.
3. Generate each comparison set with `pipeline.py generate-fits`.
4. Submit one cluster job for each generated configuration.
5. Check every fit against the Stage 8 thresholds before reading a coefficient.
6. Build each bias map with `pipeline.py collect GENERATED_FIT_DIRECTORY`.
7. Run the evaluation with `pipeline.py evaluate`.
8. Generate the figures with `pipeline.py figures`.
9. Record the commands, cluster IDs, checks, and artifact validation.

Exact commands, cluster IDs, and the run log are in `COMMANDS.txt`. The stage modules remain executable for compatibility, but `pipeline.py` is the supported interface. Do not prepare data without explicit output paths. The defaults point at the analysis directory and can replace existing prepared tables.

## Where the pipeline should go next

Ranked by what each would settle.

1. **A second benchmark in a different domain.** This is the only way to address generalisation, and nothing in the current data speaks to it.
2. **A strong-versus-weak pairing at high absolute capability.** For example a model near 0.93 against one near 0.67. This separates saturation from small-gap difficulty, which is currently the largest interpretive gap.
3. **An equal-length condition that keeps the answer.** This isolates length from answer removal and would show whether length normalisation is a deployable fix.
4. **One frontier judge on the existing comparison set.** This tests whether the small-gap plateau is a property of the method or of the judges used.
5. **An independent capability measure.** Anything not derived from the benchmark under test would let you check whether small gaps are real differences.
