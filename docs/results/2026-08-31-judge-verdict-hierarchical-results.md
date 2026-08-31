# What guides judge verdicts? Hierarchical refit results

## Main result

The corrected hierarchical Raw-L1 model predicts **72.66%** of held-out judge
verdicts (item-clustered 95% CI 71.67–73.61%; log loss 0.5379). This is
**84.4–93.0% of the empirical repeat-agreement ceiling** of 78.13–86.06%, using
direct `accuracy / ceiling`, and a **20.4% reduction in held-out deviance** from
the intercept-only model.

To obtain orthogonal axes, we projected the standardized raw composites into
one global 30-component PCA basis and fitted the same hierarchical GLM with L1
regularization. The PC model retained almost all predictive performance:
**72.54% accuracy**, **84.3–92.8% of ceiling**, and **19.4% held-out deviance
recovery**. Raw and PC L2 sensitivity fits were similarly accurate (72.61% and
72.41%), so the L1 models are the primary results below.

The substantive picture is:

1. Judges are predictable, but predictability is far below repeat agreement
   for the most internally consistent judges.
2. Judges favor efficient, rigorous, clear and directly progressing answers;
   self-correction, hedging, think-aloud language and annotated errors predict
   rejection.
3. The dominant orthogonal predictor is PC2, a direct-efficiency axis. PC1
   mostly represents trajectory breadth/length and is not strongly predictive.
4. Effects vary substantially by benchmark. In particular, PC2 is uniquely
   important for InterCode, while several structural/tool features matter for
   InterCode and SWE but little for AIME or GPQA.
5. Failures concentrate on unstable or intrinsically ambiguous verdicts:
   order inconsistency, choosing the weaker model, low cross-judge consensus,
   and small capability gaps. Reasoning-mode and release-year mismatch add
   little explanatory value.

![Overall held-out performance](../../judge_results/verdict_interpretability/hierarchical-refit-20260831-report-final/figures/performance.svg)

## Evaluation and model

- Six judges, four benchmarks and 113,951 held-out verdict rows.
- Sixteen outer folds: four held-out source-model groups × four held-out item
  folds.
- Raw representation: 112 final correlation composites, standardized within
  benchmark before forming antisymmetric left-minus-right pair features.
- PC representation: one outcome-blind global PCA basis shared across every
  fold, judge and benchmark. PC scores were divided by the square root of the
  corresponding global explained variance; the first 30 PCs retain 71.21% of
  total feature variance.
- All coefficient blocks are regularized. Population, judge, benchmark,
  judge×benchmark and item relative penalties were estimated from each
  outer-training fold by training-likelihood variance components. Only the
  global strength `C` was selected by inner grouped CV.
- The selected hierarchy ratios and `C` were frozen for the intercept-only,
  Singular, Unique, block, family and capability-companion fits. No lesion
  retuned a hyperparameter.
- Every published fold and lesion converged. Raw models have 252 labelled
  lesion specifications and 6,300 scoped effect rows; PC models have 60 and
  1,500. All final manifests and hashes validated.

The Raw-L1 fitted ratios varied across outer folds: judge 1.46–1.96,
benchmark 0.88–1.02, judge×benchmark 1.56–2.19 and item 0.89–1.78 relative to
population effects. PC-L1 ratios were larger for judge (4.86–10.17),
judge×benchmark (3.41–5.22) and item (1.64–2.52), while benchmark ratios stayed
near one (0.93–1.10). These are fitted relative shrinkage ratios, not fixed
penalties.

## Prediction ceiling by judge

The ceiling is an empirical repeat-agreement reference, not a mathematical
upper bound. The lower endpoint is the row-weighted Wilson 95% lower bound;
the upper endpoint is raw historical-to-current agreement. Because the lower
endpoint is an uncertainty bound, an accuracy ratio can slightly exceed 100%.

| Judge | Ceiling range | Raw-L1 accuracy | Accuracy / raw ceiling | Deviance recovery |
|---|---:|---:|---:|---:|
| Llama 8B | 64.9–74.2% | 62.8% | 84.6% | 3.7% |
| DeepSeek R1 | 75.5–83.9% | 75.6% | 90.1% | 22.6% |
| DeepSeek V4 | 77.4–85.6% | 73.4% | 85.7% | 24.3% |
| Gemma 4 | 84.0–91.2% | 74.2% | 81.4% | 25.0% |
| GPT-OSS | 83.1–90.4% | 74.8% | 82.7% | 21.1% |
| o4-mini | 83.5–90.8% | 75.7% | 83.4% | 26.0% |

DeepSeek R1 has the largest recovered fraction of its raw repeat ceiling.
Llama 8B has both the lowest ceiling and by far the least deviance recovery,
suggesting that much of its residual verdict variation is not stable enough to
be learned from these features.

![Accuracy divided by empirical ceiling](../../judge_results/verdict_interpretability/hierarchical-refit-20260831-report-final/figures/accuracy-vs-ceiling.svg)

## Prediction by benchmark

| Benchmark | Raw-L1 accuracy | Raw repeat ceiling | Accuracy / ceiling | Deviance recovery |
|---|---:|---:|---:|---:|
| AIME | 61.3% | 89.1% | 68.8% | 4.2% |
| GPQA | 72.9% | 85.0% | 85.8% | 22.5% |
| InterCode | 71.9% | 86.8% | 82.8% | 20.2% |
| SWE-bench Mini | 77.4% | 84.4% | 91.8% | 21.7% |

AIME is the clear failure case: accuracy is only 61.3% despite high repeat
agreement, and the model recovers only 4.2% of intercept-relative deviance.
SWE is easiest in accuracy and fraction-of-ceiling terms. Full performance,
effect and coefficient packs are available by benchmark, judge and every
observed judge×benchmark cell in the [figure index](../../judge_results/verdict_interpretability/hierarchical-refit-20260831-report-final/figures/index.md).

![Performance by benchmark](../../judge_results/verdict_interpretability/hierarchical-refit-20260831-report-final/figures/performance-by-bench.svg)

## What raw features do judges use?

`Unique` is the held-out log-loss increase after removing a feature from the
complete model. `Singular` is the held-out log-loss improvement of a model
containing only that feature relative to the intercept-only model. Positive
values indicate useful predictive information; they are predictive, not
causal, effects.

The strongest individually necessary raw predictors were:

| Feature | Unique Δ log loss | Singular Δ log loss | Conditional direction |
|---|---:|---:|---:|
| Self-corrections | 0.0048 | 0.0204 | −0.185 |
| Think-aloud language / 100 words | 0.0034 | 0.0104 | −0.050 |
| Hedging count | 0.0034 | 0.0033 | −0.164 |
| Depth | 0.0029 | 0.0418 | +0.157 |
| Efficiency | 0.0026 | 0.0434 | +0.314 |
| Rigour | 0.0015 | 0.0541 | +0.157 |
| Submit calls | 0.0010 | 0.0087 | approximately zero globally |
| Any error / error count | 0.0008 | 0.0050 | −0.198 |

The direction column is the median population slope per one-SD pair-feature
difference. Judges therefore tend to select the answer that is more efficient,
rigorous and deep, and reject the answer with more self-correction, hedging,
think-aloud language or annotated errors, conditional on the other features.

![Raw feature effects](../../judge_results/verdict_interpretability/hierarchical-refit-20260831-report-final/figures/raw-feature-effects.svg)

The grouped lesions are stronger and clearer than most individual features:

| Feature set | Unique Δ log loss | Singular Δ log loss |
|---|---:|---:|
| All annotation features | 0.0573 | 0.0990 |
| Rubric block | 0.0403 | 0.0997 |
| Hand-engineered text features | 0.0085 | 0.0738 |
| Structural features | 0.0067 | 0.0630 |
| Intermediate-error annotations | 0.0036 | 0.0090 |

Rubric/annotation variables carry the largest conditional signal. Structural
and ordinary text features still predict verdicts, but much of their
standalone signal is redundant with the broader feature set.

![Raw group effects](../../judge_results/verdict_interpretability/hierarchical-refit-20260831-report-final/figures/raw-group-effects.svg)

## Orthogonal PCA interpretation

The PCA model loses only 0.12 percentage points of accuracy relative to Raw L1
while making axes orthogonal. The most important PCs are:

| PC | Interpretation | Variance | Unique Δ log loss | Direction |
|---|---|---:|---:|---:|
| PC2 | Direct efficiency | 7.60% | 0.1381 | +0.459 |
| PC3 | Symbolic/backtracking | 5.41% | 0.0225 | −0.480 |
| PC5 | Sequential verification | 2.87% | 0.0168 | +0.352 |
| PC4 | Rigorous clarity | 4.10% | 0.0126 | +0.366 |
| PC29 | Planned backtracking | 0.88% | 0.0047 | +0.158 |
| PC1 | Trajectory breadth/length | 17.65% | 0.0035, CI includes 0 | +0.322 |

PC2 contrasts efficient, clear, prose-forward trajectories with deep,
redundant, depth-heavy ones. PC4 loads positively on rigour, clarity and
efficiency and negatively on reasoning errors and unjustified claims. PC5
loads on sequential structure, explicit verification, depth and rigour. PC1
contains much of the raw length/trajectory variation—including word/log/tool
volume—but its Unique and Singular intervals include zero. This is why word
count does not appear as the dominant verdict axis even though length-related
features contribute strongly to PC1.

![PC effects](../../judge_results/verdict_interpretability/hierarchical-refit-20260831-report-final/figures/pc-effects.svg)

![PC1 loadings](../../judge_results/verdict_interpretability/hierarchical-refit-20260831-report-final/figures/pc1-loadings.svg)

![PC2 loadings](../../judge_results/verdict_interpretability/hierarchical-refit-20260831-report-final/figures/pc2-loadings.svg)

L1 did not produce a genuinely sparse PC model: all 30 PCs were nonzero in all
16 folds. Its value here is regularization plus orthogonality; the lesion
effects, rather than zero coefficients alone, identify the small set of
important axes. Raw L1 was more selective but still retained 57/112 features
in every fold.

## Benchmark and judge heterogeneity

Effects are not well described by “one benchmark only.” The report therefore
ranks heterogeneity by the larger of the Unique or Singular max–min ranges and
provides full cross-benchmark tables.

- PC2 has a Unique effect of 0.234 on InterCode but only 0.0019 on GPQA,
  0.0169 on AIME and 0.0122 on SWE.
- PC1 is helpful alone on SWE (+0.054) and AIME (+0.023), near zero on
  InterCode, and harmful on GPQA (−0.042).
- PC3 is helpful alone on GPQA (+0.058) but harmful on SWE (−0.022).
- The bare-answer/prose composite is harmful alone on AIME and GPQA but
  helpful on InterCode and SWE.
- Depth is strongly helpful alone on GPQA and SWE, moderately helpful on
  InterCode, and harmful on AIME.
- Tool-transition features such as another-tool-after-tool and verification
  after tool use show their main standalone signal jointly on InterCode and
  SWE, rather than on only one benchmark.

![Raw benchmark heterogeneity](../../judge_results/verdict_interpretability/hierarchical-refit-20260831-report-final/figures/raw-benchmark-heterogeneity-singular.svg)

![PC benchmark heterogeneity](../../judge_results/verdict_interpretability/hierarchical-refit-20260831-report-final/figures/pc-benchmark-heterogeneity-singular.svg)

Judge×benchmark packs show important local exceptions. For example, Raw-L1
deviance recovery is negative for Llama 8B on SWE (−5.0%), but 31.7% for Gemma
and 32.7% for DeepSeek V4 on SWE. The poorest cells are predominantly AIME and
Llama cells; the strongest are GPQA/SWE cells for the larger judges.

## When can and cannot verdicts be predicted?

The PC-L1 model's overall prediction error is 27.46%. Errors are concentrated
where the observed verdict itself looks unstable or difficult:

| Factor | Change in prediction error |
|---|---:|
| Order-inconsistent verdict | +14.4 pp (95% CI +8.7 to +20.1) |
| Judge chose the weaker model | +11.3 pp (+7.5 to +15.1) |
| Configured-reasoning mismatch | +0.2 pp (−3.5 to +3.9) |
| Mixed reasoning configuration | −0.3 pp (−1.7 to +1.1) |
| Both candidates newer than judge | −1.8 pp (−5.2 to +1.7) |

Among correctly predicted verdicts, the judge chose the weaker model 34.6% of
the time; among prediction failures, this rises to 49.0%. Prediction error is
38.4% for order-inconsistent verdicts versus 23.3% for consistent verdicts.

Low cross-judge consensus is the strongest descriptive failure marker: error
falls from 39.1% in the lowest-consensus quartile to 11.9% for unanimous cases.
Small capability gaps are also harder (32.7% error in the lowest-gap quartile),
while candidate release-year gap and judge–candidate release distance show no
clear monotonic relationship. Model confidence behaves as expected, with error
falling from 44.0% to 10.9% across confidence quartiles.

![Prediction failure factors](../../judge_results/verdict_interpretability/hierarchical-refit-20260831-report-final/figures/prediction-failure-factors.svg)

![Prediction failure quartiles](../../judge_results/verdict_interpretability/hierarchical-refit-20260831-report-final/figures/prediction-failure-quartiles.svg)

## Capability-difference companion

Adding the benchmark-normalized source-model capability difference to the
already fitted Raw-L1 full model raises accuracy from 72.66% to **73.39%**
(+0.72 pp) and improves log loss by 0.0123. The PC-L1 companion gains 0.59 pp
and 0.0106 log-loss units. The companion reuses the primary model's fitted
hierarchy ratios and `C`; it is not retuned and is not lesioned.

Capability therefore contributes real predictive information, but the small
increment implies that the main verdict signal is not reducible to judges
merely recognizing the stronger source model.

![Capability companion](../../judge_results/verdict_interpretability/hierarchical-refit-20260831-report-final/figures/capability-companion.svg)

## L2 sensitivity

L2 does not change the performance conclusion: Raw L2 reaches 72.61% accuracy
and log loss 0.5394; PC L2 reaches 72.41% and 0.5460. Raw L1 is marginally best
on both accuracy and log loss, so comparisons between L1 and L2 are not a main
result.

## Artifacts and provenance

- Final results/tables/figures:
  `judge_results/verdict_interpretability/hierarchical-refit-20260831-report-final/`
- Full fit and lesion bundle:
  `judge_results/verdict_interpretability/hierarchical-refit-20260831-final/`
- Figure index:
  `judge_results/verdict_interpretability/hierarchical-refit-20260831-report-final/figures/index.md`
- Final report code: `3035cdc`.
- Final Raw-L1 downstream jobs: finalization `3494291`, summary `3494292`,
  judge×benchmark cells `3494293`.
- Definitive report render: `3494333`.
- Verification: 18 judge tests passed; four fit, four capability, four lesion
  and four cell manifests validated; final figure audit found 276 PNG/SVG
  pairs, zero SVGs without editable text, 27 tables, and a complete index.

All PNGs have same-stem editable SVGs. The report distinguishes full-model
prediction, Unique contribution, Singular contribution, coefficient direction
and PCA loading; none should be read as a causal effect.
