# Signal in saturated transcripts: a hierarchical GLM over protocols and domains

**Status:** Design, revised 2026-09-16 after discussion. Implementation exists (flat and
hierarchical fits, full data); the semester subset and the (1 | task) removal are pending.

**Question (RQ1, figure 2).** Which capability-estimation protocols beat the saturated-item
prediction, and is that consistent across domains? The results table answers this one
(domain, semester) task at a time. This analysis fits one model over every task so the
protocol and domain effects are estimated jointly.

## Data

The forecasts table written by the results runner (`analysis/results/forecasts.csv`),
restricted to one row per target and protocol with `analysis.stats.one_per_target`
(transcript protocols as they are, judge protocols as the best judge on references) and to
the variants the results table reports (pairwise `linear`, direct `raw`, embedding `full`).
The mean over judges and the accuracy oracle are excluded.

**Semesters: 2025-II and 2026-I only.** Every domain has both, so the design is balanced:
4 domains x 2 semesters = 8 tasks, 8 targets each, every protocol in every task. 16 targets
per protocol per domain, 384 rows.

One row per (target, protocol):

| column | meaning |
| --- | --- |
| `y` | squared forecast error of that target, (θ̂ − θ\*)² |
| `protocol` | `easy_items` (the saturated-item prediction), `glm_length`, `glm_regex`, `embedding`, `direct`, `pairwise` |
| `domain` | science, math, coding, cyber |
| `task` | domain × semester, kept for the cross-check's semester term |

Exact zeros are clipped to 1e-6 so the Gamma likelihood is defined.

**The baseline is a level of `protocol`, not a separate quantity.** The saturated-item
prediction forecasts every target like any other protocol, so it has its own squared errors
and its rows sit in the table with `protocol = easy_items`. It is the reference category:
its coefficient is pinned to 0 and every other protocol's coefficient is measured against it.
Nothing is subtracted or normalised beforehand.

## Model

```
squared_error ~ protocol + domain + (protocol | domain)        Gamma family, log link
```

- **Outcome**: each target's squared error, the metric the descriptive plots average into MSE.
- **Gamma family**: squared errors are positive and right-skewed, with values spanning orders
  of magnitude. Gamma assumes the spread grows with the mean, so a large error is expected to
  be noisy and does not dominate the fit the way it would under a normal family.
- **Log link**: predictors add on the log scale, so effects multiply the mean. A protocol
  coefficient is the log of its expected squared error divided by the saturated-item
  prediction's; exp(coefficient) is an error ratio, and ratios are what make "consistent
  across domains" comparable when baseline error differs several-fold between domains.
- **`protocol`**: the pooled effect of each protocol across domains, relative to the
  saturated-item prediction (reference level, 0).
- **`domain`**: how the baseline's error differs between domains. Science is the reference:
  the intercept is the baseline's error in science and each other domain's coefficient is its
  difference from that. This is the only place the reference domain matters; protocol effects
  have no reference domain, because `α[p]` is the pooled effect and every domain, science
  included, has its own deviation `δ[p, d]`. Fixed, not pooled: four domains are too few to
  estimate a spread from.
- **`(protocol | domain)`**: partial pooling across domains, within each protocol. A
  protocol's four domain effects are draws from a distribution centred on its pooled effect,
  with one spread per protocol estimated from the data. Small spread: the domain estimates
  are pulled toward the pooled effect. Large spread: each domain keeps its own. The spread is
  the "how domain-dependent is this protocol" number. Deviations are uncorrelated across
  protocols (no correlation matrix; fewer parameters than the full lme4 `(protocol | domain)`).
- **No semester or task term.** On the balanced subset every protocol is run in every task, so
  a task that is hard for everyone raises every protocol's error by the same factor and cancels
  out of every protocol comparison. Leaving it out cannot bias a protocol effect; it only leaves
  task-level variation in the residual. (On the full 14-task data, where 2025-I is an outlier
  that only some domains have and domains have 2 or 5 semesters, a `(1 | task)` intercept
  would be needed. That is not the design here.)

### Coefficient form

```
y ~ Gamma(shape = ν, rate = ν / μ)
log μ = β₀ + α[protocol] + γ[domain] + δ[protocol, domain]

β₀       ~ Normal(0, 2)          baseline error in science, log scale
α[p]     ~ Normal(0, 1)          5 protocols; α[easy_items] = 0
γ[d]     ~ Normal(0, 1)          3 domains; γ[science] = 0
σ_p      ~ HalfNormal(0.5)       one per non-baseline protocol
δ[p, d]  = σ_p · z[p, d],  z ~ Normal(0, 1);   δ[easy_items, ·] = 0
ν        ~ Exponential(1)
```

Non-centred so the sampler behaves when a σ is near 0. Priors are on the log-ratio scale
and are chosen widths: Normal(0, 1) puts the ratio between 0.14 and 7 with 95% prior
probability; Normal(0, 2) puts the baseline's squared error between 0.02 and 55; HalfNormal(0.5)
says domain deviations are typically under 1 on the log scale (within ×2.7 of the pooled
value). With 384 rows they matter little; the prior predictive check verifies they do not
exclude the data's range.

## Reported quantities

Posterior median and 95% credible interval.

| quantity | reads as |
| --- | --- |
| exp(α[p]) | pooled error ratio of protocol p to the saturated-item prediction: which protocols work |
| exp(α[p] + δ[p, d]) | the ratio in domain d: for which domains |
| σ_p | how domain-dependent protocol p's effect is; near 0 = the same everywhere |

Below 1 beats the baseline.

## Cross-check

The flat model with the same family and link, every cell its own fixed coefficient and a
semester dummy in place of the pooling:

```
squared_error ~ protocol + domain + protocol:domain + semester
```

statsmodels, Wald 95% intervals. Its cells sit beside the pooled ones so the reader sees what
the pooling changed.

## Implementation

```
src/analysis/challenges/signal_glm.py     rows(forecasts, variants, semesters) -> the table;
                                          flat_glm(rows); ratio_table(posterior, coords); ratio_figure
scripts/analyses/signal_glm/model.py      the hibayes @model (numpyro)
scripts/analyses/signal_glm/config.yaml   hibayes pipeline: features, fit, checks, communicators
scripts/analyses/signal_glm.py            cell-script runner
tests/test_signal_glm.py                  synthetic errors with a planted ratio
```

Dependencies: `hibayes` (from GitHub, `rich` cap overridden), `statsmodels`.

## Outputs

- `analysis/results/signal_glm.csv`: protocol × domain plus a pooled row per protocol:
  `ratio, lo, hi` (hierarchical), `flat_ratio, flat_lo, flat_hi`, `sigma_protocol` on pooled rows.
- `analysis/results/figures/signal_glm/ratios.{svg,png}`: error ratio on a log axis, one
  panel per domain plus pooled, filled = hierarchical, open = flat, dashed line at 1.
- hibayes diagnostics under `analysis/results/signal_glm/`.

## Out of scope

All three judges with judge as a random effect; paired-improvement or beats-the-baseline
outcomes; protocol × semester; the full 14-task data (kept runnable via the `semesters`
argument, not reported).
