# Figure one and the signal GLM — handoff

State as of 2026-09-17, 17:45. Covers figure one of the paper ("do saturated transcripts carry
capability signal beyond accuracy"), the prediction score and its test, and the hierarchical Gamma
GLM that breaks the result down by domain and judge. Nothing here launches judging; every number
comes from `analysis/results/forecasts.csv` as written by `scripts/analyses/results.py`. The
signal_glm code is untracked and uncommitted (`git status`).

Protocol names are the paper's (CLAUDE.md): Deterministic features (`glm_regex`), Embedding,
Absolute scoring (`direct`), Pairwise judge (`pairwise`). Length alone (`glm_length`) is a
diagnostic and is left out of paper figures.

## The data

- 16 target models: the 8 targets of 2025-II and the 8 of 2026-I, the two semesters every domain
  has. Each is forecast in all four domains (science, math, coding, cyber): 64 (target, domain)
  cells, 6 protocols, 384 rows.
- One variant per protocol: saturated-item prediction (the references' mean θ\*, the baseline),
  Length (3 length features), Deterministic features (length + 9 lexical rates + log tool-call
  count, 13 features), Embedding (`full`), Absolute scoring (`raw`), Pairwise judge (`linear`).
  Judge protocols use the best judge on references per task unless stated.
- All targets are open-weights and every transcript is full length. Closed-frontier targets and
  truncated transcripts are separate challenges and are not in this data.

## The prediction score and its test

Per target model i and protocol P, with squared error `error = (θ\* − θ̂)²`:

    score_P,i = 100 · (1 − error_P,i / error_baseline,i)

0 is the baseline, 100 is a perfect prediction, negative is worse than the baseline (unbounded
below). Figure one pools each model's squared error over its four domains before taking the score,
one score per model per protocol, and tests the 16 scores against 0 with a one-sample t-test.
`analysis/results/signal_score_tests.csv` (mean score, p, median error ratio):

| protocol | mean score | p | pooled MSE (baseline 1.37) |
| --- | --- | --- | --- |
| Pairwise judge | 49 | 0.0056 | 0.33 |
| Embedding | 23 | 8e-7 | 1.04 |
| Absolute scoring | 10 | 0.12 | 1.14 |
| Deterministic features | 10 | 0.62 | 0.84 |
| Length | −228 | 0.33 | 1.66 |

The t-test replaced the Wilcoxon signed-rank test at 17:14 today (the paper's test; the box
figure's 95% t-interval is its dual). Under Wilcoxon, Absolute scoring had p = 0.044 and Pairwise
judge p = 0.0008; the t-test on scores is pulled by the unbounded lower tail (Pairwise judge has
one model at −430, Length two below −1000), so Absolute scoring loses its star. Holm over the four
paper protocols keeps Embedding and Pairwise judge.

Replication on 2026-I alone (N = 8, `signal_score_tests_2026-I.csv`): Pairwise judge 51,
p = 0.004; Embedding 24, p = 0.0007; Absolute scoring 7, p = 0.24; Deterministic features −1.

No self-judging (each target judged by the best judge from another developer family,
`signal_score_tests_no_self_judging.csv`): Pairwise judge 47, p = 0.006; Absolute scoring 12,
p = 0.095. Same conclusion.

## Hierarchical GLM

`scripts/analyses/signal_glm/model.py`, fitted with hibayes (numpyro NUTS, 4 chains × 2000 draws
after 1000 warmup, target_accept 0.95):

    error ~ Gamma(shape[protocol], shape[protocol] / μ)
    log μ = intercept + protocol[p] + domain[d] + (protocol | domain)[p, d] + (1 | target model × domain) + (1 | judge within protocol)

- `protocol`: reference level is the saturated-item prediction (pinned to 0), so exp(protocol[p]) is
  the error ratio to the baseline and 100 · (1 − ratio) is the prediction score on the GLM's scale.
- `domain`: sum-to-zero coded, so the intercept is the grand mean over domains, not science.
- `(protocol | domain)`: one spread per protocol, non-centred; the "how domain-dependent" term.
  Dropping it gives m1 (`signal_gamma_additive`); keeping it is m2 (`signal_gamma_model`).
- `(1 | target model × domain)`: random intercept per cell, SD ≈ 1.0 on the log scale. Every
  protocol forecasts the same target in the same domain, so the cell is the pairing unit of the
  t-test. This is the mentor's "variance of target model" term at the finer grain; a model-level
  intercept alone was tried and helped little (cell means vary more than model means). Period is
  absorbed: each target is a target in one semester only.
- `(1 | judge)`: pinned to 0 in the best-judge table (one judge per task); free in the all-judges
  fit (SD 0.39).
- One Gamma shape per protocol (Pairwise judge's erratic gains and Embedding's small consistent
  ones need different noise), 0.40 (Pairwise judge) to 3.8 (baseline).
- Priors: intercept N(0, 2), effects N(0, 1), spreads HalfNormal(0.5), cell SD HalfNormal(1),
  shape Exponential(1). Exact hits are floored at 1e-6.

Flat cross-check: statsmodels Gamma GLM, log link, `protocol * domain + cell` fixed effects,
fitted by Newton (IRLS overflows on the 64 cell dummies). Its Wald intervals sit beside the
hierarchical ones in `ratios.png` (open markers).

### Results, best judge per task (`signal_glm_wide.csv` as scores, 100 · (1 − ratio), 95% interval)

| protocol | science | math | coding | cyber | pooled (new domain) | SD across domains |
| --- | --- | --- | --- | --- | --- | --- |
| Pairwise judge | 15 (−92, 57) | 17 (−89, 57) | 80 (39, 92) | 72 (28, 88) | 51 (−13, 77) | 0.71 (0.20–1.31) |
| Embedding | 28 (3, 47) | 33 (13, 52) | 29 (7, 47) | 21 (−14, 39) | 28 (1, 47) | 0.15 (0.01–0.59) |
| Absolute scoring | 31 (2, 57) | 23 (−7, 46) | 17 (−22, 40) | 17 (−24, 40) | 22 (−13, 46) | 0.18 (0.01–0.70) |
| Deterministic features | −2 (−84, 38) | 3 (−64, 39) | 1 (−96, 43) | 3 (−67, 40) | 2 (−66, 37) | 0.17 (0.01–0.68) |
| Length | 14 (−30, 47) | 13 (−31, 46) | −10 (−75, 24) | −24 (−129, 19) | −1 (−58, 36) | 0.30 (0.02–0.82) |

- The pooled column is the new-domain effect exp(protocol[p]), whose interval includes the
  across-domain spread; the mean of the four fitted domains' ratios is tighter (Pairwise judge 42,
  interval 2–63; Embedding 27, 11–40; Absolute scoring 21, 0–38; computed from the saved
  posterior, not in any output file). Undecided which the pooled panel should show; the mean of
  four domains is what the text asks about.
- Contrasts, Pairwise judge vs each other protocol (`signal_glm_contrasts.csv`, error ratios):
  coding 0.18–0.29 and cyber 0.22–0.34 with 98–100% posterior mass below 1; science and math
  0.83–1.25 straddling 1; pooled 0.49–0.69 with 82–94% below 1.
- LOO, m2 (protocol × domain) vs m1 (additive): m2 ahead by 4.2 elpd, difference SE 3.4, stacking
  weights 0.87 / 0.13. Weak evidence for domain dependence on the 16 targets; on 2026-I alone m2
  wins by 15.4 (SE 4.2). Both LOO runs carry the arviz warning for the bad Pareto-k rows.
- Diagnostics (`signal_glm/diagnostics.csv`, m2): 0 divergences, max r_hat 1.01, min bulk ESS 923,
  min tail ESS 1524, Pareto-k > 0.7 on 8 of 384 rows (2 above 1, the near-zero errors),
  Bayesian R² on the log scale 0.35 (0.31–0.40). m1: R² 0.32.
- Posterior predictive: `figures/signal_glm/predictive.png` (density per protocol, log scale,
  200 draws) and `fit.png` (observed vs predicted log error per row; 95% predictive coverage
  0.94–0.97 per protocol, `signal_glm/fit_coverage.csv`).
- Estimand caveat. A Gamma GLM with a log link estimates the arithmetic mean within a group, so
  with cell intercepts the protocol effect is close to the mean of per-cell ratios, which the
  heavy-tailed protocols' blow-ups dominate. Deterministic features sits at 2 here and at 26 under
  a rank test (median ratio 0.74); Length at −1 vs a median ratio of 0.94. Do not put a p-value on
  either in the paper.

### Results, all judges (`signal_glm_all_judges_wide.csv`, judge random effect)

Fitted 2026-09-17 02:03, **before** the tool-call feature was added to Deterministic features and
forecasts were refit, so its Deterministic-features row is stale; the other rows shift only
through the joint fit. Pairwise judge coding 62 (21, 80), cyber 59 (14, 79), science −9, math −10;
Embedding 27 pooled. Rerun with `JUDGES = "all"` in the runner (about 6 minutes) before quoting.

### 2026-I only (`signal_glm_2026-I*`)

`SIGNAL_SEMESTERS=2026-I uv run python scripts/analyses/signal_glm.py`. Pairwise judge coding 84
(37, 94), cyber 84 (38, 94), science −37, math −69; Absolute scoring science 49 (3, 73), math 47
(6, 72). Same picture with wider intervals.

## Mentor's points, status

- Variance of target model: cell intercept (target × domain), see above. Done.
- Sum-to-zero domain coding: done.
- LOO m1 vs m2: done, weak preference for m2.
- Contrasts against Pairwise judge: done.
- Frontier vs non-frontier targets (m3, and the stats for Q3.1): **not done**. No flag exists in
  the data; the only frontier notion in the repo is the top-3 targets per task by θ\*
  (`plots.frontier_figure`). Needs a definition before it can be a model term.
- Open/closed and truncated/full: confirmed not in this data.
- Posterior predictive plot and fit diagnostics: done, see above.

## Stronger constant guesses

Computed once in a session, not in any output file: "the new model equals the best reference"
scores pooled MSE 0.34, "the top-3 mean" 0.46. Embedding and Absolute scoring are significantly
worse than both (p ≤ 0.009); Pairwise judge is indistinguishable from them (p = 0.94) and its
pooled MSE (0.33) equals the oracle's (hard-item accuracy, 0.33). Not used in the paper; recorded
because they bound what "good" means.

## Files

Code (untracked):

- `src/analysis/challenges/signal_glm.py` — `score`, rows (`judges="best"|"best_other_family"|"all"`),
  flat GLM, ratio and contrast tables, per-model table, `model_tests` (t-test), figure one
  (`model_bar_figure` with metrics `mse`, `delta`, `ratio`, `score`; `model_box_figure`, the paper
  version; `model_violin_figure`), contrast figure, diagnostics, LOO compare, predictive and fit
  figures. Colours from CLAUDE.md, labels from `plots.LABELS`.
- `scripts/analyses/signal_glm/model.py` — `signal_gamma_model` (m2), `signal_gamma_additive` (m1).
- `scripts/analyses/signal_glm/config.yaml` — hibayes pipeline; both models, m2 first.
- `scripts/analyses/signal_glm.py` — cell runner. `JUDGES = "best"` toggle; `SIGNAL_SEMESTERS`
  env var for a subset.
- `tests/test_signal_glm.py` — synthetic errors with a planted ratio (passes; pytest itself then
  hits an internal KeyError in a plugin teardown, unrelated).

Outputs:

- `analysis/results/figures/bars/figure1_score_box.{svg,png}` — figure one, the paper version
  (`fig2_signal_box`): mean score as a filled circle, 95% t-interval box, per-model points, axis
  −50 to 100 with lower points clipped, stars from the t-test. Also `figure1_{mse,delta,ratio,score}`,
  `figure1_score_no_self_judging`, and `_2026-I` variants of each.
- `analysis/results/figures/signal_glm/` — `ratios`, `ratios_table`, `posteriors`, `contrasts`,
  `predictive`, `predictive_additive`, `fit`, `fit_additive`. (`figure1_score.*` in this directory
  is a stale 14:17 copy; delete it.)
- `analysis/results/signal_score_tests{,_no_self_judging,_2026-I,_2026-I_no_self_judging}.csv`,
  `signal_glm{,_2026-I,_all_judges}.csv`, `_wide.csv`, `_contrasts.csv`,
  `signal_glm*/{diagnostics,loo_compare,per_model,fit_coverage}.csv`.

Rerun: `uv run python scripts/analyses/signal_glm.py` (about 6 minutes for the two fits on CPU).
It reads `forecasts.csv`; rerun `scripts/analyses/results.py` first only if forecasts change.

## Changes today

1. Deterministic features gained `log_tool_calls` (13 features); forecasts, GLM and figure one
   refit. Its score is 10, p = 0.62 (was p = 0.065 under Wilcoxon without the feature).
2. Sampler draws raised to 2000 per chain after an r_hat 1.02 failure.
3. Figure-one test switched from Wilcoxon signed-rank to a one-sample t-test on scores (17:14).
4. Score scaled to 0–100 in code, tables and figures; the test table renamed from
   `signal_wilcoxon*.csv` to `signal_score_tests*.csv`.
5. Added: no-self-judging test, 2026-I-only fit, fit figure with 95% coverage, box figure.

## Open items

- **(protocol | task) random effect.** Targets within a semester share one fitted calibration per
  protocol (one ridge, one judge, one linear conversion), so the 16 scores are not independent and
  the t-test is somewhat anti-conservative. The per-semester replication is the current answer.
  The principled fix is a random effect per protocol per (domain, semester) task; 8 tasks per
  protocol, so weakly identified. Proposed, not built.
- **Model families.** glm 4.5/4.7/5.1/5.2, kimi k2-thinking/2.5/2.6, gpt-oss 20b/120b, three qwen3
  variants are among the 16; siblings have correlated errors. Not modelled.
- **Frontier flag** for m3, see above.
- **Pooled panel of the ratio figure**: new-domain effect vs mean of the four domains.
- **All-judges fit** is stale, rerun before quoting.
- **Judge and task are confounded** for the judge protocols under the best-judge table; the
  all-judges fit is the version with a judge term.

## Paper text check (results section draft)

Against the current numbers: Pairwise judge mean score 49, p = 0.0056 (so "p < .01" holds), MSE
0.33 vs 1.37; GLM coding 80 and cyber 72 hold. **Embedding's mean score is 23, not 10** (10 is
Absolute scoring and Deterministic features). "Embedding ... slightly better" should read
"Embedding predictions were also better than baseline (mean score 23%, p < .001)". The caption's
"positive for ... absolute scoring" is true of the mean (10) but not significant (p = 0.12);
consider "positive and significant for embedding and pairwise judging".
