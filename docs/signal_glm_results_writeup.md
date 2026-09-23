# Domain specificity of protocol performance

The main analysis compares prediction error after pooling across capability domains. This establishes whether transcript-based protocols improve prediction on average, but it does not show whether those gains are shared across domains or driven by a subset of settings. We therefore fit a hierarchical Gamma generalized linear model (GLM) to the squared prediction errors. The model jointly estimates each protocol's pooled effect, its domain-specific deviations, and the variation of those deviations across science, mathematics, coding, and cybersecurity.

## Data and estimand

We used the balanced 2025-II and 2026-I evaluation periods from the main analysis. These periods contain 16 target models evaluated in all four domains. Each of the resulting 64 target-by-domain cells contributed one squared prediction error for each of six prediction conditions: the saturated-item prediction, Length, Deterministic features, Embedding, Absolute scoring, and Pairwise judge. This gave 384 observations. Length was included as a diagnostic condition in the fitted model but is omitted from the paper-protocol results below.

For the judge-based protocols, each prediction task used the judge with the lowest leave-one-model-out error on its reference models, as in the main analysis. Judge identity was not included as a model term. The unit modelled was the target-level squared error,

\[
y_{mdp}=(\hat\theta_{mdp}-\theta^*_{md})^2,
\]

where \(m\), \(d\), and \(p\) index target model, domain, and protocol. Exact zero errors were clipped to \(10^{-6}\) so that the Gamma likelihood remained defined.

## Hierarchical Gamma GLM

We fitted

```text
squared_error ~ protocol + domain + (protocol | domain) + (1 | target:domain), Gamma(log)
```

with coefficient form

\[
y_{mdp} \sim \operatorname{Gamma}(\nu_p,\nu_p/\mu_{mdp}),
\qquad
\log\mu_{mdp}=\beta_0+\alpha_p+\gamma_d+\delta_{pd}+u_{md}.
\]

The saturated-item prediction was the reference protocol, so \(\alpha_{\mathrm{baseline}}=0\). Domain effects \(\gamma_d\) were constrained to sum to zero, making the intercept the baseline's grand mean on the log scale. Protocol-by-domain deviations were partially pooled separately for each protocol,

\[
\delta_{pd}=\sigma_p z_{pd}, \qquad z_{pd}\sim\mathcal N(0,1),
\]

where \(\sigma_p\) measures how strongly protocol \(p\)'s error reduction varies across domains. The random intercept \(u_{md}\) accounts for the fact that every protocol predicts the same target in the same domain. The likelihood used a separate shape parameter \(\nu_p\) for each protocol, allowing protocols with similar mean error but different dispersion to be represented separately.

We placed \(\mathcal N(0,2)\) and \(\mathcal N(0,1)\) priors on the intercept and fixed effects, \(\operatorname{HalfNormal}(0.5)\) priors on the protocol-specific between-domain standard deviations, a \(\operatorname{HalfNormal}(1)\) prior on the target-by-domain standard deviation, and \(\operatorname{Exponential}(1)\) priors on the Gamma shape parameters. We drew four chains with 1,000 warm-up and 2,000 retained iterations per chain.

Exponentiating a protocol coefficient gives its mean-squared-error (MSE) ratio relative to the saturated-item prediction. The pooled ratio is \(\exp(\alpha_p)\), while the ratio in domain \(d\) is \(\exp(\alpha_p+\delta_{pd})\). Ratios below one indicate lower squared error. We report posterior medians and 95% credible intervals.

## Domain-specific protocol effects

![Posterior MSE ratios for each protocol, pooled and by domain](../analysis/results/figures/signal_glm/appendix_effects.png)

**Figure A1. Protocol effects differ across capability domains.** Posterior MSE ratios relative to the saturated-item prediction from the hierarchical Gamma GLM. Filled circles show pooled effects; open circles, squares, triangles, and diamonds show science, mathematics, coding, and cybersecurity, respectively. Horizontal lines are 95% credible intervals. The dashed line marks equal error to the baseline. The model contains 384 protocol-by-target-by-domain observations from 16 target models; Length is fitted as a diagnostic but omitted here.

Pairwise judge had the lowest pooled posterior MSE ratio, at 0.43 (95% credible interval 0.21--0.96; posterior probability of a reduction, 0.981). Its estimated effect differed strongly across domains. The MSE ratio was 0.54 (0.29--1.17) in science and 0.74 (0.38--1.72) in mathematics, leaving substantial uncertainty about improvement in either non-agentic domain. In contrast, the ratios were 0.21 (0.09--0.58) in coding and 0.27 (0.12--0.65) in cybersecurity, providing clear evidence of lower squared error in both agentic domains.

Embedding produced a smaller but more consistent reduction. Its pooled MSE ratio was 0.70 (0.52--0.94; posterior probability of a reduction, 0.987), with domain estimates of 0.70 (0.53--0.94) in science, 0.66 (0.48--0.85) in mathematics, 0.68 (0.51--0.90) in coding, and 0.74 (0.57--1.03) in cybersecurity. Absolute scoring had a pooled ratio of 0.78 (0.55--1.14); only its science interval excluded one. Deterministic features had a pooled ratio of 1.02 (0.64--1.70), and every domain interval included one.

| Protocol | Pooled | Science | Mathematics | Coding | Cybersecurity | Between-domain SD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Deterministic features | 1.02 [0.64, 1.70] | 1.04 [0.64, 1.89] | 0.98 [0.60, 1.64] | 1.03 [0.61, 2.04] | 0.99 [0.60, 1.67] | 0.17 [0.01, 0.70] |
| Embedding | 0.70 [0.52, 0.94] | 0.70 [0.53, 0.94] | 0.66 [0.48, 0.85] | 0.68 [0.51, 0.90] | 0.74 [0.57, 1.03] | 0.13 [0.01, 0.56] |
| Absolute scoring | 0.78 [0.55, 1.14] | 0.69 [0.43, 0.98] | 0.77 [0.54, 1.07] | 0.83 [0.61, 1.24] | 0.84 [0.61, 1.25] | 0.19 [0.01, 0.69] |
| Pairwise judge | 0.43 [0.21, 0.96] | 0.54 [0.29, 1.17] | 0.74 [0.38, 1.72] | 0.21 [0.09, 0.58] | 0.27 [0.12, 0.65] | 0.61 [0.12, 1.23] |

**Table A1. Hierarchical estimates of protocol performance and domain specificity.** Entries are posterior median MSE ratios relative to the saturated-item prediction [95% credible interval]. The final column gives \(\sigma_p\), the between-domain standard deviation on the log-MSE scale.

## Pairwise contrasts and between-domain heterogeneity

![Pairwise-judge contrasts and protocol-specific between-domain variation](../analysis/results/figures/signal_glm/appendix_contrasts.png)

**Figure A2. Pairwise judge gains are concentrated in coding and cybersecurity.** **a,** Direct posterior MSE contrasts between Pairwise judge and each comparator. Values below one favour Pairwise judge; points and intervals use the same domain encoding as Figure A1. **b,** Posterior median and 95% credible interval for each protocol's between-domain standard deviation \(\sigma_p\) on the log-MSE scale. Pairwise judge has the strongest average effect and the greatest domain dependence.

Direct posterior contrasts qualify the ordering of the pooled estimates. Pairwise judge had a pooled MSE ratio of 0.42 relative to Deterministic features (0.18--1.01), 0.61 relative to Embedding (0.29--1.41), and 0.55 relative to Absolute scoring (0.25--1.27). Thus, although Pairwise judge had the lowest pooled posterior median, none of the pooled pairwise contrasts excluded equal error at the 95% level.

The domain-specific contrasts were more informative. In coding, Pairwise judge had MSE ratios of 0.20 relative to Deterministic features (0.07--0.60), 0.31 relative to Embedding (0.13--0.87), and 0.25 relative to Absolute scoring (0.10--0.72). In cybersecurity, the corresponding ratios were 0.27 (0.11--0.72), 0.36 (0.16--0.88), and 0.32 (0.13--0.79). Every coding and cybersecurity interval lay below one. By contrast, the science and mathematics intervals included one for every comparator.

Pairwise judge also had the largest estimated between-domain standard deviation: \(\sigma_{\mathrm{pairwise}}=0.61\) on the log-MSE scale (0.12--1.23). The corresponding estimates were 0.13 (0.01--0.56) for Embedding, 0.19 (0.01--0.69) for Absolute scoring, and 0.17 (0.01--0.70) for Deterministic features. This supports the interpretation that Pairwise judge's average gain is not uniform: it is substantially larger in coding and cybersecurity than in science and mathematics. Because the two former domains are agentic and the two latter domains are non-agentic, this pattern is consistent with an agenticity difference, but four domains are insufficient to separate agenticity from other domain characteristics.

## Model comparison and checks

We compared the hierarchical interaction model with an additive Gamma GLM that omitted the protocol-by-domain deviations. The interaction model increased expected leave-one-out predictive density by 3.01 (SE 3.07). This difference is small relative to its uncertainty and does not by itself provide decisive model-selection evidence for the interaction. PSIS-LOO also flagged seven observations with \(0.7<k\leq1\) and three with \(k>1\) in the interaction model, so the comparison is sensitive to influential targets.

The retained fit showed no divergent transitions. The maximum split \(\hat R\) was 1.004, the minimum bulk effective sample size was 905, and the minimum tail effective sample size was 1,088. The median Bayesian \(R^2\) on log squared error was 0.356 (95% interval 0.308--0.401). A non-hierarchical Gamma GLM with fixed target-by-domain effects also placed Pairwise judge lowest, but estimated larger improvements overall and did not reproduce the ordering of the remaining protocols. The hierarchical estimates are therefore used throughout this section.

## Interpretation and limitations

The hierarchical analysis resolves an ambiguity left by the pooled main result. Pairwise judge has the strongest average reduction in squared prediction error, but that reduction is domain-specific: the evidence is clear in coding and cybersecurity and uncertain in science and mathematics. Embedding shows the opposite pattern of heterogeneity, providing a smaller reduction that is comparatively stable across domains. Absolute scoring and Deterministic features do not show clear pooled improvements.

These estimates should be interpreted with three limitations. First, the domain interaction is inferred from only four domains, so the between-domain standard deviations remain uncertain. Second, the analysis uses the balanced set of 16 target models from two evaluation periods and does not establish that the same domain pattern holds historically. Third, judge selection is based only on reference-model performance, but uncertainty from selecting a judge is not propagated through this GLM. The robust conclusion is that protocol performance depends on the capability domain; the stronger explanation that agentic transcripts intrinsically contain more recoverable signal requires comparisons across a larger set of agentic and non-agentic domains.
