"""Signal in saturated transcripts: one GLM of forecast error over every protocol and domain.

    squared_error ~ protocol + domain + (protocol | domain) + (1 | model:domain) [+ (1 | protocol:judge)],   Gamma, log link

on the balanced semesters (2025-II, 2026-I: every protocol in every domain x semester). Rows
are one forecast per target x protocol (best judge on references); the saturated-item
prediction is the reference level of `protocol`, so every coefficient is a log error ratio to it. The hierarchical fit runs through hibayes;
this module builds its rows, reads its posterior into a ratio
table and fits the flat statsmodels cross-check; the figures are in plots.signal_glm.
"""

import arviz as az
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import build_design_matrices

from analysis.stats import CONDITIONS, chosen
from analysis.stats import LABELS as plots_labels
from analysis.stats import BASELINE, JUDGED, core_condition, one_per_target

LABELS = {p: plots_labels[p] for p in CONDITIONS}  # the paper's protocol names, one source


PROTOCOLS = [BASELINE, *CONDITIONS]  # the baseline first: it is the reference level everywhere
PAPER_PROTOCOLS = [p for p in CONDITIONS if p != "glm_length"]  # Length alone is a diagnostic, not a paper protocol: results figures leave it out
DOMAINS = ["science", "math", "coding", "cyber"]
SEMESTERS = ("2025-II", "2026-I")  # the semesters every domain has: a balanced design needs no semester term
POOLED = "all"
FLOOR = 1e-6  # a Gamma outcome must be positive; an exact hit is clipped here


def score(errors, baseline):
    """Prediction score per model, -log2(error / the model's own saturated-item error): the number
    of halvings of the error. 0 is the baseline, 1 half its error, 2 a quarter, -1 twice it.
    Symmetric in the ratio (twice as good and twice as bad are equidistant from 0), so a model
    the baseline happens to predict almost exactly cannot dominate the mean, unlike
    100 * (1 - error / baseline) (gpt-oss-20b: -114%); the same scale as the hierarchical GLM."""
    return -np.log2(errors / baseline)


def rows(forecasts: pd.DataFrame, variants: dict[str, str], semesters: tuple[str, ...] = SEMESTERS,
         judges: str = "best") -> pd.DataFrame:
    """The model's table from the forecasts table: one row per target x protocol in PROTOCOLS,
    each protocol in the variant `variants` names for it (pairwise conversion, direct
    normalisation, embedding variant; "-" when the protocol is not named), semesters in
    `semesters` only. judges "best": judge protocols as the best judge on references, one row
    per target; "best_other_family": the best judge on references among judges from another
    developer than the target (no self-judging: glm-4.7 never judges glm-5.x); "all": one row
    per target x judge on the core condition. Columns: y (squared
    error, >= FLOOR), protocol, domain, semester, task ("<domain> <semester>"), model,
    model_domain ("<model> <domain>", the cell every protocol forecasts once), judge ("-" for
    transcript protocols and under "best"), protocol_judge ("<protocol> <judge>", "-" likewise).
    Rows without a forecast are dropped."""
    assert judges in ("best", "best_other_family", "all"), judges
    f = one_per_target(forecasts) if judges == "best" else best_other_family(forecasts) if judges == "best_other_family" else core_condition(forecasts)
    f = chosen(f, variants)
    f = f[f.protocol.isin(PROTOCOLS) & f.semester.isin(semesters)].dropna(subset=["theta_hat", "theta"])
    judge = f.judge.where(f.protocol.isin(JUDGED) & (judges == "all"), "-")
    return pd.DataFrame({
        "y": ((f.theta_hat - f.theta) ** 2).clip(lower=FLOOR).to_numpy(),
        "protocol": f.protocol.to_numpy(), "domain": f.domain.to_numpy(), "semester": f.semester.to_numpy(),
        "task": (f.domain + " " + f.semester).to_numpy(), "model": f.model.to_numpy(),
        "model_domain": (f.model + " " + f.domain).to_numpy(),
        "judge": judge.to_numpy(), "protocol_judge": (f.protocol + " " + judge).where(judge != "-", "-").to_numpy(),
    })


def family(model: pd.Series) -> pd.Series:
    """The developer prefix of a model name: glm-4.7 -> glm, kimi-k2.5 -> kimi, qwen3-235b -> qwen."""
    return model.str.extract(r"^([a-z]+)", expand=False)


def best_other_family(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Like one_per_target, but each target's judge is the best (lowest reference leave-one-out
    MSE) among the core-condition judges from another developer family than the target, so no
    model is judged by its own lineage. Targets with no such judge are dropped."""
    core = core_condition(forecasts)
    judged = core[core.protocol.isin(JUDGED) & (family(core.judge) != family(core.model))]
    key = ["domain", "semester", "protocol", "variant", "model"]
    best = judged.loc[judged.groupby(key).judge_loo_mse.transform("min") == judged.judge_loo_mse].drop_duplicates(key)
    return pd.concat([forecasts[~forecasts.protocol.isin(JUDGED)], best.assign(judge="best-other-family", condition="core")], ignore_index=True)


def flat_glm(table: pd.DataFrame) -> pd.DataFrame:
    """The frequentist cross-check: Gamma GLM with a log link of y on protocol * domain +
    model_domain (a fixed intercept per (target model, domain) cell, the paired analogue of the
    hierarchical fit's random one; semester and domain main effects are implied by the cell).
    With every judge's rows the judges are pooled: no judge term, the effect is the mean over judges.
    Returns one row per non-baseline protocol x domain plus a pooled row (domain = "all", the
    mean of the four domain log ratios) with flat_ratio, flat_lo, flat_hi (95% Wald interval),
    the error ratio to the saturated-item prediction in that domain. A domain without a
    protocol's forecasts gets NaN."""
    formula = f"y ~ C(protocol, Treatment('{BASELINE}')) * C(domain, Treatment('science')) + C(model_domain)"
    # Newton, not IRLS: the 64 cell dummies and the floored exact hits make IRLS overflow
    fit = smf.glm(formula, table, family=sm.families.Gamma(sm.families.links.Log())).fit(method="newton", maxiter=5000, disp=0)
    design = fit.model.data.model_spec  # patsy DesignInfo
    first_cell = table.model_domain.min()  # any cell: it cancels in every contrast

    def contrast(protocol: str, domain: str) -> np.ndarray | None:
        if not ((table.protocol == protocol) & (table.domain == domain)).any():
            return None
        frame = pd.DataFrame({"protocol": [protocol, BASELINE], "domain": domain, "model_domain": first_cell})
        (x,) = build_design_matrices([design], frame)
        return np.asarray(x)[0] - np.asarray(x)[1]  # log mu(protocol) - log mu(baseline), same cell

    out = []
    for protocol in PROTOCOLS[1:]:
        cells = {d: contrast(protocol, d) for d in DOMAINS}
        present = [c for c in cells.values() if c is not None]
        cells[POOLED] = np.mean(present, axis=0) if present else None
        for domain, c in cells.items():
            if c is None:
                out.append(dict(protocol=protocol, domain=domain, flat_ratio=np.nan, flat_lo=np.nan, flat_hi=np.nan))
                continue
            t = fit.t_test(c)
            lo, hi = np.asarray(t.conf_int()).ravel()
            out.append(dict(protocol=protocol, domain=domain, flat_ratio=float(np.exp(np.ravel(t.effect)[0])), flat_lo=float(np.exp(lo)), flat_hi=float(np.exp(hi))))
    return pd.DataFrame(out)


def ratio_table(posterior, coords: dict) -> pd.DataFrame:
    """The hierarchical fit's error ratios from the posterior (an xarray Dataset or dict of
    arrays with leading chain/draw dims): protocol_effects [.., P], interaction [.., P, D],
    sigma_protocol [.., P]; coords["protocol"] and coords["domain"] in the model's order, the
    baseline first. One row per non-baseline protocol x domain (ratio = exp(alpha + delta))
    plus a pooled row per protocol (domain = "all", ratio = exp(alpha), sigma_protocol with
    sigma_lo, sigma_hi = the protocol's across-domain SD on the log scale). ratio, lo, hi and
    the sigma triple = posterior median and 2.5 / 97.5 percentiles."""
    alpha = np.asarray(posterior["protocol_effects"])
    delta = np.asarray(posterior["interaction"])
    sigma = np.asarray(posterior["sigma_protocol"])
    alpha, delta, sigma = (a.reshape(-1, *a.shape[-k:]) for a, k in ((alpha, 1), (delta, 2), (sigma, 1)))

    def summary(draws: np.ndarray) -> dict:
        lo, mid, hi = np.percentile(np.exp(draws), [2.5, 50, 97.5])
        return dict(ratio=mid, lo=lo, hi=hi)

    out = []
    for p, protocol in enumerate(coords["protocol"]):
        if protocol == BASELINE:
            continue
        for d, domain in enumerate(coords["domain"]):
            out.append(dict(protocol=protocol, domain=domain, **summary(alpha[:, p] + delta[:, p, d]),
                            sigma_protocol=np.nan, sigma_lo=np.nan, sigma_hi=np.nan))
        s_lo, s_mid, s_hi = np.percentile(sigma[:, p], [2.5, 50, 97.5])
        out.append(dict(protocol=protocol, domain=POOLED, **summary(alpha[:, p]), sigma_protocol=s_mid, sigma_lo=s_lo, sigma_hi=s_hi))
    return pd.DataFrame(out)


def error_table(posterior, coords: dict) -> pd.DataFrame:
    """The hierarchical fit's expected squared error per protocol x domain for a typical target
    (cell intercept 0): exp(intercept + alpha + gamma + delta), the baseline included, plus a
    pooled row per protocol (domain = "all", exp(intercept + alpha): the domain effects sum to
    zero, so this is the mean over domains on the log scale). mse, lo, hi = posterior median
    and 2.5 / 97.5 percentiles."""
    intercept = np.asarray(posterior["intercept"]).reshape(-1)
    alpha = np.asarray(posterior["protocol_effects"])
    gamma = np.asarray(posterior["domain_effects"])
    delta = np.asarray(posterior["interaction"])
    alpha, gamma, delta = (a.reshape(-1, *a.shape[-k:]) for a, k in ((alpha, 1), (gamma, 1), (delta, 2)))

    def summary(draws: np.ndarray) -> dict:
        lo, mid, hi = np.percentile(np.exp(draws), [2.5, 50, 97.5])
        return dict(mse=mid, lo=lo, hi=hi)

    out = []
    for p, protocol in enumerate(coords["protocol"]):
        for d, domain in enumerate(coords["domain"]):
            out.append(dict(protocol=protocol, domain=domain, **summary(intercept + alpha[:, p] + gamma[:, d] + delta[:, p, d])))
        out.append(dict(protocol=protocol, domain=POOLED, **summary(intercept + alpha[:, p])))
    return pd.DataFrame(out)


def wide_table(table: pd.DataFrame) -> pd.DataFrame:
    """ratio_table rows as protocol x domain: index protocol (the baseline first, ratio 1 by
    definition, no interval), columns <domain>, <domain>_lo, <domain>_hi for each domain and
    "all" (the pooled effect), then sigma, sigma_lo, sigma_hi (the across-domain SD on the log
    scale, NaN for the baseline)."""
    cols = [d for d in [*DOMAINS, POOLED] if (table.domain == d).any()]
    wide = table.pivot(index="protocol", columns="domain", values=["ratio", "lo", "hi"])
    out = pd.DataFrame(index=[BASELINE, *[p for p in PROTOCOLS[1:] if p in wide.index]])
    for d in cols:
        out[d] = wide["ratio"][d].reindex(out.index).fillna({BASELINE: 1.0})
        out[f"{d}_lo"], out[f"{d}_hi"] = wide["lo"][d].reindex(out.index), wide["hi"][d].reindex(out.index)
    pooled = table[table.domain == POOLED].set_index("protocol").reindex(out.index)
    out["sigma"], out["sigma_lo"], out["sigma_hi"] = pooled.sigma_protocol, pooled.sigma_lo, pooled.sigma_hi
    out.index.name = "protocol"
    return out


def per_model(table: pd.DataFrame) -> pd.DataFrame:
    """Squared error pooled over domains: one row per target model, one column per protocol in
    PROTOCOLS, the mean of y over the model's domains. A model missing a protocol in any domain
    gets NaN there."""
    wide = table.pivot_table(index="model", columns=["protocol", "domain"], values="y")
    complete = wide.T.groupby(level="protocol").count().T.eq(len(DOMAINS))
    return wide.T.groupby(level="protocol").mean().T.where(complete).reindex(columns=PROTOCOLS)


def model_scores(table: pd.DataFrame) -> pd.DataFrame:
    """Prediction score per target model: `score` of the model's squared error pooled over its
    domains (`per_model`) against the saturated-item prediction's pooled the same way. Pooling
    first keeps one lucky near-exact cell from dominating a model (scored per domain, an error of
    0.0000 against a baseline of 2 is 17 halvings). One row per model, one column per protocol
    in PROTOCOLS (the baseline's column is 0); NaN where a model lacks the protocol in any domain."""
    pooled = per_model(table)
    return pd.concat({p: score(pooled[p], pooled[BASELINE]) for p in pooled.columns}, axis=1).reindex(columns=PROTOCOLS)


def hodges_lehmann_interval(values: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    """The exact (1 - alpha) Hodges-Lehmann interval of one sample: order statistics of the
    Walsh averages (v_i + v_j) / 2 at the critical value of the signed-rank statistic's exact
    null distribution, so 0 lies outside it exactly when the two-sided Wilcoxon signed-rank
    test of `values` against 0 rejects at `alpha`."""
    v = np.asarray(values, float)
    n = len(v)
    walsh = np.sort([(v[i] + v[j]) / 2 for i in range(n) for j in range(i, n)])
    dist = np.zeros(n * (n + 1) // 2 + 1)
    dist[0] = 1
    for r in range(1, n + 1):  # each rank r is in T+ with probability 1/2
        dist = (dist + np.roll(dist, r)) / 2
    c = int(np.max(np.where(np.cumsum(dist) <= alpha / 2)[0]))
    return float(walsh[c]), float(walsh[len(walsh) - c - 1])


def contrast_table(posterior, coords: dict, against: str = "pairwise") -> pd.DataFrame:
    """Error ratios of `against` to every other non-baseline protocol, from the same posterior
    draws: one row per protocol x domain (exp of the difference of the two protocols' alpha +
    delta in that domain) plus a pooled row (domain "all", exp(alpha_against - alpha_p)).
    ratio, lo, hi = posterior median and 2.5 / 97.5 percentiles; p_below_1 = posterior mass
    with `against` the smaller error. Shared cell, domain and judge terms cancel in the
    difference, so these intervals are tighter than the two marginal ones suggest."""
    alpha = np.asarray(posterior["protocol_effects"]); alpha = alpha.reshape(-1, alpha.shape[-1])
    delta = np.asarray(posterior["interaction"]); delta = delta.reshape(-1, *delta.shape[-2:])
    a = coords["protocol"].index(against)
    out = []
    for p, protocol in enumerate(coords["protocol"]):
        if protocol in (BASELINE, against):
            continue
        cells = {d: (alpha[:, a] + delta[:, a, i]) - (alpha[:, p] + delta[:, p, i]) for i, d in enumerate(coords["domain"])}
        cells[POOLED] = alpha[:, a] - alpha[:, p]
        for domain, draws in cells.items():
            lo, mid, hi = np.percentile(np.exp(draws), [2.5, 50, 97.5])
            out.append(dict(against=against, protocol=protocol, domain=domain, ratio=mid, lo=lo, hi=hi, p_below_1=float((draws < 0).mean())))
    return pd.DataFrame(out)


def log_mu_draws(posterior, features: dict) -> np.ndarray:
    """log mu per draw and row (draws x rows) from the fitted effects and the row indices in
    `features` (protocol_index, domain_index, model_domain_index, protocol_judge_index)."""
    get = lambda k: np.asarray(posterior[k]).reshape(-1, *np.asarray(posterior[k]).shape[2:])
    p, d, c, j = (np.asarray(features[k]) for k in ("protocol_index", "domain_index", "model_domain_index", "protocol_judge_index"))
    return (get("intercept")[:, None] + get("protocol_effects")[:, p] + get("domain_effects")[:, d]
            + get("interaction")[:, p, d] + get("cell_effects")[:, c] + get("judge_effects")[:, j])


def fit_coverage(idata, features: dict, table: pd.DataFrame, log: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Observed against predicted per row: the observed squared error, the posterior median of mu and the 95%
    posterior predictive interval (lo, hi), all on the log scale when `log`; and per protocol in PROTOCOLS
    present in `table` (the rows in observation order) the interval's coverage and the log-scale R^2
    (protocol, n, coverage95, r2_log)."""
    obs = np.asarray(idata.observed_data["obs"])
    pp = np.asarray(idata.posterior_predictive["obs"]).reshape(-1, obs.size)
    mu = np.median(log_mu_draws(idata.posterior, features), axis=0)
    if log:
        obs, pp = np.log(obs), np.log(pp)
    else:
        mu = np.exp(mu)
    lo, hi = np.percentile(pp, [2.5, 97.5], axis=0)
    coverage = []
    for p in PROTOCOLS:
        r = (table.protocol == p).to_numpy()
        if r.any():
            x, y = (mu[r], obs[r]) if log else (np.log(mu[r]), np.log(obs[r]))
            coverage.append(dict(protocol=p, n=int(r.sum()), coverage95=((obs[r] >= lo[r]) & (obs[r] <= hi[r])).mean(),
                                 r2_log=np.corrcoef(x, y)[0, 1] ** 2))
    return obs, mu, lo, hi, pd.DataFrame(coverage)


def diagnostics(idata, features: dict) -> dict:
    """Fit and convergence summary of one model: divergences, max r_hat and min ESS over the
    free parameters, LOO (elpd, se, p_loo) with Pareto-k counts, and the Bayesian R^2 on the
    log scale (posterior median of var(log mu) / (var(log mu) + var(log y - log mu)))."""
    summ = az.summary(idata, var_names=["~cell_z", "~judge_z", "~interaction_z"], filter_vars="like", round_to="none")
    loo = az.loo(idata, pointwise=True)
    k = np.asarray(loo.pareto_k)
    y = np.log(np.asarray(idata.observed_data["obs"]))
    mu = log_mu_draws(idata.posterior, features)
    r2 = np.var(mu, axis=1) / (np.var(mu, axis=1) + np.var(y[None, :] - mu, axis=1))
    return dict(divergences=int(idata.sample_stats.diverging.sum()), r_hat_max=float(summ.r_hat.max()),
                ess_bulk_min=float(summ.ess_bulk.min()), ess_tail_min=float(summ.ess_tail.min()),
                elpd_loo=float(loo.elpd_loo), elpd_se=float(loo.se), p_loo=float(loo.p_loo),
                pareto_k_bad=int(((k > 0.7) & (k <= 1)).sum()), pareto_k_very_bad=int((k > 1).sum()), n=int(k.size),
                r2_log=float(np.median(r2)), r2_log_lo=float(np.percentile(r2, 2.5)), r2_log_hi=float(np.percentile(r2, 97.5)))


def loo_compare(idatas: dict) -> pd.DataFrame:
    """arviz.compare over the named fits (elpd_loo, elpd_diff, weight, se, dse, warning)."""
    return az.compare(idatas, ic="loo", scale="log")
