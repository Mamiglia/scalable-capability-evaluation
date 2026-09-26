"""
Tests whether a protocol's gain over the Baseline depends on the domain, with a hierarchical Gamma GLM of squared
prediction error:

    squared_error ~ protocol + domain + (protocol | domain) + (1 | target:domain),   Gamma, log link

over the targets of the main analysis, one prediction per target and protocol (judge protocols: each task's best judge
on references). The Baseline is the reference level of protocol, so every coefficient is a log error ratio to it. The
fit runs through hibayes (models in domain_glm_model.py); this module builds its rows and pipeline, checks the fit,
reads the posterior into tables of error ratios and contrasts, and draws the ratio and contrast figures.
"""

from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis import likelihood_ratio as lr
from analysis.capability_signal.by_protocol import two_lines
from analysis.stats import BASELINE

POOLED = "all"  # the domain label of an estimate pooled over domains
RATIO_TICKS = [0.05, 0.25, 1, 4]


def glm_rows(errors: pd.DataFrame, floor: float) -> pd.DataFrame:
    """
    Builds the GLM's data rows from the squared errors, raising each error to at least `floor` because a Gamma
    outcome must be positive.

    Inputs:
        errors: one prediction per target and protocol (stats.target_errors); uses err, protocol, domain, semester and
            model.
        floor: the smallest squared error kept (config.yaml's domain_glm.error_floor).
    Outputs:
        DataFrame with one row per target and protocol: y (the floored squared error), protocol, domain, semester,
        task ("<domain> <semester>"), model, model_domain ("<model> <domain>", the cell every protocol predicts once),
        judge and protocol_judge (both "-": one judge per task, so the judge term has a single level).
    """
    return pd.DataFrame({
        "y": errors.err.clip(lower=floor).to_numpy(),
        "protocol": errors.protocol.to_numpy(), "domain": errors.domain.to_numpy(), "semester": errors.semester.to_numpy(),
        "task": (errors.domain + " " + errors.semester).to_numpy(), "model": errors.model.to_numpy(),
        "model_domain": (errors.model + " " + errors.domain).to_numpy(), "judge": "-", "protocol_judge": "-",
    })


def hibayes_config(rows_csv: Path, protocols: list[str], domains: list[str], priors: dict[str, float],
                   sampler: dict) -> dict:
    """
    Builds the hibayes pipeline configuration: load the rows, index the categories (protocol and domain in the given
    order, so the Baseline is the reference level), fit the model with and without the (protocol | domain) term, run
    the prior predictive, divergence, r_hat, ESS and posterior predictive checks, and write a summary table and trace
    plots.

    Inputs:
        rows_csv: the CSV of glm_rows written for hibayes to load.
        protocols: the protocol levels, the Baseline first.
        domains: the domain levels, in order.
        priors: config.yaml's domain_glm.priors (intercept_scale, effect_scale, sigma_scale, cell_scale), passed to the
            models as prior_<name>.
        sampler: config.yaml's domain_glm.sampler (samples, warmup, chains, seed, target_accept).
    Outputs:
        dict in the layout hibayes' AnalysisConfig.from_dict reads.
    """
    model_config = {"link_function": "identity", "fit": dict(sampler), **{f"prior_{k}": v for k, v in priors.items()}}
    return {
        "data_loader": {"paths": {"extracted_data": [str(rows_csv)]}},
        "data_process": {"processors": [
            {"extract_observed_feature": {"feature_name": "y"}},
            {"extract_features": {"categorical_features": ["protocol", "domain", "model_domain", "protocol_judge"],
                                  "reference_categories": {"protocol_judge": "-"},
                                  "category_order": {"protocol": list(protocols), "domain": list(domains)}}},
        ]},
        "model": {"models": [  # the reported fit first, the additive one (the LOO comparison) second
            {"name": "domain_glm", "config": {
                **model_config, "main_effect_params": ["protocol_effects", "sigma_protocol", "sigma_cell", "sigma_judge", "shape"]}},
            {"name": "domain_glm_additive", "config": {
                **model_config, "main_effect_params": ["protocol_effects", "sigma_cell", "sigma_judge", "shape"]}},
        ]},
        "check": {"checkers": [
            {"prior_predictive_check": {"num_samples": 200}}, "divergences",
            "r_hat",  # fails on the pinned reference levels (NaN r_hat); fit() checks the free parameters
            "ess_bulk", "ess_tail",
            {"posterior_predictive_plot": {"interactive": False, "kind": "cumulative", "num_samples": 200}},
        ]},
        "communicate": {"communicators": [
            {"summary_table": {"vars": ["protocol_effects", "domain_effects", "sigma_protocol", "sigma_cell", "sigma_judge",
                                        "judge_effects", "shape"]}},
            {"trace_plot": {"vars": ["protocol_effects", "sigma_protocol", "sigma_cell"], "best_model": False}},
        ]},
        "platform": {"device_type": "cpu", "chain_method": "parallel"},
    }


def fit(rows: pd.DataFrame, out: Path, protocols: list[str], domains: list[str], priors: dict[str, float],
        sampler: dict) -> tuple:
    """
    Fits the GLM with and without the (protocol | domain) term through hibayes, and asserts r_hat <= 1.01 on every
    parameter of both fits whose r_hat is defined (the pinned reference levels have NaN r_hat).

    Inputs:
        rows: glm_rows' output.
        out: the folder for hibayes' files; created if missing.
        protocols, domains, priors, sampler: as hibayes_config.
    Outputs:
        tuple of the two hibayes ModelAnalysisState objects (with the term, then without), each holding
        inference_data, features and coords; in out: rows.csv, and hibayes' saved state, check results, summary
        table and plots.
    """
    from hibayes.analysis import AnalysisConfig, communicate, load_data, model, process_data
    from hibayes.platform import configure_computation_platform
    from hibayes.ui import ModellingDisplay

    from analysis.capability_signal import domain_glm_model  # noqa: F401  registers the models with hibayes

    out.mkdir(parents=True, exist_ok=True)
    rows.to_csv(out / "rows.csv", index=False)
    config = AnalysisConfig.from_dict(hibayes_config(out / "rows.csv", protocols, domains, priors, sampler))
    display = ModellingDisplay()
    configure_computation_platform(platform_config=config.platform, display=display)
    state = load_data(config=config.data_loader, display=display)
    state = process_data(analysis_state=state, config=config.data_process, display=display)
    state = model(analysis_state=state, models_to_run_config=config.models, checker_config=config.checkers,
                  platform_config=config.platform, display=display, out=out)
    state = communicate(analysis_state=state, communicate_config=config.communicate, display=display, out=out)
    state.save(path=out)
    fitted, additive = state.models
    for f in (fitted, additive):  # the pinned levels are constants with NaN r_hat, left out here
        rhat = f.get_summary().r_hat.dropna()
        assert (rhat <= 1.01).all(), rhat[rhat > 1.01]
    return fitted, additive


def log_mu_draws(posterior, features: dict) -> np.ndarray:
    """
    Computes the model's log mean squared error for every posterior draw and data row, adding the intercept and the
    protocol, domain, interaction, cell and judge effects at each row's indices.

    Inputs:
        posterior: the fit's posterior (chain and draw dimensions first).
        features: the fit's features; uses protocol_index, domain_index, model_domain_index and protocol_judge_index.
    Outputs:
        array of shape (draws, rows), chains flattened into draws.
    """
    get = lambda k: np.asarray(posterior[k]).reshape(-1, *np.asarray(posterior[k]).shape[2:])  # noqa: E731
    p, d, c, j = (np.asarray(features[k]) for k in ("protocol_index", "domain_index", "model_domain_index", "protocol_judge_index"))
    return (get("intercept")[:, None] + get("protocol_effects")[:, p] + get("domain_effects")[:, d]
            + get("interaction")[:, p, d] + get("cell_effects")[:, c] + get("judge_effects")[:, j])


def diagnostics(idata, features: dict) -> dict:
    """
    Summarises one fit's convergence and fit: divergent transitions, the largest r_hat and smallest bulk and tail ESS
    over the posterior variables other than the standard-normal z draws, the leave-one-out cross-validation estimate
    with its Pareto-k counts, and the Bayesian R^2 of log squared error.

    Inputs:
        idata: the fit's arviz InferenceData (posterior, sample_stats, observed_data "obs", log likelihood).
        features: the fit's features, for log_mu_draws.
    Outputs:
        dict: divergences, r_hat_max, ess_bulk_min, ess_tail_min, elpd_loo, elpd_se, p_loo, pareto_k_bad (0.7 < k <= 1),
        pareto_k_very_bad (k > 1), n (rows), r2_log with r2_log_lo and r2_log_hi (median and 2.5 / 97.5 percentiles
        over draws of var(log mu) / (var(log mu) + var(log y - log mu))).
    """
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
    """
    Compares the fits by their leave-one-out expected log predictive density (arviz.compare, log scale).

    Inputs:
        idatas: fit name -> its arviz InferenceData.
    Outputs:
        arviz.compare's DataFrame, one row per fit, best first: rank, elpd_loo, p_loo, elpd_diff, weight, se, dse,
        warning, scale.
    """
    return az.compare(idatas, ic="loo", scale="log")


def ratio_table(posterior, coords: dict) -> pd.DataFrame:
    """
    Computes each protocol's error ratio to the Baseline from the posterior: per domain exp(protocol effect + the
    protocol's deviation in that domain), and pooled exp(protocol effect), the ratio with no domain deviation.

    Inputs:
        posterior: the fit's posterior (chain and draw dimensions first); uses protocol_effects, interaction and
            sigma_protocol.
        coords: the fit's levels; uses protocol and domain.
    Outputs:
        DataFrame with columns protocol, domain, ratio, lo, hi (posterior median and 2.5 / 97.5 percentiles),
        sigma_protocol, sigma_lo, sigma_hi (the protocol's between-domain SD on the log scale, median and percentiles;
        NaN on per-domain rows); one row per protocol other than the Baseline and domain, plus one pooled row per
        protocol with domain "all".
    """
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


def wide_table(table: pd.DataFrame) -> pd.DataFrame:
    """
    Reshapes ratio_table to one row per protocol, with the Baseline added first as ratio 1 and no interval.

    Inputs:
        table: ratio_table's output.
    Outputs:
        DataFrame indexed by protocol; columns <domain>, <domain>_lo and <domain>_hi for each domain and for "all"
        (pooled), then sigma, sigma_lo and sigma_hi (the between-domain SD, NaN for the Baseline).
    """
    cols = list(dict.fromkeys(table.domain))  # the domains, then "all"
    wide = table.pivot(index="protocol", columns="domain", values=["ratio", "lo", "hi"])
    out = pd.DataFrame(index=[BASELINE, *dict.fromkeys(table.protocol)])
    for d in cols:
        out[d] = wide["ratio"][d].reindex(out.index).fillna({BASELINE: 1.0})
        out[f"{d}_lo"], out[f"{d}_hi"] = wide["lo"][d].reindex(out.index), wide["hi"][d].reindex(out.index)
    pooled = table[table.domain == POOLED].set_index("protocol").reindex(out.index)
    out["sigma"], out["sigma_lo"], out["sigma_hi"] = pooled.sigma_protocol, pooled.sigma_lo, pooled.sigma_hi
    out.index.name = "protocol"
    return out


def contrast_table(posterior, coords: dict, against: str) -> pd.DataFrame:
    """
    Computes the error ratio of the protocol `against` to every other protocol except the Baseline, per domain and
    pooled, from the same posterior draws. The cell and domain terms cancel in the difference, so these intervals
    are tighter than the two protocols' separate intervals suggest.

    Inputs:
        posterior: the fit's posterior (chain and draw dimensions first); uses protocol_effects and interaction.
        coords: the fit's levels; uses protocol and domain.
        against: the protocol in the numerator (config.yaml's domain_glm.contrast_against).
    Outputs:
        DataFrame with columns against, protocol, domain, ratio, lo, hi (posterior median and 2.5 / 97.5 percentiles)
        and p_below_1 (the posterior probability that `against` has the smaller error); one row per compared protocol
        and domain, plus one per protocol with domain "all" (pooled).
    """
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


def _domain_axis(ax, labels: list[str]) -> None:
    """
    Sets a categorical x axis with one tick per label, the bottom spine trimmed to the first and last tick.

    Inputs:
        ax: the axes.
        labels: the tick labels, left to right (pooled, then the domains).
    Outputs:
        none; `ax` is changed in place.
    """
    ax.set_xticks(range(len(labels)), labels)
    ax.set_xlim(-0.5, len(labels) - 0.5)
    ax.spines["bottom"].set_bounds(0, len(labels) - 1)


def _ratio_axis(ax, ticks: list[float], label: str | None) -> None:
    """
    Sets a log-scale y axis for error ratios, limited to the first and last tick, with a dashed line at 1 (equal
    error) and the left spine trimmed to the ticks.

    Inputs:
        ax: the axes.
        ticks: the tick values, ascending.
        label: the y-axis label, or None for no label.
    Outputs:
        none; `ax` is changed in place.
    """
    ax.axhline(1, color="0.5", ls="--", lw=0.8, zorder=0)
    ax.set(yscale="log", ylim=(ticks[0], ticks[-1]))
    ax.set_yticks(ticks, [f"{t:g}".replace("0.", ".") for t in ticks])
    ax.minorticks_off()
    ax.spines["left"].set_bounds(ticks[0], ticks[-1])
    if label:
        ax.set_ylabel(label)


def _point(ax, x: float, row, color: str, marker: str | None) -> None:
    """
    Draws one estimate: its 95% interval as a vertical line and its median as a marker, a filled circle when pooled
    and the domain's marker, open, otherwise.

    Inputs:
        ax: the axes.
        x: the x position.
        row: a row with ratio, lo and hi.
        color: the line and marker colour.
        marker: the domain's marker, or None for the pooled estimate.
    Outputs:
        none; draws on `ax`.
    """
    pooled = marker is None
    ax.plot([x, x], [row.lo, row.hi], color=color, lw=1 if pooled else 0.8, solid_capstyle="butt")
    ax.plot(x, row.ratio, marker="o" if pooled else marker, ms=4 if pooled else 3.5,
            mfc=color if pooled else "white", mec=color, mew=0.9, ls="none", clip_on=False)


def effects_figure(table: pd.DataFrame, protocols: list[str], display: dict) -> plt.Figure:
    """
    Draws each protocol's posterior error ratio to the Baseline with its 95% interval, one panel per protocol, the
    pooled estimate then each domain along x, on a shared log y axis.

    Inputs:
        table: ratio_table's output.
        protocols: the protocols to draw, in panel order; those missing from `table` are skipped.
        display: config.yaml's display section; uses protocol_names, protocol_colours, domain_markers and
            domain_names.
    Outputs:
        the matplotlib Figure (not saved).
    """
    names, colours, markers = display["protocol_names"], display["protocol_colours"], display["domain_markers"]
    protocols = [p for p in protocols if (table.protocol == p).any()]
    domains = [POOLED, *display["domain_names"]]
    fig, axes = plt.subplots(1, len(protocols), figsize=(5.5, 1.9), layout="constrained", sharey=True)
    for ax, protocol in zip(axes, protocols):
        rows = table[table.protocol == protocol].set_index("domain")
        for x, domain in enumerate(domains):
            if domain in rows.index:
                _point(ax, x, rows.loc[domain], colours[protocol], markers.get(domain))
        _domain_axis(ax, ["Pooled", *display["domain_names"].values()])
        ax.tick_params(axis="x", rotation=90)
        _ratio_axis(ax, RATIO_TICKS, "MSE ratio to baseline" if ax is axes[0] else None)
        ax.set_title(names[protocol], loc="left", fontsize=7, fontweight="bold", pad=3)
    return fig


def contrasts_figure(contrasts: pd.DataFrame, table: pd.DataFrame, protocols: list[str], display: dict) -> plt.Figure:
    """
    Draws two panels: a, the error ratio of the contrasted protocol to each other protocol with its 95% interval,
    pooled then per domain, coloured by the other protocol; b, each protocol's between-domain SD of log error with
    its 95% interval.

    Inputs:
        contrasts: contrast_table's output.
        table: ratio_table's output; its pooled rows give the SDs.
        protocols: the protocols to draw; those missing from `contrasts` or `table` are skipped.
        display: config.yaml's display section; uses protocol_names, protocol_colours, domain_markers and
            domain_names.
    Outputs:
        the matplotlib Figure (not saved).
    """
    names, colours, markers = display["protocol_names"], display["protocol_colours"], display["domain_markers"]
    against = contrasts.against.iloc[0]
    comparators = [p for p in protocols if p != against and (contrasts.protocol == p).any()]
    domains = [POOLED, *display["domain_names"]]
    fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.3), layout="constrained", width_ratios=[1.6, 1])
    ax = axes[0]
    offsets = np.linspace(-0.22, 0.22, len(comparators))
    for offset, protocol in zip(offsets, comparators):
        rows = contrasts[contrasts.protocol == protocol].set_index("domain")
        for x, domain in enumerate(domains):
            if domain in rows.index:
                _point(ax, x + offset, rows.loc[domain], colours[protocol], markers.get(domain))
    _domain_axis(ax, ["Pooled", *display["domain_names"].values()])
    ax.tick_params(axis="x", rotation=90)
    _ratio_axis(ax, RATIO_TICKS, f"{names[against].split()[0]} / comparator MSE")
    ax.text(-0.3, 1.02, "a", transform=ax.transAxes, fontweight="bold")
    handles = [plt.Line2D([], [], color=colours[p], marker="o", ms=4, lw=0, label=names[p]) for p in comparators]
    fig.legend(handles=handles, loc="outside upper center", ncol=3, frameon=False, handletextpad=0.3, columnspacing=0.8)

    ax = axes[1]
    sigma = table[table.domain == POOLED].set_index("protocol")
    drawn = [p for p in protocols if p in sigma.index]
    for x, protocol in enumerate(drawn):
        row = sigma.loc[protocol]
        ax.plot([x, x], [row.sigma_lo, row.sigma_hi], color=colours[protocol], lw=1, solid_capstyle="butt")
        ax.plot(x, row.sigma_protocol, "o", color=colours[protocol], ms=4, clip_on=False)
    sigma_ticks = [0, 0.5, 1, 1.5]
    ax.set_xticks(range(len(drawn)), [two_lines(names[p]) for p in drawn], rotation=90)
    ax.set_xlim(-0.5, len(drawn) - 0.5)
    ax.set(ylim=(sigma_ticks[0], sigma_ticks[-1]), ylabel="Between-domain SD")
    ax.set_yticks(sigma_ticks)
    ax.spines["bottom"].set_bounds(0, len(drawn) - 1)
    ax.spines["left"].set_bounds(sigma_ticks[0], sigma_ticks[-1])
    ax.text(-0.5, 1.02, "b", transform=ax.transAxes, fontweight="bold")
    return fig


def _interval(r) -> str:
    """
    Formats an estimate as "median [lo, hi]".

    Inputs:
        r: a row with ratio, lo and hi.
    Outputs:
        the formatted string.
    """
    return f"{lr.num(r.ratio)} [{lr.num(r.lo)}, {lr.num(r.hi)}]"


def effects_caption(table: pd.DataFrame, protocols: list[str], display: dict) -> str:
    """
    Builds the text written next to the ratio figure: what it draws, then per protocol every ratio to the Baseline
    (pooled and per domain) with its interval, and the protocol's between-domain SD.

    Inputs:
        table: ratio_table's output.
        protocols: the protocols listed, in order.
        display: config.yaml's display section; uses protocol_names and domain_names.
    Outputs:
        the caption as one string ending in a newline.
    """
    columns = {POOLED: "Pooled", **display["domain_names"]}
    lines = []
    for p in protocols:
        rows = table[table.protocol == p].set_index("domain")
        s = rows.loc[POOLED]
        lines.append(f"{display['protocol_names'][p]}: " + "; ".join(f"{columns[d]} {_interval(rows.loc[d])}" for d in columns)
                     + f"; between-domain SD {lr.num(s.sigma_protocol)} [{lr.num(s.sigma_lo)}, {lr.num(s.sigma_hi)}].")
    return ("Posterior MSE ratio of each protocol to the Baseline, pooled across domains (filled circles) and per domain "
            "(open markers), from the hierarchical Gamma GLM of squared error: posterior median [95% credible interval]; "
            "the dashed line is equal error to the Baseline. The between-domain SD is on the log scale.\n\n"
            + "\n".join(lines) + "\n")


def contrasts_caption(contrasts: pd.DataFrame, table: pd.DataFrame, protocols: list[str], display: dict) -> str:
    """
    Builds the text written next to the contrast figure: what its two panels draw, then every contrast (pooled and
    per domain) with its interval and posterior probability of being below 1, and every protocol's between-domain SD.

    Inputs:
        contrasts: contrast_table's output.
        table: ratio_table's output; its pooled rows give the SDs.
        protocols: the protocols listed, in order.
        display: config.yaml's display section; uses protocol_names and domain_names.
    Outputs:
        the caption as one string ending in a newline.
    """
    names, columns = display["protocol_names"], {POOLED: "Pooled", **display["domain_names"]}
    against = contrasts.against.iloc[0]
    lines = ["a:"]
    for p in [p for p in protocols if p != against]:
        rows = contrasts[contrasts.protocol == p].set_index("domain")
        lines.append(f"  {names[against]} / {names[p]}: " + "; ".join(
            f"{columns[d]} {_interval(rows.loc[d])} (P < 1: {lr.num(rows.loc[d].p_below_1)})" for d in columns) + ".")
    sigma = table[table.domain == POOLED].set_index("protocol")
    lines += ["b:"] + [f"  {names[p]}: {lr.num(sigma.loc[p].sigma_protocol)} [{lr.num(sigma.loc[p].sigma_lo)}, "
                       f"{lr.num(sigma.loc[p].sigma_hi)}]." for p in protocols]
    return (f"a, MSE ratio of the {names[against]} to each other protocol, pooled and per domain, from the same posterior "
            "draws of the hierarchical Gamma GLM: posterior median [95% credible interval], and the posterior probability "
            f"that the {names[against]} has the smaller error (P < 1). b, each protocol's between-domain standard deviation "
            "of log MSE.\n\n" + "\n".join(lines) + "\n")
