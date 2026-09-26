"""Squared error and log likelihood ratio of each protocol against the Baseline, for the 16 target models of the main
analysis: per target model pooled over its domains (Figure 3, Table 6; Figure 21 draws the same figure), and per
domain (Table 3).

The squared error is each target's own; the test is the log likelihood ratio of the target's measured θ* under the
protocol's predictive distribution against the Baseline's (analysis.likelihood_ratio), with a target model (its
domains averaged) as the unit.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from analysis import likelihood_ratio as lr
from analysis.plots import ci95, stars, unclip_markers
from analysis.stats import BASELINE

POOLED = "Pooled"


def chosen_llr(llr: pd.DataFrame, forecasts: pd.DataFrame, variants: dict[str, str],
               semesters: list[str]) -> pd.DataFrame:
    """The log likelihood ratio rows the tests use: each target's judge the task's best on references (transcript
    protocols as they are), the figures' predictive shape, `semesters` only; every reference set (dropped).

    llr: analysis/forecasts/likelihood_ratio.csv.
    """
    llr = lr.select(llr, lr.chosen_judges(forecasts, variants))
    return llr[(llr["shape"] == lr.SHAPE) & llr.semester.isin(semesters)]


def protocol_tests(llr: pd.DataFrame) -> pd.DataFrame:
    """Per protocol: n, the mean log likelihood ratio (llr), its SE and the two-sided p of the one-sample t-test
    against 0, a target model (its domains averaged) the unit.

    llr: one reference set's rows (dropped "-"), so the SE is the plain t-test's.
    """
    tests = lr.pooled_tests(llr, unit=("semester", "model"))
    tests["p"] = 2 * tests.p.clip(upper=1 - tests.p)  # two-sided: pooled_tests gives the upper tail
    return tests


def per_model(errors: pd.DataFrame, protocols: list[str], domains: list[str]) -> pd.DataFrame:
    """Each target model's squared error averaged over its domains: one row per model, one column per protocol in
    `protocols`; NaN where the model lacks the protocol in any domain.

    errors: stats.target_errors rows.
    """
    wide = errors.pivot_table(index="model", columns=["protocol", "domain"], values="err")
    complete = wide.T.groupby(level="protocol").count().T.eq(len(domains))
    return wide.T.groupby(level="protocol").mean().T.where(complete).reindex(columns=protocols)


def two_lines(label: str) -> str:
    """A tick label broken at its last space."""
    return "\n".join(label.rsplit(" ", 1))


def model_error_figure(pooled: pd.DataFrame, tests: pd.DataFrame, protocols: list[str], display: dict) -> plt.Figure:
    """Per protocol (the Baseline first), each target model's squared error pooled over domains as a small point, the
    mean over models as a large circle and its 95% t-interval as a box; stars (or n.s.) from `tests`. The axis holds
    the middle 95% of the per-model errors; a model beyond it is drawn on the edge.

    pooled: per_model's table. tests: protocol_tests' table. protocols: the protocols drawn, the Baseline first.
    display: config.yaml's display section (protocol_names, protocol_colours).
    """
    names, colours = display["protocol_names"], display["protocol_colours"]
    values = {p: pooled[p].dropna().to_numpy(float) for p in protocols}
    top = float(np.nanpercentile(np.concatenate(list(values.values())), 95)) * 1.15
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(2.7, 2.4), layout="constrained")
    for i, p in enumerate(protocols):
        v, colour = values[p], colours[p]
        mean = float(v.mean())
        lo, hi = mean - ci95(v), mean + ci95(v)
        ax.add_patch(plt.Rectangle((i - 0.3, max(lo, 0)), 0.6, min(hi, top) - max(lo, 0),
                                   facecolor=colour, edgecolor="none", alpha=0.35, zorder=1))
        out = v > top
        ax.scatter(i + rng.uniform(-0.15, 0.15, len(v)), np.minimum(v, top), s=7, color=colour, alpha=0.5,
                   marker="o", linewidths=0, edgecolors="none", zorder=3, clip_on=False)
        if out.any():
            print(f"model_error_figure: {p} above the axis: {[round(x, 1) for x in v[out]]}")
        ax.scatter([i], [min(mean, top)], s=28, color=colour, linewidths=0, edgecolors="none", zorder=5)
        if p != BASELINE:
            ax.text(i, top, stars(float(tests.set_index("protocol").loc[p, "p"]), "n.s."), ha="center", va="bottom",
                    fontsize=7, zorder=10)
    ax.set_xlim(-0.6, len(protocols) - 0.4)
    ax.set_ylim(0, top)
    ax.set_ylabel("Squared error")
    ax.set_xticks(range(len(protocols)), [two_lines(names[p]) for p in protocols])
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", multialignment="right")
    return unclip_markers(fig)


def error_table(pooled: pd.DataFrame, tests: pd.DataFrame, protocols: list[str]) -> pd.DataFrame:
    """Table 6: per protocol (the Baseline first), the mean over target models of the pooled squared error (mse) with
    its 95% t-interval (lo, hi) and number of models; for the other protocols the mean paired difference from the
    Baseline's error (delta) with its paired t-test (delta_t, delta_df, delta_p, two-sided) and the log likelihood
    ratio test (llr, llr_se, llr_p).

    pooled: per_model's table. tests: protocol_tests' table.
    """
    by = tests.set_index("protocol")
    rows = []
    for p in protocols:
        v = pooled[p].dropna().to_numpy(float)
        row = dict(protocol=p, mse=v.mean(), lo=v.mean() - ci95(v), hi=v.mean() + ci95(v), models=len(v))
        if p != BASELINE:
            d = (pooled[p] - pooled[BASELINE]).dropna()  # paired per target model
            t = scipy_stats.ttest_1samp(d, 0)
            row.update(delta=d.mean(), delta_t=t.statistic, delta_df=len(d) - 1, delta_p=t.pvalue,
                       llr=by.loc[p, "llr"], llr_se=by.loc[p, "se"], llr_p=by.loc[p, "p"])
        rows.append(row)
    return pd.DataFrame(rows)


def error_lines(table: pd.DataFrame, names: dict[str, str]) -> list[str]:
    """One caption line per error_table row: every number the figure draws and both tests."""
    lines = []
    for r in table.itertuples():
        entry = (f"{names[r.protocol]}: mean squared error {lr.num(r.mse)} (95% CI {lr.num(r.lo)} to {lr.num(r.hi)}, "
                 f"{r.models} models)")
        if r.protocol != BASELINE:
            entry += (f"; mean log likelihood ratio {lr.num(r.llr)} (SE {lr.num(r.llr_se)}), {lr.p_text(r.llr_p)}; "
                      f"squared error minus the Baseline's {lr.num(r.delta, sign=True)}, paired t({r.delta_df:.0f}) = "
                      f"{lr.num(r.delta_t)}, two-sided {lr.p_text(r.delta_p)}")
        lines.append(entry + ".")
    return lines


def error_caption(table: pd.DataFrame, names: dict[str, str], judges: str) -> str:
    """The .txt of the error-by-protocol figure: what it draws, the test, then error_lines.

    judges: the sentence on the judges the judge protocols use.
    """
    return ("Squared error of each protocol's forecast of θ* for the 16 target models of 2025-II and 2026-I (8 per "
            "semester). Each small point is one target model's squared error averaged over its four domains (Science, "
            "Maths, Coding, Cyber); the large circle is the mean over models and the box its 95% t-interval. "
            + judges + " Stars test whether a protocol's forecast explains the true θ* better than the "
            "references' own distribution, with each target model (its four domains averaged) as the unit, 15 degrees "
            "of freedom. " + lr.TEST_TEXT + " n.s.: not significant. Each protocol's line also gives the two-sided paired "
            "t-test of the models' squared error against the Baseline's (not drawn).\n\n"
            + "\n".join(error_lines(table, names)) + "\n")


def domain_table(errors: pd.DataFrame, llr: pd.DataFrame, tests: pd.DataFrame, names: dict[str, str],
                 domain_names: dict[str, str]) -> pd.DataFrame:
    """Table 3: per protocol, the mean squared error over the targets of each domain and the mean of the four
    (Pooled), to two decimals, marked */**/*** when the mean log likelihood ratio is positive and its two-sided
    one-sample t-test significant: per domain over the domain's targets, pooled with protocol_tests' test.

    errors: stats.target_errors rows. llr: the matching rows of one reference set (dropped "-"). tests: protocol_tests
    of `llr`. names: protocol -> row label, in row order. domain_names: domain -> column label, in column order.
    """
    def mark(mean: float, p: float) -> str:
        return "" if mean <= 0 else stars(p)

    pooled = tests.set_index("protocol")
    table = []
    for protocol, label in names.items():
        mse = errors[errors.protocol == protocol].groupby("domain").err.mean()
        row = {"Protocol": label, POOLED: f"{mse.reindex(list(domain_names)).mean():.2f}"}
        if protocol in pooled.index:
            row[POOLED] += mark(pooled.loc[protocol, "llr"], pooled.loc[protocol, "p"])
        for domain, column in domain_names.items():
            cell, g = f"{mse[domain]:.2f}", llr[(llr.protocol == protocol) & (llr.domain == domain)]
            if len(g):
                cell += mark(g.llr.mean(), scipy_stats.ttest_1samp(g.llr, 0).pvalue)
            row[column] = cell
        table.append(row)
    return pd.DataFrame(table)
