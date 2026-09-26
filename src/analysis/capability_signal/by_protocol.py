"""
Squared error and log likelihood ratio of each protocol against the Baseline for the targets of the main analysis,
summarised per target model (its squared error averaged over its domains) and per domain, as a figure, a table of
means and tests, and a caption text.

The squared error is each target's own. The test is the log likelihood ratio of the target's measured θ* under the
protocol's predictive distribution against the Baseline's (analysis.likelihood_ratio), tested against 0 with a target
model (its domains averaged) as the unit.
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
    """
    Selects the log likelihood ratio rows the tests use: for judge protocols only the rows of each task's best judge on
    references, for transcript protocols every row; only the Student t shape (lr.SHAPE) and only `semesters`. Rows of
    every reference set are kept (the full set, dropped "-", and each reference left out in turn).

    Inputs:
        llr: every log likelihood ratio (analysis/forecasts/likelihood_ratio.csv); uses domain, semester, model,
            protocol, judge and shape.
        forecasts: every protocol's prediction of every target (analysis/forecasts/forecasts.csv), from which each
            task's best judge is found.
        variants: protocol -> the variant it is reported in (config.yaml's protocols).
        semesters: the semesters to keep.
    Outputs:
        the selected rows of `llr`, same columns (domain, semester, model, protocol, judge, dropped, shape, llr); one
        row per target, protocol and reference set.
    """
    llr = lr.select(llr, lr.chosen_judges(forecasts, variants))
    return llr[(llr["shape"] == lr.SHAPE) & llr.semester.isin(semesters)]


def protocol_tests(llr: pd.DataFrame) -> pd.DataFrame:
    """
    Tests each protocol's mean log likelihood ratio against 0 with a one-sample t-test, a target model (the mean of
    its rows over domains, per semester) being one unit, and makes the p-value two-sided.

    Inputs:
        llr: log likelihood ratio rows of one reference set only (dropped "-"), so the standard error is the plain
            t-test's with no reference-set part; uses protocol, domain, semester, model, dropped and llr.
    Outputs:
        DataFrame with columns protocol, n (target models), llr (mean over target models), se and p (two-sided); one
        row per protocol present in `llr`.
    """
    tests = lr.pooled_tests(llr, unit=("semester", "model"))
    tests["p"] = 2 * tests.p.clip(upper=1 - tests.p)  # two-sided: pooled_tests gives the upper tail
    return tests


def per_model(errors: pd.DataFrame, protocols: list[str], domains: list[str]) -> pd.DataFrame:
    """
    Averages each target model's squared error over its domains, per protocol. A model missing a protocol's
    prediction in any of `domains` gets NaN for that protocol.

    Inputs:
        errors: one prediction per target and protocol (stats.target_errors); uses model, protocol, domain and err.
        protocols: the protocols to return, in column order.
        domains: the domains a model needs a prediction in for its mean to count.
    Outputs:
        DataFrame indexed by model, one column per protocol in `protocols`; one row per target model.
    """
    wide = errors.pivot_table(index="model", columns=["protocol", "domain"], values="err")
    complete = wide.T.groupby(level="protocol").count().T.eq(len(domains))
    return wide.T.groupby(level="protocol").mean().T.where(complete).reindex(columns=protocols)


def two_lines(label: str) -> str:
    """
    Breaks a tick label onto two lines at its last space.

    Inputs:
        label: the label.
    Outputs:
        the label with its last space replaced by a newline (unchanged if it has no space).
    """
    return "\n".join(label.rsplit(" ", 1))


def model_error_figure(pooled: pd.DataFrame, tests: pd.DataFrame, protocols: list[str], display: dict) -> plt.Figure:
    """
    Draws, per protocol, each target model's squared error averaged over domains as a small jittered point, the mean
    over models as a large circle and its 95% t-interval as a box, with the significance stars (or n.s.) of the log
    likelihood ratio test above every protocol but the Baseline. The y axis runs to 1.15 times the 95th percentile of
    all per-model errors; a model above it is drawn on the top edge and printed.

    Inputs:
        pooled: per_model's table (one row per target model, one column per protocol).
        tests: protocol_tests' table; uses protocol and p.
        protocols: the protocols drawn, left to right, the Baseline first.
        display: config.yaml's display section; uses protocol_names and protocol_colours.
    Outputs:
        the matplotlib Figure (not saved).
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
    """
    Summarises each protocol's per-model squared errors: their mean with its 95% t-interval, and for every protocol
    but the Baseline the paired t-test of its errors against the Baseline's and its log likelihood ratio test.

    Inputs:
        pooled: per_model's table (one row per target model, one column per protocol); must include the Baseline's
            column, which the paired differences use.
        tests: protocol_tests' table; uses protocol, llr, se and p.
        protocols: the protocols in row order.
    Outputs:
        DataFrame with one row per protocol: protocol, mse (mean over target models), lo and hi (its 95% t-interval),
        models (their number); for protocols other than the Baseline also delta (mean paired difference from the
        Baseline's error), delta_t, delta_df and delta_p (two-sided paired t-test), llr, llr_se and llr_p (from
        `tests`). The Baseline's row has NaN in those columns.
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
    """
    Writes one caption line per protocol: its mean squared error with the 95% interval and number of models, and for
    protocols other than the Baseline the log likelihood ratio test and the paired t-test against the Baseline.

    Inputs:
        table: error_table's output.
        names: protocol -> display name.
    Outputs:
        list of strings, one per row of `table`.
    """
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
    """
    Builds the text written next to the error-by-protocol figure: what the figure draws, which judges are used, how
    the log likelihood ratio test works, then error_lines.

    Inputs:
        table: error_table's output.
        names: protocol -> display name.
        judges: the sentence describing the judges the judge protocols use.
    Outputs:
        the caption as one string ending in a newline.
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
    """
    Tabulates each protocol's mean squared error over the targets of each domain, and the mean of the domain means
    (Pooled), as text to two decimals. A cell gets */**/*** when the mean log likelihood ratio is positive and its
    two-sided test is significant: per domain a one-sample t-test over the domain's targets, for Pooled the test in
    `tests` (a target model the unit).

    Inputs:
        errors: one prediction per target and protocol (stats.target_errors); uses protocol, domain and err.
        llr: the matching log likelihood ratio rows of one reference set (dropped "-"); uses protocol, domain and llr.
        tests: protocol_tests of `llr`; uses protocol, llr and p.
        names: protocol -> row label, in row order.
        domain_names: domain -> column label, in column order.
    Outputs:
        DataFrame with columns Protocol, Pooled and one per domain label; one row per protocol in `names`.
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
