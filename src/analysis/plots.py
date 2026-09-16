"""Figures and tables of the results section.

forecast_figure   one panel per domain, semesters along x; within a semester the targets
                  sorted by theta*, black = true theta, orange = forecast, dotted connectors
bar_figure        four bars (the conditions), coloured by method, at the mean of a per-(domain,
                  semester) metric; one open circle per domain at its pooled value, joined
                  across conditions by a domain-styled line; easy-item prediction as a dashed
                  line
semester_table    semester x condition for one domain, or pooled over the domains that have
                  the semester, with an "all" row

Judge protocols appear as the best judge on references (task_metrics has no other judge
rows); one variant per protocol is drawn, chosen by the runner's toggles.

matplotlib.use("Agg") below is deliberate: figures go to files, nothing renders inline in an
IDE cell.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from analysis.stats import BASELINE, BEST, JUDGED, METRICS, one_per_target

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial"], "font.size": 8, "axes.labelsize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.8,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "svg.fonttype": "none", "savefig.dpi": 300,
})
CONDITIONS = ["glm_regex", "embedding", "direct", "pairwise"]  # the bars, in this order
LABELS = {BASELINE: "easy-item prediction", "glm_regex": "length+regex", "embedding": "embedding",
          "direct": "direct judge", "pairwise": "pairwise judge"}
COLORS = dict(zip(CONDITIONS, ["#1baf7a", "#4a3aa7", "#eb6834", "#2a78d6"]))  # one hue per method
DOMAIN_STYLES = {"science": "-", "math": "--", "coding": ":", "cyber": "-."}  # line style identifies the domain
TRUE, PRED = "black", "#eb6834"
METRIC_LABELS = {"mse": "MSE (θ units)", "delta": "MSE − easy-item MSE", "z": "z of MSE − easy-item MSE (paired bootstrap)"}


def ci95(values: np.ndarray) -> float:
    """Half-width of the t-based 95% interval of the mean of `values`; NaN with fewer than 3."""
    n = len(values)
    if n < 3:
        return float("nan")
    return float(scipy_stats.t.ppf(0.975, n - 1) * np.std(values, ddof=1) / np.sqrt(n))


def chosen(metrics: pd.DataFrame, variants: dict[str, str]) -> pd.DataFrame:
    """The metric rows of the chosen variant per protocol ("-" for protocols without one)."""
    return metrics[[variants.get(p, "-") == v for p, v in zip(metrics.protocol, metrics.variant)]]


def forecast_figure(forecasts: pd.DataFrame, protocol: str, variant: str, path, semesters: list[str] | None = None) -> None:
    """True theta (black) and the condition's forecast (orange) for every target: one panel per
    domain (panel width follows its number of semesters), semesters along x, targets within
    a semester sorted by theta*, a dotted connector per target. `semesters` restricts the
    x-axis (e.g. the latest one). Judge protocols are drawn as the best judge on references."""
    f = one_per_target(forecasts)
    baseline = f[f.protocol == BASELINE].groupby(["domain", "semester"]).theta_hat.first()  # one value per task
    f = f[(f.protocol == protocol) & (f.variant == variant)]
    if semesters:
        f = f[f.semester.isin(semesters)]
    domains = list(dict.fromkeys(f.domain))
    widths = [f[f.domain == d].semester.nunique() for d in domains]
    fig, axes = plt.subplots(1, len(domains), figsize=(0.95 * sum(widths) + 0.5 * len(domains), 2.1), sharey=True,
                             gridspec_kw={"width_ratios": widths}, layout="constrained")
    for ax, dom in zip(np.atleast_1d(axes), domains):
        d = f[f.domain == dom]
        sems = sorted(dict.fromkeys(d.semester))  # "2024-I" < "2024-II" < "2025-I": chronological
        for i, sem in enumerate(sems):
            g = d[d.semester == sem].sort_values("theta")
            x = i + np.linspace(-0.36, 0.36, len(g))  # targets spread within the semester's slot
            ax.vlines(x, np.minimum(g.theta, g.theta_hat), np.maximum(g.theta, g.theta_hat), color="#b0aea6", lw=0.6, zorder=1)
            if (dom, sem) in baseline.index:  # the easy-item prediction: one level per task
                ax.hlines(baseline[(dom, sem)], i - 0.45, i + 0.45, color="grey", lw=0.8, ls="--", zorder=0)
            ax.scatter(x, g.theta, s=16, facecolor="white", edgecolor=TRUE, linewidth=0.8, zorder=3)
            ax.scatter(x, g.theta_hat, s=16, facecolor=PRED, edgecolor="white", linewidth=0.5, zorder=4)
        ax.set_xticks(range(len(sems)), sems, rotation=90 if len(sems) > 3 else 0)
        ax.text(0, 1.02, dom, transform=ax.transAxes, color="grey", ha="left")
    np.atleast_1d(axes)[0].set_ylabel("θ")
    handles = [plt.Line2D([], [], marker="o", ms=4, ls="", markerfacecolor="white", markeredgecolor=TRUE, label="true θ"),
               plt.Line2D([], [], marker="o", ms=4, ls="", color=PRED, label=f"forecast ({LABELS.get(protocol, protocol)})"),
               plt.Line2D([], [], color="grey", lw=0.8, ls="--", label=LABELS[BASELINE])]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def bar_figure(metrics: pd.DataFrame, metric: str, variants: dict[str, str], path, semester: str | None = None) -> None:
    """One bar per condition in CONDITIONS, coloured by method, at the mean over the (domain,
    semester) tasks of the metric ("mse", "delta" or "z"), with a capless 95% t-interval over
    those tasks (none below 3); one open circle per domain at the domain's pooled value
    (semester "all", or the given semester's value), the circles of a domain joined across the
    conditions by a black line whose style is DOMAIN_STYLES[domain]; the easy-item prediction
    as a dashed grey line (its mean raw MSE, or 0 for delta and z). A plain single axis: the
    y-limits are set by the values within Tukey's far-out fence (3 IQR) and anything beyond
    is not drawn but printed, so it can be marked by hand."""
    assert metric in METRICS, metric
    m = chosen(metrics, variants)
    m = m[m.domain != "all"]
    tasks = m[m.semester != "all"]
    if semester:
        tasks = tasks[tasks.semester == semester]
    per_domain = tasks if semester else m[m.semester == "all"]
    fig, ax = plt.subplots(figsize=(3.2, 2.4), layout="constrained")
    x = {p: i for i, p in enumerate(CONDITIONS)}
    drawn = []
    for p, i in x.items():
        values = tasks[tasks.protocol == p][metric].dropna().to_numpy()
        if len(values) == 0:
            continue
        ax.bar(i, values.mean(), color=COLORS[p], width=0.72, zorder=1)
        ax.hlines(values.mean(), i - 0.36, i + 0.36, color="black", lw=1.0, zorder=4)
        if np.isfinite(half := ci95(values)):
            ax.errorbar(i, values.mean(), yerr=half, color="black", lw=0.8, capsize=0, zorder=4)
        drawn += [values.mean()]
    for d, style in DOMAIN_STYLES.items():
        g = per_domain[per_domain.domain == d].set_index("protocol")[metric].reindex(CONDITIONS)
        keep = [p for p in CONDITIONS if np.isfinite(g[p])]
        if not keep:
            continue
        ax.plot([x[p] for p in keep], g[keep], color="black", lw=0.7, ls=style, zorder=3)
        ax.scatter([x[p] for p in keep], g[keep], s=18, facecolor="white", edgecolor="black", linewidth=0.8, zorder=5)
        drawn += list(g[keep])
    baseline = tasks[tasks.protocol == BASELINE].mse.mean() if metric == "mse" else 0.0
    ax.axhline(baseline, color="grey", lw=0.8, ls="--", zorder=0)
    drawn = np.asarray(drawn + [baseline], float)
    q1, q3 = np.percentile(drawn, [25, 75])
    inside = drawn[(drawn >= q1 - 3 * (q3 - q1)) & (drawn <= q3 + 3 * (q3 - q1))]
    if len(inside) < len(drawn):  # off-axis values are left to the user to mark by hand
        print(f"{path}: off-axis {sorted(set(drawn) - set(inside))}")
        pad = 0.08 * max(np.ptp(inside), 1e-9)
        ax.set_ylim(min(inside.min(), 0) - pad, inside.max() + pad)
    ax.set_xticks(range(len(CONDITIONS)), [LABELS[p] for p in CONDITIONS])
    ax.tick_params(axis="x", rotation=90)
    ax.set_ylabel(METRIC_LABELS[metric])
    domains = [d for d in DOMAIN_STYLES if d in set(per_domain.domain)]
    handles = [plt.Line2D([], [], color="black", lw=0.7, ls=DOMAIN_STYLES[d], marker="o", ms=4, markerfacecolor="white", label=d) for d in domains]
    handles += [plt.Line2D([], [], color="black", lw=0.8, label="mean ± 95% CI over tasks"),
                plt.Line2D([], [], color="grey", lw=0.8, ls="--", label=LABELS[BASELINE])]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False, handlelength=2.2)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def semester_table(metrics: pd.DataFrame, metric: str, variants: dict[str, str], domain: str | None = None) -> pd.DataFrame:
    """Semester x condition (easy-item prediction first, then CONDITIONS) for one domain, or pooled when
    `domain` is None: the mean over the domains that have that semester, plus an "all" row
    (the domain's pooled row, or the mean over every (domain, semester) cell). Columns are
    the LABELS; a condition without a value is NaN."""
    assert metric in METRICS, metric
    m = chosen(metrics, variants)
    m = m[m.domain == domain] if domain else m[(m.domain != "all") & (m.semester != "all")]
    per_task = m[m.semester != "all"].pivot_table(index="semester", columns="protocol", values=metric, aggfunc="mean").sort_index()
    if domain:
        pooled = m[m.semester == "all"].set_index("protocol")[metric]
    else:
        pooled = m.groupby("protocol")[metric].mean()
    table = pd.concat([per_task, pooled.to_frame("all").T])
    columns = [p for p in [BASELINE, *CONDITIONS] if p in table.columns]
    return table[columns].rename(columns=LABELS)
