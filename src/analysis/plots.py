"""Figures and tables of the results section.

forecast_figure   one panel per domain, semesters along x; within a semester the targets
                  sorted by theta*, black = true theta, orange = forecast, dotted connectors
bar_figure        four bars (the conditions) at the mean of a per-(domain, semester) metric,
                  jittered circles for the individual (domain, semester) values, the
                  intercept as a dashed line
semester_table    semester x condition for one domain, or pooled over the domains that have
                  the semester, with an "all" row

Judge protocols appear as the best judge on references (task_metrics has no other judge
rows); one variant per protocol is drawn, chosen by the runner's toggles.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.stats import BASELINE, BEST, JUDGED, METRICS, one_per_target

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial"], "font.size": 8, "axes.labelsize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.8,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "svg.fonttype": "none", "savefig.dpi": 300,
})
CONDITIONS = ["glm_regex", "embedding", "direct", "pairwise"]  # the four bars, in this order
LABELS = {BASELINE: "intercept", "glm_regex": "length+regex", "embedding": "embedding", "direct": "direct judge",
          "pairwise": "pairwise judge"}
COLORS = dict(zip(CONDITIONS, ["#1baf7a", "#4a3aa7", "#eb6834", "#2a78d6"]))
TRUE, PRED = "black", "#eb6834"
METRIC_LABELS = {"mse": "MSE (θ units)", "delta": "MSE − intercept MSE", "z": "z of MSE − intercept (paired bootstrap)"}


def chosen(metrics: pd.DataFrame, variants: dict[str, str]) -> pd.DataFrame:
    """The metric rows of the chosen variant per protocol ("-" for protocols without one)."""
    return metrics[[variants.get(p, "-") == v for p, v in zip(metrics.protocol, metrics.variant)]]


def offset_spines(ax) -> None:
    """Nature-panel axes: left and bottom spines 3pt outward."""
    for side in ("left", "bottom"):
        ax.spines[side].set_position(("outward", 3))


def forecast_figure(forecasts: pd.DataFrame, protocol: str, variant: str, path, semesters: list[str] | None = None) -> None:
    """True theta (black) and the condition's forecast (orange) for every target: one panel per
    domain (panel width follows its number of semesters), semesters along x, targets within
    a semester sorted by theta*, a dotted connector per target. `semesters` restricts the
    x-axis (e.g. the latest one). Judge protocols are drawn as the best judge on references."""
    f = one_per_target(forecasts)
    f = f[(f.protocol == protocol) & (f.variant == variant)]
    if semesters:
        f = f[f.semester.isin(semesters)]
    domains = list(dict.fromkeys(f.domain))
    widths = [f[f.domain == d].semester.nunique() for d in domains]
    fig, axes = plt.subplots(1, len(domains), figsize=(0.9 * sum(widths) + 0.6 * len(domains), 2.2), sharey=True,
                             gridspec_kw={"width_ratios": widths})
    for ax, dom in zip(np.atleast_1d(axes), domains):
        d = f[f.domain == dom]
        sems = sorted(dict.fromkeys(d.semester))  # "2024-I" < "2024-II" < "2025-I": chronological
        for i, sem in enumerate(sems):
            g = d[d.semester == sem].sort_values("theta")
            x = i + np.linspace(-0.35, 0.35, len(g))  # targets spread within the semester's slot
            for xi, t, p in zip(x, g.theta, g.theta_hat):
                ax.plot([xi, xi], [t, p], color="grey", lw=0.6, ls=":")
            ax.plot(x, g.theta, "o", color=TRUE, ms=3)
            ax.plot(x, g.theta_hat, "o", color=PRED, ms=3)
        ax.set_xticks(range(len(sems)), sems, rotation=90 if len(sems) > 3 else 0)
        ax.text(0, 1.02, dom, transform=ax.transAxes, color="grey", ha="left")
        offset_spines(ax)
    np.atleast_1d(axes)[0].set_ylabel("θ")
    fig.legend(handles=[plt.Line2D([], [], color=TRUE, marker="o", ms=3, ls="", label="true θ"),
                        plt.Line2D([], [], color=PRED, marker="o", ms=3, ls="", label=f"forecast ({LABELS.get(protocol, protocol)})")],
               loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.15))
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def bar_figure(metrics: pd.DataFrame, metric: str, variants: dict[str, str], path, seed: int = 0) -> None:
    """Four bars, one per condition in CONDITIONS, at the mean over (domain, semester) of the
    metric ("mse", "delta" or "z"); the individual (domain, semester) values as jittered
    circles; the intercept as a dashed grey line (its mean raw MSE, or 0 for delta and z)."""
    assert metric in METRICS, metric
    m = chosen(metrics, variants)
    m = m[(m.semester != "all") & (m.domain != "all")]
    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(figsize=(2.6, 2.0))
    for i, p in enumerate(CONDITIONS):
        values = m[m.protocol == p][metric].dropna().to_numpy()
        if len(values) == 0:
            continue
        ax.bar(i, values.mean(), color=COLORS[p], width=0.7)
        ax.plot(i + rng.uniform(-0.2, 0.2, len(values)), values, "o", color="black", ms=2, alpha=0.6, mew=0)
    baseline = m[m.protocol == BASELINE].mse.mean() if metric == "mse" else 0.0
    ax.axhline(baseline, color="grey", lw=0.8, ls="--")
    ax.set_xticks(range(len(CONDITIONS)), [LABELS[p] for p in CONDITIONS])
    ax.tick_params(axis="x", rotation=90)
    ax.set_ylabel(METRIC_LABELS[metric])
    offset_spines(ax)
    fig.legend(handles=[plt.Line2D([], [], color="grey", lw=0.8, ls="--", label="intercept"),
                        plt.Line2D([], [], color="black", marker="o", ms=2, ls="", label="one domain × semester")],
               loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def semester_table(metrics: pd.DataFrame, metric: str, variants: dict[str, str], domain: str | None = None) -> pd.DataFrame:
    """Semester x condition (intercept first, then CONDITIONS) for one domain, or pooled when
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
