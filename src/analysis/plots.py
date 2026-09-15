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
    np.atleast_1d(axes)[0].set_ylabel("θ")
    fig.legend(handles=[plt.Line2D([], [], color=TRUE, marker="o", ms=3, ls="", label="true θ"),
                        plt.Line2D([], [], color=PRED, marker="o", ms=3, ls="", label=f"forecast ({LABELS.get(protocol, protocol)})")],
               loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.15))
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def outlier_split(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(inliers, outliers) of a 1-D array: an outlier lies beyond 3 interquartile ranges from
    the quartiles (Tukey's far-out fence). With fewer than 4 values nothing is an outlier."""
    values = np.asarray(values, float)
    if len(values) < 4:
        return values, values[:0]
    q1, q3 = np.percentile(values, [25, 75])
    fence = 3 * (q3 - q1)
    far = (values > q3 + fence) | (values < q1 - fence)
    return values[~far], values[far]


def zigzag(ax, y: float, size: float = 0.02) -> None:
    """A small zigzag across the left spine at axes-fraction height `y`, marking an axis break."""
    x = np.array([-1, -0.5, 0.5, -0.5, 0.5, 1]) * size
    yy = y + np.linspace(-1.5, 1.5, len(x)) * size
    ax.plot(x, yy, color="black", lw=0.8, clip_on=False, transform=ax.transAxes, solid_capstyle="round")


def bar_figure(metrics: pd.DataFrame, metric: str, variants: dict[str, str], path, seed: int = 0) -> None:
    """Four bars, one per condition in CONDITIONS, at the mean over (domain, semester) of the
    metric ("mse", "delta" or "z"); the individual (domain, semester) values as jittered
    circles; the intercept as a dashed grey line (its mean raw MSE, or 0 for delta and z).
    Values beyond Tukey's far-out fence (outlier_split) go to a short upper panel above a
    zigzag axis break, so one blown-up fit cannot squash the others."""
    assert metric in METRICS, metric
    m = chosen(metrics, variants)
    m = m[(m.semester != "all") & (m.domain != "all")]
    rng = np.random.default_rng(seed)
    points = {p: m[m.protocol == p][metric].dropna().to_numpy() for p in CONDITIONS}
    every = np.concatenate([v for v in points.values() if len(v)]) if any(len(v) for v in points.values()) else np.array([])
    baseline = m[m.protocol == BASELINE].mse.mean() if metric == "mse" else 0.0
    inliers, outliers = outlier_split(every)
    if len(outliers):
        fig, (top, ax) = plt.subplots(2, 1, figsize=(2.6, 2.6), sharex=True, layout="constrained",
                                      gridspec_kw={"height_ratios": [1, 3]})
        panels = [top, ax]
    else:
        fig, ax = plt.subplots(figsize=(2.6, 2.2), layout="constrained")
        top, panels = None, [ax]
    for panel in panels:
        for i, p in enumerate(CONDITIONS):
            values = points[p]
            if len(values) == 0:
                continue
            panel.bar(i, values.mean(), color=COLORS[p], width=0.7)
            panel.plot(i + rng.uniform(-0.2, 0.2, len(values)), values, "o", color="black", ms=2, alpha=0.6, mew=0)
        panel.axhline(baseline, color="grey", lw=0.8, ls="--")
    if top is not None:
        pad = 0.1 * max(np.ptp(inliers), 1e-9)  # np.ptp: the ndarray method is gone in NumPy 2.x
        ax.set_ylim(min(inliers.min(), baseline) - pad, max(inliers.max(), baseline) + pad)
        span = max(np.ptp(outliers), pad)
        top.set_ylim(outliers.min() - 0.5 * span, outliers.max() + 0.5 * span)
        top.spines["bottom"].set_visible(False)
        top.tick_params(axis="x", bottom=False)
        zigzag(ax, 1.0)
        zigzag(top, 0.0)
    ax.set_xticks(range(len(CONDITIONS)), [LABELS[p] for p in CONDITIONS])
    ax.tick_params(axis="x", rotation=90)
    (top or ax).set_ylabel(METRIC_LABELS[metric]) if top is None else fig.supylabel(METRIC_LABELS[metric], fontsize=8)
    # constrained layout reserves room for an "outside" legend, in both the one- and two-panel branches
    fig.legend(handles=[plt.Line2D([], [], color="grey", lw=0.8, ls="--", label="intercept"),
                        plt.Line2D([], [], color="black", marker="o", ms=2, ls="", label="one domain × semester")],
               loc="outside lower center", ncol=2, frameon=False)
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
