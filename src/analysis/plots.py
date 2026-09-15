"""Figures and tables of the results section.

forecast_figure   one panel per domain, semesters along x; within a semester the targets
                  sorted by theta*, black = true theta, orange = forecast, dotted connectors
bar_figure        six bars (the conditions) at the mean of a per-(domain, semester) metric,
                  jittered circles for the individual (domain, semester) values, easy-item
                  prediction as a dashed line
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
CONDITIONS = ["glm_regex", "embedding", "direct", "direct_ensemble", "pairwise", "pairwise_ensemble"]  # the bars, in this order
LABELS = {BASELINE: "easy-item prediction", "glm_regex": "length+regex", "embedding": "embedding",
          "direct": "direct judge", "direct_ensemble": "direct judges, pooled",
          "pairwise": "pairwise judge", "pairwise_ensemble": "pairwise judges, pooled"}
COLORS = dict(zip(CONDITIONS, ["#1baf7a", "#4a3aa7", "#eb6834", "#f4a581", "#2a78d6", "#8fb8e8"]))
TRUE, PRED = "black", "#eb6834"
METRIC_LABELS = {"mse": "MSE (θ units)", "delta": "MSE − easy-item MSE", "z": "z of MSE − easy-item MSE (paired bootstrap)"}
DOMAIN_COLORS = {"science": "#2a78d6", "math": "#eb6834", "coding": "#1baf7a", "cyber": "#eda100"}
BAR = "#d9d7d0"  # neutral bar; colour is reserved for the domain of a point


def swarm_offsets(values: np.ndarray, width: float = 0.3, min_gap: float | None = None) -> np.ndarray:
    """Deterministic beeswarm x-offsets in [-width, width] for 1-D `values`: points are placed in
    value order, each at the offset nearest the centre that keeps it at least `min_gap` (in the
    values' units; default 1/25 of their range) from every already-placed point closer than
    that vertically. No random jitter, so a figure is reproducible."""
    values = np.asarray(values, float)
    n = len(values)
    if n == 0:
        return values
    gap = min_gap if min_gap is not None else max(np.ptp(values), 1e-9) / 25
    slots = np.linspace(-width, width, 2 * n + 1)
    slots = slots[np.argsort(np.abs(slots), kind="stable")]  # centre first, then outward
    offsets = np.zeros(n)
    placed: list[tuple[float, float]] = []  # (value, offset)
    for i in np.argsort(values, kind="stable"):
        for s in slots:
            if all(abs(values[i] - v) >= gap or abs(s - o) >= width / max(n, 1) * 2 for v, o in placed):
                offsets[i] = s
                break
        else:
            offsets[i] = slots[-1]
        placed.append((values[i], offsets[i]))
    return offsets


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


def bar_figure(metrics: pd.DataFrame, metric: str, variants: dict[str, str], path, semester: str | None = None) -> None:
    """Bars per condition in CONDITIONS (neutral fill) at the mean over (domain, semester) of the
    metric ("mse", "delta" or "z"), a black line at the mean with a capless 95% t-interval over
    those points (none below 3 points), the points as a deterministic beeswarm coloured by
    domain, the easy-item prediction as a dashed grey line (its mean raw MSE, or 0 for delta
    and z). `semester` restricts to one semester (points are then the domains). Values beyond
    Tukey's far-out fence go to a short upper panel above a zigzag axis break."""
    assert metric in METRICS, metric
    m = chosen(metrics, variants)
    m = m[(m.semester != "all") & (m.domain != "all")]
    if semester:
        m = m[m.semester == semester]
    groups = {p: m[m.protocol == p].dropna(subset=[metric]) for p in CONDITIONS}
    every = np.concatenate([g[metric].to_numpy() for g in groups.values() if len(g)]) if any(len(g) for g in groups.values()) else np.array([])
    baseline = m[m.protocol == BASELINE].mse.mean() if metric == "mse" else 0.0
    inliers, outliers = outlier_split(every)
    if len(outliers):
        fig, (top, ax) = plt.subplots(2, 1, figsize=(3.4, 2.8), sharex=True, layout="constrained",
                                      gridspec_kw={"height_ratios": [1, 3]})
    else:
        fig, ax = plt.subplots(figsize=(3.4, 2.4), layout="constrained")
        top = None
    for i, p in enumerate(CONDITIONS):  # the lower panel draws everything and clips
        g = groups[p]
        if g.empty:
            continue
        values = g[metric].to_numpy()
        ax.bar(i, values.mean(), color=BAR, width=0.72, zorder=1)
        ax.hlines(values.mean(), i - 0.36, i + 0.36, color="black", lw=1.0, zorder=4)
        if np.isfinite(half := ci95(values)):
            ax.errorbar(i, values.mean(), yerr=half, color="black", lw=0.8, capsize=0, zorder=4)
        ax.scatter(i + swarm_offsets(values), values, s=14, c=[DOMAIN_COLORS.get(d, "black") for d in g.domain],
                   edgecolor="white", linewidth=0.5, zorder=3)
    ax.axhline(baseline, color="grey", lw=0.8, ls="--", zorder=0)
    if top is not None:
        pad = 0.1 * max(np.ptp(inliers), 1e-9)
        ax.set_ylim(min(inliers.min(), baseline) - pad, max(inliers.max(), baseline) + pad)
        span = max(np.ptp(outliers), pad)
        top.set_ylim(outliers.min() - 0.5 * span, outliers.max() + 0.5 * span)
        lo, hi = top.get_ylim()
        for i, p in enumerate(CONDITIONS):  # the upper panel draws only the outlier points and, per condition, a bar/mean line only if its mean is in view
            g = groups[p]
            if g.empty:
                continue
            values = g[metric].to_numpy()
            far = np.isin(values, outliers)
            if far.any():
                far_values = values[far]
                top.scatter(i + swarm_offsets(far_values), far_values, s=14,
                            c=[DOMAIN_COLORS.get(d, "black") for d in g.domain.to_numpy()[far]],
                            edgecolor="white", linewidth=0.5, zorder=3)
            if lo <= values.mean() <= hi:
                top.bar(i, values.mean(), color=BAR, width=0.72, zorder=1)
                top.hlines(values.mean(), i - 0.36, i + 0.36, color="black", lw=1.0, zorder=4)
                if np.isfinite(half := ci95(values)):
                    top.errorbar(i, values.mean(), yerr=half, color="black", lw=0.8, capsize=0, zorder=4)
        top.axhline(baseline, color="grey", lw=0.8, ls="--", zorder=0)
        top.spines["bottom"].set_visible(False)
        top.tick_params(axis="x", bottom=False)
        zigzag(ax, 1.0)
        zigzag(top, 0.0)
    ax.set_xticks(range(len(CONDITIONS)), [LABELS[p] for p in CONDITIONS])
    ax.tick_params(axis="x", rotation=90)
    if top is None:
        ax.set_ylabel(METRIC_LABELS[metric])
    else:
        fig.supylabel(METRIC_LABELS[metric], fontsize=8)
    domains = [d for d in DOMAIN_COLORS if d in set(m.domain)]
    handles = [plt.Line2D([], [], marker="o", ms=4, ls="", color=DOMAIN_COLORS[d], markeredgecolor="white", label=d) for d in domains]
    handles += [plt.Line2D([], [], color="black", lw=1, label="mean ± 95% CI over tasks"),
                plt.Line2D([], [], color="grey", lw=0.8, ls="--", label=LABELS[BASELINE])]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False, handletextpad=0.4, columnspacing=1.0)
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
