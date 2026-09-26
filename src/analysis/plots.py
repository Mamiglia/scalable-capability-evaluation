"""
Shared plotting code: the figure style, protocol and domain names and colours, the axis helpers other figure modules
import (unclip_markers, ci95, stars), and the prediction figures and tables of the exploratory runner
scripts/analyses/results.py (forecast_figure, also drawn by scripts/analysis/capability_signal.py; bar_figure,
semester_table, task_figure, timeline_figure, frontier_figure).
Judge protocols are drawn as the best judge on references, in the one variant per protocol the caller chooses.
matplotlib.use("Agg") below is deliberate: figures go to files, nothing renders inline in an IDE cell.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from analysis.stats import BASELINE, BEST, METRICS, one_per_target

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial"], "font.size": 8, "axes.labelsize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.8,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "svg.fonttype": "none", "pdf.fonttype": 42, "savefig.dpi": 300,
})
CONDITIONS = ["glm_length", "glm_regex", "embedding", "direct", "pairwise"]  # the bars, in this order
# Display names of the protocols (CLAUDE.md, figure colour palette); Length alone is a diagnostic.
LABELS = {BASELINE: "Baseline", "glm_length": "Length", "glm_regex": "Deterministic features",
          "embedding": "Embedding", "direct": "Absolute scoring", "pairwise": "Pairwise judge"}
DOMAIN_LABELS = {"science": "Science", "math": "Maths", "coding": "Coding", "cyber": "Cyber"}
# the project palette (CLAUDE.md, scripts/analysis/config.yaml display.protocol_colours)
COLORS = {"glm_length": "#f1a48a", "glm_regex": "#eb6834", "embedding": "#4a3aa7", "direct": "#1baf7a",
          "pairwise": "#2a78d6"}
# the pooled-judge conditions are not bars, but task figures may draw them
LABELS.update({"direct_ensemble": "Direct scoring, pooled judges", "pairwise_ensemble": "Pairwise judges, pooled"})
# the palette's tenuous colour, and a shade of the Pairwise judge's
COLORS.update({"direct_ensemble": "#f8cbbc", "pairwise_ensemble": "#6aa1e2"})
# domains in colours no protocol uses (config.yaml display.domain_colours)
DOMAIN_COLORS = {"science": "#cc79a7", "math": "#d55e00", "coding": "#6a3d9a", "cyber": "#c9a800"}
STAGGER = 0.05  # the four domains' circles sit within ±0.05 of the bar centre
TRUE, PRED = "black", "#eb6834"  # PRED: a prediction of a protocol without a palette colour
METRIC_LABELS = {"mse": "Squared error", "delta": "MSE − Baseline MSE", "z": "z of MSE − Baseline MSE",
                 "pct": "Relative MSE (%)"}


def with_pct(metrics: pd.DataFrame) -> pd.DataFrame:
    """
    Adds the MSE as a percentage of the Baseline's MSE on the same targets (100 = the Baseline, below 100 = better).

    Inputs:
        metrics: per-task metrics (stats.task_metrics output); uses mse and baseline_mse.
    Outputs:
        a copy of `metrics` with a `pct` column, 100 * mse / baseline_mse per row.
    """
    return metrics.assign(pct=100 * metrics.mse / metrics.baseline_mse)


def compress(values: np.ndarray, extent: np.ndarray | None = None) -> tuple[np.ndarray, dict[float, float]]:
    """
    Moves far-high values into a band above the in-range values, keeping their order, so that an outlier does not
    squash the rest of the axis. With at least 10 values, a value more than 3 interquartile ranges above the upper
    quartile is off-axis; every other value keeps its position. The distinct off-axis values are spread in rank order
    from 52.5% to 70% of the in-range span above the in-range maximum (a single one at 61%).

    Inputs:
        values: every y value that will be drawn (bar means, domain values, the Baseline level).
        extent: further y values that must stay in range, e.g. the tops of the in-range error bars; they widen the
            in-range span but are not placed themselves. None for no such values.
    Outputs:
        (drawn position of each value, dict from each off-axis value to its drawn position); the dict is empty and the
        positions equal the values when nothing is off-axis.
    """
    values = np.asarray(values, float)
    q1, q3 = np.percentile(values, [25, 75])
    fence = 3 * (q3 - q1)
    # only values far ABOVE the rest are compressed (low values stay where they are, so a negative
    # delta is never drawn above the axis); with fewer than 10 values the quartiles are meaningless
    off = (values > q3 + fence) if len(values) >= 10 else np.zeros(len(values), bool)
    if not off.any():
        return values.copy(), {}
    inside = values[~off]
    if extent is not None and len(extent):
        inside = np.concatenate([inside, np.asarray(extent, float)[np.isfinite(extent)]])  # error bars stay in range
    span = max(np.ptp(inside), 1e-9)
    top = inside.max() + 0.35 * span
    distinct = np.unique(values[off])
    band = np.linspace(top + 0.175 * span, top + 0.35 * span, len(distinct)) if len(distinct) > 1 else np.array([top + 0.26 * span])
    mapping = dict(zip(distinct.tolist(), band.tolist()))
    positions = values.copy()
    positions[off] = [mapping[v] for v in values[off]]
    return positions, mapping


def break_strip(ax, raw: np.ndarray, mapping: dict[float, float], extent: np.ndarray | None = None) -> None:
    """
    Draws a white strip across the panel and its left spine in the gap between the in-range values and the compressed
    band, so the axis break is visible; the break glyph itself is left to be added by hand.

    Inputs:
        ax: the axes to draw on.
        raw: the true values passed to compress.
        mapping: compress's dict from off-axis value to drawn position; nothing is drawn when it is empty.
        extent: the in-range error-bar tops passed to compress, or None.
    Outputs:
        none; draws on ax.
    """
    if not mapping:
        return
    lo, hi = break_edges(raw, mapping, extent)
    ax.axhspan(lo, hi, xmin=-0.04, xmax=1.0, facecolor="white", edgecolor="none", zorder=9, clip_on=False)


def break_edges(raw: np.ndarray, mapping: dict[float, float], extent: np.ndarray | None = None) -> tuple[float, float]:
    """
    Computes where the white break strip goes, in drawn units: just above the highest in-range value (or error-bar top)
    and just below the compressed band. Callers cap error bars at the bottom edge so nothing pokes through the strip.

    Inputs:
        raw: the true values passed to compress.
        mapping: compress's dict from off-axis value to drawn position.
        extent: the in-range error-bar tops, or None.
    Outputs:
        (bottom, top) of the strip: 5% of the gap above the highest in-range value and 25% of the gap below the band's
        lowest position; (inf, inf) when there is no band.
    """
    if not mapping:
        return float("inf"), float("inf")
    lo = max([v for v in raw if v not in mapping] + ([float(np.nanmax(extent))] if extent is not None and len(extent) else []))
    hi = min(mapping.values())
    return lo + 0.05 * (hi - lo), hi - 0.25 * (hi - lo)


def unclip_markers(fig):
    """
    Makes every marker draw in full: scatter points and lines with markers are no longer clipped at the axes, so a
    point on an axis limit is not cut in half. Lines without markers, bands and boxes stay clipped.

    Inputs:
        fig: a matplotlib figure.
    Outputs:
        the same figure, modified in place.
    """
    from matplotlib.collections import PathCollection
    for ax in fig.axes:
        for c in ax.collections:
            if isinstance(c, PathCollection):
                c.set_clip_on(False)
        for line in ax.lines:
            if line.get_marker() not in (None, "None", "", " "):
                line.set_clip_on(False)
    return fig


def ci95(values: np.ndarray) -> float:
    """
    Computes the half-width of the t-based 95% confidence interval of a sample's mean.

    Inputs:
        values: the sample, a 1-D array.
    Outputs:
        float, t(0.975, n - 1) * sd / sqrt(n); NaN with fewer than three values.
    """
    n = len(values)
    if n < 3:
        return float("nan")
    return float(scipy_stats.t.ppf(0.975, n - 1) * np.std(values, ddof=1) / np.sqrt(n))


AXES_HEIGHT = 1.6  # inches; bar_figure and task_figure share it so the panels sit side by side


def fix_axes_height(fig, ax, inches: float = AXES_HEIGHT, rounds: int = 3) -> None:
    """
    Resizes the figure so that the axes' drawn height is `inches`, whatever the legend and tick labels take up; the
    constrained layout re-runs on each draw, so a few rounds converge.

    Inputs:
        fig: the figure.
        ax: the axes whose height is fixed.
        inches: the axes height to reach (default AXES_HEIGHT, shared so that figures placed side by side line up).
        rounds: how many draw-and-resize rounds to run.
    Outputs:
        none; changes the figure's height.
    """
    for _ in range(rounds):
        fig.canvas.draw()
        current = ax.get_position().height * fig.get_figheight()
        fig.set_figheight(fig.get_figheight() + inches - current)


def chosen(metrics: pd.DataFrame, variants: dict[str, str]) -> pd.DataFrame:
    """
    Keeps the rows of each protocol's chosen variant.

    Inputs:
        metrics: any table with protocol and variant columns (e.g. stats.task_metrics output).
        variants: protocol -> the variant to keep; a protocol not named keeps its "-" rows.
    Outputs:
        the matching rows of `metrics`, every column kept.
    """
    return metrics[[variants.get(p, "-") == v for p, v in zip(metrics.protocol, metrics.variant)]]


def forecast_figure(forecasts: pd.DataFrame, protocol: str, variant: str, path, semesters: list[str] | None = None) -> None:
    """
    Draws every target's true theta* and one protocol's prediction of it: one panel per domain (its width set by its
    number of semesters), semesters along x, the targets within a semester sorted by theta*, a grey connector from
    truth to prediction, and the Baseline as a dashed level per task. Judge protocols are drawn as the best judge on
    references (stats.one_per_target).

    Inputs:
        forecasts: the forecasts table (calibration.forecasts rows, every column); the drawing uses domain, semester,
            protocol, variant, theta and theta_hat.
        protocol: the protocol to draw, e.g. "pairwise".
        variant: its variant, e.g. "linear" ("-" for a protocol without variants).
        path: the image file to write.
        semesters: draw only these semesters (e.g. the latest); None draws all.
    Outputs:
        the figure, written to path: open black circles for the truth, orange filled circles for the prediction.
    """
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
            if (dom, sem) in baseline.index:  # the Baseline: one level per task
                ax.hlines(baseline[(dom, sem)], i - 0.45, i + 0.45, color="grey", lw=0.8, ls="--", zorder=0)
            ax.scatter(x, g.theta, s=16, facecolor="white", edgecolor=TRUE, linewidth=0.8, zorder=3)
            ax.scatter(x, g.theta_hat, s=16, facecolor=COLORS.get(protocol, PRED), edgecolor="none", zorder=4)
        ax.set_xticks(range(len(sems)), sems, rotation=90 if len(sems) > 3 else 0)
        ax.text(0, 1.02, DOMAIN_LABELS.get(dom, dom), transform=ax.transAxes, color="grey", ha="left")
    np.atleast_1d(axes)[0].set_ylabel("Model Capability (θ)")
    handles = [plt.Line2D([], [], marker="o", ms=4, ls="", markerfacecolor="white", markeredgecolor=TRUE, label="Target models (true θ)"),
               plt.Line2D([], [], marker="o", ms=4, ls="", color=COLORS.get(protocol, PRED),
                          label=f"{LABELS.get(protocol, protocol)} prediction"),
               plt.Line2D([], [], color="grey", lw=0.8, ls="--", label=LABELS[BASELINE])]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def bar_figure(metrics: pd.DataFrame, metric: str, variants: dict[str, str], path, semester: str | None = None,
               llr: pd.DataFrame | None = None) -> None:
    """
    Draws one per-task metric as bars: one bar per protocol in CONDITIONS at the mean over the (domain, semester)
    tasks, with a capless 95% t-interval over those tasks (none with fewer than three); one filled circle per domain at
    the mean over its semesters, the domains staggered across the bar and each domain's circles joined across the
    protocols by a line in its colour; and the Baseline as a dashed grey line (its mean MSE over the tasks for "mse",
    100 for "pct", 0 for "delta" and "z"). Far-high values are drawn in a compressed band above a break strip, with
    ticks at their true values (see compress).

    Inputs:
        metrics: per-task metrics (stats.task_metrics output); uses domain, semester, protocol, variant, mse,
            baseline_mse and the metric's column; rows with domain or semester "all" are ignored.
        metric: "mse", "delta", "z" or "pct" (the MSE as a percentage of the Baseline's).
        variants: protocol -> the variant to draw (see chosen).
        path: the image file to write.
        semester: draw only this semester's tasks; None uses every semester.
        llr: log likelihood ratio rows (analysis.likelihood_ratio, one shape and judge per target) for the stars; None
            draws no stars.
    Outputs:
        the figure, written to path. A star above a bar marks the one-sided log likelihood ratio test against the
        references' distribution with each task as the unit (*, **, *** for p below .05, .01, .001); a protocol
        without log likelihood ratios gets none.
    """
    assert metric in (*METRICS, "pct"), metric
    m = chosen(with_pct(metrics), variants)
    m = m[m.domain != "all"]
    tasks = m[m.semester != "all"]
    if semester:
        tasks = tasks[tasks.semester == semester]
    # one value per domain: the mean over its semesters of the per-task metric, the same
    # aggregation as the bar (a pooled z would grow with the number of targets and not match)
    per_domain = tasks.groupby(["domain", "protocol"], as_index=False)[metric].mean()
    domains = [d for d in DOMAIN_COLORS if d in set(per_domain.domain)]
    # collect everything to draw first, so the off-axis compression sees every value
    means, halves = {}, {}
    for p in CONDITIONS:
        values = tasks[tasks.protocol == p][metric].dropna().to_numpy()
        if len(values):
            means[p], halves[p] = float(values.mean()), ci95(values)
    points = {d: per_domain[per_domain.domain == d].set_index("protocol")[metric].reindex(CONDITIONS) for d in domains}
    baseline = float(tasks[tasks.protocol == BASELINE].mse.mean()) if metric == "mse" else 100.0 if metric == "pct" else 0.0
    raw = np.array(list(means.values()) + [v for s in points.values() for v in s.dropna()] + [baseline])
    _, first = compress(raw)  # which values are off-axis, before the error bars widen the in-range span
    extent = np.array([means[p] + halves[p] for p in means if means[p] not in first and np.isfinite(halves[p])])
    positions, mapping = compress(raw, extent)
    place = dict(zip(raw.tolist(), positions.tolist()))
    fig, ax = plt.subplots(figsize=(2.7, 2.4), layout="constrained")
    x = {p: i for i, p in enumerate(CONDITIONS)}
    for p, mean in means.items():
        ax.bar(x[p], place[mean], color=COLORS[p], width=0.72, zorder=1)
        if np.isfinite(halves[p]) and mean not in mapping:  # an interval around a compressed mean is meaningless
            cap = break_edges(raw, mapping, extent)[0]
            ax.plot([x[p], x[p]], [mean - halves[p], min(mean + halves[p], cap)], color="black", lw=0.8, zorder=4)  # capped at the break strip
    offsets = np.linspace(-STAGGER, STAGGER, len(domains)) if len(domains) > 1 else np.zeros(1)
    for d, dx in zip(domains, offsets):
        s = points[d]
        keep = [p for p in CONDITIONS if np.isfinite(s[p])]
        if not keep:
            continue
        xs = [x[p] + dx for p in keep]
        ys = [place[float(s[p])] for p in keep]
        ax.plot(xs, ys, color=DOMAIN_COLORS[d], lw=0.8, zorder=3)
        ax.scatter(xs, ys, s=18, facecolor=DOMAIN_COLORS[d], edgecolor="black", linewidth=0.4, zorder=5)
    ax.axhline(place[baseline], color="grey", lw=0.8, ls="--", zorder=0)
    # significance: the log likelihood ratio against the references' distribution, one unit per task
    from analysis import likelihood_ratio as lr
    shown = llr.merge(tasks[["domain", "semester"]].drop_duplicates()) if llr is not None else pd.DataFrame(columns=["protocol"])
    for p, mean in means.items():
        rows = shown[shown.protocol == p]
        if rows.empty:
            continue
        pval = float(lr.unit_test(rows, unit=("domain", "semester"))["p"])
        stars = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
        if not stars:
            continue
        top = max([place[mean] + (halves[p] if np.isfinite(halves[p]) and mean not in mapping else 0)]
                  + [place[float(v)] for d in domains for v in [points[d][p]] if np.isfinite(v)])
        ax.text(x[p], top, stars, ha="center", va="bottom", fontsize=7, zorder=6)
    if mapping:  # true-value ticks for the compressed band, after the in-range ticks
        inside = [t for t in ax.get_yticks() if t <= max(v for v in raw if v not in mapping)]
        ax.set_yticks(inside + list(mapping.values()), [f"{t:g}" for t in inside] + [f"{v:.0f}" for v in mapping])
        break_strip(ax, raw, mapping, extent)
    ax.set_xticks(range(len(CONDITIONS)), [LABELS[p] for p in CONDITIONS])
    ax.tick_params(axis="x", rotation=90)
    ax.set_ylabel(METRIC_LABELS[metric])
    handles = [plt.Line2D([], [], color=DOMAIN_COLORS[d], lw=0.8, marker="o", ms=4, markeredgecolor="black", markeredgewidth=0.4, label=DOMAIN_LABELS.get(d, d)) for d in domains]
    ax.text(len(CONDITIONS) - 0.5, place[baseline], LABELS[BASELINE], color="grey", fontsize=7,
            ha="right", va="bottom", zorder=6)  # the Baseline named at its line
    fig.legend(handles=handles, loc="outside lower center", ncol=2, frameon=False, handlelength=2.0)
    fix_axes_height(fig, ax)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def semester_table(metrics: pd.DataFrame, metric: str, variants: dict[str, str], domain: str | None = None) -> pd.DataFrame:
    """
    Tabulates one metric by semester and protocol, for one domain or pooled over the domains; pooled, a semester's
    value is the mean over the domains that have that semester.

    Inputs:
        metrics: per-task metrics (stats.task_metrics output); uses domain, semester, protocol, variant and the metric.
        metric: "mse", "delta" or "z".
        variants: protocol -> the variant to show (see chosen).
        domain: a domain's code key (e.g. "math"); None pools over the domains.
    Outputs:
        DataFrame with one row per semester plus a row "all" (for a domain, its pooled row from stats.task_metrics;
        pooled, the mean over every (domain, semester) cell), and one column per protocol, the Baseline first and then
        CONDITIONS, named by LABELS; NaN where a protocol has no value.
    """
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


def task_figure(forecasts: pd.DataFrame, domain: str, semester: str, conditions: list[tuple[str, str]], path,
                reasoning: set[str] | None = None) -> None:
    """
    Draws one prediction task: the references' theta* as grey filled circles in the left slot and, in the right slot,
    the targets' true theta* (open black circles, sorted by theta*) with each protocol's prediction as a filled circle
    in the protocol's colour (COLORS) joined to the truth by a thin connector, and the Baseline (the references' mean)
    as a dashed level across the target slot. Judge protocols are drawn as the best judge on references.

    Inputs:
        forecasts: the forecasts table (calibration.forecasts rows, every column).
        domain: the domain's code key; the references' theta* are read from its frozen tables.
        semester: the task's semester; the left slot is labelled with the semester before it.
        conditions: (protocol, variant) pairs to draw, e.g. [("pairwise", "linear"), ("glm_regex", "-")].
        path: the image file to write.
        reasoning: names of target models whose transcripts show chain of thought; their truth is drawn as an open
            diamond and named in the legend. None for no such models.
    Outputs:
        the figure, written to path.
    """
    from analysis import loading  # local import: plots stays importable without configs
    from reference import tables
    models, _ = tables.load(loading.domain(domain))
    refs = models.loc[models[f"{semester}_role"] == "reference", f"{semester}_theta"].sort_values()
    f = one_per_target(forecasts)
    f = f[(f.domain == domain) & (f.semester == semester)]
    names = list(tables.load_semesters())
    ref_label = names[names.index(semester) - 1] if semester in names and names.index(semester) > 0 else "references"
    fig, ax = plt.subplots(figsize=(2.0, 2.2), layout="constrained")
    ax.scatter(np.linspace(-0.3, 0.3, len(refs)), refs, s=18, facecolor="#9a9891", edgecolor="none", zorder=3)
    ax.hlines(refs.mean(), 0.6, 1.4, color="grey", lw=0.8, ls="--", zorder=0)
    truth = f.drop_duplicates("model").set_index("model").theta.sort_values()
    x = pd.Series(1 + np.linspace(-0.3, 0.3, len(truth)), index=truth.index)
    handles = []
    for protocol, variant in conditions:
        g = f[(f.protocol == protocol) & (f.variant == variant)].set_index("model").theta_hat.reindex(truth.index)
        colour = COLORS.get(protocol, PRED)
        ax.vlines(x, np.minimum(truth, g), np.maximum(truth, g), color=colour, lw=0.5, alpha=0.6, zorder=1)
        ax.scatter(x, g, s=18, facecolor=colour, edgecolor="none", zorder=5)
        handles.append(plt.Line2D([], [], marker="o", ms=4, ls="", color=colour, label=f"{LABELS.get(protocol, protocol)} prediction"))
    is_reasoning = np.array([m in (reasoning or set()) for m in truth.index])
    ax.scatter(x[~is_reasoning], truth[~is_reasoning], s=18, facecolor="white", edgecolor=TRUE, linewidth=0.8, zorder=6)
    if is_reasoning.any():
        ax.scatter(x[is_reasoning], truth[is_reasoning], s=26, marker="D", facecolor="white", edgecolor=TRUE, linewidth=0.8, zorder=6)
    ax.set_xticks([0, 1], [f"{ref_label}\n(References)", f"{semester}\n(Targets)"])
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylabel("Model Capability (θ)")
    ax.text(0, 1.02, DOMAIN_LABELS.get(domain, domain), transform=ax.transAxes, color="grey", ha="left")
    truth_handles = [plt.Line2D([], [], marker="o", ms=4, ls="", markerfacecolor="white", markeredgecolor=TRUE,
                                label="Target models" + (", no reasoning" if is_reasoning.any() else ""))]
    if is_reasoning.any():
        truth_handles.append(plt.Line2D([], [], marker="D", ms=4, ls="", markerfacecolor="white", markeredgecolor=TRUE, label="Target models, reasoning"))
    handles = [plt.Line2D([], [], marker="o", ms=4, ls="", color="#9a9891", label="Reference models"),
               *truth_handles, *handles, plt.Line2D([], [], color="grey", lw=0.8, ls="--", label=LABELS[BASELINE])]
    fig.legend(handles=handles, loc="outside lower center", ncol=2, frameon=False)
    fix_axes_height(fig, ax)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


TIMELINE = ["glm_regex", "embedding", "direct", "pairwise"]  # the four protocols these figures draw, in this order


def timeline_figure(metrics: pd.DataFrame, domain: str, variants: dict[str, str], path, scores: pd.DataFrame | None = None,
                    metric: str = "mse", forecasts: pd.DataFrame | None = None, band: bool = False, shade: bool = False,
                    cut: float | None = None, llr: pd.DataFrame | None = None) -> None:
    """
    Draws one domain's tasks over time: x = semester, y = the task's metric, one line with filled circles per protocol
    in TIMELINE (colours from COLORS), slightly offset in x so the points do not overlap, and the Baseline as a dashed
    grey line drawn on top. Judge protocols are the best judge on references. By default, far-high values (more than 3
    interquartile ranges above the upper quartile) run off the top of the axis, which spans the other values and their
    error bars, and are printed.

    Inputs:
        metrics: per-task metrics (stats.task_metrics output); uses domain, semester, protocol, variant, mse,
            baseline_mse and the metric's column.
        domain: the domain's code key.
        variants: protocol -> the variant to draw (see chosen).
        path: the image file to write.
        scores: stats.scores output; when given, each point gets a capless error bar, the task's percentile-bootstrap
            95% interval of the MSE over its targets (lo, hi), shifted by the Baseline's MSE for "delta" and divided
            by it for "pct". None draws no error bars.
        metric: "mse"; "delta", the MSE minus the Baseline's MSE (Baseline at 0); or "pct", the MSE as a percentage of
            the Baseline's (Baseline at 100).
        forecasts: the forecasts table; given together with `scores` and metric "delta" or "pct", the error bars are
            instead the 95% t-interval of the paired difference of squared errors (protocol minus Baseline) over the
            task's targets, in the metric's units.
        band: draw the far-high values in a compressed band above a break strip (as bar_figure does) instead of
            letting them run off the axis.
        shade: draw each protocol's intervals as a translucent band along its line instead of vertical bars (not
            across compressed values).
        cut: fix the top of the axis at this value; points above it are drawn as upward triangles on the cut line,
            the top tick reads ">cut", lines and bands are clipped there, and nothing is compressed.
        llr: log likelihood ratio rows (analysis.likelihood_ratio, one shape and judge per target); when given, a
            star above a point gives the one-sided log likelihood ratio test of the task's targets against the
            references' distribution (*, **, *** for p below .05, .01, .001).
    Outputs:
        the figure, written to path; values cut off the top of the axis are printed.
    """
    assert metric in ("mse", "delta", "pct"), metric
    m = chosen(with_pct(metrics), variants)
    m = m[(m.domain == domain) & (m.semester != "all")]
    semesters = sorted(m.semester.unique())
    x = {s: i for i, s in enumerate(semesters)}
    series = {p: m[m.protocol == p].set_index("semester")[metric].reindex(semesters) for p in [BASELINE, *TIMELINE]}
    base_mse = m[m.protocol == BASELINE].set_index("semester").mse.reindex(semesters)  # to convert error bars to %
    raw = np.array([v for s in series.values() for v in s.dropna()])
    sc = None
    if scores is not None:
        sc = scores[(scores.domain == domain) & (scores.semester != "all") & scores.judge.isin(["-", BEST])]
        sc = sc[[variants.get(pr, "-") == v for pr, v in zip(sc.protocol, sc.variant)]].set_index(["protocol", "semester"])
        base_of = sc.reset_index().semester.map(base_mse).to_numpy()
        if metric == "delta":  # the bootstrap interval is on the MSE; shift it by the task's Baseline MSE
            sc = sc.assign(lo=sc.lo.to_numpy() - base_of, hi=sc.hi.to_numpy() - base_of)
        elif metric == "pct":  # or as a percentage of the task's Baseline MSE
            sc = sc.assign(lo=100 * sc.lo.to_numpy() / base_of, hi=100 * sc.hi.to_numpy() / base_of)
    _, first = compress(raw)  # which values are off-axis, before the error bars widen the in-range span
    extent = np.array([sc.hi.get((p, sem), np.nan) for p in TIMELINE for sem, v in series[p].dropna().items()
                       if float(v) not in first]) if sc is not None else None
    positions, mapping = compress(raw, extent)
    place = dict(zip(raw.tolist(), positions.tolist()))
    fig, ax = plt.subplots(figsize=(2.4, 2.4), layout="constrained")
    base = series[BASELINE]
    # on top: a regression whose coefficients shrink to zero predicts the references' mean and sits exactly on it
    ax.plot([x[s] for s in semesters], [place[float(v)] for v in base], color="grey", lw=0.8, ls="--", zorder=4)
    offsets = dict(zip(TIMELINE, np.linspace(-0.09, 0.09, len(TIMELINE))))  # slight x offset per protocol
    task_p = None
    if llr is not None:
        from analysis import likelihood_ratio as lr
        task_p = lr.task_tests(llr[llr.domain == domain]).set_index(["protocol", "semester"]).p
    errors = None
    if forecasts is not None:  # per-target squared errors, paired with the Baseline's on the same targets
        f = one_per_target(forecasts)
        f = f[(f.domain == domain).to_numpy() & np.array([variants.get(pr, "-") == v for pr, v in zip(f.protocol, f.variant)])]
        f = f.assign(err=(f.theta_hat - f.theta) ** 2)
        base_err = f[f.protocol == BASELINE].set_index(["semester", "model"]).err
        errors = f.assign(base=f.set_index(["semester", "model"]).index.map(base_err))
    paired = metric in ("delta", "pct") and errors is not None and scores is not None  # scores=None: no intervals at all
    if paired:  # error bars = 95% t-interval of the paired difference over the task's targets, in the metric's units
        rows = {}
        for (pr, sem), e in errors.dropna(subset=["err", "base"]).groupby(["protocol", "semester"]):
            d = (e.err - e.base).to_numpy()
            if len(d) < 3:
                continue
            half = float(scipy_stats.t.ppf(0.975, len(d) - 1) * np.std(d, ddof=1) / np.sqrt(len(d)))
            centre = float(d.mean())
            b = float(base_mse[sem])
            lo, hi = (centre - half, centre + half) if metric == "delta" else (100 * (b + centre - half) / b, 100 * (b + centre + half) / b)
            rows[(pr, sem)] = (lo, hi)
        sc = pd.DataFrame(rows, index=["lo", "hi"]).T
        sc.index = pd.MultiIndex.from_tuples(sc.index, names=["protocol", "semester"])
        extent = np.array([sc.hi.get((p, sem), np.nan) for p in TIMELINE for sem, v in series[p].dropna().items() if float(v) not in first])
        positions, mapping = compress(raw, extent)
        place = dict(zip(raw.tolist(), positions.tolist()))
    if cut is not None:  # fixed cut: nothing is compressed, the axis just stops at `cut`
        mapping = {}
        place = dict(zip(raw.tolist(), raw.tolist()))
    if cut is None and not band and mapping:  # cut off: keep true positions, let the far-high values run off the top
        print(f"{path}: off the top of the axis: {[round(v, 2) for v in sorted(mapping)]}")
        top_in_range = max([v for v in raw if v not in mapping] + ([float(np.nanmax(extent))] if extent is not None and len(extent) else []))
        mapping = {}
        place = dict(zip(raw.tolist(), raw.tolist()))
        ax.set_ylim(top=top_in_range + 0.05 * abs(top_in_range))
    cap = break_edges(raw, mapping, extent)[0]
    for p in TIMELINE:
        s = series[p].dropna()
        xs = [x[i] + offsets[p] for i in s.index]
        ys = [place[float(v)] for v in s]
        if sc is not None and shade:  # one translucent band per protocol along its line
            los = [sc.lo.get((p, sem), np.nan) if float(v) not in mapping else np.nan for sem, v in s.items()]
            his = [min(sc.hi.get((p, sem), np.nan), cap) if float(v) not in mapping else np.nan for sem, v in s.items()]
            ax.fill_between(xs, los, his, color=COLORS[p], alpha=0.2, linewidth=0, zorder=1)
        elif sc is not None:
            for xi, sem, v in zip(xs, s.index, s):
                lo, hi = sc.lo.get((p, sem), np.nan), sc.hi.get((p, sem), np.nan)
                if float(v) not in mapping and np.isfinite(lo) and np.isfinite(hi):  # no interval around a compressed value
                    ax.plot([xi, xi], [lo, min(hi, cap)], color=COLORS[p], lw=0.7, zorder=2)  # capped at the break strip
        ax.plot(xs, ys, color=COLORS[p], lw=1.0, zorder=2)
        if cut is not None:
            above = np.array(ys) > cut
            ax.scatter(np.array(xs)[~above], np.array(ys)[~above], s=16, facecolor=COLORS[p], edgecolor="none", zorder=3)
            ax.scatter(np.array(xs)[above], np.full(above.sum(), cut), s=22, marker="^", facecolor=COLORS[p], edgecolor="none", zorder=6, clip_on=False)
        else:
            ax.scatter(xs, ys, s=16, facecolor=COLORS[p], edgecolor="none", zorder=3)
        if task_p is not None:
            for xi, sem, v, yi in zip(xs, s.index, s, ys):
                pval = float(task_p.get((p, sem), np.nan))
                if not np.isfinite(pval):
                    continue
                stars = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
                if stars:
                    top = yi if float(v) in mapping or sc is None else max(yi, min(sc.hi.get((p, sem), yi), cap))
                    if cut is not None:
                        top = min(top, cut * 0.93)
                    ax.text(xi, top, stars, ha="center", va="bottom", fontsize=6, color=COLORS[p], zorder=6)
    if metric in ("mse", "pct"):
        ax.set_ylim(bottom=0)  # MSE (and its ratio) cannot be negative
    if cut is not None:
        ax.set_ylim(top=cut)
        ticks = [t for t in ax.get_yticks() if ax.get_ylim()[0] <= t < cut] + [cut]
        unit = "%" if metric == "pct" else ""
        ax.set_yticks(ticks, [f"{t:g}" for t in ticks[:-1]] + [f">{cut:g}{unit}"])
    if mapping:
        inside = [t for t in ax.get_yticks() if t <= max(v for v in raw if v not in mapping) and (metric not in ("mse", "pct") or t >= 0)]
        ax.set_yticks(inside + list(mapping.values()), [f"{t:g}" for t in inside] + [f"{v:.0f}" for v in mapping])
        break_strip(ax, raw, mapping, extent)
    ax.set_xticks(range(len(semesters)), semesters)
    ax.tick_params(axis="x", rotation=90)
    ax.set_ylabel(METRIC_LABELS[metric])
    ax.text(0, 1.02, DOMAIN_LABELS.get(domain, domain), transform=ax.transAxes, color="grey", ha="left")
    handles = [plt.Line2D([], [], color=COLORS[p], lw=1.0, marker="o", ms=4, label=LABELS[p]) for p in TIMELINE]
    ax.text(len(semesters) - 0.55, place[float(base.iloc[-1])] if np.isfinite(base.iloc[-1]) else 0, LABELS[BASELINE],
            color="grey", fontsize=7, ha="right", va="bottom", zorder=6)  # the Baseline named at its line
    fig.legend(handles=handles, loc="outside lower center", ncol=2, frameon=False)
    fix_axes_height(fig, ax)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def frontier_figure(forecasts: pd.DataFrame, domain: str, variants: dict[str, str], path, top: int = 3,
                    mse_panel: bool = False) -> None:
    """
    Draws the strongest targets of each of one domain's tasks: x = semester, with one sub-slot per such target (weakest
    to strongest), y = theta. Each target's true theta* is an open black circle on a short black bar across its
    sub-slot; each protocol in TIMELINE (colours from COLORS) puts a filled circle at its prediction, offset within the
    sub-slot, with a thin connector to the truth; the Baseline is a dashed grey segment per semester. Judge protocols
    are the best judge on references. The theta axis spans the truths and the Baseline levels plus a margin.

    Inputs:
        forecasts: the forecasts table (calibration.forecasts rows, every column).
        domain: the domain's code key.
        variants: protocol -> the variant to draw.
        path: the image file to write.
        top: how many of each semester's strongest targets to draw; the number of targets per task (8) draws them all.
        mse_panel: add a lower panel with, per semester, each protocol's MSE over those targets as a percentage of the
            Baseline's MSE on the same targets (lines with circles, dashed line at 100); its axis spans the values
            within 3 interquartile ranges above the upper quartile.
    Outputs:
        the figure, written to path; predictions outside the theta axis, and lower-panel values above its axis, are
        printed.
    """
    f = one_per_target(forecasts)
    f = f[(f.domain == domain) & np.array([variants.get(pr, "-") == v for pr, v in zip(f.protocol, f.variant)])]
    semesters = sorted(f.semester.unique())
    x = {s: i for i, s in enumerate(semesters)}
    truth = f.drop_duplicates(["semester", "model"]).sort_values("theta")
    frontier = truth.groupby("semester").tail(top)  # the `top` strongest targets per task
    base = f[f.protocol == BASELINE].groupby("semester").theta_hat.first()
    # targets of one semester sit within ±0.22 of its tick, so the gap between semesters (>= 0.56) is wider
    # than the gap between neighbouring targets; the protocols spread inside each target's sub-slot
    slots = np.linspace(-0.22, 0.22, top) if top > 1 else np.zeros(1)
    width = (slots[1] - slots[0]) if top > 1 else 0.2
    offsets = dict(zip(TIMELINE, np.linspace(-0.3 * width, 0.3 * width, len(TIMELINE))))
    fig_width = 0.5 * len(semesters) + 0.8 if top <= 3 else 0.4 * top * len(semesters) + 0.8  # the all-targets version needs the room
    figsize = (fig_width, 3.6 if mse_panel else 2.4)
    if mse_panel:
        fig, (ax, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True, layout="constrained", gridspec_kw={"height_ratios": [2, 1]})
    else:
        fig, ax = plt.subplots(figsize=figsize, layout="constrained")
        ax2 = None
    for sem in semesters:
        ax.hlines(base[sem], x[sem] - 0.3, x[sem] + 0.3, color="grey", lw=0.8, ls="--", zorder=1)
        for slot, (_, row) in zip(slots, frontier[frontier.semester == sem].iterrows()):
            xc, t = x[sem] + slot, row.theta
            ax.hlines(t, xc - 0.42 * width, xc + 0.42 * width, color=TRUE, lw=0.7, zorder=3)  # the truth level every connector ends on
            for p in TIMELINE:
                g = f[(f.protocol == p) & (f.semester == sem) & (f.model == row.model)]
                if g.empty or not np.isfinite(g.theta_hat.iloc[0]):
                    continue
                xi, yi = xc + offsets[p], float(g.theta_hat.iloc[0])
                ax.plot([xi, xi], [min(t, yi), max(t, yi)], color=COLORS[p], lw=0.5, alpha=0.6, zorder=2)
                ax.scatter(xi, yi, s=12, facecolor=COLORS[p], edgecolor="none", zorder=4)
            ax.scatter(xc, t, s=18, facecolor="white", edgecolor=TRUE, linewidth=0.8, zorder=5)
    # the axis spans the truths and the Baseline levels; a prediction far outside runs off the panel
    shown = pd.concat([frontier.theta, base])
    lo, hi = float(shown.min()), float(shown.max())
    ax.set_ylim(lo - 0.15 * (hi - lo) - 0.2, hi + 0.15 * (hi - lo) + 0.2)
    keys = frontier.set_index(["semester", "model"]).index
    off = f[f.protocol.isin(TIMELINE) & f.set_index(["semester", "model"]).index.isin(keys)
            & ((f.theta_hat < ax.get_ylim()[0]) | (f.theta_hat > ax.get_ylim()[1]))]
    if len(off):
        print(f"{path}: off the panel: " + ", ".join(f"{r.protocol} {r.semester} {r.model} {r.theta_hat:.1f}" for r in off.itertuples()))
    ax.set_ylabel("Model Capability (θ)")
    ax.text(0, 1.02, DOMAIN_LABELS.get(domain, domain), transform=ax.transAxes, color="grey", ha="left")
    if ax2 is not None:  # MSE over the frontier targets as % of the Baseline's, per semester and protocol
        on_frontier = f[f.set_index(["semester", "model"]).index.isin(keys)].assign(err=lambda d: (d.theta_hat - d.theta) ** 2)
        mse = on_frontier.groupby(["protocol", "semester"]).err.mean()
        pct = {(p, s): 100 * mse.get((p, s), np.nan) / mse.get((BASELINE, s), np.nan) for p in TIMELINE for s in semesters}
        ax2.axhline(100, color="grey", lw=0.8, ls="--", zorder=4)
        values = []
        for p in TIMELINE:
            ys = [pct[(p, s)] for s in semesters]
            ax2.plot([x[s] for s in semesters], ys, color=COLORS[p], lw=1.0, marker="o", ms=3.5, zorder=3)
            values += [v for v in ys if np.isfinite(v)]
        values = np.array(values + [100.0])
        q1, q3 = np.percentile(values, [25, 75])
        fence = q3 + 3 * (q3 - q1) if len(values) >= 10 else np.inf
        inside = values[values <= fence]
        ax2.set_ylim(0, inside.max() * 1.1)
        if (values > fence).any():
            print(f"{path}: lower panel, off the top (%): {sorted(round(float(v)) for v in values[values > fence])}")
        ax2.set_ylabel(f"MSE, top-{top} (% of saturated-item)" if top < 8 else "MSE (% of saturated-item)")
    (ax2 or ax).set_xticks(range(len(semesters)), semesters)
    (ax2 or ax).tick_params(axis="x", rotation=90)
    handles = [plt.Line2D([], [], marker="o", ms=4, ls="", markerfacecolor="white", markeredgecolor=TRUE,
                          label=("Target models (true θ)" if top >= 8 else f"Top-{top} target models (true θ)"))]
    handles += [plt.Line2D([], [], marker="o", ms=4, ls="", color=COLORS[p], label=LABELS[p]) for p in TIMELINE]
    handles.append(plt.Line2D([], [], color="grey", lw=0.8, ls="--", label=LABELS[BASELINE]))
    fig.legend(handles=handles, loc="outside lower center", ncol=2, frameon=False)
    fix_axes_height(fig, ax)  # the theta panel keeps the shared height; the lower panel is half of it
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def stars(p: float, not_significant: str = "") -> str:
    """
    Returns the significance mark of a p-value.

    Inputs:
        p: the p-value.
        not_significant: the mark for p of .05 or above (e.g. "n.s."); empty by default.
    Outputs:
        "***" for p < .001, "**" for p < .01, "*" for p < .05, else `not_significant`.
    """
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else not_significant
