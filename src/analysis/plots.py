"""Figures and tables of the results section.

forecast_figure   one panel per domain, semesters along x; within a semester the targets
                  sorted by theta*, black = true theta, orange = forecast, dotted connectors
bar_figure        four bars (the conditions), coloured by method, at the mean of a per-(domain,
                  semester) metric; one staggered open circle per domain at its pooled value,
                  joined across conditions by a solid line in the domain's colour; saturated-item
                  prediction as a dashed line; off-axis values compressed into a band above
                  the in-range axis
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
    "svg.fonttype": "none", "pdf.fonttype": 42, "savefig.dpi": 300,
})
CONDITIONS = ["glm_length", "glm_regex", "embedding", "direct", "pairwise"]  # the bars, in this order
# Protocol names as the paper uses them (CLAUDE.md, figure colour palette); Length alone is a diagnostic, not a paper protocol.
LABELS = {BASELINE: "Prediction from saturated items", "glm_length": "Length", "glm_regex": "Deterministic features",
          "embedding": "Embedding", "direct": "Absolute scoring", "pairwise": "Pairwise judge"}
DOMAIN_LABELS = {"science": "Science", "math": "Maths", "coding": "Coding", "cyber": "Cyber"}
# regressions in greens, light to dark (length, length + prose, embedding); judges in blues (direct
# light, pairwise dark); domains in warm hues, so the two encodings never share a colour
COLORS = dict(zip(CONDITIONS, ["#b8e0c2", "#3f9c73", "#0d4a30", "#9dc6e6", "#1f4c85"]))  # muted, print-like; wider steps for visibility
# the pooled-judge conditions are not bars, but task figures may draw them
LABELS.update({"direct_ensemble": "Direct scoring, pooled judges", "pairwise_ensemble": "Pairwise judges, pooled"})
COLORS.update({"direct_ensemble": "#c9d9ec", "pairwise_ensemble": "#7a93b8"})
DOMAIN_COLORS = {"science": "#b4485a", "math": "#e0955a", "coding": "#a97aa8", "cyber": "#cdb04c"}
STAGGER = 0.05  # the four domains' circles sit within ±0.05 of the bar centre
TRUE, PRED = "black", "#eb6834"
METRIC_LABELS = {"mse": "Squared error", "delta": "MSE − saturated-item MSE", "z": "z of MSE − saturated-item MSE",
                 "pct": "Relative MSE (%)"}


def with_pct(metrics: pd.DataFrame) -> pd.DataFrame:
    """metrics with a `pct` column: 100 * mse / baseline_mse per row, the MSE as a percentage of
    the prediction from saturated items's MSE on the same targets (100 = the baseline, below = better)."""
    return metrics.assign(pct=100 * metrics.mse / metrics.baseline_mse)


def compress(values: np.ndarray, extent: np.ndarray | None = None) -> tuple[np.ndarray, dict[float, float]]:
    """Map off-axis values into a band above the in-range values, keeping their order.

    values: every y that will be drawn (bar means, domain values, the baseline); `extent`:
    further y's that must stay in range, e.g. the ends of the in-range error bars. Returns
    (position per value, {true off-axis value: drawn position}). Values within 3 IQR above the
    upper quartile, and every value below it, keep their position; only far-high values (and
    only when there are at least 10 values) are off-axis. Off-axis values are placed above
    max(in-range) + a gap of 35% of the in-range span, spread over a band of another 35%
    in rank order (a single off-axis value sits at the band's centre). The dict is empty
    when nothing is off-axis."""
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
    """A white strip across the panel and its left spine in the gap between the in-range
    values and the compressed band, so the axis break is visible; the glyph is the user's."""
    if not mapping:
        return
    lo, hi = break_edges(raw, mapping, extent)
    ax.axhspan(lo, hi, xmin=-0.04, xmax=1.0, facecolor="white", edgecolor="none", zorder=9, clip_on=False)


def break_edges(raw: np.ndarray, mapping: dict[float, float], extent: np.ndarray | None = None) -> tuple[float, float]:
    """(bottom, top) of the white break strip in drawn units: just above the highest in-range
    value and just below the compressed band. Error bars are capped at the bottom edge so
    nothing pokes through the strip into the band. (inf, inf) without a band."""
    if not mapping:
        return float("inf"), float("inf")
    lo = max([v for v in raw if v not in mapping] + ([float(np.nanmax(extent))] if extent is not None and len(extent) else []))
    hi = min(mapping.values())
    return lo + 0.05 * (hi - lo), hi - 0.25 * (hi - lo)


def unclip_markers(fig):
    """Every marker drawn in full: scatter points and marker lines are never clipped by the axes, so a point
    on an axis limit is not cut in half. Lines, bands and boxes stay clipped. Returns the figure."""
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
    """Half-width of the t-based 95% interval of the mean of `values`; NaN with fewer than 3."""
    n = len(values)
    if n < 3:
        return float("nan")
    return float(scipy_stats.t.ppf(0.975, n - 1) * np.std(values, ddof=1) / np.sqrt(n))


AXES_HEIGHT = 1.6  # inches; bar_figure and task_figure share it so the panels sit side by side


def fix_axes_height(fig, ax, inches: float = AXES_HEIGHT, rounds: int = 3) -> None:
    """Grow or shrink the figure so the axes' drawn height equals `inches`, whatever the legend
    and tick labels take up. Constrained layout re-runs on each draw, so a few rounds converge."""
    for _ in range(rounds):
        fig.canvas.draw()
        current = ax.get_position().height * fig.get_figheight()
        fig.set_figheight(fig.get_figheight() + inches - current)


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
            if (dom, sem) in baseline.index:  # the prediction from saturated items: one level per task
                ax.hlines(baseline[(dom, sem)], i - 0.45, i + 0.45, color="grey", lw=0.8, ls="--", zorder=0)
            ax.scatter(x, g.theta, s=16, facecolor="white", edgecolor=TRUE, linewidth=0.8, zorder=3)
            ax.scatter(x, g.theta_hat, s=16, facecolor=PRED, edgecolor="none", zorder=4)
        ax.set_xticks(range(len(sems)), sems, rotation=90 if len(sems) > 3 else 0)
        ax.text(0, 1.02, DOMAIN_LABELS.get(dom, dom), transform=ax.transAxes, color="grey", ha="left")
    np.atleast_1d(axes)[0].set_ylabel("Model Capability (θ)")
    handles = [plt.Line2D([], [], marker="o", ms=4, ls="", markerfacecolor="white", markeredgecolor=TRUE, label="Target models (true θ)"),
               plt.Line2D([], [], marker="o", ms=4, ls="", color=PRED, label=f"{LABELS.get(protocol, protocol)} prediction"),
               plt.Line2D([], [], color="grey", lw=0.8, ls="--", label=LABELS[BASELINE])]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def bar_figure(metrics: pd.DataFrame, metric: str, variants: dict[str, str], path, semester: str | None = None,
               llr: pd.DataFrame | None = None) -> None:
    """One bar per condition in CONDITIONS, coloured by method, at the mean over the (domain,
    semester) tasks of the metric ("mse", "delta" or "z"), with a capless 95% t-interval over
    those tasks (none below 3); one open circle per domain at the domain's pooled value
    (semester "all", or the given semester's value), the domains staggered across the bar
    and each domain's circles joined across the conditions by a solid line in its colour; the
    prediction from saturated items as a dashed grey line (its mean raw MSE, or 0 for delta and z); stars
    above a bar, given `llr` (likelihood_ratio rows, one shape and judge per target), are the one-sided
    log likelihood ratio test against the references' distribution with each task as the unit
    (*, **, *** for p below .05, .01, .001; a protocol without log LRs gets none). Off-axis values (beyond 3 IQR) are drawn in a compressed band above the in-range values
    with ticks at their true values (see compress); the break glyph is left to the user."""
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
    ax.text(len(CONDITIONS) - 0.5, place[baseline], LABELS[BASELINE].replace(" from ", " from\n"), color="grey", fontsize=7,
            ha="right", va="bottom", zorder=6)  # the baseline named at its line, as in the draft
    fig.legend(handles=handles, loc="outside lower center", ncol=2, frameon=False, handlelength=2.0)
    fix_axes_height(fig, ax)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def semester_table(metrics: pd.DataFrame, metric: str, variants: dict[str, str], domain: str | None = None) -> pd.DataFrame:
    """Semester x condition (prediction from saturated items first, then CONDITIONS) for one domain, or pooled when
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


def task_figure(forecasts: pd.DataFrame, domain: str, semester: str, conditions: list[tuple[str, str]], path,
                reasoning: set[str] | None = None) -> None:
    """One forecasting task: the references' known theta* (grey filled circles, left slot) and,
    in the right slot, the targets' true theta (open black circles) with each condition's
    forecast as filled circles in the condition's bar colour (COLORS), joined to the truth by a
    thin connector in the same colour, targets sorted by theta*; the prediction from saturated items (the
    references' mean) as a dashed level across the target slot. `conditions` are (protocol,
    variant) pairs, e.g. [("pairwise", "linear"), ("glm_regex", "-")]. Judge protocols are drawn
    as the best judge on references. Reads the references' theta* from the frozen tables.
    `reasoning`: model names whose transcripts carry visible chain of thought; their truth is
    drawn as an open diamond instead of a circle (and named in the legend)."""
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


TIMELINE = ["glm_regex", "embedding", "direct", "pairwise"]  # the four main approaches, in this order


def timeline_figure(metrics: pd.DataFrame, domain: str, variants: dict[str, str], path, scores: pd.DataFrame | None = None,
                    metric: str = "mse", forecasts: pd.DataFrame | None = None, band: bool = False, shade: bool = False,
                    cut: float | None = None, llr: pd.DataFrame | None = None) -> None:
    """One domain's tasks over time: x = semester, y = the task's `metric` ("mse"; "delta": MSE minus
    the saturated-item MSE, baseline at 0; "pct": MSE as a percentage of the saturated-item MSE, baseline
    at 100), one line with filled
    circles per approach in TIMELINE (bar colours), and the prediction from saturated items as a dashed
    grey line drawn on top. Judge approaches are the best judge on references. The approaches
    are offset slightly in x so their points do not overlap. With `scores` (stats.scores
    output) each point carries a capless error bar: the task's percentile-bootstrap 95%
    interval of the MSE over its 8 targets (lo, hi); for "delta" and "pct" with `forecasts` it is
    instead the 95% t-interval of the paired difference of squared errors over the task's
    targets. With `llr` (likelihood_ratio rows, one shape and judge per target) a star above a
    point gives the one-sided log likelihood ratio test of the task's targets against the
    references' distribution (*, **, *** for p below .05, .01, .001). Far-high
    values (beyond Tukey's far-out fence) are cut off by default: the axis spans the in-range
    values and their error bars and the outliers run off the top (printed); with `band=True`
    they are drawn in a compressed band above a break strip instead (the bar plots' convention).
    `shade=True` draws each approach's interval as a translucent band along its line instead of
    vertical bars (not drawn across compressed values). `cut` fixes the top of the axis at that
    value: points above it are drawn as upward triangles on the cut line and the top tick reads
    ">cut"; lines and bands are clipped there."""
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
        if metric == "delta":  # the bootstrap interval is on the MSE; shift it by the task's saturated-item MSE
            sc = sc.assign(lo=sc.lo.to_numpy() - base_of, hi=sc.hi.to_numpy() - base_of)
        elif metric == "pct":  # or as a percentage of the task's saturated-item MSE
            sc = sc.assign(lo=100 * sc.lo.to_numpy() / base_of, hi=100 * sc.hi.to_numpy() / base_of)
    _, first = compress(raw)  # which values are off-axis, before the error bars widen the in-range span
    extent = np.array([sc.hi.get((p, sem), np.nan) for p in TIMELINE for sem, v in series[p].dropna().items()
                       if float(v) not in first]) if sc is not None else None
    positions, mapping = compress(raw, extent)
    place = dict(zip(raw.tolist(), positions.tolist()))
    fig, ax = plt.subplots(figsize=(2.4, 2.4), layout="constrained")
    base = series[BASELINE]
    ax.plot([x[s] for s in semesters], [place[float(v)] for v in base], color="grey", lw=0.8, ls="--", zorder=4)  # on top: a fit that collapses to the intercept sits exactly on it
    offsets = dict(zip(TIMELINE, np.linspace(-0.09, 0.09, len(TIMELINE))))  # slight x offset per approach
    task_p = None
    if llr is not None:
        from analysis import likelihood_ratio as lr
        task_p = lr.task_tests(llr[llr.domain == domain]).set_index(["protocol", "semester"]).p
    errors = None
    if forecasts is not None:  # per-target squared errors, paired with the prediction from saturated items's on the same targets
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
        if sc is not None and shade:  # one translucent band per approach along its line
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
    ax.text(len(semesters) - 0.55, place[float(base.iloc[-1])] if np.isfinite(base.iloc[-1]) else 0, LABELS[BASELINE].replace(" from ", " from\n"),
            color="grey", fontsize=7, ha="right", va="bottom", zorder=6)  # the baseline named at its line, as in the draft
    fig.legend(handles=handles, loc="outside lower center", ncol=2, frameon=False)
    fix_axes_height(fig, ax)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def frontier_figure(forecasts: pd.DataFrame, domain: str, variants: dict[str, str], path, top: int = 3,
                    mse_panel: bool = False) -> None:
    """The `top` frontier targets of each task in one domain: x = semester (one sub-slot per
    frontier model, weakest to strongest), y = theta. Open black circles are the true theta*
    of the semester's `top` strongest targets; each approach in TIMELINE (bar colours) puts a
    filled circle at its forecast for that model, offset slightly within the sub-slot, with a
    thin connector to the truth, which is also marked by a short black bar across the sub-slot;
    the prediction from saturated items is a dashed grey segment per
    semester. Judge approaches are the best judge on references. The y-axis spans the truths
    and saturated-item levels (plus a margin); a forecast far outside runs off the panel and is
    printed. With `mse_panel` a second panel below shows, per semester, each approach's MSE
    over those `top` targets as a percentage of the prediction from saturated items's MSE on the same
    targets (lines with circles; dashed line at 100); its axis spans the values within Tukey's
    far-out fence, higher ones run off and are printed."""
    f = one_per_target(forecasts)
    f = f[(f.domain == domain) & np.array([variants.get(pr, "-") == v for pr, v in zip(f.protocol, f.variant)])]
    semesters = sorted(f.semester.unique())
    x = {s: i for i, s in enumerate(semesters)}
    truth = f.drop_duplicates(["semester", "model"]).sort_values("theta")
    frontier = truth.groupby("semester").tail(top)  # the `top` strongest targets per task
    base = f[f.protocol == BASELINE].groupby("semester").theta_hat.first()
    # targets of one semester sit within ±0.22 of its tick, so the gap between semesters (>= 0.56) is wider
    # than the gap between neighbouring targets; the approaches spread inside each target's sub-slot
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
    # the axis spans the truths and the saturated-item levels; a forecast far outside runs off the panel
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
    if ax2 is not None:  # MSE over the frontier targets as % of the prediction from saturated items's, per semester and approach
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


def reasoning_figure(forecasts: pd.DataFrame, domain: str, semester: str, variants: dict[str, str], reasoning: set[str], path,
                     cut: float = 2.0, mse_cut: float = 400.0) -> None:
    """One task split by whether the target is a reasoning model: two panels side by side.
    Left: targets in two slots (no reasoning | reasoning), sorted by theta* within a slot, the
    truth as a short black bar with an open circle (no reasoning) or diamond (reasoning), and
    each approach in TIMELINE's forecast as a filled circle offset inside the target's
    sub-slot with a thin connector; the prediction from saturated items as a dashed segment per slot.
    The axis stops at `cut`: a forecast above it is drawn as a triangle on the cut line and the
    top tick reads ">cut". Right: MSE per approach over each
    group as paired bars (reasoning hatched) with 95% t-intervals over the group's squared
    errors, and the prediction from saturated items's MSE per group as dashed / dotted lines; the axis
    stops at `mse_cut`, a bar above it is drawn to the cut with a triangle on the cut line and
    the top tick reads ">mse_cut"."""
    f = one_per_target(forecasts)
    f = f[(f.domain == domain) & (f.semester == semester) & np.array([variants.get(pr, "-") == v for pr, v in zip(f.protocol, f.variant)])]
    truth = f.drop_duplicates("model").set_index("model").theta
    groups = [("no reasoning", [m for m in truth.index if m not in reasoning]), ("reasoning", [m for m in truth.index if m in reasoning])]
    base = float(f[f.protocol == BASELINE].theta_hat.iloc[0])
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(4.6, 2.4), layout="constrained", gridspec_kw={"width_ratios": [3, 2]})
    for gi, (name, models) in enumerate(groups):
        models = sorted(models, key=lambda m: truth[m])
        slots = np.linspace(-0.3, 0.3, len(models)) if len(models) > 1 else np.zeros(1)
        width = (slots[1] - slots[0]) if len(models) > 1 else 0.2
        offsets = dict(zip(TIMELINE, np.linspace(-0.3 * width, 0.3 * width, len(TIMELINE))))
        ax.hlines(base, gi - 0.4, gi + 0.4, color="grey", lw=0.8, ls="--", zorder=1)
        for slot, m in zip(slots, models):
            xc, t = gi + slot, truth[m]
            ax.hlines(t, xc - 0.42 * width, xc + 0.42 * width, color=TRUE, lw=0.7, zorder=3)  # the truth level every connector ends on
            for p in TIMELINE:
                g = f[(f.protocol == p) & (f.model == m)]
                if g.empty or not np.isfinite(g.theta_hat.iloc[0]):
                    continue
                xi, yi = xc + offsets[p], float(g.theta_hat.iloc[0])
                if yi > cut:  # off the top: a triangle on the cut line
                    ax.plot([xi, xi], [t, cut], color=COLORS[p], lw=0.5, alpha=0.6, zorder=2)
                    ax.scatter(xi, cut, s=18, marker="^", facecolor=COLORS[p], edgecolor="none", zorder=6, clip_on=False)
                    continue
                ax.plot([xi, xi], [min(t, yi), max(t, yi)], color=COLORS[p], lw=0.5, alpha=0.6, zorder=2)
                ax.scatter(xi, yi, s=12, facecolor=COLORS[p], edgecolor="none", zorder=4)
            ax.scatter(xc, t, s=18 if gi == 0 else 24, marker="o" if gi == 0 else "D", facecolor="white", edgecolor=TRUE, linewidth=0.8, zorder=5)
    ax.set_xticks([0, 1], [f"{n}\n(n={len(ms)})" for n, ms in groups])
    ax.set_xlim(-0.6, 1.6)
    lo = min(f[f.protocol.isin(TIMELINE)].theta_hat.min(), truth.min(), base)
    ax.set_ylim(lo - 0.2, cut)
    ticks = [t for t in ax.get_yticks() if ax.get_ylim()[0] <= t < cut] + [cut]
    ax.set_yticks(ticks, [f"{t:g}" for t in ticks[:-1]] + [f">{cut:g}"])
    ax.set_ylabel("Model Capability (θ)")
    ax.text(0, 1.02, f"{DOMAIN_LABELS.get(domain, domain)} {semester}", transform=ax.transAxes, color="grey", ha="left")
    # right: MSE per approach and group, with a t-interval over the group's squared errors
    err = f.assign(err=(f.theta_hat - f.theta) ** 2, group=f.model.map(lambda m: "reasoning" if m in reasoning else "no reasoning"))
    base_mse = err[err.protocol == BASELINE].groupby("group").err.mean()  # the group's baseline MSE
    err = err.assign(err=100 * err.err / err.group.map(base_mse))  # relative: % of the group's baseline MSE
    mse = err.groupby(["protocol", "group"]).err.mean()
    for i, p in enumerate(TIMELINE):
        for gi, (name, _) in enumerate(groups):
            xi, v = i + (-0.19 if gi == 0 else 0.19), mse.get((p, name), np.nan)
            ax2.bar(xi, v, width=0.36, color=COLORS[p], alpha=0.45, hatch=None if gi == 0 else "///", edgecolor="white" if gi else "none", linewidth=0.4)
            pts = err[(err.protocol == p) & (err.group == name)].err.dropna().to_numpy()  # one solid circle per model
            pts_in = pts[pts <= mse_cut]
            ax2.scatter(np.full(len(pts_in), xi) + np.linspace(-0.08, 0.08, len(pts_in)), pts_in, s=12, facecolor=COLORS[p],
                        edgecolor="black", linewidth=0.4, zorder=5)
            if (pts > mse_cut).any():
                ax2.scatter(np.full((pts > mse_cut).sum(), xi), np.full((pts > mse_cut).sum(), mse_cut), s=16, marker="^",
                            facecolor=COLORS[p], edgecolor="black", linewidth=0.4, zorder=6, clip_on=False)
    ax2.axhline(100, color="grey", lw=0.8, ls="--", zorder=3)  # both groups' baselines are 100% by construction
    for i, p in enumerate(TIMELINE):  # bars above the cut: a triangle on the cut line
        for gi, (name, _) in enumerate(groups):
            if mse.get((p, name), 0) > mse_cut:
                ax2.scatter(i + (-0.19 if gi == 0 else 0.19), mse_cut, s=22, marker="^", facecolor=COLORS[p], edgecolor="none", zorder=6, clip_on=False)
    ax2.set_ylim(0, mse_cut)
    ticks = [t for t in ax2.get_yticks() if 0 <= t < mse_cut] + [mse_cut]
    ax2.set_yticks(ticks, [f"{t:g}" for t in ticks[:-1]] + [f">{mse_cut:g}%"])
    ax2.set_xticks(range(len(TIMELINE)), [LABELS[p] for p in TIMELINE])
    ax2.tick_params(axis="x", rotation=90)
    ax2.set_ylabel(METRIC_LABELS["pct"])
    handles = [plt.Line2D([], [], marker="o", ms=4, ls="", markerfacecolor="white", markeredgecolor=TRUE, label="Target models, no reasoning"),
               plt.Line2D([], [], marker="D", ms=4, ls="", markerfacecolor="white", markeredgecolor=TRUE, label="Target models, reasoning")]
    handles += [plt.Line2D([], [], marker="o", ms=4, ls="", color=COLORS[p], label=LABELS[p]) for p in TIMELINE]
    handles += [plt.Line2D([], [], color="grey", lw=0.8, ls="--", label=LABELS[BASELINE]),
                plt.Line2D([], [], marker="o", ms=3.5, ls="", color="#888888", markeredgecolor="black", markeredgewidth=0.4, label="one model (squared error)")]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False)
    fix_axes_height(fig, ax)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def stars(p: float, not_significant: str = "") -> str:
    """The significance mark of a p-value: *** p < .001, ** p < .01, * p < .05, else `not_significant`."""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else not_significant
