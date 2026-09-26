"""Research question 3: when does the capability signal generalise?

Three analyses that only reshape `analysis/results/forecasts.csv`: the prediction score over
target semesters, the forecast error against how far a target sits from its references, and the
reasoning-model targets of 2025-I, whose references are all non-reasoning models. Transcripts
from one domain calibrated on another domain's theta* live in cross_domain.py.

Judge protocols are the best judge on references, one variant per protocol as in figure one.
The prediction score is signal_glm's, -log2(error / the saturated-item prediction's error): the
number of halvings of the error, 0 at the baseline. Squared errors are pooled before the score is
taken, because a target the saturated-item prediction happens to nail has a near-zero baseline
error and would otherwise dominate.

Design: docs/superpowers/specs/2026-09-17-generalisation-design.md
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from analysis.challenges.signal_glm import COLORS, DOMAIN_COLORS, DOMAINS, LABELS, PROTOCOLS, score, score_axis, stars
from analysis.plots import DOMAIN_LABELS, chosen, ci95, fix_axes_height, unclip_markers
from analysis.stats import BASELINE, one_per_target

DOMAIN_LABELS = {**DOMAIN_LABELS, "math": "Maths"}  # the manuscript spells the domain with an s

MARKERS = {"science": "o", "math": "s", "coding": "^", "cyber": "D"}  # CLAUDE.md, one shape per domain
# 2025-I targets whose transcripts carry a reasoning trace: deepseek-r1 and minimax-m1 always
# reason, the three Qwen3 models were collected with `reasoning: {enabled: true}` (configs/models).
REASONING_2025_I = ("deepseek-r1-0528", "minimax-m1", "qwen3-235b-a22b", "qwen3-30b-a3b", "qwen3-32b")
SCORE_LABEL = "Prediction score\n(halvings of error)"


def rows(forecasts: pd.DataFrame, variants: dict[str, str], semesters: tuple[str, ...] | None = None) -> pd.DataFrame:
    """One row per target x protocol in PROTOCOLS: domain, semester, model, protocol, theta_hat,
    theta and err (the squared error). Judge protocols are the best judge on references; every
    protocol is taken in the variant `variants` names for it. Rows without a forecast are dropped."""
    f = chosen(one_per_target(forecasts), variants)
    f = f[f.protocol.isin(PROTOCOLS)]
    if semesters is not None:
        f = f[f.semester.isin(semesters)]
    f = f.dropna(subset=["theta_hat", "theta"])
    return f.assign(err=(f.theta_hat - f.theta) ** 2)[["domain", "semester", "model", "protocol", "theta_hat", "theta", "err"]]


def with_baseline(table: pd.DataFrame) -> pd.DataFrame:
    """`table` with a `base` column: the saturated-item prediction's squared error on the same
    (domain, semester, model). Baseline rows are kept."""
    base = table[table.protocol == BASELINE].set_index(["domain", "semester", "model"]).err
    return table.assign(base=table.set_index(["domain", "semester", "model"]).index.map(base))


def task_scores(table: pd.DataFrame, draws: int = 2000, seed: int = 0, metric: str = "score") -> pd.DataFrame:
    """Per (domain, semester, protocol), from the task's targets: n, mse, baseline_mse, score =
    -log2(mse / baseline_mse), and `value`, `lo`, `hi`, `p` in the chosen metric, from `draws`
    bootstrap resamples of the task's targets (each target's pair of squared errors moves together).
    metric "score": value = the score, lo/hi its 2.5/97.5 percentiles, p the two-sided test of the
    score against 0. metric "error": value = the mean squared error, lo/hi its own 2.5/97.5
    percentiles, p the two-sided test of the paired difference from the saturated-item
    prediction's error. The figures replace p by the log likelihood ratio's
    (likelihood_ratio.task_tests)."""
    assert metric in ("score", "error"), metric
    rng = np.random.default_rng(seed)
    out = []
    for (domain, semester, protocol), g in with_baseline(table).dropna(subset=["base"]).groupby(["domain", "semester", "protocol"]):
        err, base = g.err.to_numpy(float), g.base.to_numpy(float)
        idx = rng.integers(0, len(err), size=(draws, len(err)))
        boot_err, boot_base = err[idx].mean(axis=1), base[idx].mean(axis=1)
        boot = score(boot_err, boot_base) if metric == "score" else boot_err
        gap = boot if metric == "score" else boot_err - boot_base  # what the test holds against 0
        keep = np.isfinite(boot) & np.isfinite(gap)
        lo, hi = np.percentile(boot[keep], [2.5, 97.5]) if len(g) >= 3 else (np.nan, np.nan)
        tail = min((gap[keep] <= 0).mean(), (gap[keep] >= 0).mean()) if len(g) >= 3 else np.nan
        value = float(score(err.mean(), base.mean())) if metric == "score" else float(err.mean())
        out.append(dict(domain=domain, semester=semester, protocol=protocol, n=len(g), mse=float(err.mean()),
                        baseline_mse=float(base.mean()), score=float(score(err.mean(), base.mean())), value=value,
                        p=np.nan if protocol == BASELINE else float(min(1.0, 2 * tail)),
                        lo=float(lo), hi=float(hi)))
    return pd.DataFrame(out).sort_values(["domain", "semester", "protocol"], ignore_index=True)


def padded_axis(values: np.ndarray, margin: float = 0.15) -> tuple[float, float]:
    """score_axis widened by `margin` halvings, so a point never sits on the spine."""
    lo, hi = score_axis(values)
    return lo - margin, hi + margin


def cut_axis(values: np.ndarray, keep: float = 0.9, margin: float = 0.15) -> tuple[float, float]:
    """Axis limits holding the middle `keep` of `values`, rounded out to whole halvings and
    widened by `margin`, always including 0. A protocol that collapses on one task (Deterministic
    features loses 7 halvings in math 2025-I) would otherwise flatten every other line."""
    lo, hi = np.percentile(values[np.isfinite(values)], [50 * (1 - keep), 50 * (1 + keep)])
    return padded_axis(np.array([lo, hi]), margin)


BASELINE_COLOUR, BASELINE_LABEL = "#9a9a9a", "Baseline"
ERROR_LABEL = "Squared error"


def time_figure(scores: pd.DataFrame, by: str = "protocol", stack: bool = False, box: bool = False,
                metric: str = "score", log: bool = False) -> plt.Figure:
    """Prediction score over target semesters, in halvings of the error. `by` "protocol": one panel
    per protocol, one line per domain in the domain's colour and marker; `by` "domain": one panel
    per domain, one line per protocol in the protocol's colour, every panel on one axis so the
    domains compare. Each line carries a translucent band one standard error either side of the
    paired difference on the score scale, clipped to the panel, and the saturated-item prediction
    is the dashed line at 0. Stars sit above each bar: `scores.p`, which the figures take from the
    log likelihood ratio of the task's targets (likelihood_ratio.task_tests). Coding and cyber
    start at 2025-II, the first semester they have. `log` (metric "error" only) puts the squared
    error on a log axis from 0.01 to 10."""
    assert by in ("protocol", "domain"), by
    assert not log or metric == "error", "only squared error goes on a log axis"
    assert metric == "score" or by == "domain", "the error version draws the baseline as a series: by domain only"
    # stacked: one panel per row on a shared semester axis, so the panels line up vertically
    protocols = [p for p in PROTOCOLS[1:] if p != "glm_length"]
    domains = [d for d in DOMAINS if d in set(scores.domain)]
    series = protocols if metric == "score" else [BASELINE, *protocols]  # errors show the baseline's own
    panels, lines = (protocols, domains) if by == "protocol" else (domains, series)
    colour = (lambda key: DOMAIN_COLORS[key]) if by == "protocol" else (
        lambda key: BASELINE_COLOUR if key == BASELINE else COLORS[key])
    marker = (lambda key: MARKERS[key]) if by == "protocol" else (lambda key: "o")
    label = (lambda key: DOMAIN_LABELS[key]) if by == "protocol" else (
        lambda key: BASELINE_LABEL if key == BASELINE else LABELS[key])
    shown = scores[scores.protocol.isin(series)]
    if metric == "score":
        wide = cut_axis(shown.value.to_numpy(float))
        inside = shown[shown.value.between(*wide)]
        # room for the bars, but no more than 5 halvings: one interval reaches 8 and would flatten the rest
        wide = (float(min(wide[0], np.floor(inside.lo.min()))), float(min(max(wide[1], np.ceil(inside.hi.max())), 5.0)))
    else:  # errors start at 0; the axis holds the middle 90% of the task errors, a blow-up sits on the edge
        wide = (0.01, 10.0) if log else (0.0, float(np.ceil(np.nanpercentile(shown.value, 90) * 1.25 * 2) / 2))
    # by protocol every panel shares the semester axis; by domain a panel spans only its own
    # semesters, so coding and cyber are two slots wide rather than five with three empty
    spans = {key: sorted(scores.semester.unique()) if by == "protocol" or stack
             else sorted(scores[scores.domain == key].semester.unique()) for key in panels}
    if stack:
        fig, axes = plt.subplots(len(panels), 1, figsize=(2.4, 0.95 * len(panels) + 1.0), layout="constrained",
                                 sharex=True, sharey=by == "domain")
    else:
        fig, axes = plt.subplots(1, len(panels), figsize=(0.7 + 0.24 * sum(len(v) for v in spans.values()) + 0.35 * len(panels), 2.2),
                                 layout="constrained", sharey=by == "domain", width_ratios=[len(spans[k]) for k in panels])
    for ax, key in zip(np.atleast_1d(axes), panels):
        panel = scores[scores.protocol == key] if by == "protocol" else scores[scores.domain == key]
        semesters = spans[key]
        x = {s: i for i, s in enumerate(semesters)}
        bottom, top = padded_axis(panel.value.to_numpy(float)) if by == "protocol" else wide
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#d5d5d5", lw=0.6, zorder=0)  # one gridline per whole halving mark
        if metric == "score":
            ax.axhline(0, color="black", lw=0.8, zorder=1)  # the baseline reads as the figure's x axis
        if by == "protocol":  # thin rule between semesters; the by-domain paper version has none
            for i in range(len(semesters) - 1):
                ax.axvline(i + 0.5, color="lightgrey", lw=0.5, zorder=0)
        dodge = dict(zip(lines, np.linspace(-0.22, 0.22, len(lines))))  # the lines share a semester slot
        for k, line in enumerate(lines):
            d = (panel[panel.domain == line] if by == "protocol" else panel[panel.protocol == line]).sort_values("semester")
            xs = np.array([x[s] + dodge[line] for s in d.semester], float)
            inset = 0.02 * (top - bottom)  # the marker for an off-axis score sits just inside the spine
            raw = d.value.to_numpy(float)
            ys = np.clip(raw, bottom * 1.08, top / 1.08) if log else np.clip(raw, bottom + inset, top - inset)
            out = (raw < bottom) | (raw > top)
            if out.any():
                print(f"time_figure: {key} {line} off the axis: {[round(v, 1) for v in raw[out]]}")
            width = 0.7 / len(lines)  # the slot shared out between the panel's lines
            for xi, lo, hi in zip(xs, np.clip(d.lo, bottom, top), np.clip(d.hi, bottom, top)):
                if box:  # figure one's box: the interval as a translucent rectangle
                    ax.add_patch(plt.Rectangle((xi - width / 2, lo), width, max(hi - lo, 1e-6), facecolor=colour(line),
                                               edgecolor="none", alpha=0.35, zorder=2))
                else:
                    ax.plot([xi, xi], [lo, hi], color=colour(line), lw=0.8, zorder=2)  # bootstrap 95% interval
            ax.plot(xs, ys, color=colour(line), lw=0.4, alpha=0.7, zorder=2)  # very thin connector
            ax.scatter(xs[~out], ys[~out], s=12, marker=marker(line), color=colour(line), linewidths=0,
                       edgecolors="none", zorder=3, clip_on=False)
            for side, tri in ((raw > top, "^"), (raw < bottom, "v")):  # off the axis: a triangle on the edge, pointing out
                ax.scatter(xs[side], ys[side], s=16, marker=tri, color=colour(line), linewidths=0, edgecolors="none",
                           zorder=3, clip_on=False)
            for xi, yi, pval in zip(xs, np.clip(d.hi.to_numpy(float), bottom, top), d.p.to_numpy(float)):  # star above the bar
                if np.isfinite(pval) and stars(pval):
                    ax.text(xi, yi, stars(pval), ha="center", va="bottom", fontsize=6,
                            color=colour(line), zorder=5, clip_on=False)
        ax.set_xticks(range(len(semesters)), semesters)
        ax.tick_params(axis="x", rotation=90 if not stack else 0)
        if stack and key != panels[-1]:  # stacked rows share the semester axis: only the last one carries it
            ax.spines["bottom"].set_visible(False)
            ax.tick_params(axis="x", length=0)
        ax.set_xlim(-0.5, len(semesters) - 0.5)  # room for the dodged points at the ends
        if log:
            ax.set_yscale("log")
            ax.set_yticks([0.01, 0.1, 1, 10], ["0.01", "0.1", "1", "10"])
            ax.yaxis.set_minor_locator(plt.NullLocator())
        else:
            ax.yaxis.set_major_locator(plt.MaxNLocator(5, steps=[1, 2, 5, 10]))
        ax.set_ylim(bottom, top)
        ax.spines["left"].set_bounds(bottom, top)
        if by == "domain":  # the axis runs from the first semester to the last, as in the manuscript
            ax.spines["bottom"].set_bounds(0, len(semesters) - 1)
        title = DOMAIN_LABELS[key] if by == "domain" else LABELS[key]
        ax.text(0, 1.10, title,  # clear of a star on a bar that reaches the top
                transform=ax.transAxes, ha="left", fontsize=8 if by == "domain" else 7,
                **({"fontweight": "bold"} if by == "domain" else {"color": "grey"}))
    ylabel = SCORE_LABEL if metric == "score" else ERROR_LABEL
    if stack:  # one label for the column, on the middle row
        np.atleast_1d(axes)[len(panels) // 2].set_ylabel(ylabel)
    else:
        np.atleast_1d(axes)[0].set_ylabel(ylabel)
    handles = [plt.Line2D([], [], color=colour(l), ls="none", marker=marker(l), ms=3.5, label=label(l)) for l in lines]
    if by == "protocol" or stack:
        fig.legend(handles=handles, loc="outside lower center", ncol=4 if by == "protocol" else (2 if metric == "score" else 3),
                   frameon=False)
        return unclip_markers(fig)
    # the manuscript's layout: a boxed legend at the top right, and an arrow from one note to the 2025-I slot of
    # Science and Maths, the semester whose targets are the first reasoning models (their references are not)
    fig.legend(handles=handles, loc="outside upper right", ncol=2, frameon=True, edgecolor="black", fancybox=False,
               framealpha=1, borderpad=0.3, handletextpad=0.1, columnspacing=0.8, fontsize=7)
    pointed = [(ax, spans[key].index("2025-I")) for ax, key in zip(np.atleast_1d(axes), panels)
               if key in ("science", "math") and "2025-I" in spans[key]]
    if pointed:
        fig.canvas.draw()  # fix the layout, then place the note in figure coordinates
        fig.set_layout_engine("none")
        to_fig = fig.transFigure.inverted()
        xs = [to_fig.transform(ax.transData.transform((i, 1)))[0] for ax, i in pointed]
        top = to_fig.transform(pointed[0][0].transAxes.transform((0, 1)))[1]
        note = top + 0.17
        text = fig.text(float(np.mean(xs)), note, "Reasoning models introduced in early 2025", ha="center", va="bottom",
                        fontsize=7)
        renderer = fig.canvas.get_renderer()  # keep the note clear of the legend box
        clash = (to_fig.transform(text.get_window_extent(renderer).get_points())[1, 0]
                 - to_fig.transform(fig.legends[0].get_window_extent(renderer).get_points())[0, 0] + 0.015)
        if clash > 0:
            text.set_x(float(np.mean(xs)) - clash)
        for x in xs:
            fig.add_artist(plt.matplotlib.patches.FancyArrowPatch((x, note - 0.01), (x, top + 0.01), transform=fig.transFigure,
                                                                 arrowstyle="-|>", mutation_scale=5, lw=0.6, color="black"))
    return unclip_markers(fig)


def reference_theta(domain: str, semester: str) -> pd.Series:
    """The references' theta* for one task, from the frozen model table."""
    from analysis import loading
    from reference import tables
    models, _ = tables.load(loading.domain(domain))
    return models.loc[models[f"{semester}_role"] == "reference", f"{semester}_theta"].dropna()


def distances(table: pd.DataFrame) -> pd.DataFrame:
    """`table` with the signed error (theta_hat - theta) and two distances per target: `above_top`,
    its theta* minus the strongest reference's, and `above_mean`, its theta* minus the references'
    mean. The saturated-item prediction's signed error is exactly -above_mean."""
    refs = {key: reference_theta(*key) for key in table.groupby(["domain", "semester"]).groups}
    key = table.set_index(["domain", "semester"]).index
    return table.assign(signed=table.theta_hat - table.theta,
                        above_top=table.theta.to_numpy() - np.array([refs[k].max() for k in key]),
                        above_mean=table.theta.to_numpy() - np.array([refs[k].mean() for k in key]))


def distance_slopes(table: pd.DataFrame, draws: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Per protocol and distance, the OLS slope of the signed error on the distance with a 95%
    percentile-bootstrap interval over targets. A slope of 0 means the error does not grow as
    targets move away from the references; -1 means the protocol tracks the saturated-item
    prediction, whose signed error is -above_mean by construction. Targets within a task share
    references, so the interval is optimistic."""
    rng = np.random.default_rng(seed)
    out = []
    for distance in ("above_top", "above_mean"):
        for p in PROTOCOLS:
            g = table[table.protocol == p].dropna(subset=[distance, "signed"])
            x, y = g[distance].to_numpy(float), g.signed.to_numpy(float)
            if len(x) < 3:
                continue
            idx = rng.integers(0, len(x), size=(draws, len(x)))
            boot = [np.polyfit(x[i], y[i], 1)[0] for i in idx]
            out.append(dict(distance=distance, protocol=p, n=len(x), slope=float(np.polyfit(x, y, 1)[0]),
                            lo=float(np.percentile(boot, 2.5)), hi=float(np.percentile(boot, 97.5))))
    return pd.DataFrame(out)


def with_bins(table: pd.DataFrame, distance: str = "above_top", bins: int = 3,
              within: tuple[float, float] = (-1.0, 1.0)) -> pd.DataFrame:
    """`table` with `bin`, 0 to bins - 1: the target's equal-count bin of `distance` within its
    domain, over the targets whose distance lies in `within`; the rest get no bin."""
    x = table[distance].where(table[distance].between(*within))
    return table.assign(bin=x.groupby(table.domain).transform(lambda v: pd.qcut(v, bins, labels=False)))


def binned(x: np.ndarray, y: np.ndarray, bins: int = 4) -> tuple[list[float], list[float]]:
    """The means of x and y in each of `bins` equal-count bins of x."""
    edges = np.quantile(x, np.linspace(0, 1, bins + 1))
    which = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, bins - 1)
    full = [b for b in range(bins) if (which == b).any()]
    return [x[which == b].mean() for b in full], [y[which == b].mean() for b in full]


DISTANCE_LABELS = {"above_top": "θ* above the strongest reference", "above_mean": "θ* above the mean reference"}


def distance_figure(table: pd.DataFrame, distance: str = "above_mean", metric: str = "signed",
                    points: bool = True, tests: pd.DataFrame | None = None) -> plt.Figure:
    """One distance from the references against, with `metric` "signed", the signed forecast error
    (a perfect forecast sits on the black zero rule, and for `above_mean` the dashed line is the
    saturated-item prediction's own error, which is -above_mean exactly), or with "score" that
    target's prediction score, -log2(its squared error / the saturated-item prediction's on the
    same target), where 0 on the black rule is the baseline. A 2 x 2 grid of domains sized for one
    column. Each protocol's mean in each quarter of the distance is joined up, over one faint point
    per target in the protocol's colour; `points` False leaves the means alone, because four
    protocols of points overlap into mud. The y-axis holds the
    middle 96% of the values, shared across panels; a point outside it is a triangle on the edge
    and is printed. `tests` (domain, protocol, p, from likelihood_ratio.slope_tests) puts the
    stars of each protocol's log likelihood ratio slope on the distance at the right end of its line."""
    assert distance in DISTANCE_LABELS, distance
    assert metric in ("signed", "score"), metric
    protocols = [p for p in PROTOCOLS[1:] if p != "glm_length"]
    table = with_baseline(table).dropna(subset=["base"])
    table = table.assign(y=table.signed if metric == "signed" else score(table.err, table.base))
    shown = table[table.protocol.isin(protocols)]
    lo, hi = np.percentile(shown.y, [2, 98])
    pad = 0.25 * (hi - lo)
    lo, hi = (float(lo - pad), float(hi + pad)) if points else (-2.0, 2.0)  # the means alone need no room for outliers
    # the binned panels hold four lines each and fit in a row; the point clouds need a 2 x 2 grid
    shape, size = ((1, len(DOMAINS)), (4.4, 1.6)) if not points else ((2, 2), (2.6, 2.4))
    fig, axes = plt.subplots(*shape, figsize=size, layout="constrained", sharex=True, sharey=True)
    for ax, domain in zip(np.ravel(axes), DOMAINS):
        panel = shown[shown.domain == domain]
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#d5d5d5", lw=0.6, zorder=0)
        if distance == "above_mean" and metric == "signed":  # the saturated-item prediction's own signed error, exactly
            span = np.array([shown[distance].min(), shown[distance].max()])
            ax.plot(span, -span, color="black", lw=0.8, ls="--", zorder=6)
        elif metric == "signed":  # against the strongest reference it is not a line, so its binned means
            b = table[(table.domain == domain) & (table.protocol == BASELINE)]
            bx, by = binned(b[distance].to_numpy(float), b.signed.to_numpy(float))
            ax.plot(bx, np.clip(by, lo, hi), color="black", lw=0.8, ls="--", zorder=6, clip_on=False)
        for p in protocols:
            d = panel[panel.protocol == p]
            y = d.y.to_numpy(float)
            inside = (y >= lo) & (y <= hi)
            if points:
                ax.scatter(d[distance][inside], y[inside], s=8, color=COLORS[p], alpha=0.22,
                           linewidths=0, edgecolors="none", zorder=3, clip_on=False)
            x = d[distance].to_numpy(float)  # the cloud is four protocols deep, so each one also gets its
            if len(x) >= 8:                  # mean over four equal-count bins of the distance, joined up
                bx, by = binned(x, y)
                ax.plot(bx, np.clip(by, lo, hi), color=COLORS[p], lw=1.2, marker="o", ms=3, zorder=5, clip_on=False)
                if tests is not None:
                    star = stars(float(tests.set_index(["domain", "protocol"]).loc[(domain, p), "p"]))
                    ax.text(bx[-1] + 0.15, np.clip(by[-1], lo, hi), star, color=COLORS[p], fontsize=9,
                            ha="left", va="center", zorder=7)
            for edge, tri in [(hi, "^"), (lo, "v")]:
                out = ((y > hi) if tri == "^" else (y < lo)) & points
                if out.any():
                    ax.scatter(d[distance][out], np.full(out.sum(), edge), s=14, marker=tri, alpha=0.55,
                               facecolor=COLORS[p], linewidths=0, edgecolors="none", zorder=4, clip_on=False)
                    print(f"distance_figure {distance} {metric}: {domain} {LABELS[p]} outside the axis: {[round(v, 1) for v in y[out]]}")
        if not points:
            ax.set_xlim(*((0.0, 2.0) if distance == "above_mean" else (-1.5, 1.5)))
            ax.set_xticks([0, 1, 2] if distance == "above_mean" else [-1, 0, 1])
            ax.set_yticks([-2, 2])
        else:
            ax.xaxis.set_major_locator(plt.MaxNLocator(3))
            ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        ax.set_ylim(lo, hi)
        if points:  # the point clouds read better with the axes crossing at the origin
            for side, span in (("left", (lo, hi)), ("bottom", ax.get_xlim())):
                ax.spines[side].set_position(("data", 0))
                ax.spines[side].set_bounds(*span)
        else:  # the binned row keeps its axes outside, with zero as a rule across the panel
            ax.axhline(0, color="black", lw=0.8, zorder=2)
        ax.text(0, 1.03, DOMAIN_LABELS[domain], transform=ax.transAxes, color="grey", ha="left", fontsize=7)
    fig.supxlabel(DISTANCE_LABELS[distance], fontsize=8)  # one label for the grid, not one per panel
    fig.supylabel("Predicted − true θ" if metric == "signed" else SCORE_LABEL.replace("\n", " "), fontsize=8)
    if not points:
        fig.set_layout_engine("constrained", w_pad=0.02)
    # the legend sits below the shared x label; bbox_inches="tight" on save keeps it in frame
    handles = [plt.Line2D([], [], color=COLORS[p], marker="o", ls="none", ms=3.5, alpha=0.55, label=LABELS[p])
               for p in protocols]
    if metric == "signed":
        handles.insert(0, plt.Line2D([], [], color="black", lw=0.8, ls="--", label=BASELINE_LABEL))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False)
    return fig


def sliding_curves(table: pd.DataFrame, distance: str = "above_top", k: int = 5, draws: int = 2000,
                   seed: int = 0) -> pd.DataFrame:
    """Per domain and protocol (the saturated-item prediction and the paper protocols), the mean
    squared error over each run of `k` targets adjacent in `distance` (the k nearest to one another,
    stepping one target at a time), placed at the run's mean distance, with the 95%
    percentile-bootstrap interval over the run's targets. The window is narrow where targets are
    dense and wide where they are sparse, and every point rests on k targets. Columns: domain,
    protocol, centre, first, last (the run's distances), n, mean, lo, hi."""
    protocols = [BASELINE, *[p for p in PROTOCOLS[1:] if p != "glm_length"]]
    rng = np.random.default_rng(seed)
    rows = []
    for domain in DOMAINS:
        for p in protocols:
            d = table[(table.domain == domain) & (table.protocol == p)].dropna(subset=["err"]).sort_values(distance)
            x, e = d[distance].to_numpy(float), d.err.to_numpy(float)
            for i in range(len(x) - k + 1):
                v = e[i:i + k]
                boot = v[rng.integers(0, k, (draws, k))].mean(axis=1)
                rows.append(dict(domain=domain, protocol=p, centre=float(x[i:i + k].mean()), first=float(x[i]),
                                 last=float(x[i + k - 1]), n=k, mean=float(v.mean()),
                                 lo=float(np.percentile(boot, 2.5)), hi=float(np.percentile(boot, 97.5))))
    return pd.DataFrame(rows)


def degradation_tests(table: pd.DataFrame, distance: str = "above_top") -> pd.DataFrame:
    """Per domain and protocol (the saturated-item prediction and the paper protocols), the
    least-squares slope of squared error on `distance` over the domain's targets and the two-sided
    t-test (n - 2 degrees of freedom) that it is 0; a positive slope: the protocol's error grows as
    targets sit further above the references. Targets share their task's references, so p is optimistic.
    Columns: domain, protocol, n, slope, se, p."""
    protocols = [BASELINE, *[p for p in PROTOCOLS[1:] if p != "glm_length"]]
    rows = []
    for domain in DOMAINS:
        for p in protocols:
            d = table[(table.domain == domain) & (table.protocol == p)].dropna(subset=["err", distance])
            fit = scipy_stats.linregress(d[distance], d.err)
            rows.append(dict(domain=domain, protocol=p, n=len(d), slope=fit.slope, se=fit.stderr, p=fit.pvalue))
    return pd.DataFrame(rows)


def sliding_distance_figure(curves: pd.DataFrame, distance: str = "above_top",
                            tests: pd.DataFrame | None = None) -> plt.Figure:
    """`curves` (sliding_curves) as one line per protocol with its interval shaded, one panel per
    domain with its own distance axis; the saturated-item prediction black and dashed. The squared
    error axis is shared and fits the lines, so a band may run off it. `tests` (degradation_tests)
    puts a star at the right end of each line whose squared error rises significantly with distance."""
    protocols = [BASELINE, *[p for p in PROTOCOLS[1:] if p != "glm_length"]]
    fig, axes = plt.subplots(1, len(DOMAINS), figsize=(4.4, 1.6), layout="constrained", sharey=True)
    top = float(curves["mean"].max()) * 1.1
    for ax, domain in zip(axes, DOMAINS):
        marks = []  # (y, x, star, colour) at each line's right end, spread apart below
        for p in protocols:
            c = curves[(curves.domain == domain) & (curves.protocol == p)]
            colour = "black" if p == BASELINE else COLORS[p]
            ax.fill_between(c.centre, c.lo, c.hi, color=colour, alpha=0.12, linewidth=0, zorder=2)
            ax.plot(c.centre, c["mean"], color=colour, lw=0.8 if p == BASELINE else 1, ls="--" if p == BASELINE else "-",
                    zorder=3)
            if tests is not None and len(c):
                star = stars(float(tests.set_index(["domain", "protocol"]).loc[(domain, p), "p"]))
                if star:
                    marks.append([float(c["mean"].iloc[-1]), float(c.centre.iloc[-1]), star, colour])
        gap = 0.07 * top  # at least one line of text between marks, moving the higher mark up
        marks.sort()
        for i in range(1, len(marks)):
            marks[i][0] = max(marks[i][0], marks[i - 1][0] + gap)
        for y, x, star, colour in marks:
            ax.text(x, y, " " + star, color=colour, fontsize=6, ha="left", va="center", zorder=5, clip_on=False)
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.text(0, 1.03, DOMAIN_LABELS[domain], transform=ax.transAxes, color="grey", ha="left", fontsize=7)
    for ax in axes:
        ax.set_ylim(0, top)
    fig.supxlabel(DISTANCE_LABELS[distance], fontsize=8)
    fig.supylabel("Squared error", fontsize=8)
    handles = [plt.Line2D([], [], color="black", lw=0.8, ls="--", label=BASELINE_LABEL),
               *[plt.Line2D([], [], color=COLORS[p], lw=1, label=LABELS[p]) for p in protocols[1:]]]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False, fontsize=7)
    return unclip_markers(fig)


def bin_boxes(table: pd.DataFrame, distance: str = "above_top", draws: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Per domain, distance bin (with_bins) and series (the saturated-item prediction and the paper
    protocols): the bin's mean distance, its targets, the mean squared error over them and its 95%
    percentile-bootstrap interval. Columns: domain, bin, protocol, x, n, mean, lo, hi."""
    series = [BASELINE, *[p for p in PROTOCOLS[1:] if p != "glm_length"]]
    rng = np.random.default_rng(seed)
    rows = []
    for (domain, b, p), g in table[table.protocol.isin(series)].dropna(subset=["bin"]).groupby(["domain", "bin", "protocol"]):
        v = g.err.to_numpy(float)
        boot = v[rng.integers(0, len(v), (draws, len(v)))].mean(axis=1)
        rows.append(dict(domain=domain, bin=b, protocol=p, x=float(g[distance].mean()), n=len(v), mean=float(v.mean()),
                         lo=float(np.percentile(boot, 2.5)), hi=float(np.percentile(boot, 97.5))))
    return pd.DataFrame(rows)


def bin_box_figure(boxes: pd.DataFrame, table: pd.DataFrame, distance: str = "above_top") -> plt.Figure:
    """`boxes` (bin_boxes): per domain and bin, one dot per protocol at the bin's mean squared error with a
    translucent box over its 95% interval, the protocols side by side within the bin. The saturated-item
    prediction is its least-squares line over the domain's targets in `table`, dashed black with its 95%
    confidence band. No test. Each domain has its own distance axis; the squared-error axis is shared and
    holds every box."""
    series = [p for p in PROTOCOLS[1:] if p != "glm_length"]
    colour = lambda p: BASELINE_COLOUR if p == BASELINE else COLORS[p]  # noqa: E731
    fig, axes = plt.subplots(1, len(DOMAINS), figsize=(4.4, 1.6), layout="constrained", sharey=True)
    for ax, domain in zip(axes, DOMAINS):
        panel = boxes[boxes.domain == domain]
        xs = panel.drop_duplicates("bin").x
        gap = float(np.diff(np.sort(xs.to_numpy(float))).min())  # the series share a bin's slot: fit the closest pair
        step = min(0.06 * float(xs.max() - xs.min()), 0.85 * gap / len(series))
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#d5d5d5", lw=0.6, zorder=0)
        base = table[(table.domain == domain) & (table.protocol == BASELINE)]
        line_x = np.linspace(float(xs.min()) - 0.2, float(xs.max()) + 0.2, 50)
        fit, band = fit_band(base[distance].to_numpy(float), base.err.to_numpy(float), line_x)
        ax.plot(line_x, fit, color="black", lw=0.8, ls="--", zorder=4)
        ax.fill_between(line_x, np.maximum(fit - band, 0), fit + band, color="black", alpha=0.12, linewidth=0, zorder=1)
        for k, p in enumerate(series):
            d = panel[panel.protocol == p]
            x = d.x.to_numpy(float) + step * (k - (len(series) - 1) / 2)
            for xi, lo, hi in zip(x, d.lo, d.hi):
                ax.add_patch(plt.Rectangle((xi - 0.4 * step, lo), 0.8 * step, hi - lo, facecolor=colour(p),
                                           edgecolor="none", alpha=0.35, zorder=2))
            ax.scatter(x, d["mean"], s=10, color=colour(p), linewidths=0, edgecolors="none", zorder=3)
        margin = max(step * len(series) / 2 + 0.05 * float(xs.max() - xs.min()), 0.2)  # the line runs 0.2 past the bins
        ax.set_xlim(float(xs.min()) - margin, float(xs.max()) + margin)
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.text(0, 1.03, DOMAIN_LABELS[domain], transform=ax.transAxes, color="grey", ha="left", fontsize=7)
    for ax in axes:
        ax.set_ylim(0, float(boxes[boxes.protocol.isin(series)].hi.max()) * 1.05)
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    fig.supxlabel(DISTANCE_LABELS[distance], fontsize=8)
    fig.supylabel("Squared error", fontsize=8)
    handles = [plt.Line2D([], [], color="black", lw=0.8, ls="--", label=BASELINE_LABEL),
               *[plt.Line2D([], [], color=colour(p), marker="o", ls="none", ms=3.5, label=LABELS[p]) for p in series]]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False, fontsize=7)
    return unclip_markers(fig)


def fit_band(x: np.ndarray, y: np.ndarray, xs: np.ndarray, degree: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """The least-squares polynomial of `degree` through (x, y) at `xs`, and its 95% confidence half-width."""
    X, Xs = np.vander(x, degree + 1), np.vander(xs, degree + 1)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid, dof = y - X @ beta, len(x) - degree - 1
    cov = resid @ resid / dof * np.linalg.inv(X.T @ X)
    return Xs @ beta, scipy_stats.t.ppf(0.975, dof) * np.sqrt(np.einsum("ij,jk,ik->i", Xs, cov, Xs))


def bin_distance_figure(table: pd.DataFrame, slopes: pd.DataFrame, bin_tests: pd.DataFrame,
                             distance: str = "above_top", metric: str = "signed", lines: bool = True,
                             log: bool = False, bars: bool = False, dot_slopes: bool = False,
                             fits: bool = False, degree: int = 1) -> plt.Figure:
    """A 1 x 4 row of domains: per protocol, the OLS line of the signed (`metric` "signed") or squared
    ("squared") error on `distance` over the domain's targets, drawn 0.2 beyond the outermost dots, with the
    stars of `slopes` (likelihood_ratio.slope_tests) at its right end; and one dot per `bin` (with_bins) at the mean
    distance and mean error of its targets, filled where `bin_tests` (likelihood_ratio.group_tests,
    group "bin") puts that bin's mean log likelihood ratio above 0 at p < .05, open where not. Targets without a bin are in
    the lines but not the dots. The saturated-item prediction
    is its own OLS line, black and dashed, shaded with the line's 95% confidence band (targets in a
    task share references, so the band is optimistic). `log` puts squared error on a log axis up to 10 (the lines are
    still fitted to the squared error, so they bend). `lines` False keeps only that one, and so drops the slope
    stars. `bars` gives each dot the 95% percentile-bootstrap interval of its mean over the bin's
    targets and, since a bin's protocols share one x, dodges them sideways by 0.07. A dot beyond
    the y-axis is a triangle on the edge."""
    assert metric in ("signed", "squared"), metric
    protocols = [p for p in PROTOCOLS[1:] if p != "glm_length"]
    rng = np.random.default_rng(0)
    y_col = "signed" if metric == "signed" else "err"
    assert not log or metric == "squared", "only squared error goes on a log axis"
    lo, hi = (-2.0, 2.0) if metric == "signed" else (0.05, 10.0) if log else (0.0, 5.0)
    slope_p = slopes.set_index(["domain", "protocol"]).p
    bin_p = bin_tests.set_index(["domain", "bin", "protocol"]).p if bin_tests is not None else None  # None: no tests
    if metric == "squared" and not log:  # the y-axis fits the dots and the baseline's line, shared by every panel
        tops = []
        for domain in DOMAINS:
            panel = table[table.domain == domain]
            dots = panel.groupby("bin")[distance].mean()
            ends = np.array([dots.min() - 0.2, dots.max() + 0.2])
            base = panel[panel.protocol == BASELINE]
            base_slope, base_intercept = np.polyfit(base[distance], base[y_col], 1)
            tops += [panel[panel.protocol.isin(protocols)].dropna(subset=["bin"]).groupby(["protocol", "bin"])[y_col].mean().max(),
                     (base_intercept + base_slope * ends).max()]
        hi = float(max(tops)) * 1.08
    fig, axes = plt.subplots(1, len(DOMAINS), figsize=(4.4, 1.6), layout="constrained", sharey=True)
    for ax, domain in zip(axes, DOMAINS):
        panel = table[table.domain == domain]
        dots = panel.groupby("bin")[distance].mean()  # the lines span the dots, so no panel's is cut by the y-axis
        span = np.array([dots.min() - 0.2, dots.max() + 0.2])
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#d5d5d5", lw=0.6, zorder=0)
        if not log:
            ax.axhline(0, color="black", lw=0.8, zorder=2)
        else:
            ax.set_yscale("log")
        labels = []  # (colour, slope) for the panel's list of slopes
        for p in [BASELINE, *protocols]:
            d = table[(table.domain == domain) & (table.protocol == p)]
            slope, intercept = np.polyfit(d[distance], d[y_col], 1)
            if p == BASELINE:
                labels.append(("black", slope))
            xs = np.linspace(*span, 50)  # enough points that the line bends with the band on a log axis
            if p == BASELINE or fits:  # the least-squares fit (degree `degree`) over the domain's targets, 95% band
                fit, band = fit_band(d[distance].to_numpy(float), d[y_col].to_numpy(float), xs, degree)
                colour = "black" if p == BASELINE else COLORS[p]
                ax.plot(xs, fit, color=colour, lw=0.8 if p == BASELINE else 1,
                        ls="--" if p == BASELINE else "-", zorder=6 if p == BASELINE else 4)
                ax.fill_between(xs, np.maximum(fit - band, lo), fit + band, color=colour,
                                alpha=0.12 if p == BASELINE else 0.1, linewidth=0, zorder=3)
            if p == BASELINE:
                continue
            if lines:
                ax.plot(xs, intercept + slope * xs, color=COLORS[p], lw=0.8, zorder=4)
                end = float(np.clip(intercept + slope * span[1], lo, hi))
                ax.text(span[1] + 0.05, end, stars(float(slope_p[(domain, p)])), color=COLORS[p], fontsize=8,
                        ha="left", va="center", zorder=7, clip_on=False)
            dot_xy = []
            for b, g in d.groupby("bin"):
                x, y = float(g[distance].mean()), float(g[y_col].mean())
                dot_xy.append((x, y))
                if bars:
                    x += 0.07 * (protocols.index(p) - (len(protocols) - 1) / 2)
                    v = g[y_col].to_numpy(float)
                    boot = v[rng.integers(0, len(v), (2000, len(v)))].mean(axis=1)
                    ax.plot([x, x], np.clip(np.percentile(boot, [2.5, 97.5]), lo, hi), color=COLORS[p], lw=0.8,
                            zorder=4, clip_on=False)
                out = y > hi or y < lo
                better = bin_p is None or float(bin_p[(domain, b, p)]) < 0.05  # without tests every dot is filled
                ax.scatter([x], [np.clip(y, lo, hi)], s=14, marker=("^" if y > hi else "v") if out else "o",
                           facecolors=COLORS[p] if better else "white", edgecolors=COLORS[p],
                           linewidths=0 if better else 0.8, zorder=5, clip_on=False)
                if out:
                    print(f"bin_distance_figure {metric}: {domain} bin {b} {LABELS[p]} outside the axis: {y:.1f}")
            if dot_slopes and len(dot_xy) >= 2:  # a thin least-squares line through the protocol's dots, its slope listed
                bx, by = np.array(dot_xy).T
                fit_slope, fit_intercept = np.polyfit(bx, by, 1)
                ends = np.array([bx.min(), bx.max()])
                ax.plot(ends, fit_intercept + fit_slope * ends, color=COLORS[p], lw=0.8, zorder=4)
                labels.append((COLORS[p], fit_slope))
        if dot_slopes:  # the slopes between the title and the plot, in the lines' colours, three to a row
            for i, (colour, value) in enumerate(labels):
                ax.text(0.33 * (i % 3), 1.12 - 0.1 * (i // 3), f"{value:+.2f}".replace("-", "−"), transform=ax.transAxes,
                        ha="left", va="bottom", fontsize=6, color=colour)
        ax.set_xlim(span[0] - 0.05, span[1] + 0.05)  # each domain's own range: its dots and lines
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.set_ylim(lo, hi)
        if metric == "squared" and not log:
            ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        else:
            ax.set_yticks([lo, hi] if metric == "signed" else [0.1, 1, 10], ["0.1", "1", "10"] if log else None)
        ax.text(0, 1.25 if dot_slopes else 1.03, DOMAIN_LABELS[domain], transform=ax.transAxes, color="grey", ha="left",
                fontsize=7)
    fig.supxlabel(DISTANCE_LABELS[distance], fontsize=8)
    fig.supylabel("Predicted − true θ" if metric == "signed" else "Squared error", fontsize=8)
    fig.set_layout_engine("constrained", w_pad=0.02)
    handles = [plt.Line2D([], [], color="black", lw=0.8, ls="--", label=BASELINE_LABEL),
               *[plt.Line2D([], [], color=COLORS[p], lw=0.8 if lines else 0, marker="o", ms=3.5, label=LABELS[p])
                 for p in protocols]]
    if bin_p is not None:
        handles += [plt.Line2D([], [], color="grey", ls="none", marker="o", ms=3.5, label="Better than baseline"),
                    plt.Line2D([], [], color="grey", ls="none", marker="o", ms=3.5, markerfacecolor="white",
                               markeredgewidth=0.8, label="Not better")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False)
    return unclip_markers(fig)


def reasoning_rows(table: pd.DataFrame) -> pd.DataFrame:
    """The 2025-I math and science targets with a `group` column, "reasoning" or "non-reasoning"
    (REASONING_2025_I), and the signed error. Their references, the 2024-II roster, are all
    non-reasoning models."""
    t = table[(table.semester == "2025-I") & table.domain.isin(["math", "science"])]
    group = np.where(t.model.isin(REASONING_2025_I), "reasoning", "non-reasoning")
    return t.assign(group=group, signed=t.theta_hat - t.theta)


def reasoning_scores(table: pd.DataFrame) -> pd.DataFrame:
    """Per domain, protocol and group: n, mse, baseline_mse and the prediction score from the
    group's pooled squared errors. Five reasoning and three non-reasoning targets, so no test."""
    out = []
    for (domain, protocol, group), g in with_baseline(reasoning_rows(table)).groupby(["domain", "protocol", "group"]):
        mse, base = float(g.err.mean()), float(g.base.mean())
        out.append(dict(domain=domain, protocol=protocol, group=group, n=len(g), mse=mse,
                        baseline_mse=base, score=float(score(mse, base))))
    return pd.DataFrame(out)


def reasoning_model_scores(table: pd.DataFrame) -> pd.DataFrame:
    """One prediction score per 2025-I target and protocol, its squared errors pooled over math
    and science as figure one pools over the four domains: model, group, protocol, score."""
    t = with_baseline(reasoning_rows(table)).dropna(subset=["base"])
    pooled = t.groupby(["model", "group", "protocol"])[["err", "base"]].mean().reset_index()
    return pooled.assign(score=score(pooled.err, pooled.base))


def reasoning_score_figure(table: pd.DataFrame) -> plt.Figure:
    """Prediction score of the 2025-I targets, math and science pooled per model: one x slot per
    protocol, two slots side by side for the five reasoning and the three non-reasoning targets,
    a large filled circle at the group's mean score, a translucent box over its 95% t-interval as
    in figure one (clipped to the axis; with three and five models it is wide) and the per-model
    scores as small points. Every mean and interval is printed."""
    protocols = [p for p in PROTOCOLS[1:] if p != "glm_length"]
    s = reasoning_model_scores(table)
    bottom, top = padded_axis(s[s.protocol.isin(protocols)].score.to_numpy(float))
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(3.4, 2.4), layout="constrained")
    ax.axhline(0, color="black", lw=0.8, zorder=1)  # the baseline reads as the figure's x axis
    for i, p in enumerate(protocols):
        for j, (group, alpha) in enumerate([("non-reasoning", 0.35), ("reasoning", 1.0)]):
            v = s[(s.protocol == p) & (s.group == group)].score.to_numpy(float)
            x = i + (j - 0.5) * 0.36
            mean, half = float(v.mean()), ci95(v)
            if np.isfinite(half):  # the box, clipped to the axis
                hi, lo = min(mean + half, top), max(mean - half, bottom)
                if hi > lo:
                    ax.add_patch(plt.Rectangle((x - 0.13, lo), 0.26, hi - lo, facecolor=COLORS[p],
                                               edgecolor="none", alpha=0.35 * alpha, zorder=2))
                print(f"reasoning_score_figure: {LABELS[p]} {group} mean {mean:.1f} [{mean - half:.1f}, {mean + half:.1f}]")
            ax.scatter(x + rng.uniform(-0.08, 0.08, len(v)), v, s=8, clip_on=False, color=COLORS[p],
                       alpha=0.5 * alpha + 0.2, linewidths=0, edgecolors="none", zorder=3)
            ax.scatter([x], [mean], s=30, color=COLORS[p], alpha=alpha, linewidths=0, edgecolors="none", zorder=5)
    ax.set_xticks(range(len(protocols)), [LABELS[p] for p in protocols])
    ax.tick_params(axis="x", rotation=90)
    ax.set_xlim(-0.6, len(protocols) - 0.4)
    ax.set_ylim(bottom, top)
    ax.spines["left"].set_bounds(bottom, top)
    ax.set_ylabel(SCORE_LABEL)
    fig.legend(handles=[plt.Line2D([], [], color="grey", marker="o", ls="none", ms=5, alpha=a, label=l)
                        for a, l in [(0.35, "Non-reasoning target (3)"), (1.0, "Reasoning target (5)")]],
               loc="outside lower center", ncol=2, frameon=False)
    return fig


def reasoning_scatter_figure(table: pd.DataFrame, domain: str = "math") -> plt.Figure:
    """Forecast against true theta* for one domain's 2025-I targets, one panel per protocol in a
    2 x 2 grid of small square panels whose axes cross at the origin: open
    circles for the three non-reasoning targets, filled diamonds for the five reasoning ones. The
    solid diagonal is a perfect forecast, the dashed horizontal line the saturated-item
    prediction (the references' mean theta*). Both axes are theta on the same square scale; a
    forecast outside it is drawn on the edge as a triangle and printed."""
    protocols = [p for p in PROTOCOLS[1:] if p != "glm_length"]
    t = reasoning_rows(table)
    t = t[(t.domain == domain) & t.protocol.isin(protocols)]
    ref = float(reference_theta(domain, "2025-I").mean())
    reach = max(abs(float(min(t.theta.min(), ref))), abs(float(max(t.theta.max(), ref)))) + 0.4
    lo, hi = -reach, reach  # symmetric, so 0 sits at the centre of both axes
    lo = min(lo, float(t.theta_hat.min()) - 0.2)  # reach down to the lowest forecast, so nothing sits on the floor
    # 2 x 2 of small square panels, the whole figure inside one column of a two-column paper
    fig, axes = plt.subplots(2, 2, figsize=(2.1, 2.2), layout="constrained", sharex=True, sharey=True)
    for ax, p in zip(axes.ravel(), protocols):
        ax.plot([lo, hi], [lo, hi], color="grey", lw=0.8, zorder=1)
        ax.axhline(ref, color="grey", lw=0.8, ls="--", zorder=1)
        for group, marker, filled in [("non-reasoning", "o", False), ("reasoning", "D", True)]:
            g = t[(t.protocol == p) & (t.group == group)]
            y = g.theta_hat.to_numpy(float)
            inside = (y >= lo) & (y <= hi)
            ax.scatter(g.theta[inside], y[inside], s=18, marker=marker, zorder=3, clip_on=False,
                       facecolor=COLORS[p] if filled else "none", edgecolor=COLORS[p], linewidth=0.8)
            if not inside.all():
                print(f"reasoning_scatter_figure: {domain} {LABELS[p]} {group} outside the axis: {[round(v, 1) for v in y[~inside]]}")
                for edge, tri in [(hi, "^"), (lo, "v")]:
                    out = (y > hi) if tri == "^" else (y < lo)
                    ax.scatter(g.theta[out], np.full(out.sum(), edge), s=20, marker=tri,
                               facecolor=COLORS[p], edgecolor="none", zorder=4, clip_on=False)
        ax.set_xlim(float(min(t.theta.min(), ref)) - 0.6, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        for side, span in (("left", (lo, hi)), ("bottom", ax.get_xlim())):  # the two axes cross at the origin
            ax.spines[side].set_position(("data", 0))
            ax.spines[side].set_bounds(*span)
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        ax.text(0, 1.03, LABELS[p], transform=ax.transAxes, color="grey", ha="left", fontsize=7)
    for ax in axes[1]:  # spines cross at the origin, so the labels are pinned outside the frame
        ax.set_xlabel("True θ*")
        ax.xaxis.set_label_coords(0.5, -0.16)
    for ax in axes[:, 0]:
        ax.set_ylabel("Predicted θ")
        ax.yaxis.set_label_coords(-0.16, 0.5)
    fig.legend(handles=[plt.Line2D([], [], color="grey", marker=m, ls="none", ms=4, markerfacecolor=f, label=l)
                        for m, f, l in [("o", "none", "Non-reasoning target (3)"), ("D", "grey", "Reasoning target (5)")]],
               loc="outside lower center", ncol=2, frameon=False)
    return unclip_markers(fig)


# the manuscript's simplified_reasoning.svg: non-reasoning targets as grey circles, reasoning targets
# as the pairwise judge's blue in diamonds, each marker with an outline, labelled in place
REASONING_STYLE = {"non-reasoning": dict(marker="o", face="#9a9a9a", edge="#333333", mean="#222222"),
                   "reasoning": dict(marker="D", face="#9cc3ee", edge="#2a78d6", mean="#2a78d6"),
                   # beside a reasoning-removed slot: the judge's dark shade shown, its light shade removed
                   "shown": dict(marker="D", face="#90afd5", edge="#2260ab", mean="#2260ab"),
                   "removed": dict(marker="D", face="#b4d0f0", edge="#6aa1e2", mean="#6aa1e2")}
# Absolute scoring's reasoning targets, in its green; reasoning removed in the same green, hollow
REASONING_DIRECT = dict(marker="D", face="#a4e0ca", edge="#1baf7a", mean="#1baf7a")


def reasoning_pairwise_figure(table: pd.DataFrame, domain: str = "math", metric: str = "score",
                              removed: pd.DataFrame | None = None,
                              tests: list[tuple[int, int, float]] | None = None,
                              protocol: str = "pairwise") -> plt.Figure:
    """The pairwise judge (or `protocol`, e.g. "direct", reasoning shown only) alone on one domain's 2025-I targets, whose references are all
    non-reasoning models, in the layout of the manuscript's simplified_reasoning.svg. Left: predicted
    against true capability on ordinary outside axes, the diagonal a perfect forecast, each group
    labelled beside its points. Right: one small marker per target and a large one at the group's
    mean; `metric` "score" plots each target's prediction score against the saturated-item
    prediction (halvings of the error), "error" its squared error (theta* - predicted)^2, the
    quantity the score is built from. The error version adds figure one's box, the mean's 95%
    t-interval floored at 0; the score version has none, as in the manuscript draft.
    `removed` (error only): the same judge's forecasts with the reasoning removed from the
    transcripts (domain, model, theta, theta_hat). The reasoning targets' are drawn hollow in the judge's
    light blue (the shown ones in its dark blue) on the left and as a third slot on the right; the non-reasoning targets carry no reasoning, so theirs
    equal the shown forecasts and are not drawn again.
    `tests` (error only): (slot, slot, p) per comparison, drawn as a bracket over the two slots with
    its stars or "n.s." on top, the wider bracket higher."""
    assert metric in ("score", "error"), metric
    assert removed is None or metric == "error", "the reasoning-removed slot is drawn in squared error only"
    assert not tests or metric == "error", "brackets are drawn in squared error only"
    styles = REASONING_STYLE if protocol == "pairwise" else {
        **REASONING_STYLE, "reasoning": REASONING_DIRECT, "shown": REASONING_DIRECT, "removed": REASONING_DIRECT}
    t = reasoning_rows(table)
    t = t[t.domain == domain]
    judge = with_baseline(t[t.protocol.isin([protocol, BASELINE])]).dropna(subset=["base"])
    judge = judge[judge.protocol == protocol].assign(score=lambda d: score(d.err, d.base))
    value = "score" if metric == "score" else "err"
    wide = removed is not None  # a third slot on the right needs room for its tick label
    fig, (left, right) = plt.subplots(1, 2, figsize=(3.8 if wide else 3.1, 1.5), layout="constrained",
                                      width_ratios=[1, 1.6 if wide else 1])
    reach = max(2.0, float(np.ceil(np.abs(judge[["theta", "theta_hat"]].to_numpy()).max() * 2) / 2)) + 0.2
    left.plot([-reach, reach], [-reach, reach], color="grey", lw=0.8, zorder=1)
    rng = np.random.default_rng(0)
    for i, group in enumerate(("non-reasoning", "reasoning")):
        g, st = judge[judge.group == group], styles["shown" if wide and group == "reasoning" else group]
        left.scatter(g.theta, g.theta_hat, s=22, marker=st["marker"], facecolor=st["face"], edgecolor=st["edge"],
                     linewidths=0.7, zorder=3, clip_on=False)
        name = "Non-reasoning\ntargets" if group == "non-reasoning" else "Reasoning\ntargets"
        if group == "non-reasoning":  # label to the left of the cloud, reasoning to the right
            left.text(g.theta.min() - 0.25, g.theta_hat.min(), name, ha="right", va="center", fontsize=5.5, color=st["edge"])
        else:
            left.text(g.theta.max() + 0.25, g.theta_hat.mean(), name, ha="left", va="center", fontsize=5.5, color=st["edge"])
        v = g[value].to_numpy(float)
        half = ci95(v)
        if metric == "error" and np.isfinite(half):  # figure one's box: the mean's 95% t-interval, floored at 0
            lo = max(v.mean() - half, 0.0)
            right.add_patch(plt.Rectangle((i - 0.25, lo), 0.5, v.mean() + half - lo, facecolor=st["face"],
                                          edgecolor="none", alpha=0.45, zorder=2))
        right.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=9, marker=st["marker"], facecolor=st["face"],
                      edgecolor=st["edge"], linewidths=0.5, zorder=3, clip_on=False)
        right.scatter([i], [v.mean()], s=34, marker=st["marker"], color=st["mean"], linewidths=0, zorder=5)
        print(f"reasoning_pairwise_figure {metric}: {domain} {group} mean {v.mean():.2f}, n = {len(v)}")
    if removed is not None:  # hollow, in their own colour: the reasoning targets with their reasoning removed
        st = styles["removed"]
        r = removed[(removed.domain == domain) & removed.model.isin(REASONING_2025_I)]
        left.scatter(r.theta, r.theta_hat, s=22, marker="D", facecolor="white", edgecolor=st["edge"],
                     linewidths=0.7, zorder=3, clip_on=False)
        left.text(r.theta.max() + 0.25, r.theta_hat.max(), "Reasoning\nremoved", ha="left", va="center",
                  fontsize=5.5, color=st["edge"])
        v = ((r.theta_hat - r.theta) ** 2).to_numpy(float)
        mean, half = v.mean(), ci95(v)
        lo = max(mean - half, 0.0)
        right.add_patch(plt.Rectangle((2 - 0.25, lo), 0.5, mean + half - lo, facecolor=st["face"],
                                      edgecolor="none", alpha=0.45, zorder=2))
        right.scatter(2 + rng.uniform(-0.12, 0.12, len(v)), v, s=9, marker="D", facecolor="white",
                      edgecolor=st["edge"], linewidths=0.5, zorder=3, clip_on=False)
        right.scatter([2], [mean], s=34, marker="D", facecolor="white", edgecolor=st["mean"], linewidths=1, zorder=5)
        print(f"reasoning_pairwise_figure {metric}: {domain} reasoning removed mean {mean:.2f}, n = {len(v)}")
    left.set_xlim(-reach, reach)
    left.set_ylim(-reach, reach)
    left.set_aspect("equal")
    left.set_xticks([-1.5, 0, 1.5], ["−1.5", "0", "1.5"])
    left.set_yticks([-2, 0, 2])
    # mathtext for the hat: Arial's combining circumflex renders as a prime
    left.set_xlabel(r"True capability ($\theta^*$)")
    left.set_ylabel(r"Predicted capability ($\hat{\theta}$)")
    slots = ["Non-\nreasoning\ntargets", "Reasoning\ntargets"] + (["Reasoning\nremoved"] if removed is not None else [])
    right.set_xticks(range(len(slots)), slots)
    right.set_xlim(-0.6, len(slots) - 0.4)
    if metric == "score":
        right.set_ylabel("Prediction score")
        right.yaxis.set_major_locator(plt.MultipleLocator(2))
    else:
        right.set_ylabel("Squared error")
        right.set_ylim(bottom=0)
        right.autoscale_view()  # the boxes are patches, which the y-axis does not follow on its own
        right.set_ylim(0, max(right.get_ylim()[1], *(p.get_y() + p.get_height() for p in right.patches)))
    if tests:  # brackets above the data; the axis line stops at its last tick below them
        top = right.get_ylim()[1]
        ticks = [t for t in right.get_yticks() if 0 <= t <= top]
        right.set_yticks(ticks)
        right.spines["left"].set_bounds(0, ticks[-1])
        for k, (a, b, p) in enumerate(sorted(tests, key=lambda t: abs(t[1] - t[0]))):
            y = top * (1.04 + 0.14 * k)
            right.plot([a, a, b, b], [y - 0.03 * top, y, y, y - 0.03 * top], color="black", lw=0.6, clip_on=False)
            right.text((a + b) / 2, y, stars(p) or "n.s.", ha="center", va="bottom", fontsize=7)
        right.set_ylim(0, top * (1.04 + 0.14 * len(tests)))
    return unclip_markers(fig)


def reasoning_figure(table: pd.DataFrame) -> plt.Figure:
    """The 2025-I task, whose references are all non-reasoning models: one panel per domain,
    one x slot per protocol, a point per target at its signed error — open circles for the three
    non-reasoning targets, filled diamonds for the five reasoning ones, as in the task figures —
    with a short bar at each group's mean. A point below 0 is an under-forecast. The axis spans
    the middle 90% of the errors; a point outside it is drawn on the edge as a triangle and
    printed, and a group mean outside it is printed instead of drawn."""
    protocols = [p for p in PROTOCOLS[1:] if p != "glm_length"]
    t = reasoning_rows(table)
    lo, hi = np.percentile(t[t.protocol.isin(protocols)].signed, [5, 95])
    lo, hi = float(min(lo, -0.5)) * 1.3, float(max(hi, 0.5)) * 1.3
    fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.4), layout="constrained", sharey=True)
    for ax, domain in zip(axes, ("math", "science")):
        ax.axhline(0, color="black", lw=0.8, zorder=1)
        for i, p in enumerate(protocols):
            for j, (group, marker, filled) in enumerate([("non-reasoning", "o", False), ("reasoning", "D", True)]):
                g = t[(t.domain == domain) & (t.protocol == p) & (t.group == group)]
                x, y = i + (j - 0.5) * 0.36, g.signed.to_numpy(float)
                inside = (y >= lo) & (y <= hi)
                ax.scatter(np.full(inside.sum(), x), y[inside], s=14, marker=marker, zorder=3, clip_on=False,
                           facecolor=COLORS[p] if filled else "none", edgecolor=COLORS[p], linewidth=0.7)
                if not inside.all():
                    print(f"reasoning_figure: {domain} {LABELS[p]} {group} outside the axis: {[round(v, 1) for v in y[~inside]]}")
                    for sign, edge, tri in [(1, hi, "^"), (-1, lo, "v")]:
                        out = (y > hi) if sign > 0 else (y < lo)
                        ax.scatter(np.full(out.sum(), x), np.full(out.sum(), edge), s=16, marker=tri,
                                   facecolor=COLORS[p], edgecolor="none", zorder=4, clip_on=False)
                if len(g) and lo <= y.mean() <= hi:
                    ax.plot([x - 0.13, x + 0.13], [y.mean()] * 2, color=COLORS[p], lw=1.2, zorder=4)
                elif len(g):
                    print(f"reasoning_figure: {domain} {LABELS[p]} {group} mean outside the axis: {y.mean():.1f}")
        ax.set_xticks(range(len(protocols)), [LABELS[p] for p in protocols])
        ax.tick_params(axis="x", rotation=90)
        ax.set_xlim(-0.6, len(protocols) - 0.4)
        ax.text(0, 1.06 + len(protocols) * 0.09, DOMAIN_LABELS[domain], transform=ax.transAxes, color="grey", ha="left")
    axes[0].set_ylabel("Predicted − true θ")
    fig.legend(handles=[plt.Line2D([], [], color="grey", marker=m, ls="none", ms=4, markerfacecolor=f, label=l)
                        for m, f, l in [("o", "none", "Non-reasoning target (3)"), ("D", "grey", "Reasoning target (5)")]],
               loc="outside lower center", ncol=2, frameon=False)
    return fig


def model_error_figure(forecasts: pd.DataFrame, variants: dict[str, str], tests: pd.DataFrame) -> plt.Figure:
    """Figure one in squared error: per target model of 2025-II and 2026-I, the mean squared error
    over its four domains, for the saturated-item prediction and each paper protocol. A large circle
    at the mean over models, the per-model values as small points and a box over the mean's 95%
    t-interval; no line at the saturated-item prediction's mean, which the stars do not test. Stars from `tests`
    (protocol, p): the log likelihood ratio against the references' distribution
    (likelihood_ratio.pooled_tests, a target model pooled over its domains). The axis holds the
    middle 95% of the per-model errors; a model beyond it is a triangle on the edge."""
    from analysis.challenges import signal_glm as sg
    pooled = sg.per_model(sg.rows(forecasts, variants))  # one row per model, one column per protocol
    protocols = [BASELINE, *sg.PAPER_PROTOCOLS]
    values = {p: pooled[p].dropna().to_numpy(float) for p in protocols}
    top = float(np.nanpercentile(np.concatenate(list(values.values())), 95)) * 1.15
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(2.7, 2.4), layout="constrained")
    for i, p in enumerate(protocols):
        v, colour = values[p], BASELINE_COLOUR if p == BASELINE else COLORS[p]
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
            star = stars(float(tests.set_index("protocol").loc[p, "p"]))
            ax.text(i, top, star or "n.s.", ha="center", va="bottom", fontsize=7, zorder=10)
    ax.set_xlim(-0.6, len(protocols) - 0.4)
    ax.set_ylim(0, top)
    ax.set_ylabel(ERROR_LABEL)
    ax.set_xticks(range(len(protocols)), [sg.two_lines(BASELINE_LABEL if p == BASELINE else LABELS[p]) for p in protocols])
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", multialignment="right")  # figure one
    return unclip_markers(fig)


def release_dates() -> dict[str, pd.Timestamp]:
    """Each model's release date from configs/models, keyed by its data-dir shortname (the config
    name's last segment, lowercased) and by the config's file stem; a date range keeps its start."""
    import yaml
    from pathlib import Path
    out = {}
    for path in Path("configs/models").glob("*/*.yml"):
        config = yaml.safe_load(path.read_text()) or {}
        if config.get("release_date"):
            date = pd.Timestamp(str(config["release_date"]).split("/")[0])
            out[path.stem] = date
            if "name" in config:
                out[config["name"].split("/")[-1].lower()] = date
    return out


def prediction_example_figure(forecasts: pd.DataFrame, variants: dict[str, str], domain: str, semester: str, model: str,
                              protocols: list[str], width_days: float = 90) -> tuple[plt.Figure, pd.DataFrame]:
    """What one target's forecast looks like under `protocols` and the saturated-item prediction: the
    references' θ* at their release dates (grey), the target's true θ* (black) and, from the target's
    release date, each rule's Student t distribution of θ* (likelihood_ratio.log_p) as a one-sided
    violin in the protocol's colour, cut 3 spreads either side of its forecast or at the axis, with a
    line across it at the forecast. The black line at θ* reaches the widest violin, so its length inside
    each violin is that distribution's density at the truth; a protocol's log LR is the log of its
    density there over the saturated-item prediction's. Every violin shares one width per unit of
    density, the widest `width_days` long, so a narrower distribution draws wider at its centre. Judge
    protocols use the task's best judge on references (forecast and spread from `forecasts`); feature
    protocols are refitted on the references, so SMALL_JUDGE_DATA and SMALL_JUDGE_EMBEDDINGS must be set.
    Also returns the drawn values: per rule its forecast, spread, squared error and log LR, and the judge."""
    import matplotlib.dates as mdates

    from analysis import loading
    from analysis import likelihood_ratio as lr
    from protocols.embeddings import EmbeddingRegression
    from analysis.stats import core_condition
    from protocols import PROTOCOLS
    task = loading.tasks(loading.domain(domain), [semester])[0]
    refs, truth, df = task.references.theta, float(task.truth[model]), len(task.references.theta) - 1
    judged = core_condition(forecasts).merge(lr.chosen_judges(forecasts, variants),
                                             on=["domain", "semester", "protocol", "model", "judge"])
    spreads = {BASELINE: (float(refs.mean()), float(np.sqrt(lr.loo_mse(lr.saturated, refs))))}
    judges = {BASELINE: "-"}
    for p in protocols:
        if p in ("pairwise", "direct"):
            row = judged[(judged.domain == domain) & (judged.semester == semester) & (judged.protocol == p)
                         & (judged.variant == variants[p]) & (judged.model == model)].iloc[0]
            spreads[p], judges[p] = (float(row.theta_hat), float(np.sqrt(row.judge_loo_mse))), row.judge
        else:
            feature = PROTOCOLS[p] if p != "embedding" else EmbeddingRegression(variants["embedding"])
            rule = lr.feature_rule(feature.estimator, pd.concat([feature.get_features(task.references),
                                                                 feature.get_features(task.test)]))
            spreads[p], judges[p] = (float(rule(refs, [model]).iloc[0]), float(np.sqrt(lr.loo_mse(rule, refs)))), "-"
    log_p = {p: float(scipy_stats.t.logpdf(truth, df, c, s)) for p, (c, s) in spreads.items()}
    values = pd.DataFrame([dict(protocol=p, judge=judges[p], theta=truth, forecast=c, spread=s, squared_error=(c - truth) ** 2,
                                log_lr=log_p[p] - log_p[BASELINE], references=len(refs)) for p, (c, s) in spreads.items()])
    print(values.round(3).to_string(index=False))
    dates = release_dates()
    fig, ax = plt.subplots(figsize=(2.8, 2.5), layout="constrained")
    ax.scatter([dates[m] for m in refs.index], refs.to_numpy(), s=14, marker=MARKERS[domain], color=BASELINE_COLOUR,
               linewidths=0, zorder=3)
    base_c, base_s = spreads[BASELINE]
    extent = [*refs, truth, base_c - 3 * base_s, base_c + 3 * base_s, *(c for c, _ in spreads.values())]
    lo, hi = np.floor(min(extent) * 2) / 2, np.ceil(max(extent) * 2) / 2
    x0 = mdates.date2num(dates[model])
    scale = width_days / max(scipy_stats.t.pdf(0, df, 0, s) for _, s in spreads.values())  # days per unit of density
    for p, (centre, spread) in spreads.items():
        colour = BASELINE_COLOUR if p == BASELINE else COLORS[p]
        ys = np.linspace(max(centre - 3 * spread, lo), min(centre + 3 * spread, hi), 400)  # the violin's cut
        edge = x0 + scale * scipy_stats.t.pdf(ys, df, centre, spread)
        ax.fill_betweenx(ys, x0, edge, color=colour, alpha=0.25, lw=0, zorder=1)
        ax.plot(edge, ys, color=colour, lw=1, zorder=2)
        ax.plot([x0, x0 + scale * scipy_stats.t.pdf(centre, df, centre, spread)], [centre] * 2, color=colour, lw=1, zorder=2)
    reach = max(scipy_stats.t.pdf(truth, df, c, s) for c, s in spreads.values())
    ax.plot([x0, x0 + scale * reach], [truth] * 2, color="black", lw=0.8, zorder=4)
    ax.scatter([x0], [truth], s=18, marker=MARKERS[domain], color="black", linewidths=0, zorder=5)
    for p in protocols:  # each protocol's squared error, left of the violins at its forecast
        centre = spreads[p][0]
        ax.text(x0 - 5, centre, f"squared error {(centre - truth) ** 2:.2f}", color=COLORS[p], ha="right", va="center", fontsize=7)
    ax.set_ylim(lo, hi)
    ax.set_yticks(np.arange(np.ceil(lo), np.floor(hi) + 1))
    ax.spines["left"].set_bounds(np.ceil(lo), np.floor(hi))
    ax.set_ylabel("Model Capability (θ)")
    ax.set_xlim(mdates.date2num(min(dates[m] for m in refs.index).to_period("Q").to_timestamp()), x0 + 1.05 * width_days)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    ticks = [t for t in ax.get_xticks() if ax.get_xlim()[0] <= t <= ax.get_xlim()[1]]
    ax.spines["bottom"].set_bounds(ticks[0], ticks[-1])
    ax.set_xlabel("Release date")
    point = lambda colour: plt.Line2D([], [], color=colour, marker=MARKERS[domain], ls="none", ms=4, markeredgewidth=0)  # noqa: E731
    patch = lambda colour: plt.matplotlib.patches.Patch(facecolor=colour, alpha=0.5, edgecolor="none")  # noqa: E731
    handles = [point(BASELINE_COLOUR), point("black"), *(patch(BASELINE_COLOUR if p == BASELINE else COLORS[p]) for p in spreads)]
    labels = ["References", "Target", BASELINE_LABEL,
              *(f"{LABELS[p]}, log LR {log_p[p] - log_p[BASELINE]:.2f}".replace("-", "−") for p in protocols)]
    fig.legend(handles=handles, labels=labels, loc="outside lower center", ncol=2, frameon=False)
    return unclip_markers(fig), values
