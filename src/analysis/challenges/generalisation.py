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
from analysis.plots import DOMAIN_LABELS, chosen, ci95, fix_axes_height
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
    score against 0. metric "error": value = the mean squared error, p the two-sided test of the
    paired difference from the saturated-item prediction's error, and lo/hi that difference's
    percentiles added to the baseline's error, so the interval clears the baseline exactly when the
    test is significant (the baseline's own rows keep the percentiles of its error). Either way
    the test comes from the same resamples as the interval, relative to the saturated-item
    prediction."""
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
        if metric == "score" or protocol == BASELINE:
            lo, hi = np.percentile(boot[keep], [2.5, 97.5]) if len(g) >= 3 else (np.nan, np.nan)
        else:  # the paired difference's interval, placed on the baseline's error: it clears the baseline exactly
            lo, hi = (base.mean() + np.percentile(gap[keep], [2.5, 97.5])) if len(g) >= 3 else (np.nan, np.nan)
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


BASELINE_COLOUR, BASELINE_LABEL = "#9a9a9a", "Saturated-item prediction"
ERROR_LABEL = "Mean squared error"


def time_figure(scores: pd.DataFrame, by: str = "protocol", stack: bool = False, box: bool = False,
                metric: str = "score") -> plt.Figure:
    """Prediction score over target semesters, in halvings of the error. `by` "protocol": one panel
    per protocol, one line per domain in the domain's colour and marker; `by` "domain": one panel
    per domain, one line per protocol in the protocol's colour, every panel on one axis so the
    domains compare. Each line carries a translucent band one standard error either side of the
    paired difference on the score scale, clipped to the panel, and the saturated-item prediction
    is the dashed line at 0. Stars sit above the panel, one row per line: the paired t-test of that
    task's squared errors against the baseline's. Coding and cyber start at 2025-II, the first
    semester they have."""
    assert by in ("protocol", "domain"), by
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
        wide = (0.0, float(np.ceil(np.nanpercentile(shown.value, 90) * 1.25 * 2) / 2))
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
        for i in range(len(semesters) - 1):  # thin rule between semesters
            ax.axvline(i + 0.5, color="lightgrey", lw=0.5, zorder=0)
        dodge = dict(zip(lines, np.linspace(-0.22, 0.22, len(lines))))  # the lines share a semester slot
        for k, line in enumerate(lines):
            d = (panel[panel.domain == line] if by == "protocol" else panel[panel.protocol == line]).sort_values("semester")
            xs = np.array([x[s] + dodge[line] for s in d.semester], float)
            inset = 0.02 * (top - bottom)  # the marker for an off-axis score sits just inside the spine
            raw = d.value.to_numpy(float)
            ys = np.clip(raw, bottom + inset, top - inset)
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
            ax.scatter(xs[out], ys[out], s=16, marker="v", color=colour(line), linewidths=0, edgecolors="none",
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
        ax.set_ylim(bottom, top)
        ax.yaxis.set_major_locator(plt.MaxNLocator(5, steps=[1, 2, 5, 10]))
        ax.spines["left"].set_bounds(bottom, top)
        title = DOMAIN_LABELS[key] if by == "domain" else LABELS[key]
        ax.text(0, 1.10, title,  # clear of a star on a bar that reaches the top
                transform=ax.transAxes, color="grey", ha="left", fontsize=7)
    ylabel = SCORE_LABEL if metric == "score" else ERROR_LABEL
    if stack:  # one label for the column, on the middle row
        np.atleast_1d(axes)[len(panels) // 2].set_ylabel(ylabel)
    else:
        np.atleast_1d(axes)[0].set_ylabel(ylabel)
    handles = [plt.Line2D([], [], color=colour(l), ls="none", marker=marker(l), ms=3.5, label=label(l)) for l in lines]
    fig.legend(handles=handles, loc="outside lower center", ncol=4 if by == "protocol" else (2 if metric == "score" else 3),
               frameon=False)
    return fig


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


DISTANCE_LABELS = {"above_top": "θ* above the strongest reference", "above_mean": "θ* above the mean reference"}


def distance_figure(table: pd.DataFrame, distance: str = "above_mean", metric: str = "signed",
                    points: bool = True) -> plt.Figure:
    """One distance from the references against, with `metric` "signed", the signed forecast error
    (a perfect forecast sits on the black zero rule, and for `above_mean` the dashed line is the
    saturated-item prediction's own error, which is -above_mean exactly), or with "score" that
    target's prediction score, -log2(its squared error / the saturated-item prediction's on the
    same target), where 0 on the black rule is the baseline. A 2 x 2 grid of domains sized for one
    column. Each protocol's mean in each quarter of the distance is joined up, over one faint point
    per target in the protocol's colour; `points` False leaves the means alone, because four
    protocols of points overlap into mud. The y-axis holds the
    middle 96% of the values, shared across panels; a point outside it is a triangle on the edge
    and is printed."""
    assert distance in DISTANCE_LABELS, distance
    assert metric in ("signed", "score"), metric
    protocols = [p for p in PROTOCOLS[1:] if p != "glm_length"]
    table = with_baseline(table).dropna(subset=["base"])
    table = table.assign(y=table.signed if metric == "signed" else score(table.err, table.base))
    shown = table[table.protocol.isin(protocols)]
    lo, hi = np.percentile(shown.y, [2, 98])
    pad = 0.25 * (hi - lo)
    lo, hi = float(lo - pad), float(hi + pad)
    # the binned panels hold four lines each and fit in a row; the point clouds need a 2 x 2 grid
    shape, size = ((1, len(DOMAINS)), (4.4, 1.6)) if not points else ((2, 2), (2.6, 2.4))
    fig, axes = plt.subplots(*shape, figsize=size, layout="constrained", sharex=True, sharey=True)
    for ax, domain in zip(np.ravel(axes), DOMAINS):
        panel = shown[shown.domain == domain]
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#d5d5d5", lw=0.6, zorder=0)
        if distance == "above_mean" and metric == "signed":  # the saturated-item prediction's own signed error, exactly
            span = np.array([shown[distance].min(), shown[distance].max()])
            ax.plot(span, -span, color="grey", lw=0.8, ls="--", zorder=1)
        for p in protocols:
            d = panel[panel.protocol == p]
            y = d.y.to_numpy(float)
            inside = (y >= lo) & (y <= hi)
            if points:
                ax.scatter(d[distance][inside], y[inside], s=8, color=COLORS[p], alpha=0.22,
                           linewidths=0, edgecolors="none", zorder=3, clip_on=False)
            x = d[distance].to_numpy(float)  # the cloud is four protocols deep, so each one also gets its
            if len(x) >= 8:                  # mean over four equal-count bins of the distance, joined up
                edges = np.quantile(x, np.linspace(0, 1, 5))
                which = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, 3)
                bx = [x[which == b].mean() for b in range(4) if (which == b).any()]
                by = [np.clip(y[which == b].mean(), lo, hi) for b in range(4) if (which == b).any()]
                ax.plot(bx, by, color=COLORS[p], lw=1.2, marker="o", ms=3, zorder=5, clip_on=False)
            for edge, tri in [(hi, "^"), (lo, "v")]:
                out = ((y > hi) if tri == "^" else (y < lo)) & points
                if out.any():
                    ax.scatter(d[distance][out], np.full(out.sum(), edge), s=14, marker=tri, alpha=0.55,
                               facecolor=COLORS[p], linewidths=0, edgecolors="none", zorder=4, clip_on=False)
                    print(f"distance_figure {distance} {metric}: {domain} {LABELS[p]} outside the axis: {[round(v, 1) for v in y[out]]}")
        if not points:  # the means alone need no room for the outlying targets
            lo, hi = -2.0, 2.0
            ax.set_xlim(0.0 if distance == "above_mean" else -2.0, 2.0)
            ax.set_xticks([0, 1, 2] if distance == "above_mean" else [-2, 0, 2])
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
    fig.legend(handles=[plt.Line2D([], [], color=COLORS[p], marker="o", ls="none", ms=3.5, alpha=0.55, label=LABELS[p])
                        for p in protocols], loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False)
    return fig


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
    return fig


# the manuscript's simplified_reasoning.svg: non-reasoning targets as grey circles, reasoning targets
# as the pairwise judge's blue in diamonds, each marker with an outline, labelled in place
REASONING_STYLE = {"non-reasoning": dict(marker="o", face="#9a9a9a", edge="#333333", mean="#222222"),
                   "reasoning": dict(marker="D", face="#9cc3ee", edge="#2a78d6", mean="#2a78d6")}


def reasoning_pairwise_figure(table: pd.DataFrame, domain: str = "math", metric: str = "score") -> plt.Figure:
    """The pairwise judge alone on one domain's 2025-I targets, whose references are all
    non-reasoning models, in the layout of the manuscript's simplified_reasoning.svg. Left: predicted
    against true capability on ordinary outside axes, the diagonal a perfect forecast, each group
    labelled beside its points. Right: one small marker per target and a large one at the group's
    mean; `metric` "score" plots each target's prediction score against the saturated-item
    prediction (halvings of the error), "error" its squared error (theta* - predicted)^2, the
    quantity the score is built from. The error version adds figure one's box, the mean's 95%
    t-interval floored at 0; the score version has none, as in the manuscript draft."""
    assert metric in ("score", "error"), metric
    t = reasoning_rows(table)
    t = t[t.domain == domain]
    judge = with_baseline(t[t.protocol.isin(["pairwise", BASELINE])]).dropna(subset=["base"])
    judge = judge[judge.protocol == "pairwise"].assign(score=lambda d: score(d.err, d.base))
    value = "score" if metric == "score" else "err"
    fig, (left, right) = plt.subplots(1, 2, figsize=(3.1, 1.5), layout="constrained", width_ratios=[1, 1])
    reach = max(2.0, float(np.ceil(np.abs(judge[["theta", "theta_hat"]].to_numpy()).max() * 2) / 2)) + 0.2
    left.plot([-reach, reach], [-reach, reach], color="grey", lw=0.8, zorder=1)
    rng = np.random.default_rng(0)
    for i, group in enumerate(("non-reasoning", "reasoning")):
        g, st = judge[judge.group == group], REASONING_STYLE[group]
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
    left.set_xlim(-reach, reach)
    left.set_ylim(-reach, reach)
    left.set_aspect("equal")
    left.set_xticks([-1.5, 0, 1.5], ["−1.5", "0", "1.5"])
    left.set_yticks([-2, 0, 2])
    # mathtext for the hat: Arial's combining circumflex renders as a prime
    left.set_xlabel(r"True capability ($\theta^*$)")
    left.set_ylabel(r"Predicted capability ($\hat{\theta}$)")
    right.set_xticks([0, 1], ["Non-\nreasoning\ntargets", "Reasoning\ntargets"])
    right.set_xlim(-0.6, 1.6)
    if metric == "score":
        right.set_ylabel("Prediction score")
        right.yaxis.set_major_locator(plt.MultipleLocator(2))
    else:
        right.set_ylabel("Squared error")
        right.set_ylim(bottom=0)
        right.autoscale_view()  # the boxes are patches, which the y-axis does not follow on its own
        right.set_ylim(0, max(right.get_ylim()[1], *(p.get_y() + p.get_height() for p in right.patches)))
    return fig


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


def model_error_figure(forecasts: pd.DataFrame, variants: dict[str, str]) -> tuple[plt.Figure, pd.DataFrame]:
    """Figure one in squared error: per target model of 2025-II and 2026-I, the mean squared error
    over its four domains, for the saturated-item prediction and each paper protocol. A large circle
    at the mean over models, the per-model values as small points, and a box: for a protocol the 95%
    t-interval of its paired difference from the saturated-item prediction, placed on that
    prediction's mean, so the box clears the dashed line exactly when the star is significant; for
    the saturated-item prediction its own 95% t-interval.
    and the saturated-item prediction's mean as a dashed line. Stars: paired t-test over models of
    each protocol's error against the saturated-item prediction's on the same models, so the test
    is still relative to the baseline. The axis holds the middle 95% of the per-model errors; a
    model beyond it is a triangle on the edge. Returns the figure and the tests (protocol, n,
    mean error, baseline's mean error on the same models, p)."""
    from analysis.challenges import signal_glm as sg
    pooled = sg.per_model(sg.rows(forecasts, variants))  # one row per model, one column per protocol
    protocols = [BASELINE, *sg.PAPER_PROTOCOLS]
    tests = []
    for p in sg.PAPER_PROTOCOLS:
        pair = pooled[[p, BASELINE]].dropna()
        tests.append(dict(protocol=p, n=len(pair), mean_error=float(pair[p].mean()),
                          baseline_error=float(pair[BASELINE].mean()),
                          p=float(scipy_stats.ttest_rel(pair[p], pair[BASELINE]).pvalue)))
    tests = pd.DataFrame(tests)
    values = {p: pooled[p].dropna().to_numpy(float) for p in protocols}
    top = float(np.nanpercentile(np.concatenate(list(values.values())), 95)) * 1.15
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(2.7, 2.4), layout="constrained")
    ax.axhline(float(values[BASELINE].mean()), color="grey", lw=0.8, ls="--", zorder=0)
    for i, p in enumerate(protocols):
        v, colour = values[p], BASELINE_COLOUR if p == BASELINE else COLORS[p]
        mean = float(v.mean())
        if p == BASELINE:
            lo, hi = mean - ci95(v), mean + ci95(v)
        else:  # the paired t-interval of the difference, placed on the baseline: it clears the dashed line exactly
            pair = pooled[[p, BASELINE]].dropna()  # when the paired t-test is significant
            diff = (pair[p] - pair[BASELINE]).to_numpy(float)
            lo, hi = pair[BASELINE].mean() + diff.mean() - ci95(diff), pair[BASELINE].mean() + diff.mean() + ci95(diff)
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
    ax.set_xticks(range(len(protocols)), [BASELINE_LABEL if p == BASELINE else LABELS[p] for p in protocols])
    ax.tick_params(axis="x", rotation=90)
    return fig, tests
