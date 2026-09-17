"""Research question 3: when does the capability signal generalise?

Three analyses that only reshape `analysis/results/forecasts.csv`: the prediction score over
target semesters, the forecast error against how far a target sits from its references, and the
reasoning-model targets of 2025-I, whose references are all non-reasoning models. Transcripts
from one domain calibrated on another domain's theta* live in cross_domain.py.

Judge protocols are the best judge on references, one variant per protocol as in figure one.
Squared errors are pooled before a prediction score is taken: a target sitting on the references'
mean gives the saturated-item prediction a near-zero error, and its own score runs to minus
thousands.

Design: docs/superpowers/specs/2026-09-17-generalisation-design.md
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from analysis.challenges.signal_glm import COLORS, DOMAINS, LABELS, PROTOCOLS, score, stars
from analysis.plots import DOMAIN_LABELS, chosen, ci95, fix_axes_height
from analysis.stats import BASELINE, one_per_target

MARKERS = {"science": "o", "math": "s", "coding": "^", "cyber": "D"}  # CLAUDE.md, one shape per domain
# 2025-I targets whose transcripts carry a reasoning trace: deepseek-r1 and minimax-m1 always
# reason, the three Qwen3 models were collected with `reasoning: {enabled: true}` (configs/models).
REASONING_2025_I = ("deepseek-r1-0528", "minimax-m1", "qwen3-235b-a22b", "qwen3-30b-a3b", "qwen3-32b")
SCORE_FLOOR = -100  # scores below this are drawn on the axis floor as triangles


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


def task_scores(table: pd.DataFrame) -> pd.DataFrame:
    """Prediction score per (domain, semester, protocol) from the task's targets: n, mse,
    baseline_mse, score = 100 * (1 - mse / baseline_mse), `p`, the paired t-test of the
    protocol's squared errors against the saturated-item prediction's over the same targets, and
    `lo`, `hi`: that paired difference's 95% t-interval carried onto the score scale, the same
    interval the results timelines draw. It is the dual of `p`, so a band clear of 0 agrees with
    the star."""
    out = []
    for (domain, semester, protocol), g in with_baseline(table).dropna(subset=["base"]).groupby(["domain", "semester", "protocol"]):
        mse, base = float(g.err.mean()), float(g.base.mean())
        pval = float(scipy_stats.ttest_rel(g.err, g.base).pvalue) if protocol != BASELINE and len(g) >= 3 else np.nan
        d = (g.err - g.base).to_numpy(float)
        half = float(scipy_stats.t.ppf(0.975, len(d) - 1) * np.std(d, ddof=1) / np.sqrt(len(d))) if len(d) >= 3 else np.nan
        out.append(dict(domain=domain, semester=semester, protocol=protocol, n=len(g), mse=mse,
                        baseline_mse=base, score=float(score(mse, base)), p=pval,
                        lo=-100 * (d.mean() + half) / base, hi=-100 * (d.mean() - half) / base))
    return pd.DataFrame(out).sort_values(["domain", "semester", "protocol"], ignore_index=True)


def time_figure(scores: pd.DataFrame) -> plt.Figure:
    """Prediction score over target semesters: one panel per domain (width by its number of
    semesters), x = target semester, y = score, one line with filled circles per protocol in the
    protocol's colour, the saturated-item prediction as a dashed line at 0, and a translucent
    band along each line for the 95% t-interval of the paired difference on the score scale
    (clipped to the axis). Stars sit above the panel, one row per protocol in its
    colour: the paired t-test of that task's squared errors against the
    baseline's. Scores below SCORE_FLOOR are drawn as downward triangles on the floor and printed."""
    protocols = [p for p in PROTOCOLS[1:] if p != "glm_length"]
    domains = [d for d in DOMAINS if d in set(scores.domain)]
    counts = [scores[scores.domain == d].semester.nunique() for d in domains]
    fig, axes = plt.subplots(1, len(domains), figsize=(1.0 + 0.45 * sum(counts) + 0.5 * len(domains), 2.4),
                             layout="constrained", sharey=True, width_ratios=counts)
    for ax, domain in zip(np.atleast_1d(axes), domains):
        d = scores[scores.domain == domain]
        semesters = sorted(d.semester.unique())
        x = {s: i for i, s in enumerate(semesters)}
        ax.axhline(0, color="grey", lw=0.8, ls="--", zorder=1)
        for i in range(len(semesters) - 1):  # thin rule between semesters
            ax.axvline(i + 0.5, color="lightgrey", lw=0.5, zorder=0)
        for k, p in enumerate(protocols):
            s = d[d.protocol == p].set_index("semester").reindex(semesters)
            xs, ys = np.array([x[i] for i in semesters], float), s.score.to_numpy(float)
            below = ys < SCORE_FLOOR
            if below.any():
                print(f"time_figure: {domain} {p} off the bottom: {[round(v) for v in ys[below]]}")
            drawn = np.where(below, SCORE_FLOOR, ys)
            band = (np.clip(s.lo.to_numpy(float), SCORE_FLOOR, 100), np.clip(s.hi.to_numpy(float), SCORE_FLOOR, 100))
            ax.fill_between(xs, *band, color=COLORS[p], alpha=0.18, linewidth=0, zorder=1)
            ax.plot(xs, drawn, color=COLORS[p], lw=1.0, zorder=2)
            ax.scatter(xs[~below], drawn[~below], s=16, color=COLORS[p], linewidth=0, zorder=3)
            ax.scatter(xs[below], drawn[below], s=22, marker="v", color=COLORS[p], linewidth=0, zorder=3, clip_on=False)
            for xi, pval in zip(xs, s.p.to_numpy(float)):  # one row of marks per protocol, all above the panel
                if np.isfinite(pval) and stars(pval):
                    ax.text(xi, 1.02 + k * 0.09, stars(pval), ha="center", va="bottom", fontsize=6, color=COLORS[p],
                            zorder=4, clip_on=False, transform=ax.get_xaxis_transform())
        ax.set_xticks(range(len(semesters)), semesters)
        ax.tick_params(axis="x", rotation=90)
        ax.set_xlim(-0.4, len(semesters) - 0.6)
        ax.set_ylim(SCORE_FLOOR, 100)
        ax.set_yticks([SCORE_FLOOR, -50, 0, 50, 100], [f"<{SCORE_FLOOR}", "-50", "0", "50", "100"])
        ax.spines["left"].set_bounds(SCORE_FLOOR, 100)
        ax.text(0, 1.06 + len(protocols) * 0.09, DOMAIN_LABELS[domain], transform=ax.transAxes, color="grey", ha="left")
    np.atleast_1d(axes)[0].set_ylabel("Prediction score (%)")
    fig.legend(handles=[plt.Line2D([], [], color=COLORS[p], lw=1.0, marker="o", ms=4, label=LABELS[p]) for p in protocols],
               loc="outside lower center", ncol=2, frameon=False)
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


def distance_figure(table: pd.DataFrame, slopes: pd.DataFrame) -> plt.Figure:
    """Signed forecast error against distance from the references: one row per distance
    (top: above the strongest reference, bottom: above the references' mean), one panel per
    protocol, one point per target in the protocol's colour with the domain's marker shape.
    A perfect forecast sits on 0; the dashed line is the saturated-item prediction's own error,
    which is -above_mean exactly. Each panel prints its OLS slope. The axis spans the middle 98%
    of the signed errors; a forecast outside it is drawn as a triangle on the edge and printed."""
    protocols = [p for p in PROTOCOLS[1:] if p != "glm_length"]
    lo, hi = np.percentile(table[table.protocol.isin(protocols)].signed, [1, 99])
    lo, hi = float(min(lo, -0.5)) * 1.2, float(max(hi, 0.5)) * 1.2
    fig, axes = plt.subplots(2, len(protocols), figsize=(1.5 * len(protocols) + 0.6, 4.2), layout="constrained",
                             sharex="row", sharey=True)
    for row, distance in enumerate(("above_top", "above_mean")):
        for col, p in enumerate(protocols):
            ax = axes[row, col]
            g = table[table.protocol == p]
            ax.axhline(0, color="grey", lw=0.8, zorder=1)
            if distance == "above_mean":  # the saturated-item prediction's own signed error, exactly
                span = np.array([table[distance].min(), table[distance].max()])
                ax.plot(span, -span, color="grey", lw=0.8, ls="--", zorder=1)
            for domain in DOMAINS:
                d = g[g.domain == domain]
                y = d.signed.to_numpy(float)
                inside = (y >= lo) & (y <= hi)
                ax.scatter(d[distance][inside], y[inside], s=12, marker=MARKERS[domain], facecolor="none",
                           edgecolor=COLORS[p], linewidth=0.7, zorder=3)
                for sign, edge, marker in [(1, hi, "^"), (-1, lo, "v")]:
                    out = (y > hi) if sign > 0 else (y < lo)
                    if out.any():
                        ax.scatter(d[distance][out], np.full(out.sum(), edge), s=16, marker=marker,
                                   facecolor=COLORS[p], edgecolor="none", zorder=4, clip_on=False)
                        print(f"distance_figure: {LABELS[p]} {domain} outside the axis: {[round(v, 1) for v in y[out]]}")
            ax.set_ylim(lo, hi)
            s = slopes[(slopes.protocol == p) & (slopes.distance == distance)]
            if len(s):
                ax.text(0.03, 0.03, f"slope {float(s.slope.iloc[0]):.2f}", transform=ax.transAxes, fontsize=6, color="grey")
            if row == 0:
                ax.text(0, 1.03, LABELS[p], transform=ax.transAxes, color="grey", ha="left", fontsize=7)
        axes[row, 0].set_ylabel("Forecast − true θ")
        axes[row, 0].set_xlabel("θ* above the strongest reference" if distance == "above_top" else "θ* above the mean reference")
    fig.legend(handles=[plt.Line2D([], [], color="grey", marker=MARKERS[d], ls="none", ms=4, label=DOMAIN_LABELS[d]) for d in DOMAINS],
               loc="outside lower center", ncol=4, frameon=False)
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
    scores as small points. Scores below the axis are drawn on the floor as triangles and printed."""
    protocols = [p for p in PROTOCOLS[1:] if p != "glm_length"]
    s = reasoning_model_scores(table)
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(3.4, 2.4), layout="constrained")
    ax.axhline(0, color="grey", lw=0.8, ls="--", zorder=1)
    for i, p in enumerate(protocols):
        for j, (group, alpha) in enumerate([("non-reasoning", 0.35), ("reasoning", 1.0)]):
            v = s[(s.protocol == p) & (s.group == group)].score.to_numpy(float)
            x = i + (j - 0.5) * 0.36
            below = v < SCORE_FLOOR
            if below.any():
                print(f"reasoning_score_figure: {LABELS[p]} {group} off the bottom: {[round(u) for u in v[below]]}")
            mean, half = float(v.mean()), ci95(v)
            if np.isfinite(half):  # the box, clipped to the axis
                top, bottom = min(mean + half, 100), max(mean - half, SCORE_FLOOR)
                if top > bottom:
                    ax.add_patch(plt.Rectangle((x - 0.13, bottom), 0.26, top - bottom, facecolor=COLORS[p],
                                               edgecolor="none", alpha=0.35 * alpha, zorder=2))
                print(f"reasoning_score_figure: {LABELS[p]} {group} mean {mean:.0f} [{mean - half:.0f}, {mean + half:.0f}]")
            ax.scatter(x + rng.uniform(-0.08, 0.08, len(v)), np.clip(v, SCORE_FLOOR, None), s=8,
                       color=COLORS[p], alpha=0.5 * alpha + 0.2, linewidth=0, zorder=3, clip_on=False)
            ax.scatter([x], [max(mean, SCORE_FLOOR)], s=30, color=COLORS[p], alpha=alpha, linewidth=0, zorder=5,
                       marker="v" if mean < SCORE_FLOOR else "o", clip_on=False)
            if mean < SCORE_FLOOR:
                print(f"reasoning_score_figure: {LABELS[p]} {group} mean off the bottom: {mean:.0f}")
    ax.set_xticks(range(len(protocols)), [LABELS[p] for p in protocols])
    ax.tick_params(axis="x", rotation=90)
    ax.set_xlim(-0.6, len(protocols) - 0.4)
    ax.set_ylim(SCORE_FLOOR, 100)
    ax.set_yticks([SCORE_FLOOR, -50, 0, 50, 100], [f"<{SCORE_FLOOR}", "-50", "0", "50", "100"])
    ax.spines["left"].set_bounds(SCORE_FLOOR, 100)
    ax.set_ylabel("Prediction score (%)")
    fig.legend(handles=[plt.Line2D([], [], color="grey", marker="o", ls="none", ms=5, alpha=a, label=l)
                        for a, l in [(0.35, "Non-reasoning target (3)"), (1.0, "Reasoning target (5)")]],
               loc="outside lower center", ncol=2, frameon=False)
    return fig


def reasoning_scatter_figure(table: pd.DataFrame, domain: str = "math") -> plt.Figure:
    """Forecast against true theta* for one domain's 2025-I targets, one panel per protocol: open
    circles for the three non-reasoning targets, filled diamonds for the five reasoning ones. The
    solid diagonal is a perfect forecast, the dashed horizontal line the saturated-item
    prediction (the references' mean theta*). Both axes are theta on the same square scale; a
    forecast outside it is drawn on the edge as a triangle and printed."""
    protocols = [p for p in PROTOCOLS[1:] if p != "glm_length"]
    t = reasoning_rows(table)
    t = t[(t.domain == domain) & t.protocol.isin(protocols)]
    ref = float(reference_theta(domain, "2025-I").mean())
    lo = float(min(t.theta.min(), ref)) - 0.6
    hi = float(max(t.theta.max(), ref)) + 0.6
    lo = min(lo, float(t.theta_hat.min()) - 0.2)  # reach down to the lowest forecast, so nothing sits on the floor
    fig, axes = plt.subplots(1, len(protocols), figsize=(1.0 * len(protocols) + 0.45, 1.8), layout="constrained", sharey=True)
    for ax, p in zip(axes, protocols):
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
        ax.set_xlabel("True θ*")
        ax.text(0, 1.03, LABELS[p], transform=ax.transAxes, color="grey", ha="left", fontsize=7)
    axes[0].set_ylabel("Forecast θ")
    fig.legend(handles=[plt.Line2D([], [], color="grey", marker=m, ls="none", ms=4, markerfacecolor=f, label=l)
                        for m, f, l in [("o", "none", "Non-reasoning target (3)"), ("D", "grey", "Reasoning target (5)")]],
               loc="outside lower center", ncol=2, frameon=False)
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
        ax.axhline(0, color="grey", lw=0.8, zorder=1)
        for i, p in enumerate(protocols):
            for j, (group, marker, filled) in enumerate([("non-reasoning", "o", False), ("reasoning", "D", True)]):
                g = t[(t.domain == domain) & (t.protocol == p) & (t.group == group)]
                x, y = i + (j - 0.5) * 0.36, g.signed.to_numpy(float)
                inside = (y >= lo) & (y <= hi)
                ax.scatter(np.full(inside.sum(), x), y[inside], s=14, marker=marker, zorder=3,
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
    axes[0].set_ylabel("Forecast − true θ")
    fig.legend(handles=[plt.Line2D([], [], color="grey", marker=m, ls="none", ms=4, markerfacecolor=f, label=l)
                        for m, f, l in [("o", "none", "Non-reasoning target (3)"), ("D", "grey", "Reasoning target (5)")]],
               loc="outside lower center", ncol=2, frameon=False)
    return fig
