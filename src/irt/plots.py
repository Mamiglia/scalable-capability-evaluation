"""Figures for IRT validation, in the pilot house style (Arial 8pt, 400 dpi)."""

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#33322f", "#4a4844", "#d3d2c9", "#ffffff"
COLOR = {"overall_base": "#a8a79d", "model_base": "#1baf7a",
         "item_base": "#eb6834", "irt": "#2a78d6"}
LABEL = {"overall_base": "overall (no info)", "model_base": "model %-correct",
         "item_base": "item %-correct", "irt": "IRT (1PL)"}

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial"],
    "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "svg.fonttype": "none", "lines.linewidth": 1.4,
    "figure.dpi": 400, "savefig.dpi": 400,
})


def _style(ax):
    """Shared chrome: y grid, drop top/right spines, mute the rest."""
    ax.grid(axis="y", color=GRID, lw=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=MUTED)


def _save(fig, out_stub):
    """Tight layout, write <out_stub>.png and .svg."""
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(f"{out_stub}.{ext}", facecolor=SURFACE)
    plt.close(fig)


def _logx(ax, x):
    """Log x with integer ticks at the sampled x values."""
    ax.set_xscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(v)) for v in x])


def _curves(ax, agg, xcol, series):
    """Draw each series' mean +/- 1.96 SEM band; return the x values."""
    x = agg[xcol].to_numpy()
    for s in series:
        m = agg[f"{s}_mean"].to_numpy()
        e = 1.96 * agg[f"{s}_sem"].to_numpy()
        ax.fill_between(x, m - e, m + e, color=COLOR[s], alpha=0.15, lw=0)
        ax.plot(x, m, "-o", color=COLOR[s], label=LABEL[s], ms=4, zorder=3)
    return x


def _legend(ax):
    leg = ax.legend(frameon=False, loc="lower right")
    for t in leg.get_texts():
        t.set_color(INK2)


def curve_plot(agg, xcol, xlabel, out_stub,
               series=("overall_base", "model_base", "item_base", "irt")):
    """Plot each series' mean +/- 1.96 SEM vs xcol; save <out_stub>.png and .svg."""
    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    x = _curves(ax, agg, xcol, series)
    _logx(ax, x)
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel("held-out accuracy", color=INK2)
    _style(ax)
    _legend(ax)
    _save(fig, out_stub)


def sweep_gap_plot(agg, raw, xcol, xlabel, out_stub,
                   series=("overall_base", "model_base", "item_base", "irt"),
                   a="irt", b="model_base"):
    """
    Sweep figure led by the gap: curves on top, the a-minus-b gain below.

    The gap is paired per resample (raw has one row per resample), so its band is
    the SEM of the per-resample gap, not the two curves' SEMs differenced.
    """
    fig, (top, bot) = plt.subplots(
        2, 1, sharex=True, figsize=(4.6, 4.5),
        gridspec_kw={"height_ratios": [2.3, 1]},
    )
    x = _curves(top, agg, xcol, series)
    top.set_ylabel("held-out accuracy", color=INK2)
    top.tick_params(labelbottom=False)
    _style(top)
    _legend(top)

    g = raw.assign(gap=raw[a] - raw[b]).groupby(xcol)["gap"].agg(["mean", "sem"]).reset_index()
    gx, gm, ge = g[xcol].to_numpy(), g["mean"].to_numpy(), 1.96 * g["sem"].to_numpy()
    bot.axhline(0, color=MUTED, lw=0.8)
    bot.fill_between(gx, gm - ge, gm + ge, color=COLOR[a], alpha=0.15, lw=0)
    bot.plot(gx, gm, "-o", color=COLOR[a], ms=4, zorder=3)
    bot.set_ylabel(f"{LABEL[a]} −\n{LABEL[b]}", color=INK2)
    bot.set_xlabel(xlabel, color=INK2)
    _style(bot)
    _logx(bot, x)
    _save(fig, out_stub)


def _note(ax, note, note_xy):
    if note:
        va = "top" if note_xy[1] > 0.5 else "bottom"
        ax.text(*note_xy, note, transform=ax.transAxes, va=va, ha="left",
                color=MUTED, fontsize=7.5)


def scatter(df, xcol, ycol, xlabel, ylabel, out_stub,
            color=INK2, note=None, note_xy=(0.03, 0.97), identity=False):
    """Scatter ycol vs xcol in house style; optional y=x line and corner note."""
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    if identity:
        lo = min(df[xcol].min(), df[ycol].min())
        hi = max(df[xcol].max(), df[ycol].max())
        ax.plot([lo, hi], [lo, hi], color=MUTED, lw=0.8, zorder=1)
    ax.scatter(df[xcol], df[ycol], s=14, color=color, alpha=0.75, lw=0, zorder=3)
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    _note(ax, note, note_xy)
    _style(ax)
    _save(fig, out_stub)


def cat_scatter(df, xcol, ycol, group, order, colors, xlabel, ylabel, out_stub,
                note=None, note_xy=(0.03, 0.97)):
    """Scatter coloured by a categorical column, legend in `order`."""
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for g in order:
        d = df[df[group] == g]
        ax.scatter(d[xcol], d[ycol], s=16, color=colors[g], alpha=0.82, lw=0,
                   label=g, zorder=3)
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    _note(ax, note, note_xy)
    _style(ax)
    leg = ax.legend(frameon=False, loc="best", fontsize=6.5,
                    handletextpad=0.2, labelspacing=0.28)
    for t in leg.get_texts():
        t.set_color(INK2)
    _save(fig, out_stub)


def bars(labels, means, errs, colors, ylabel, out_stub, baseline=None):
    """Vertical bars with 1.96*SEM caps; optional dashed baseline."""
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    x = list(range(len(labels)))
    ax.bar(x, means, width=0.68, color=colors, zorder=3)
    ax.errorbar(x, means, yerr=[1.96 * e for e in errs], fmt="none",
                ecolor=INK2, elinewidth=1, capsize=3, zorder=4)
    if baseline is not None:
        ax.axhline(baseline, color=MUTED, lw=1, ls="--", zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel, color=INK2)
    _style(ax)
    _save(fig, out_stub)


def strip(labels, values, colors, ylabel, out_stub, seed=0):
    """Per label: jittered individual resamples + mean with 1.96*SEM. Y autoscales."""
    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    for i, (lab, v) in enumerate(zip(labels, values)):
        v = np.asarray(v, float)
        c = colors[i] if isinstance(colors, (list, tuple)) else colors
        ax.scatter(i + rng.uniform(-0.16, 0.16, len(v)), v, s=10, color=c, alpha=0.3, lw=0, zorder=2)
        m, e = v.mean(), 1.96 * v.std(ddof=1) / np.sqrt(len(v))
        ax.errorbar(i, m, yerr=e, fmt="o", color=c, ecolor=INK2, elinewidth=1.2, capsize=4,
                    ms=7, zorder=4, markeredgecolor="white", markeredgewidth=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel, color=INK2)
    ax.margins(x=0.08)
    _style(ax)
    _save(fig, out_stub)


def wright_map(theta, b, out_stub):
    """Item-person map: model ability and item difficulty share one latent scale."""
    fig, (lt, rt) = plt.subplots(1, 2, sharey=True, figsize=(4.6, 3.6))
    lt.hist(theta, bins=24, orientation="horizontal", color=COLOR["model_base"], alpha=0.85)
    rt.hist(b, bins=24, orientation="horizontal", color=COLOR["item_base"], alpha=0.85)
    lt.invert_xaxis()
    lt.set_xlabel("models (θ)", color=INK2)
    rt.set_xlabel("items (b)", color=INK2)
    lt.set_ylabel("latent scale", color=INK2)
    for ax in (lt, rt):
        _style(ax)
    fig.subplots_adjust(wspace=0.06)
    _save(fig, out_stub)
