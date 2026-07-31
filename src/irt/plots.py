"""Figures for IRT validation, in the pilot house style (Arial, 400 dpi)."""

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#33322f", "#4a4844", "#d3d2c9", "#ffffff"
COLOR = {"overall_base": "#a8a79d", "model_base": "#1baf7a",
         "item_base": "#eb6834", "irt": "#2a78d6"}
LABEL = {"overall_base": "average accuracy", "model_base": "model %-correct",
         "item_base": "item %-correct", "irt": "IRT (1PL)"}

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial"],
    "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "svg.fonttype": "none", "lines.linewidth": 1.6,
    "figure.dpi": 400, "savefig.dpi": 400,
})


def _ci(a):
    """Mean and empirical 95% interval (2.5, 97.5 percentiles)."""
    a = np.asarray(a, float)
    return a.mean(), np.percentile(a, 2.5), np.percentile(a, 97.5)


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


def _band(ax, raw, xcol, s, label=None):
    """Line at the mean with an empirical-95% band, from per-resample rows."""
    xs = np.array(sorted(raw[xcol].unique()))
    m, lo, hi = np.array([_ci(raw.loc[raw[xcol] == x, s]) for x in xs]).T
    ax.fill_between(xs, lo, hi, color=COLOR[s], alpha=0.15, lw=0)
    ax.plot(xs, m, "-o", color=COLOR[s], label=label, ms=5, zorder=3)
    return xs


def _legend(ax, loc="lower right"):
    leg = ax.legend(frameon=False, loc=loc)
    for t in leg.get_texts():
        t.set_color(INK2)


def _note(ax, note, note_xy):
    if note:
        va = "top" if note_xy[1] > 0.5 else "bottom"
        ax.text(*note_xy, note, transform=ax.transAxes, va=va, ha="left",
                color=MUTED, fontsize=8)


def curve_plot(raw, xcol, xlabel, out_stub,
               series=("overall_base", "model_base", "item_base", "irt")):
    """Each series' mean + empirical-95% band vs xcol, from per-resample rows."""
    fig, ax = plt.subplots(figsize=(3.7, 2.6))
    for s in series:
        x = _band(ax, raw, xcol, s, LABEL[s])
    _logx(ax, x)
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel("held-out accuracy", color=INK2)
    _style(ax)
    _legend(ax)
    _save(fig, out_stub)


def sweep_gap_plot(raw, xcol, xlabel, out_stub,
                   series=("overall_base", "model_base", "item_base", "irt"),
                   a="irt", b="model_base"):
    """Sweep led by the gap: curves on top, the a-minus-b gain (empirical CI) below."""
    fig, (top, bot) = plt.subplots(
        2, 1, sharex=True, figsize=(3.8, 3.7),
        gridspec_kw={"height_ratios": [2.3, 1]},
    )
    for s in series:
        x = _band(top, raw, xcol, s, LABEL[s])
    top.set_ylabel("held-out accuracy", color=INK2)
    top.tick_params(labelbottom=False)
    _style(top)
    _legend(top)

    gap = raw.assign(_g=raw[a] - raw[b])
    xs = np.array(sorted(gap[xcol].unique()))
    m, lo, hi = np.array([_ci(gap.loc[gap[xcol] == v, "_g"]) for v in xs]).T
    bot.axhline(0, color=MUTED, lw=0.8)
    bot.fill_between(xs, lo, hi, color=COLOR[a], alpha=0.15, lw=0)
    bot.plot(xs, m, "-o", color=COLOR[a], ms=5, zorder=3)
    bot.set_ylabel(f"{LABEL[a]} −\n{LABEL[b]}", color=INK2)
    bot.set_xlabel(xlabel, color=INK2)
    _style(bot)
    _logx(bot, x)
    _save(fig, out_stub)


def strip(labels, values, colors, ylabel, out_stub, seed=0):
    """Per label: jittered individual resamples + mean with empirical 95% CI. Y autoscales."""
    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(figsize=(3.9, 2.7))
    for i, (lab, v) in enumerate(zip(labels, values)):
        v = np.asarray(v, float)
        c = colors[i] if isinstance(colors, (list, tuple)) else colors
        ax.scatter(i + rng.uniform(-0.16, 0.16, len(v)), v, s=11, color=c, alpha=0.3, lw=0, zorder=2)
        m, lo, hi = _ci(v)
        ax.errorbar(i, m, yerr=[[m - lo], [hi - m]], fmt="o", color=c, ecolor=INK2,
                    elinewidth=1.2, capsize=4, ms=7, zorder=4,
                    markeredgecolor="white", markeredgewidth=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel, color=INK2)
    ax.margins(x=0.08)
    _style(ax)
    _save(fig, out_stub)


def scatter(df, xcol, ycol, xlabel, ylabel, out_stub,
            color=INK2, note=None, note_xy=(0.03, 0.97), identity=False):
    """Scatter ycol vs xcol; optional y=x line and corner note."""
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    if identity:
        lo = min(df[xcol].min(), df[ycol].min())
        hi = max(df[xcol].max(), df[ycol].max())
        ax.plot([lo, hi], [lo, hi], color=MUTED, lw=0.8, zorder=1)
    ax.scatter(df[xcol], df[ycol], s=16, color=color, alpha=0.75, lw=0, zorder=3)
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    _note(ax, note, note_xy)
    _style(ax)
    _save(fig, out_stub)


def cat_scatter(df, xcol, ycol, group, order, colors, xlabel, ylabel, out_stub,
                note=None, note_xy=(0.03, 0.97)):
    """Scatter coloured by a categorical column, legend in `order`."""
    fig, ax = plt.subplots(figsize=(4.2, 2.9))
    for g in order:
        d = df[df[group] == g]
        ax.scatter(d[xcol], d[ycol], s=18, color=colors[g], alpha=0.82, lw=0, label=g, zorder=3)
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    _note(ax, note, note_xy)
    _style(ax)
    leg = ax.legend(frameon=False, loc="best", fontsize=7, handletextpad=0.2, labelspacing=0.3)
    for t in leg.get_texts():
        t.set_color(INK2)
    _save(fig, out_stub)


def wright_map(theta, b, out_stub):
    """Item-person map: model ability and item difficulty share one latent scale."""
    fig, (lt, rt) = plt.subplots(1, 2, sharey=True, figsize=(3.7, 2.9))
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


def heatmap(mat, out_stub, xlabel, ylabel, cbar_label="held-out accuracy (IRT)"):
    """Grid of IRT accuracy over two sweep axes; annotated, one-hue (Blues)."""
    fig, ax = plt.subplots(figsize=(4.2, 3.3))
    vals = mat.values.astype(float)
    vmin, vmax = np.nanmin(vals), np.nanmax(vals)
    im = ax.imshow(vals, aspect="auto", cmap="Blues", origin="lower", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(mat.columns)))
    ax.set_xticklabels([str(c) for c in mat.columns])
    ax.set_yticks(range(len(mat.index)))
    ax.set_yticklabels([str(r) for r in mat.index])
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    thr = vmin + 0.6 * (vmax - vmin)
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if v > thr else INK2)
    ax.tick_params(colors=MUTED, labelcolor=MUTED)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label(cbar_label, color=INK2)
    cb.ax.tick_params(colors=MUTED, labelcolor=MUTED)
    _save(fig, out_stub)
