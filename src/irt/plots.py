"""Figures for IRT validation, in the pilot house style (Arial, 400 dpi)."""

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#33322f", "#4a4844", "#d3d2c9", "#ffffff"
COLOR = {"overall_base": "#a8a79d", "model_base": "#1baf7a",
         "item_base": "#eb6834", "both_add": "#4a3aa7", "both_logit": "#e87ba4",
         "irt": "#2a78d6", "irt_1pl": "#2a78d6", "irt_2pl": "#0d3b66"}
LABEL = {"overall_base": "average accuracy", "model_base": "model %-correct",
         "item_base": "item %-correct", "both_add": "model + item (add)",
         "both_logit": "model + item (logit)", "irt": "IRT (1PL)",
         "irt_1pl": "IRT (1PL)", "irt_2pl": "IRT (2PL)"}

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
               series=("overall_base", "model_base", "item_base", "both_add", "both_logit", "irt")):
    """Each series' mean + empirical-95% band vs xcol, from per-resample rows."""
    fig, ax = plt.subplots(figsize=(3.26, 2.29))
    for s in series:
        x = _band(ax, raw, xcol, s, LABEL[s])
    _logx(ax, x)
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel("held-out MAE", color=INK2)
    _style(ax)
    _legend(ax, loc="upper right")
    _save(fig, out_stub)


def sweep_gap_plot(raw, xcol, xlabel, out_stub,
                   series=("overall_base", "model_base", "item_base", "both_add", "both_logit", "irt"),
                   a="irt", b="model_base"):
    """Sweep led by the gain: curves on top, how much lower a's error is than b below."""
    fig, (top, bot) = plt.subplots(
        2, 1, sharex=True, figsize=(3.34, 3.26),
        gridspec_kw={"height_ratios": [2.3, 1]},
    )
    for s in series:
        x = _band(top, raw, xcol, s, LABEL[s])
    top.set_ylabel("held-out MAE", color=INK2)
    top.tick_params(labelbottom=False)
    _style(top)
    _legend(top, loc="upper right")

    gap = raw.assign(_g=raw[b] - raw[a])
    xs = np.array(sorted(gap[xcol].unique()))
    m, lo, hi = np.array([_ci(gap.loc[gap[xcol] == v, "_g"]) for v in xs]).T
    bot.axhline(0, color=MUTED, lw=0.8)
    bot.fill_between(xs, lo, hi, color=COLOR[a], alpha=0.15, lw=0)
    bot.plot(xs, m, "-o", color=COLOR[a], ms=5, zorder=3)
    bot.set_ylabel(f"{LABEL[b]} −\n{LABEL[a]} MAE", color=INK2)
    bot.set_xlabel(xlabel, color=INK2)
    _style(bot)
    _logx(bot, x)
    _save(fig, out_stub)


def strip(labels, values, colors, ylabel, out_stub, seed=0, max_show=40, width=3.9, rot=0,
          connect=False):
    """Per label: up to max_show jittered resamples + mean with empirical 95% CI (from all)."""
    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(figsize=(width, 2.4))
    means = []
    for i, (lab, v) in enumerate(zip(labels, values)):
        v = np.asarray(v, float)
        c = colors[i] if isinstance(colors, (list, tuple)) else colors
        show = v if len(v) <= max_show else rng.choice(v, max_show, replace=False)
        ax.scatter(i + rng.uniform(-0.16, 0.16, len(show)), show, s=11, color=c, alpha=0.3, lw=0, zorder=2)
        m, lo, hi = _ci(v)
        means.append(m)
        ax.errorbar(i, m, yerr=[[m - lo], [hi - m]], fmt="o", color=c, ecolor=INK2,
                    elinewidth=1.2, capsize=0, ms=7, zorder=4,
                    markeredgecolor="white", markeredgewidth=0.8)
    if connect:
        lc = colors if isinstance(colors, str) else INK2
        ax.plot(range(len(labels)), means, "-", color=lc, lw=1.4, alpha=0.7, zorder=3)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=rot, ha="right" if rot else "center")
    ax.set_ylabel(ylabel, color=INK2)
    ax.margins(x=0.08)
    _style(ax)
    _save(fig, out_stub)


def scatter(df, xcol, ycol, xlabel, ylabel, out_stub,
            color=INK2, note=None, note_xy=(0.03, 0.97), identity=False):
    """Scatter ycol vs xcol; optional y=x line and corner note."""
    fig, ax = plt.subplots(figsize=(3.08, 2.29))
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
                note=None, note_xy=(0.03, 0.97), identity=False):
    """Scatter coloured by a categorical column, legend in `order`; optional y=x line."""
    fig, ax = plt.subplots(figsize=(3.7, 2.55))
    if identity:
        lo = min(df[xcol].min(), df[ycol].min())
        hi = max(df[xcol].max(), df[ycol].max())
        ax.plot([lo, hi], [lo, hi], color=MUTED, lw=0.8, zorder=1)
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


def _kde(data, grid, bw_scale=1.0):
    """Gaussian KDE density on grid, Silverman bandwidth (numpy only)."""
    data = np.asarray(data, float)
    n = len(data)
    bw = max(1.06 * data.std(ddof=1) * n ** (-0.2) * bw_scale, 1e-3)
    u = (grid[:, None] - data[None, :]) / bw
    return np.exp(-0.5 * u ** 2).sum(1) / (n * bw * np.sqrt(2 * np.pi))


def _kde_marg(ax, df, col, group, order, colors, horiz, bw_scale):
    """Per-group KDE + stacked median/IQR markers on a marginal axis."""
    lo, hi = df[col].min(), df[col].max()
    grid = np.linspace(lo - (hi - lo) * 0.04, hi + (hi - lo) * 0.04, 200)
    maxd, qs = 0.0, []
    for g in order:
        v = df.loc[df[group] == g, col]
        dens = _kde(v, grid, bw_scale)
        maxd = max(maxd, dens.max())
        qs.append((g, np.percentile(v, [25, 50, 75])))
        (ax.fill_betweenx if horiz else ax.fill_between)(grid, dens, color=colors[g], alpha=0.12, lw=0)
        ax.plot(*((dens, grid) if horiz else (grid, dens)), color=colors[g], lw=1.6)
    step = maxd * 0.16
    for i, (g, (q1, med, q3)) in enumerate(qs):
        off, err = maxd * 1.12 + i * step, [[med - q1], [q3 - med]]
        kw = dict(color=colors[g], ms=7, markeredgecolor="white", markeredgewidth=0.6,
                  ecolor=colors[g], elinewidth=1.4, capsize=0, clip_on=False, zorder=5)
        if horiz:
            ax.errorbar(off, med, yerr=err, fmt="<", **kw)
        else:
            ax.errorbar(med, off, xerr=err, fmt="v", **kw)
    lim = maxd * 1.12 + len(qs) * step
    (ax.set_xlim if horiz else ax.set_ylim)(0, lim)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)


def _joint(axm, axt, axr, df, xcol, ycol, group, order, colors, xlabel, ylabel,
           bw_scale=1.0, note=None, note_xy=(0.03, 0.97), legend=True, title=None):
    """Draw a coloured scatter + top/right KDE marginals into the given three axes."""
    for g in order:
        d = df[df[group] == g]
        axm.scatter(d[xcol], d[ycol], s=15, color=colors[g], alpha=0.82, lw=0, label=g, zorder=3)
    _kde_marg(axt, df, xcol, group, order, colors, horiz=False, bw_scale=bw_scale)
    _kde_marg(axr, df, ycol, group, order, colors, horiz=True, bw_scale=bw_scale)
    axm.set_xlabel(xlabel, color=INK2)
    axm.set_ylabel(ylabel, color=INK2)
    _note(axm, note, note_xy)
    _style(axm)
    if title:
        axt.set_title(title, color=INK2, fontsize=9, pad=3)
    if legend:
        leg = axm.legend(frameon=False, loc="best", fontsize=6.5, handletextpad=0.2, labelspacing=0.25)
        for t in leg.get_texts():
            t.set_color(INK2)


def cat_scatter_joint(df, xcol, ycol, group, order, colors, xlabel, ylabel, out_stub,
                      note=None, note_xy=(0.03, 0.97), bw_scale=1.0):
    """Coloured scatter with smooth per-group KDE marginals on top (x) and right (y)."""
    fig = plt.figure(figsize=(3.9, 3.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[3.4, 1], height_ratios=[1, 3.4],
                          wspace=0.05, hspace=0.05)
    axm = fig.add_subplot(gs[1, 0])
    axt = fig.add_subplot(gs[0, 0], sharex=axm)
    axr = fig.add_subplot(gs[1, 1], sharey=axm)
    _joint(axm, axt, axr, df, xcol, ycol, group, order, colors, xlabel, ylabel, bw_scale, note, note_xy)
    _save(fig, out_stub)


def joint_pair(specs, out_stub, bw_scale=0.9):
    """Two coloured scatters, each with KDE marginals, side by side for comparison."""
    fig = plt.figure(figsize=(6.9, 3.3))
    gs = fig.add_gridspec(2, 5, width_ratios=[3, 1, 0.8, 3, 1], height_ratios=[1, 3.4],
                          wspace=0.05, hspace=0.05)
    for k, s in enumerate(specs):
        c = 0 if k == 0 else 3
        axm = fig.add_subplot(gs[1, c])
        axt = fig.add_subplot(gs[0, c], sharex=axm)
        axr = fig.add_subplot(gs[1, c + 1], sharey=axm)
        _joint(axm, axt, axr, s["df"], s["xcol"], s["ycol"], s["group"], s["order"], s["colors"],
               s["xlabel"], s["ylabel"], bw_scale, legend=(k == 0), title=s.get("title"))
    _save(fig, out_stub)


def scree_plot(pcs, var, out_stub, highlight=None):
    """Variance-explained bar per principal component; optional highlighted PC."""
    fig, ax = plt.subplots(figsize=(3.2, 2.3))
    cols = [COLOR["item_base"] if highlight and p == highlight else COLOR["irt"] for p in pcs]
    ax.bar(pcs, var, color=cols, width=0.7, zorder=3)
    ax.set_xticks(list(pcs))
    ax.set_xlabel("principal component", color=INK2)
    ax.set_ylabel("variance explained", color=INK2)
    _style(ax)
    _save(fig, out_stub)


def density_plot(df, valcol, group, order, colors, xlabel, out_stub, bw_scale=0.8, pooled=True):
    """Per-group KDE of valcol (+ optional pooled) with stacked median/IQR markers."""
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    lo, hi = df[valcol].min(), df[valcol].max()
    grid = np.linspace(lo - (hi - lo) * 0.04, hi + (hi - lo) * 0.04, 200)
    maxd, qs = 0.0, []
    series = ([("all", INK2)] if pooled else []) + [(g, colors[g]) for g in order]
    for name, c in series:
        v = df[valcol] if name == "all" else df.loc[df[group] == name, valcol]
        dens = _kde(v, grid, bw_scale)
        maxd = max(maxd, dens.max())
        qs.append((c, np.percentile(v, [25, 50, 75])))
        ax.fill_between(grid, dens, color=c, alpha=0.10, lw=0)
        ax.plot(grid, dens, color=c, lw=1.8, label=name)
    step = maxd * 0.15
    for i, (c, (q1, med, q3)) in enumerate(qs):
        ax.errorbar(med, maxd * 1.1 + i * step, xerr=[[med - q1], [q3 - med]], fmt="v", color=c,
                    ms=7, markeredgecolor="white", markeredgewidth=0.6, ecolor=c, elinewidth=1.4,
                    capsize=0, clip_on=False, zorder=5)
    ax.set_ylim(0, maxd * 1.1 + len(qs) * step)
    ax.set_yticks([])
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel("density", color=INK2)
    _style(ax)
    leg = ax.legend(frameon=False, loc="upper right", fontsize=7)
    for t in leg.get_texts():
        t.set_color(INK2)
    _save(fig, out_stub)


def wright_map(theta, b, out_stub):
    """Item-person map: model ability and item difficulty share one latent scale."""
    fig, (lt, rt) = plt.subplots(1, 2, sharey=True, figsize=(3.26, 2.55))
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


def heatmap(mat, out_stub, xlabel, ylabel, cbar_label="held-out MAE (IRT)", cmap="Blues_r"):
    """Grid of IRT held-out MAE over two sweep axes; annotated, one-hue."""
    fig, ax = plt.subplots(figsize=(3.7, 2.9))
    vals = mat.values.astype(float)
    vmin, vmax = np.nanmin(vals), np.nanmax(vals)
    cm = plt.get_cmap(cmap)
    im = ax.imshow(vals, aspect="auto", cmap=cm, origin="lower", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(mat.columns)))
    ax.set_xticklabels([str(c) for c in mat.columns])
    ax.set_yticks(range(len(mat.index)))
    ax.set_yticklabels([str(r) for r in mat.index])
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            if not np.isnan(v):
                norm = (v - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                r, g, bl, _ = cm(norm)
                lum = 0.299 * r + 0.587 * g + 0.114 * bl
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if lum < 0.5 else INK2)
    ax.tick_params(colors=MUTED, labelcolor=MUTED)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label(cbar_label, color=INK2)
    cb.ax.tick_params(colors=MUTED, labelcolor=MUTED)
    _save(fig, out_stub)
