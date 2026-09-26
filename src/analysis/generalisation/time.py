"""
Prediction error over target semesters: each semester's targets predicted from the previous semester's references,
and the mean squared error of every task and protocol with a percentile-bootstrap interval over the task's targets.
Science and Maths run from 2024-I; Coding and Cyber from 2025-II, their first semester.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.likelihood_ratio import num
from analysis.plots import unclip_markers
from analysis.stats import BASELINE, with_baseline


def task_errors(errors: pd.DataFrame, draws: int, seed: int) -> pd.DataFrame:
    """
    Per task and protocol, the mean squared error over the task's targets, the Baseline's on the same targets, and
    the 2.5 and 97.5 percentiles of the mean over `draws` bootstrap resamples of the targets (NaN with fewer than 3
    targets). One random generator runs through the tasks and protocols in sorted order, so every protocol in `errors`
    (Length included) moves the draws of those after it.

    Inputs:
        errors: per-target errors from stats.target_errors (domain, semester, model, protocol, err); a target without
            a Baseline error is dropped.
        draws: bootstrap resamples per task and protocol.
        seed: the generator's seed.
    Outputs:
        DataFrame with columns domain, semester, protocol, n, mse, baseline_mse, lo, hi; one row per task and
        protocol, sorted by domain, semester and protocol.
    """
    rng = np.random.default_rng(seed)
    out = []
    paired = with_baseline(errors).dropna(subset=["base"])
    for (domain, semester, protocol), g in paired.groupby(["domain", "semester", "protocol"]):
        err, base = g.err.to_numpy(float), g.base.to_numpy(float)
        boot = err[rng.integers(0, len(err), size=(draws, len(err)))].mean(axis=1)
        lo, hi = np.percentile(boot[np.isfinite(boot)], [2.5, 97.5]) if len(g) >= 3 else (np.nan, np.nan)
        out.append(dict(domain=domain, semester=semester, protocol=protocol, n=len(g), mse=float(err.mean()),
                        baseline_mse=float(base.mean()), lo=float(lo), hi=float(hi)))
    return pd.DataFrame(out).sort_values(["domain", "semester", "protocol"], ignore_index=True)


def error_over_time_figure(tasks: pd.DataFrame, domains: list[str], protocols: list[str], display: dict,
                           reasoning_semester: str, log: bool = True, box: bool = True) -> plt.Figure:
    """
    Draws one panel per domain over its own semesters, with a dodged series for the Baseline and for each protocol:
    the task's mean squared error as a dot and its bootstrap interval as a translucent box or a vertical line. A value
    beyond the axis is drawn as a triangle on its edge, and a note with arrows marks `reasoning_semester` in the
    domains that have it.

    Inputs:
        tasks: task_errors rows (domain, semester, protocol, mse, lo, hi).
        domains: domain keys, in panel order.
        protocols: the protocols drawn after the Baseline, in drawing order.
        display: config.yaml's display section; uses protocol_names, protocol_colours and domain_names.
        reasoning_semester: the semester the note "Reasoning models introduced in early 2025" points to.
        log: a log squared-error axis from 0.01 to 10; if False, a linear axis from 0 to 1.25 times the 90th
            percentile of the plotted errors, rounded up to a multiple of 0.5.
        box: intervals as translucent boxes; if False, as vertical lines.
    Outputs:
        the matplotlib Figure; prints every value that falls off the axis.
    """
    names, colours = display["protocol_names"], display["protocol_colours"]
    lines = [BASELINE, *protocols]
    panels = [d for d in domains if d in set(tasks.domain)]
    shown = tasks[tasks.protocol.isin(lines)]
    bottom, top = (0.01, 10.0) if log else (0.0, float(np.ceil(np.nanpercentile(shown.mse, 90) * 1.25 * 2) / 2))
    spans = {key: sorted(tasks[tasks.domain == key].semester.unique()) for key in panels}
    fig, axes = plt.subplots(1, len(panels), figsize=(0.7 + 0.24 * sum(len(v) for v in spans.values()) + 0.35 * len(panels), 2.2),
                             layout="constrained", sharey=True, width_ratios=[len(spans[k]) for k in panels])
    for ax, key in zip(np.atleast_1d(axes), panels):
        panel = tasks[tasks.domain == key]
        semesters = spans[key]
        x = {s: i for i, s in enumerate(semesters)}
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#d5d5d5", lw=0.6, zorder=0)
        dodge = dict(zip(lines, np.linspace(-0.22, 0.22, len(lines))))  # the series share a semester slot
        width = 0.7 / len(lines)
        for line in lines:
            d = panel[panel.protocol == line].sort_values("semester")
            xs = np.array([x[s] + dodge[line] for s in d.semester], float)
            raw = d.mse.to_numpy(float)
            inset = 0.02 * (top - bottom)  # the marker for an off-axis value sits just inside the spine
            ys = np.clip(raw, bottom * 1.08, top / 1.08) if log else np.clip(raw, bottom + inset, top - inset)
            out = (raw < bottom) | (raw > top)
            if out.any():
                print(f"error_over_time_figure: {key} {line} off the axis: {[round(v, 1) for v in raw[out]]}")
            for xi, lo, hi in zip(xs, np.clip(d.lo, bottom, top), np.clip(d.hi, bottom, top)):
                if box:  # the interval as a translucent rectangle
                    ax.add_patch(plt.Rectangle((xi - width / 2, lo), width, max(hi - lo, 1e-6), facecolor=colours[line],
                                               edgecolor="none", alpha=0.35, zorder=2))
                else:
                    ax.plot([xi, xi], [lo, hi], color=colours[line], lw=0.8, zorder=2)
            ax.plot(xs, ys, color=colours[line], lw=0.4, alpha=0.7, zorder=2)  # very thin connector
            ax.scatter(xs[~out], ys[~out], s=12, marker="o", color=colours[line], linewidths=0, edgecolors="none",
                       zorder=3, clip_on=False)
            for side, tri in ((raw > top, "^"), (raw < bottom, "v")):  # off the axis: a triangle on the edge, pointing out
                ax.scatter(xs[side], ys[side], s=16, marker=tri, color=colours[line], linewidths=0, edgecolors="none",
                           zorder=3, clip_on=False)
        ax.set_xticks(range(len(semesters)), semesters)
        ax.tick_params(axis="x", rotation=90)
        ax.set_xlim(-0.5, len(semesters) - 0.5)  # room for the dodged points at the ends
        if log:
            ax.set_yscale("log")
            ax.set_yticks([0.01, 0.1, 1, 10], ["0.01", "0.1", "1", "10"])
            ax.yaxis.set_minor_locator(plt.NullLocator())
        else:
            ax.yaxis.set_major_locator(plt.MaxNLocator(5, steps=[1, 2, 5, 10]))
        ax.set_ylim(bottom, top)
        ax.spines["left"].set_bounds(bottom, top)
        ax.spines["bottom"].set_bounds(0, len(semesters) - 1)
        ax.text(0, 1.10, display["domain_names"][key], transform=ax.transAxes, ha="left", fontsize=8, fontweight="bold")
    np.atleast_1d(axes)[0].set_ylabel("Squared error")
    handles = [plt.Line2D([], [], color=colours[l], ls="none", marker="o", ms=3.5, label=names[l]) for l in lines]
    fig.legend(handles=handles, loc="outside upper right", ncol=2, frameon=True, edgecolor="black", fancybox=False,
               framealpha=1, borderpad=0.3, handletextpad=0.1, columnspacing=0.8, fontsize=7)
    pointed = [(ax, spans[key].index(reasoning_semester)) for ax, key in zip(np.atleast_1d(axes), panels)
               if reasoning_semester in spans[key]]
    if pointed:  # a note above the panels, an arrow down to each marked slot
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


def caption(tasks: pd.DataFrame, domains: list[str], protocols: list[str], display: dict, draws: int) -> str:
    """
    The text of the figure's .txt file: what the figure draws, then for every domain and semester each series' mean
    squared error and 95% interval.

    Inputs:
        tasks: task_errors rows.
        domains: domain keys, in order.
        protocols: the protocols listed after the Baseline.
        display: config.yaml's display section; uses protocol_names and domain_names.
        draws: the number of bootstrap resamples, quoted in the text.
    Outputs:
        str, the caption.
    """
    names = display["protocol_names"]
    lines = []
    for domain in domains:
        lines.append(f"{display['domain_names'][domain]}:")
        for semester, g in tasks[tasks.domain == domain].groupby("semester"):
            rows = g.set_index("protocol")
            parts = [f"{names[p]} {num(rows.loc[p, 'mse'])} ({num(rows.loc[p, 'lo'])} to {num(rows.loc[p, 'hi'])})"
                     for p in [BASELINE, *protocols] if p in rows.index]
            lines.append(f"  {semester} ({int(g.n.iloc[0])} targets): " + "; ".join(parts) + ".")
    return ("Squared error over target semesters, one panel per domain. Each point is the mean squared error over a "
            "task's targets for the baseline (grey) and each protocol; the box is its 95% percentile-bootstrap interval "
            f"over the task's targets ({draws:,} resamples). Science and Maths run from 2024-I to 2026-I; Coding and "
            "Cyber start at 2025-II, their first semester. Judge protocols use each task's best judge on references. The "
            "squared-error axis is logarithmic, from 0.01 to 10; points above it are drawn as triangles on its edge. No "
            "test is drawn per semester: with 8 targets and 7 or 8 references per task it has little power; the protocols "
            "are tested over all 16 target models of 2025-II and 2026-I in the figure of squared error by protocol."
            "\n\nMean squared error (95% interval):\n" + "\n".join(lines) + "\n")
