"""Figures of which part of a transcript carries the capability signal: the channel and thirds analyses'
panels and their writers. Statistics and captions: analysis.challenges.contributions."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.challenges.contributions import DOMAIN_TITLES, baseline_stats, caption_lines, domain_stats
from plots.results import unclip_markers
from plots.signal_glm import DOMAIN_COLORS

FINAL = Path("analysis/manuscript/final_figs")  # the manuscript's final figures
# the final figures' names and the caption sentences that differ between the two
FINAL_NAMES = {"only": "kept_alone", "without": "removed"}
PIE_COLORS = {"CoT": "#4a3aa7", "Messages": "#eda100", "Tools": "#1baf7a"}
# the pairwise channel figure's hues when coloured by channel (the pie chart's; the combined tools column takes a
# darker shade of the tool-calls green)
CHANNEL_COLORS = {"CoT": PIE_COLORS["CoT"], "Messages": PIE_COLORS["Messages"], "Tools": PIE_COLORS["Tools"],
                  "Tool calls": "#7fd0b2", "Tool outputs": "#e87ba4", "Full transcript": "0.35"}
# the channel figure keeps the tool calls and their outputs together; the csv still carries them apart
DRAWN = ("CoT", "Messages", "Tools")
WIDTH = 0.5


def rc() -> None:
    """The contribution figures' rcParams (Arial, 8 pt, no top/right spines)."""
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"], "font.size": 8, "axes.labelsize": 8,
                         "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.8,
                         "xtick.major.width": 0.8, "ytick.major.width": 0.8, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none"})


def stars(p: float) -> str:
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def draw_row(axes, stats: pd.DataFrame, parts: list[str], measure: str, colours: dict[str, str],
             titles: bool = True, alpha: float | None = None, baselines: pd.DataFrame | None = None) -> None:
    """One panel per domain of `stats` into `axes` (one per domain, in DOMAIN_TITLES order), one row
    per part kept alone (`measure` "only") or removed ("without"). Dot = mean change in squared error
    from the full transcript, box = 95% t-interval of the paired change; the dashed line at 0
    is the full transcript. Each mark sits just past the panel's widest box. No grid, no line at the
    saturated-item prediction, which the marks do not test. `stats`: domain, part, measure, estimate,
    lo, hi, p. `colours`: part -> colour, or domain -> colour when every part of a domain shares one.
    Every panel shares one axis, from the first tick under the lowest interval to the first tick past
    the widest. `baselines` (baseline_stats) adds a last row, easy-item prediction's change from the full
    transcript as a grey point, for scale; no interval, no test mark."""
    domains = [d for d in DOMAIN_TITLES if d in set(stats.domain)]
    rows = [f"{measure.capitalize()} {p[0].lower()}{p[1:]}" for p in parts]
    width = 0.5
    picked = stats[stats.measure == measure]
    if baselines is not None:
        rows.append("Easy-item prediction")
        picked = pd.concat([picked, baselines])
    low, high = min(0.0, float(picked.lo.min())), max(0.0, float(picked.hi.max()))  # one axis for every panel
    for step in (m * 10.0 ** e for e in range(-2, 2) for m in (1, 2, 5)):  # the finest nice step with at most 5 ticks
        if np.ceil(high / step) - np.floor(low / step) <= 4:
            break
    ticks = list(np.arange(np.floor(low / step), np.ceil(high / step) + 1) * step)  # through 0, covering every interval
    span = ticks[-1] - ticks[0]
    gap, text = 0.04 * span, 0.2 * span  # marks, in axis units
    reach = ticks[-1] + text / 2  # how far right any panel's marks go: one x limit for every panel, past the last label
    for ax, domain in zip(axes, domains):
        part = picked[picked.domain == domain]
        ax.vlines(0, -0.5, len(rows) - 0.4, colors="0.5", lw=0.8, linestyles="--", zorder=1.5)  # the full transcript
        drawn = []
        for y, label in enumerate(parts):
            pick = part[part.part == label]
            if not pick.empty:
                drawn.append((y, colours.get(label, colours.get(domain, "0.55")), pick.iloc[0]))
        for y, colour, row in drawn:
            ax.add_patch(plt.Rectangle((row.lo, y - width / 2), row.hi - row.lo, width, color=colour,
                                       alpha=alpha or (0.6 if measure == "without" else 0.35), linewidth=0, zorder=1))
            ax.plot(row.estimate, y, "o", ms=3.5, mfc="white" if measure == "without" else colour, mec="black", mew=0.4, zorder=2)
        edge = max(r.hi for _, _, r in drawn) + gap  # the marks start just past the panel's widest box
        for y, _, row in drawn:
            ax.text(edge, y, stars(row.p), ha="left", va="center", fontsize=6)
        if baselines is not None and not (row := baselines[baselines.domain == domain]).empty:
            row, y = row.iloc[0], len(parts)
            ax.plot(row.estimate, y, "o", ms=3.5, mfc="0.6", mec="black", mew=0.4, zorder=2)
        reach = max(reach, edge + text)
        ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
        ax.spines["bottom"].set_bounds(ticks[0], ticks[-1])
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_ylim(len(rows) - 0.4, -0.6)  # the first part on top
        if titles:
            ax.set_title(DOMAIN_TITLES[domain], loc="left", fontsize=7, fontweight="bold")
    for ax in axes:
        ax.set_xlim(ticks[0], reach)
    axes[0].set_yticks(range(len(rows)), rows)


def figure(stats: pd.DataFrame, parts: list[str], measure: str, colours: dict[str, str],
           baselines: pd.DataFrame | None = None) -> plt.Figure:
    """draw_row on its own figure, 5.2 in wide, with the shared x label."""
    domains = [d for d in DOMAIN_TITLES if d in set(stats.domain)]
    fig, axes = plt.subplots(1, len(domains), figsize=(5.2, 1.3 + 0.3 * (baselines is not None)), sharey=True,
                             gridspec_kw={"wspace": 0.08})
    draw_row(np.atleast_1d(axes), stats, parts, measure, colours, baselines=baselines)
    fig.supxlabel("Δ Squared error vs full transcript", fontsize=8, y=-0.12)
    return unclip_markers(fig)


def grid(stats: dict[str, pd.DataFrame], parts: list[str], measure: str, colours: dict[str, str],
         alpha: float | None = None) -> plt.Figure:
    """One draw_row per protocol (`stats`: label -> its stats table), stacked, domain titles on the
    top row only, each row's protocol label to the left of its condition labels, its own x axis."""
    domains = [d for d in DOMAIN_TITLES if d in set(pd.concat(stats.values()).domain)]
    fig, axes = plt.subplots(len(stats), len(domains), figsize=(5.2, 1.3 * len(stats)), sharey="row", squeeze=False,
                             gridspec_kw={"wspace": 0.08, "hspace": 0.55})
    for i, (label, table) in enumerate(stats.items()):
        draw_row(axes[i], table, parts, measure, colours, titles=i == 0, alpha=alpha)
        axes[i, 0].annotate(label, xy=(0, 0.5), xycoords="axes fraction", xytext=(-92, 0), textcoords="offset points",
                            rotation=90, ha="center", va="center", fontsize=7, fontweight="bold")
    fig.supxlabel("Δ Squared error vs full transcript", fontsize=8, y=-0.12 / len(stats))
    return unclip_markers(fig)


def save(fig: plt.Figure, out: Path, name: str, final: str | None) -> None:
    """The figure as .png and .svg under `out`, and under FINAL when `final` names it."""
    for folder, stem in [(out, name)] + ([(FINAL, final)] if final else []):
        for suffix in (".png", ".svg", ".pdf"):
            fig.savefig(folder / f"{stem}{suffix}", dpi=300, bbox_inches="tight")
        print(folder / f"{stem}.png")
    plt.close(fig)


def save_figures(errors: pd.DataFrame, parts: dict[str, tuple[str, str]], colours: dict[str, str],
                 out: Path, name: str, final: str | None = None, caption: str = "",
                 measures: tuple[str, ...] = ("only", "without"), baseline: bool = False) -> None:
    """part_stats for every domain of `errors`, written to <name>_delta_squared_error.csv, and
    the "only" and "without" figures to <name>_only and <name>_without (.png, .svg) under `out`. With
    `final`, also into FINAL as <final>_kept_alone and <final>_removed, each with a .txt caption.
    `measures` limits the figures, e.g. to ("without",) when no part was run alone; `baseline` adds the
    easy-item prediction point (baseline_stats) for scale."""
    stats, fulls = domain_stats(errors, parts)
    stats.to_csv(out / f"{name}_delta_squared_error.csv", index=False)
    targets = errors[errors.condition == "full"].groupby("domain").size()
    for measure in measures:
        save(figure(stats, list(parts), measure, colours, baseline_stats(errors) if baseline else None), out, f"{name}_{measure}", final and f"{final}_{FINAL_NAMES[measure]}")
        if final:
            (FINAL / f"{final}_{FINAL_NAMES[measure]}.txt").write_text("\n".join(caption_lines(stats, fulls, targets, measure, caption)) + "\n")


def save_grid(errors: pd.DataFrame, protocols: dict[str, str], parts: dict[str, tuple[str, str]],
              colours: dict[str, str], out: Path, name: str, final: str | None = None, caption: str = "",
              alpha: float | None = None) -> None:
    """save_figures for several protocols at once (`errors` carries a protocol column; `protocols`:
    key -> row label), one grid figure per measure with a row of panels per protocol."""
    stats, fulls, targets = {}, {}, {}
    for key, label in protocols.items():
        e = errors[errors.protocol == key]
        stats[label], fulls[label] = domain_stats(e, parts)
        targets[label] = e[e.condition == "full"].groupby("domain").size()
    pd.concat(stats, names=["protocol"]).reset_index(level=0).to_csv(out / f"{name}_delta_squared_error.csv", index=False)
    for measure in ("only", "without"):
        save(grid(stats, list(parts), measure, colours, alpha), out, f"{name}_{measure}", final and f"{final}_{FINAL_NAMES[measure]}")
        if final:
            lines = [caption]
            for label in protocols.values():
                lines += ["", f"== {label}"] + caption_lines(stats[label], fulls[label], targets[label], measure, "")[1:]
            (FINAL / f"{final}_{FINAL_NAMES[measure]}.txt").write_text("\n".join(lines) + "\n")


def channel_figure(stats, fulls, order, out, name, color_by):
    """One panel per domain, the channels as rows: to the right of zero the prediction score with only that channel
    (scale on top), to the left the reduction when it is removed (scale on the bottom). Rows are shared by every
    panel, so a channel sits at the same height throughout and a domain without tool calls leaves those rows empty.
    `color_by`: "domain" (one hue per panel) or "channel" (the pie chart's hues). Values are in bits. `order`: the
    domains, in panel order. Written to <out>/<name>.png and .svg."""
    stats = stats.assign(channel=stats.channel.str.replace("\n", " "))
    colors = {k.replace("\n", " "): v for k, v in CHANNEL_COLORS.items()}
    rows = ["Full transcript"] + list(DRAWN)
    domains = [d for d in order if d in set(stats.domain)]
    left = float(np.floor(min(0, -stats.loc[stats.measure == "unique", "hi"].max())))
    right = float(np.ceil(max(stats.loc[stats.measure == "singular", "hi"].max(), fulls.hi.max())))
    fig, axes = plt.subplots(1, len(domains), figsize=(5.2, 2.1), sharey=True, sharex=True,
                             gridspec_kw={"wspace": 0.1})
    axes = np.atleast_1d(axes)
    for ax, domain in zip(axes, domains):
        part = stats[stats.domain == domain]
        color = colors["Full transcript"] if color_by == "channel" else DOMAIN_COLORS[domain]
        full = fulls.loc[domain]
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color="0.92", lw=0.5)  # one rule per channel row, to carry the eye across panels
        ax.axvline(full.full_prediction_score, color="0.5", lw=0.8, ls="--", zorder=0)
        ax.axvline(0, color="black", lw=0.8, zorder=0)
        ax.add_patch(plt.Rectangle((full.lo, -WIDTH / 2), full.hi - full.lo, WIDTH, color=color, alpha=0.3,
                                   linewidth=0, zorder=1))
        ax.plot(full.full_prediction_score, 0, "o", ms=4, mfc=color, mec="black", mew=0.4, zorder=2)
        for label in rows[1:]:
            if color_by == "channel":
                color = colors[label]
            position = rows.index(label)
            for measure, sign, alpha, face in (("singular", 1, 0.3, color), ("unique", -1, 0.6, "white")):
                row = part[(part.channel == label) & (part.measure == measure)]
                if row.empty:
                    continue
                row = row.iloc[0]
                lo, hi = sorted((sign * row.lo, sign * row.hi))
                ax.add_patch(plt.Rectangle((lo, position - WIDTH / 2), hi - lo, WIDTH, color=color, alpha=alpha,
                                           linewidth=0, zorder=1))
                ax.plot(sign * row.estimate, position, "o", ms=4, mfc=face, mec="black", mew=0.4, zorder=2)
                # one column of marks per side, pinned just inside the panel edges, rather than hugging each estimate
                ax.text(0.98 if sign > 0 else 0.02, position, stars(row.p), va="center",
                        ha="right" if sign > 0 else "left", fontsize=6.5, transform=ax.get_yaxis_transform())
        ax.set_xlim(left - 1.1, right + 1.1)
        ax.set_ylim(len(rows) - 0.4, -0.7)  # the full transcript on top, channels under it
        ax.set_xticks(np.arange(left, 0.01, 2), [f"{abs(v):g}" for v in np.arange(left, 0.01, 2)])
        ax.spines["bottom"].set_bounds(left, 0)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        top = ax.secondary_xaxis("top")
        top.set_xticks(np.arange(0, right + 0.01, 2))
        top.spines["top"].set_bounds(0, right)
        top.tick_params(labelsize=7, width=0.8)
        ax.set_title(domain.capitalize(), loc="left", fontsize=7, color="0.4", pad=14)
    axes[0].set_yticks(range(len(rows)), rows)
    fig.supxlabel("Reduction (bits)", fontsize=8, y=-0.02)
    fig.text(0.5, 1.12, "Prediction score (bits)", ha="center", fontsize=8)
    handles = [plt.Line2D([], [], marker="o", ls="none", ms=4, mfc="0.6", mec="black", mew=0.4),
               plt.Line2D([], [], marker="o", ls="none", ms=4, mfc="white", mec="black", mew=0.4),
               plt.Line2D([], [], color="0.5", lw=0.8, ls="--")]
    fig.legend(handles, ["Only this channel (right)", "Without this channel (left)", "Full transcript"],
               frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.18), handlelength=1.2,
               columnspacing=1.2, borderaxespad=0)
    for suffix in (".png", ".svg"):
        fig.savefig(out / f"{name}{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(out / f"{name}.png")


def shares_figure(share, out):
    """One pie per domain (`share`: domain x channel), the share of model-written characters per channel, to
    <out>/channel_shares.png and .svg."""
    fig, axes = plt.subplots(1, len(share), figsize=(3.8, 1.4))
    for ax, domain in zip(np.atleast_1d(axes), share.index):
        values = share.loc[domain]
        ax.pie(values, colors=[PIE_COLORS[c] for c in values.index], startangle=90, counterclock=False,
               wedgeprops={"linewidth": 0.5, "edgecolor": "white"},
               autopct=lambda pct: f"{pct:.0f}%" if pct >= 10 else "", pctdistance=0.6,
               textprops={"fontsize": 7, "color": "white"})
        ax.set_title(domain.capitalize(), fontsize=7, color="0.4", pad=2)
    fig.legend([plt.Rectangle((0, 0), 1, 1, color=PIE_COLORS[c]) for c in PIE_COLORS], list(PIE_COLORS), frameon=False,
               ncol=len(PIE_COLORS), loc="lower center", bbox_to_anchor=(0.5, -0.08), handlelength=1, columnspacing=1)
    for suffix in (".png", ".svg"):
        fig.savefig(out / f"channel_shares{suffix}", dpi=300, bbox_inches="tight")
    print(out / "channel_shares.png")
