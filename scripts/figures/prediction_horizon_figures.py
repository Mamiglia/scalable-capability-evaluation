"""Figures for predicting further ahead (docs/static_method_plan.md): squared error per target of the
Pairwise judge and the baseline, over the series of tasks, against distance above the
references, and by target semester. Science only. Styled as the manuscript's generalisation figures
(plots.generalisation: log error axis 0.01-10, translucent interval boxes, grey panel tags).

    uv run python scripts/figures/prediction_horizon_figures.py

The new tasks (40 train items) come from artifacts/prediction-horizon/; the one-semester tasks are the
paper's best judge on references (linear variant, core condition, 100 items) from the main checkout's
analysis/results/forecasts.csv. The baseline is the references' mean theta*. The tasks
with 2024-I references (GPT-4o) predict that mean for every target: the judge's reference-reference
verdicts give the linear calibration a slope <= 0.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import plots.results  # noqa: F401  (house rcParams)
from analysis.challenges import likelihood_ratio as lr
from plots.generalisation import BASELINE_COLOUR
from plots.signal_glm import COLORS
from plots.results import unclip_markers

OUT = Path("artifacts/prediction-horizon")
MAIN = Path("../small-judge-remote")
FIGURES = OUT / "figures"
PAIRWISE = COLORS["pairwise"]
TRANSCRIPT = {"glm_regex": "-", "embedding": "full"}  # the protocols without a judge: (key, the paper's variant)
# (column, paper name, colour, light shade = 2024-II references, dark shade = 2025-I references), left to right
SHOWN = ("pairwise",)  # protocols drawn in the combined figure (all three were too busy)
PROTOCOL_SHADES = [("glm_regex", "Deterministic features", COLORS["glm_regex"], "#f5b394", "#a8401a"),
                   ("embedding", "Embedding", COLORS["embedding"], "#a79ee0", "#2a1f6e"),
                   ("pairwise", "Pairwise judge", COLORS["pairwise"], "#9cc3ef", "#123f7a")]
BASELINE_LABEL = "Baseline"  # the user's name for the references' mean θ*, not the manuscript code's label
ORDER = ["2023", "2024-I", "2024-II", "2025-I", "2025-II", "2026-I"]
SERIES = [  # (panel tag, [(references, targets)]); x = semesters from references to targets
    ("References fixed at 2024-II", [("2024-II", t) for t in ("2023", "2024-I", "2025-I", "2025-II", "2026-I")]),
    ("Targets fixed at 2026-I", [(r, "2026-I") for r in ("2025-II", "2025-I", "2024-II", "2024-I", "2023")]),
    ("References fixed at 2025-I", [("2025-I", t) for t in ("2024-I", "2024-II", "2025-II", "2026-I")]),
]
OTHER = [("2023", "2024-I"), ("2024-I", "2024-II"), ("2024-I", "2025-I"), ("2024-I", "2025-II")]  # by-target only
LOG = (0.01, 10.0)
PRIMARY = (0.01, 20.0)  # the primary figure: the 2023 references' errors are above 10

plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
                     "font.family": "sans-serif", "font.sans-serif": ["Arial"], "svg.fonttype": "none"})


def ahead(references: str, targets: str) -> int:
    return ORDER.index(targets) - ORDER.index(references)


def rows() -> pd.DataFrame:
    """One row per (task, target): theta, theta_hat, the baseline, distance above the
    strongest reference, and both squared errors."""
    meta = pd.read_csv(OUT / "meta/science/models.csv", index_col="model")
    frozen = pd.read_csv(MAIN / "data/meta/science/models.csv", index_col="model")
    forecasts = pd.read_csv(MAIN / "analysis/results/forecasts.csv")
    chosen = lr.chosen_judges(forecasts[forecasts.domain == "science"], {"pairwise": "linear"})
    core = forecasts.merge(chosen, on=["domain", "semester", "protocol", "model", "judge"])
    core = core[core.variant == "linear"]
    tasks = {pair for _, pairs in SERIES for pair in pairs} | set(OTHER)
    out = []
    for references, targets in sorted(tasks):
        if ahead(references, targets) == 1:  # an existing semester task
            table, key = frozen, targets
            got = core[core.semester == targets].drop_duplicates("model").set_index("model")
        else:
            table, key = meta, f"{targets}_from_{references}"
            got = pd.read_csv(OUT / f"forecasts_{key}.csv").set_index("model")
        role, theta = table[f"{key}_role"], table[f"{key}_theta"]
        refs, tgts = theta[role == "reference"], theta[role == "target"]
        if ahead(references, targets) == 1:  # the paper's protocols without a judge
            science = forecasts[(forecasts.domain == "science") & (forecasts.semester == targets)]
            other = {p: science[(science.protocol == p) & (science.variant == v)].set_index("model").theta_hat
                     for p, v in TRANSCRIPT.items()}
        else:
            t = pd.read_csv(OUT / f"forecasts_transcript_{key}.csv")
            other = {p: t[t.protocol == p].set_index("model").theta_hat for p in TRANSCRIPT}
        out.append(pd.DataFrame({"references": references, "targets": targets, "judge": got.judge.iloc[0],
                                 "model": tgts.index, "theta": tgts.values,
                                 "theta_hat": got.theta_hat.reindex(tgts.index).values,
                                 **{f"theta_hat_{p}": other[p].reindex(tgts.index).values for p in TRANSCRIPT},
                                 "baseline": refs.mean(), "distance": tgts.values - refs.max()}))
    frame = pd.concat(out, ignore_index=True)
    frame["pairwise"] = (frame.theta_hat - frame.theta) ** 2
    frame["base"] = (frame.baseline - frame.theta) ** 2
    for p in TRANSCRIPT:
        frame[p] = (frame[f"theta_hat_{p}"] - frame.theta) ** 2
    return frame


def mean_ci(values: np.ndarray) -> tuple[float, float, float]:
    """Mean and 95% t-interval over a task's targets (N = 8)."""
    m, se = values.mean(), values.std(ddof=1) / np.sqrt(len(values))
    h = se * stats.t.ppf(0.975, len(values) - 1)
    return m, m - h, m + h


def log_axis(ax, limits: tuple[float, float] = LOG) -> None:
    """The manuscript's squared-error axis: log, 0.01 to 10 unless `limits` says otherwise, a gridline per decade."""
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#d5d5d5", lw=0.6, zorder=0)
    ax.set_yscale("log")
    ticks = [10.0 ** k for k in range(int(np.log10(limits[0])), int(np.log10(limits[1])) + 1)]
    labels = [f"{t:g}" for t in ticks]
    if limits[1] not in ticks:  # the axis line ends at a tick, left unlabelled
        ticks, labels = ticks + [limits[1]], labels + [""]
    ax.set_yticks(ticks, labels)
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.set_ylim(*limits)
    ax.spines["left"].set_bounds(*limits)


def mark(ax, xs, values: list[np.ndarray], colour, width: float, limits: tuple[float, float] = LOG) -> None:
    """Per task a translucent 95% t-interval box and the mean dot, joined by a very thin line; a mean
    off the axis is a triangle on the edge, pointing out."""
    bottom, top = limits
    means = []
    for x, v in zip(xs, values):
        m, lo, hi = mean_ci(v)
        lo, hi = np.clip([lo, hi], bottom, top)
        ax.add_patch(plt.Rectangle((x - width / 2, lo), width, max(hi - lo, 1e-6), facecolor=colour,
                                   edgecolor="none", alpha=0.35, zorder=2))
        means.append(m)
    means = np.array(means)
    ys = np.clip(means, bottom * 1.08, top / 1.08)
    ax.plot(xs, ys, color=colour, lw=0.4, alpha=0.7, zorder=2)
    out = means > top
    ax.scatter(np.array(xs)[~out], ys[~out], s=12, color=colour, linewidths=0, edgecolors="none", zorder=3, clip_on=False)
    ax.scatter(np.array(xs)[out], ys[out], s=16, marker="^", color=colour, linewidths=0, edgecolors="none", zorder=3,
               clip_on=False)
    if out.any():
        print("off the axis:", [round(v, 1) for v in means[out]])


def legend(fig, entries: list[tuple[str, str]], ncol: int) -> None:
    handles = [plt.Line2D([], [], color=c, ls="none", marker="o", ms=3.5, label=l) for l, c in entries]
    fig.legend(handles=handles, loc="outside lower center", ncol=ncol, frameon=False)


def save(fig, name: str) -> None:
    unclip_markers(fig)
    for ext in ("png", "svg"):
        fig.savefig(FIGURES / f"{name}.{ext}", dpi=300, bbox_inches="tight")


def series_figure(frame: pd.DataFrame) -> None:
    """One panel per series; x = semesters from the references to the targets (negative: older targets)."""
    widths = [max(ahead(*p) for p in pairs) - min(ahead(*p) for p in pairs) + 1 for _, pairs in SERIES]
    fig, axes = plt.subplots(1, len(SERIES), figsize=(5.2, 2.2), layout="constrained", sharey=True, width_ratios=widths)
    for ax, (tag, pairs) in zip(axes, SERIES):
        xs = [ahead(*p) for p in pairs]
        order = np.argsort(xs)
        for column, colour, dx in (("base", BASELINE_COLOUR, -0.14), ("pairwise", PAIRWISE, 0.14)):
            values = [frame[(frame.references == r) & (frame.targets == t)][column].to_numpy() for r, t in pairs]
            mark(ax, np.array(xs)[order] + dx, [values[i] for i in order], colour, 0.24)
        log_axis(ax)
        ticks = sorted(xs)  # no 0: references never predict their own semester
        ax.set_xticks(ticks, [f"−{-x}" if x < 0 else str(x) for x in ticks])
        ax.set_xlim(min(xs) - 0.5, max(xs) + 0.5)
        ax.spines["bottom"].set_bounds(min(xs), max(xs))
        ax.text(0, 1.03, tag, transform=ax.transAxes, color="grey", ha="left", fontsize=7)
    axes[0].set_ylabel("Squared error")
    axes[1].set_xlabel("Semesters from references to targets")
    legend(fig, [(BASELINE_LABEL, BASELINE_COLOUR), ("Pairwise judge", PAIRWISE)], ncol=2)
    save(fig, "horizon_series")


def distance_figure(frame: pd.DataFrame) -> None:
    """Every target of every task: squared error against theta* above the strongest reference."""
    fig, ax = plt.subplots(figsize=(2.6, 2.3), layout="constrained")
    for column, colour in (("base", BASELINE_COLOUR), ("pairwise", PAIRWISE)):
        v = frame[column].to_numpy()
        y = v.clip(LOG[0] * 1.08, LOG[1] / 1.08)
        for side, marker, size in ((v >= LOG[0]) & (v <= LOG[1]), "o", 8), (v < LOG[0], "v", 12), (v > LOG[1], "^", 12):
            # off the axis: a triangle on the edge, pointing out
            ax.scatter(frame.distance[side], y[side], s=size, marker=marker, color=colour, alpha=0.6, linewidths=0,
                       edgecolors="none", zorder=3, clip_on=False)
    log_axis(ax)
    ax.axvline(0, color="grey", lw=0.8, ls="--", zorder=1)
    lo, hi = np.floor(frame.distance.min()), np.ceil(frame.distance.max())
    ax.set_xticks(np.arange(lo, hi + 1), [f"{x:g}".replace("-", "−") for x in np.arange(lo, hi + 1)])
    ax.set_xlim(lo - 0.2, hi + 0.2)
    ax.spines["bottom"].set_bounds(lo, hi)
    ax.set_xlabel("θ* above the strongest reference")
    ax.set_ylabel("Squared error")
    legend(fig, [(BASELINE_LABEL, BASELINE_COLOUR), ("Pairwise judge", PAIRWISE)], ncol=1)
    save(fig, "horizon_distance")


def target_figure(frame: pd.DataFrame) -> None:
    """Pairwise judge squared error by target semester, one line per reference semester, coloured along
    viridis from oldest to newest references."""
    targets = [t for t in ORDER if t in set(frame.targets)]
    semesters = [r for r in ORDER if r in set(frame.references)]
    colours = dict(zip(semesters, plt.get_cmap("viridis")(np.linspace(0.05, 0.8, len(semesters)))))
    dodge = dict(zip(semesters, np.linspace(-0.22, 0.22, len(semesters))))
    fig, ax = plt.subplots(figsize=(2.8, 2.4), layout="constrained")
    for references in semesters:
        g = frame[frame.references == references]
        tasks = sorted(set(g.targets), key=ORDER.index)
        xs = np.array([targets.index(t) + dodge[references] for t in tasks])
        mark(ax, xs, [g[g.targets == t].pairwise.to_numpy() for t in tasks], colours[references], 0.7 / len(semesters))
    log_axis(ax)
    ax.set_xticks(range(len(targets)), targets, rotation=90)
    ax.set_xlim(-0.5, len(targets) - 0.5)
    ax.spines["bottom"].set_bounds(0, len(targets) - 1)
    ax.set_xlabel("Target semester")
    ax.set_ylabel("Squared error")
    handles = [plt.Line2D([], [], color=colours[r], ls="none", marker="o", ms=3.5, label=r) for r in semesters]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False, title="References",
               title_fontsize=7)
    save(fig, "horizon_by_target")


def primary_figure(frame: pd.DataFrame) -> None:
    """The primary figure: targets fixed at 2026-I, x = the reference semester (oldest left); writes the plotted
    values to a caption file beside it."""
    references = [r for r in ORDER if ((frame.references == r) & (frame.targets == "2026-I")).any()]
    fig, ax = plt.subplots(figsize=(2.4, 2.3), layout="constrained")
    lines = []
    for column, colour, dx, label in (("base", BASELINE_COLOUR, -0.14, BASELINE_LABEL),
                                      ("pairwise", PAIRWISE, 0.14, "Pairwise judge")):
        values = [frame[(frame.references == r) & (frame.targets == "2026-I")][column].to_numpy() for r in references]
        mark(ax, np.arange(len(references)) + dx, values, colour, 0.24, PRIMARY)
        lines += [f"{label}, references {r}: mean {mean_ci(v)[0]:.2f}, 95% t-interval "
                  f"[{mean_ci(v)[1]:.2f}, {mean_ci(v)[2]:.2f}], n = {len(v)}" for r, v in zip(references, values)]
    log_axis(ax, PRIMARY)
    ax.set_xticks(range(len(references)), references, rotation=90)
    ax.set_xlim(-0.5, len(references) - 0.5)
    ax.spines["bottom"].set_bounds(0, len(references) - 1)
    ax.set_xlabel("Reference semester")
    ax.set_ylabel("Squared error")
    legend(fig, [(BASELINE_LABEL, BASELINE_COLOUR), ("Pairwise judge", PAIRWISE)], ncol=1)
    save(fig, "horizon_primary")
    judges = frame[frame.targets == "2026-I"].drop_duplicates("references").set_index("references").judge
    (FIGURES / "horizon_primary.txt").write_text(
        "Science, targets = the eight 2026-I models, squared error per target; mean and 95% t-interval over targets.\n"
        "Judges: " + ", ".join(f"{r} {judges[r]}" for r in references) + ".\n"
        "2025-II references: the paper's task (100 items); the rest 40 items. 2024-I references (GPT-4o): the linear\n"
        "calibration slope is <= 0, so the Pairwise judge predicts the references' mean.\n" + "\n".join(lines) + "\n")


def back_forward_figure(frame: pd.DataFrame) -> None:
    """References fixed at 2024-II and at 2025-I, targets both older and newer: x = semesters from the references
    to the targets. If error depends on distance alone it rises on both sides of 0; if it depends on the targets
    being in the future, only on the right. Writes the plotted values to a caption file beside it."""
    panels = [pairs for tag, pairs in SERIES if tag.startswith("References fixed")]
    spans = [(min(ahead(*p) for p in pairs), max(ahead(*p) for p in pairs)) for pairs in panels]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.6, 2.3), layout="constrained", sharey=True,
                             width_ratios=[hi - lo + 1 for lo, hi in spans])
    lines = []
    for ax, pairs, (lo, hi) in zip(axes, panels, spans):
        pairs = sorted(pairs, key=lambda p: ahead(*p))
        xs = np.array([ahead(*p) for p in pairs])
        for column, colour, dx, label in (("base", BASELINE_COLOUR, -0.14, BASELINE_LABEL),
                                          ("pairwise", PAIRWISE, 0.14, "Pairwise judge")):
            values = [frame[(frame.references == r) & (frame.targets == t)][column].to_numpy() for r, t in pairs]
            mark(ax, xs + dx, values, colour, 0.24)
            lines += [f"{label}, references {r}, targets {t} ({ahead(r, t):+d}): mean {mean_ci(v)[0]:.2f}, 95% t-interval "
                      f"[{mean_ci(v)[1]:.2f}, {mean_ci(v)[2]:.2f}], n = {len(v)}" for (r, t), v in zip(pairs, values)]
        log_axis(ax)
        ax.axvline(0, color="grey", lw=0.8, ls="--", zorder=1)  # the references' own semester
        ax.set_xticks(xs, [f"−{-x}" if x < 0 else f"+{x}" for x in xs])
        ax.set_xlim(lo - 0.5, hi + 0.5)
        ax.spines["bottom"].set_bounds(lo, hi)
        ax.text(0, 1.03, f"References {pairs[0][0]}", transform=ax.transAxes, color="grey", ha="left", fontsize=7)
    axes[0].set_ylabel("Squared error")
    fig.supxlabel("Semesters from references to targets", fontsize=8)
    # above the panels: below, constrained layout stacks it onto the shared x label
    handles = [plt.Line2D([], [], color=c, ls="none", marker="o", ms=3.5, label=l)
               for l, c in ((BASELINE_LABEL, BASELINE_COLOUR), ("Pairwise judge", PAIRWISE))]
    fig.legend(handles=handles, loc="outside upper center", ncol=2, frameon=False)
    save(fig, "horizon_back_forward")
    judges = frame.drop_duplicates(["references", "targets"]).set_index(["references", "targets"]).judge
    (FIGURES / "horizon_back_forward.txt").write_text(
        "Science, squared error per target; mean and 95% t-interval over each task's eight targets. References fixed at\n"
        "2024-II (judge Qwen2.5-72B) and 2025-I (judge MiniMax-M1); negative = older targets. The +1 tasks are the paper's\n"
        "(100 items), the rest 40 items.\n" + "\n".join(lines) + "\n")


def combined_figure(frame: pd.DataFrame) -> None:
    """One figure, two panels on one log axis to 20. Left: targets fixed at 2026-I, x = the reference semester,
    the baseline and three protocols in the paper's colours. Right: references 2024-II (light shade) and 2025-I
    (dark shade) of each protocol, x = semesters from the references to the targets, -2 to +2, lines broken at 0.
    Writes the plotted values to a caption file beside it."""
    references = [r for r in ORDER if ((frame.references == r) & (frame.targets == "2026-I")).any()]
    offsets = [-2, -1, 1, 2]
    near = frame[frame.references.isin(["2024-II", "2025-I"])]
    near = near.assign(offset=[ahead(r, t) for r, t in zip(near.references, near.targets)])
    fig, (left, right) = plt.subplots(1, 2, figsize=(4.4 if len(SHOWN) == 1 else 5.2, 2.4), layout="constrained", sharey=True,
                                      width_ratios=[len(references), len(offsets)])
    shown = [row for row in PROTOCOL_SHADES if row[0] in SHOWN]
    series = [("base", BASELINE_LABEL, BASELINE_COLOUR)] + [(c, n, col) for c, n, col, _, _ in shown]
    dodge = dict(zip([c for c, *_ in series], np.linspace(-0.14, 0.14, len(series)) if len(series) == 2
                     else np.linspace(-0.3, 0.3, len(series))))
    lines = []
    for column, label, colour in series:
        values = [frame[(frame.references == r) & (frame.targets == "2026-I")][column].to_numpy() for r in references]
        mark(left, np.arange(len(references)) + dodge[column], values, colour, 0.56 / len(series), PRIMARY)
        lines += [f"Left, {label}, references {r}: mean {mean_ci(v)[0]:.2f}, 95% t-interval [{mean_ci(v)[1]:.2f}, "
                  f"{mean_ci(v)[2]:.2f}], n = {len(v)}" for r, v in zip(references, values)]
    slots = [(c, n, sh, r) for c, n, _, light, dark in shown for r, sh in (("2024-II", light), ("2025-I", dark))]
    spread = 0.14 if len(slots) == 2 else 0.33
    shift = dict(zip([(c, r) for c, _, _, r in slots], np.linspace(-spread, spread, len(slots))))
    for column, label, colour, semester in slots:
        g = near[near.references == semester]
        values = [g[g.offset == o][column].to_numpy() for o in offsets]
        for side in (offsets[:2], offsets[2:]):  # no line across the references' own semester
            idx = [offsets.index(o) for o in side]
            mark(right, np.array(side, float) + shift[(column, semester)], [values[i] for i in idx], colour,
                 0.48 / len(slots), PRIMARY)
        lines += [f"Right, {label}, references {semester}, {o:+d} semesters: mean {mean_ci(v)[0]:.2f}, 95% t-interval "
                  f"[{mean_ci(v)[1]:.2f}, {mean_ci(v)[2]:.2f}], n = {len(v)}" for o, v in zip(offsets, values)]
    log_axis(left, PRIMARY)
    log_axis(right, PRIMARY)
    left.set_xticks(range(len(references)), references, rotation=90)
    left.set_xlim(-0.5, len(references) - 0.5)
    left.spines["bottom"].set_bounds(0, len(references) - 1)
    left.set_xlabel("Reference semester")
    left.set_ylabel("Squared error")
    left.text(0, 1.03, "Targets 2026-I", transform=left.transAxes, color="grey", ha="left", fontsize=7)
    right.axvline(0, color="grey", lw=0.8, ls="--", zorder=1)  # the references' own semester
    right.set_xticks(offsets, [f"−{-o}" if o < 0 else f"+{o}" for o in offsets])
    right.set_xlim(-2.5, 2.5)
    right.spines["bottom"].set_bounds(-2, 2)
    right.set_xlabel("Semesters from references\nto targets")
    right.legend(handles=[plt.Line2D([], [], color=sh, ls="none", marker="o", ms=3.5, label=f"References {r}")
                          for r, sh in (("2024-II", shown[-1][3]), ("2025-I", shown[-1][4]))],
                 loc="lower left", bbox_to_anchor=(0, 1.0), ncol=1, frameon=False, handletextpad=0.1, borderaxespad=0)
    legend(fig, [(n, c) for _, n, c in series], ncol=len(series))
    save(fig, "horizon_combined")
    (FIGURES / "horizon_combined.txt").write_text(
        "Science, squared error per target; mean and 95% t-interval over each task's eight targets. Pairwise judge:\n"
        "each reference semester's best judge (2023 GPT-4-0613, 2024-I GPT-4o, 2024-II Qwen2.5-72B, 2025-I MiniMax-M1,\n"
        "2025-II MiniMax-M2); GPT-4o's linear calibration slope is <= 0, so at 2024-I the Pairwise judge predicts the\n"
        "references' mean. The one-semester-ahead tasks are the paper's (100 items), the rest 40 items. Right: light =\n"
        "2024-II references (Qwen2.5-72B), dark = 2025-I references (MiniMax-M1); negative = older targets.\n"
        + ("Embedding (full transcript) and Deterministic features: ridge fitted on the references, all 100 train items.\n"
           if len(SHOWN) > 1 else "") + "\n".join(lines) + "\n")

if __name__ == "__main__":
    FIGURES.mkdir(parents=True, exist_ok=True)
    frame = rows()
    frame.to_csv(OUT / "squared_errors.csv", index=False, float_format="%.4f")
    print(frame.groupby(["references", "targets", "judge"])[["pairwise", "base"]].mean().round(2).to_string())
    series_figure(frame)
    distance_figure(frame)
    target_figure(frame)
    primary_figure(frame)
    back_forward_figure(frame)
    combined_figure(frame)
