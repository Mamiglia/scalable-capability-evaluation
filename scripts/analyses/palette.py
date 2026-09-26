"""The protocol/domain palette: swatch sheet + example plots (synthetic data).

    uv run python scripts/analyses/palette.py

Writes analysis/palette.{png,svg} and analysis/palette_examples.{png,svg}.
Import PROTOCOL / DOMAIN / MATCHUP from here in real figures.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.stats import t

sys.path.insert(0, str(Path(__file__).parent))
from plots import INK, INK2, MUTED, legend, save, style  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

PROTOCOL = {
    "Length": "#f1a48a",
    "Length+regex": "#eb6834",
    "Embedding": "#4a3aa7",
    "LLM Rater": "#1baf7a",
    "LLM Matchup": "#2a78d6",
}
# Triad around LLM Matchup, for its variants; TENUOUS for weak/greyed-out marks.
MATCHUP = {"Pairwise": "#2a78d6", "Mixed": "#d62a78", "Ref pairs": "#78d62a"}
TENUOUS = "#f8cbbc"
DOMAIN = {"SCIENCE": "o", "MATH": "s", "CODING": "^", "CYBER": "D"}
YMAX = 200  # % of ref-mean; anything above is drawn as a caret at the ceiling


def offset(ax, grid_axis="y"):
    style(ax, grid_axis=grid_axis)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_position(("outward", 5))


def _blocks(ax, tag, rows, markers=False):
    """One row of big colour blocks (or domain markers), labelled underneath."""
    ax.text(0, 1.30, tag, fontsize=7, color=MUTED, ha="left", va="bottom")
    for i, (name, c) in enumerate(rows.items()):
        if markers:
            ax.plot(i + 0.5, 0.5, c, ms=6, color=INK, mec=INK)
        else:
            ax.add_patch(plt.Rectangle((i + 0.06, 0), 0.88, 1.15, color=c, lw=0))
            ax.text(i + 0.5, -0.5, c, fontsize=5.5, color=MUTED, ha="center", va="top")
        ax.text(i + 0.5, -0.12, name, fontsize=7, color=INK2, ha="center", va="top")
    ax.set_xlim(0, 5)
    ax.set_ylim(-1.1, 1.3)
    ax.set_axis_off()


def sheet(out):
    fig, axes = plt.subplots(3, 1, figsize=(4.2, 2.6))
    _blocks(axes[0], "Protocols", PROTOCOL)
    _blocks(axes[1], "Matchup variants (triad) + tenuous",
            {**MATCHUP, "Tenuous": TENUOUS})
    _blocks(axes[2], "Domains", DOMAIN, markers=True)
    fig.set_layout_engine("constrained", h_pad=0.0)
    save(fig, out)


def examples(out):
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 3, figsize=(4.4, 2.1), width_ratios=[1, 0.8, 0.8],
                             layout="constrained")

    # (a) MSE vs item budget: continuous x -> shaded bands, one line per protocol.
    ax = axes[0]
    n = np.array([5, 10, 25, 50, 100])
    for i, (name, c) in enumerate(PROTOCOL.items()):
        floor = 0.95 - 0.15 * i
        mse = floor + 1.6 / np.sqrt(n)
        se = 0.12 / np.sqrt(n) + 0.02
        ax.plot(n, mse, color=c, lw=1, label=name)
        ax.fill_between(n, mse - se, mse + se, color=c, alpha=0.18, lw=0)
    ax.set_xscale("log")
    ax.set_xticks([5, 25, 100], ["5", "25", "100"])
    ax.set_xlim(5, 100)
    ax.minorticks_off()
    ax.set_ylim(0, 1.6)
    ax.set_yticks([0, 0.8, 1.6])
    ax.set_xlabel("Items per task")
    ax.set_ylabel("MSE (θ)")
    offset(ax)

    # (b) Per-protocol mean +- 95% CI across judges (unit = judge, N = 6).
    ax = axes[1]
    for i, (name, c) in enumerate(PROTOCOL.items()):
        m, ci = 1.0 - 0.14 * i, 0.1 + 0.02 * i
        ax.errorbar(m, -i, xerr=ci, color=c, lw=1, capsize=0, marker="o", ms=4)
    ax.set_yticks([])
    ax.set_ylim(-len(PROTOCOL) + 0.5, 0.5)
    ax.set_xlim(0, 1.6)
    ax.set_xticks([0, 0.8, 1.6])
    ax.set_xlabel("MSE (θ)")
    offset(ax, grid_axis="x")
    ax.spines["left"].set_visible(False)

    # (c) Matchup variants x domains: triad colors, domain markers.
    ax = axes[2]
    for j, (dom, mk) in enumerate(DOMAIN.items()):
        for k, (var, c) in enumerate(MATCHUP.items()):
            x = j + (k - 1) * 0.22
            y = 0.6 + 0.12 * k + rng.normal(0, 0.06)
            ax.errorbar(x, y, yerr=0.12, color=c, lw=1, capsize=0, marker=mk, ms=4,
                        label=var if j == 0 else None)
    ax.set_xticks(range(len(DOMAIN)))
    ax.set_xticklabels(list(DOMAIN))
    ax.set_ylim(0, 1.6)
    ax.set_yticks([0, 0.8, 1.6])
    ax.set_ylabel("MSE (θ)")
    offset(ax)
    ax.tick_params(axis="x", labelrotation=90)

    # Figure-level legends below the row: constrained layout reserves the space,
    # so the axes keep the full canvas height instead of being squeezed by them.
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower left", ncol=3,
               frameon=False, handlelength=1.2, columnspacing=1.0, handletextpad=0.4)
    fig.legend(*axes[2].get_legend_handles_labels(), loc="outside lower right", ncol=1,
               frameon=False, handlelength=1.2, handletextpad=0.4)
    save(fig, out)


def real(out):
    """Same palette on real numbers: analysis/judge/protocol_compare.csv."""
    import pandas as pd

    df = pd.read_csv("analysis/judge/protocol_compare.csv").query("targets == 'all'")
    cols = {"glm": "Length+regex", "embedding": "Embedding",
            "direct": "LLM Rater", "pairwise": "LLM Matchup"}
    fig, axes = plt.subplots(1, 2, figsize=(4.2, 1.9), width_ratios=[1.35, 1],
                             sharey=True, layout="constrained")

    # Left: MSE as % of the ref-mean baseline, per domain. Unit = semester;
    # N >= 3 gets a t-based 95% CI, N < 3 shows the per-semester points instead.
    ax = axes[0]
    for j, (dom, mk) in enumerate(DOMAIN.items()):
        sub = df[df.domain == dom.lower()]
        for k, (col, name) in enumerate(cols.items()):
            pct = 100 * sub[col] / sub.baseline
            x = j + (k - 1.5) * 0.18
            c = PROTOCOL[name]
            if len(pct) >= 3:
                m, se = pct.mean(), pct.sem()
                if m > YMAX:  # off-scale mean: caret at the ceiling, no bar
                    ax.plot(x, YMAX, "^", ms=3.5, color=c, clip_on=False)
                    continue
                ax.errorbar(x, m, yerr=t.ppf(0.975, len(pct) - 1) * se, color=c, lw=1,
                            capsize=0, marker=mk, ms=4)
            else:
                ax.plot([x] * len(pct), pct.clip(upper=YMAX), mk, ms=3, mfc="none",
                        mew=0.8, color=c)
    ax.axhline(100, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.set_xticks(range(len(DOMAIN)))
    ax.set_xticklabels(list(DOMAIN))
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(0, YMAX)
    ax.set_yticks([0, 100, YMAX])
    ax.set_ylabel("MSE (% of ref-mean)")
    offset(ax)
    ax.tick_params(axis="x", labelrotation=90)

    # Right: the same cells by semester, pooled over domains.
    ax = axes[1]
    sems = sorted(df.semester.unique())
    for col, name in cols.items():
        g = df.groupby("semester").apply(lambda s: 100 * (s[col] / s.baseline).mean(),
                                         include_groups=False)
        y = np.array([g[s] for s in sems])
        ax.plot(range(len(sems)), np.minimum(y, YMAX), color=PROTOCOL[name], lw=1,
                marker="o", ms=3, label=name)
        for i in np.flatnonzero(y > YMAX):  # off-scale cell -> caret at the ceiling
            ax.plot(i, YMAX, "^", ms=3.5, color=PROTOCOL[name], clip_on=False)
    ax.axhline(100, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.set_xticks(range(len(sems)))
    ax.set_xticklabels(sems)
    ax.set_xlim(-0.3, len(sems) - 0.7)
    offset(ax)
    ax.tick_params(axis="x", labelrotation=90)

    fig.legend(*axes[1].get_legend_handles_labels(), loc="outside lower center", ncol=2,
               frameon=False, handlelength=1.2, columnspacing=1.0, handletextpad=0.4)
    save(fig, out)


if __name__ == "__main__":
    out = Path("analysis")
    out.mkdir(exist_ok=True)
    sheet(out / "palette")
    examples(out / "palette_examples")
    real(out / "palette_real")
    assert set(PROTOCOL) & set(MATCHUP) == set()  # triad reserved for matchup variants
    print(f"wrote {out}/palette.png, {out}/palette_examples.png, {out}/palette_real.png")
