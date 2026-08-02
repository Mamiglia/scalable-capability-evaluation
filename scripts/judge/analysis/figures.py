"""Generate the judge-pilot figures from the stored tables.

Reads verdicts.csv, models.csv, summary.csv, judge_fits.csv; writes numbered
png+svg figures to <dir>/figures/. Judges keep a fixed colour across figures.

Usage:
  uv run python scripts/judge/analysis/figures.py
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from comparisons import COMPARISONS, MODELS
from plots import INK2, MUTED, PALETTE, chance, dot_ci, legend, plt, save, style
from stats import counterbalanced, sem_ci, wilson

JUDGES = ["gemini-3.1-flash-lite", "qwen3.5-122b-a10b", "gemma-4-31b-it",
          "qwen3.6-35b-a3b", "qwen3-32b", "gemini-2.5-flash", "gpt-4o"]
JCOLOR = dict(zip(JUDGES, PALETTE))


def lab(model):
    return MODELS.get(model, {}).get("label", model)


def run_key(row):
    return (row["judge"], row["major"], row["minor"])


def ordered(summary):
    """Summary rows in fixed judge order, by gap within judge; self-pairs first."""
    summary = summary.copy()
    summary["jorder"] = summary["judge"].map({j: i for i, j in enumerate(JUDGES)})
    return summary.sort_values(["jorder", "d_logit"]).reset_index(drop=True)


def row_positions(summary):
    """y position per row, with a gap between judge blocks (top row = y 0)."""
    ys, y = [], 0.0
    prev = None
    for _, r in summary.iterrows():
        if prev is not None and r["judge"] != prev:
            y += 1.2
        ys.append(y)
        y += 1.0
        prev = r["judge"]
    return np.array(ys)


def pair_label(r):
    if r["major"] == r["minor"]:
        return f"{lab(r['major'])} (self)"
    return f"{lab(r['major'])} v {lab(r['minor'])}"


# --- 1. per-run overview -------------------------------------------------

def fig_run_overview(summary, out):
    s = ordered(summary)
    ys = row_positions(s)
    fig, ax = plt.subplots(figsize=(4.6, 0.16 * len(s) + 1.2))
    for y, (_, r) in zip(ys, s.iterrows()):
        note = f"{r.score:.2f} (n={r.n_items}) {r.stars}"
        dot_ci(ax, y, r.score, r.score_lo, r.score_hi, JCOLOR[r.judge],
               note=note, open_marker=r.major == r.minor)
    for j in s.judge.unique():
        block = ys[(s.judge == j).to_numpy()]
        ax.annotate(j, xy=(-0.02, block.min() - 0.9), xycoords=("axes fraction", "data"),
                    ha="right", va="center", color=INK2, fontsize=7.5, fontweight="bold",
                    annotation_clip=False)
    ax.set_yticks(ys)
    ax.set_yticklabels([pair_label(r) for _, r in s.iterrows()], fontsize=6.5)
    ax.invert_yaxis()
    chance(ax, 0.5, axis="x")
    ax.set_xlabel("judge score (counterbalanced)")
    ax.set_xlim(0, 1)
    style(ax, grid_axis="x")
    save(fig, out / "1_run_overview")


# --- 2. gap curve --------------------------------------------------------

def fig_gap_curve(summary, fits, out):
    fig, ax = plt.subplots(figsize=(4.0, 2.9))
    for _, r in summary.iterrows():
        c = JCOLOR[r.judge]
        ax.errorbar(r.d_logit, r.score, yerr=[[r.score - r.score_lo], [r.score_hi - r.score]],
                    fmt="o", ms=4.5, color=c, ecolor=c, elinewidth=0.9, capsize=0,
                    mfc="white" if r.major == r.minor else c, mew=1.1, alpha=0.9, zorder=3)
    ladder = summary[summary.groups.fillna("").str.contains("gap_ladder")]
    for _, f in fits.iterrows():
        hi = ladder.loc[ladder.judge == f.judge, "d_logit"].max()
        x = np.linspace(0, hi, 100)
        ax.plot(x, 1 / (1 + np.exp(-(f.a + f.d * x))), color=JCOLOR[f.judge],
                lw=1.4, label=f"{f.judge}  (d={f.d:.1f})", zorder=2)
    chance(ax)
    ax.set_xlabel("capability gap (dLogit)")
    ax.set_ylabel("judge score")
    ax.set_ylim(0, 1)
    style(ax)
    legend(ax, loc="lower right", fontsize=6.5)
    save(fig, out / "2_gap_curve")


# --- 3. controls ---------------------------------------------------------

def fig_controls(summary, out):
    s = ordered(summary)
    ys = row_positions(s)
    fig, axes = plt.subplots(1, 3, sharey=True, figsize=(7.0, 0.155 * len(s) + 1.1))
    cols = [("slot_a", "slot-A pick rate"), ("disagree_rate", "order disagreement"),
            ("noanswer_rate", "no-answer rate")]
    for ax, (col, xlabel) in zip(axes, cols):
        for y, (_, r) in zip(ys, s.iterrows()):
            if col == "slot_a":
                dot_ci(ax, y, r.slot_a, r.slot_a_lo, r.slot_a_hi, JCOLOR[r.judge],
                       open_marker=r.major == r.minor)
            else:
                ax.plot(r[col], y, "o", ms=4.5, color=JCOLOR[r.judge],
                        mfc="white" if r.major == r.minor else JCOLOR[r.judge],
                        mew=1.1, zorder=3)
        if col == "slot_a":
            chance(ax, 0.5, axis="x")
        ax.set_xlabel(xlabel)
        ax.set_xlim(-0.02, 1.02)
        style(ax, grid_axis="x")
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([pair_label(r) for _, r in s.iterrows()], fontsize=6.5)
    axes[0].invert_yaxis()
    for j in s.judge.unique():
        block = ys[(s.judge == j).to_numpy()]
        axes[0].annotate(j, xy=(-0.04, block.min() - 0.9), xycoords=("axes fraction", "data"),
                         ha="right", va="center", color=INK2, fontsize=7.5,
                         fontweight="bold", annotation_clip=False)
    save(fig, out / "3_controls")


# --- 4. judge strength ---------------------------------------------------

def fig_judge_strength(verdicts, models, out):
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    pairs = sorted({(maj, mnr) for _, maj, mnr in COMPARISONS["judge_strength"]})
    for maj, mnr in pairs:
        pts = []
        for j, m2, n2 in COMPARISONS["judge_strength"]:
            if (m2, n2) != (maj, mnr):
                continue
            run = verdicts[(verdicts.judge == j) & (verdicts.major == maj) & (verdicts.minor == mnr)]
            if not len(run):
                continue
            m, lo, hi = sem_ci(counterbalanced(run))
            pts.append((models.loc[j, "accuracy"], m, lo, hi, j))
        pts.sort()
        xs = [p[0] for p in pts]
        ax.plot(xs, [p[1] for p in pts], "-", color=MUTED, lw=1.0, zorder=2)
        for x, m, lo, hi, j in pts:
            ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], fmt="o", ms=5, color=JCOLOR[j],
                        ecolor=JCOLOR[j], elinewidth=0.9, capsize=0, zorder=3)
        ax.annotate(f"{lab(maj)} v {lab(mnr)}", xy=(xs[-1], pts[-1][1]),
                    xytext=(4, 3), textcoords="offset points",
                    color=INK2, fontsize=6.5)
    chance(ax)
    ax.set_xlabel("judge accuracy on GPQA")
    ax.set_ylabel("judge score")
    style(ax)
    save(fig, out / "4_judge_strength")


# --- 5. confidence calibration -------------------------------------------

def fig_calibration(verdicts, out):
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    parsed = verdicts.dropna(subset=["picked_major", "confidence"])
    parsed = parsed[parsed.major != parsed.minor]
    for j in JUDGES:
        d = parsed[parsed.judge == j]
        if not len(d):
            continue
        xs, ms, los, his = [], [], [], []
        for conf, cell in d.groupby(d.confidence.round().astype(int)):
            if len(cell) < 10:
                continue
            k = int(cell.picked_major.sum())
            lo, hi = wilson(k, len(cell))
            xs.append(conf), ms.append(k / len(cell)), los.append(lo), his.append(hi)
        ax.errorbar(xs, ms, yerr=[np.array(ms) - los, np.array(his) - ms],
                    fmt="-o", ms=4.5, color=JCOLOR[j], ecolor=JCOLOR[j],
                    elinewidth=0.9, capsize=0, lw=1.3, label=j)
    chance(ax)
    ax.set_xlabel("stated confidence (order-level)")
    ax.set_ylabel("P(picked stronger)")
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_ylim(0, 1)
    style(ax)
    legend(ax, loc="lower left", fontsize=6)
    save(fig, out / "5_calibration")


# --- 6. length confound --------------------------------------------------

def fig_length(verdicts, summary, out):
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.0, 2.7),
                                      gridspec_kw={"width_ratios": [1.2, 1]})
    parsed = verdicts.dropna(subset=["picked_major", "len_major", "len_minor"])
    parsed = parsed[parsed.len_major != parsed.len_minor]
    chosen_len = np.where(parsed.picked == parsed.slot_of_major, parsed.len_major, parsed.len_minor)
    other_len = np.where(parsed.picked == parsed.slot_of_major, parsed.len_minor, parsed.len_major)
    parsed = parsed.assign(picked_longer=chosen_len > other_len)
    for j in JUDGES:
        d = parsed[parsed.judge == j]
        if not len(d):
            continue
        k = int(d.picked_longer.sum())
        lo, hi = wilson(k, len(d))
        y = JUDGES.index(j)
        dot_ci(left, y, k / len(d), lo, hi, JCOLOR[j], note=f"n={len(d)}")
    left.set_yticks(range(len(JUDGES)))
    left.set_yticklabels(JUDGES, fontsize=7)
    left.invert_yaxis()
    chance(left, 0.5, axis="x")
    left.set_xlabel("P(picked the longer transcript)")
    style(left, grid_axis="x")

    per_run = (verdicts.groupby(["judge", "major", "minor"])
               .apply(lambda d: (d.len_major - d.len_minor).mean() / 1000, include_groups=False)
               .rename("dlen").reset_index())
    m = summary.merge(per_run, on=["judge", "major", "minor"])
    for _, r in m.iterrows():
        right.plot(r.dlen, r.score, "o", ms=4.5, color=JCOLOR[r.judge],
                   mfc="white" if r.major == r.minor else JCOLOR[r.judge], mew=1.1)
    chance(right)
    right.axvline(0, color=MUTED, lw=0.8)
    right.set_xlabel("mean length gap, stronger − weaker (k chars)")
    right.set_ylabel("judge score")
    style(right)
    save(fig, out / "6_length_confound")


# --- 7. item difficulty --------------------------------------------------

def fig_item_difficulty(verdicts, out, difficulties=Path("irt_results/epoch_gpqa/gpqa_difficulties.csv")):
    cross = verdicts[verdicts.major != verdicts.minor].dropna(subset=["picked_major"])
    per_item = (cross.groupby(["judge", "major", "minor", "item"])["picked_major"].mean()
                .groupby("item").mean().rename("score").reset_index())
    d = per_item.merge(pd.read_csv(difficulties)[["item", "b_1pl"]], on="item")
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    ax.scatter(d.b_1pl, d.score, s=12, color=MUTED, alpha=0.45, lw=0, zorder=2)
    qs = pd.qcut(d.b_1pl, 5)
    for interval, cell in d.groupby(qs, observed=True):
        m, lo, hi = sem_ci(cell.score.to_numpy())
        x = cell.b_1pl.mean()
        ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], fmt="o", ms=6, color=PALETTE[0],
                    ecolor=PALETTE[0], elinewidth=1.2, capsize=0, zorder=4,
                    markeredgecolor="white", markeredgewidth=0.8)
    chance(ax)
    ax.set_xlabel("item difficulty b (1PL)")
    ax.set_ylabel("mean judge score")
    ax.set_ylim(0, 1)
    style(ax)
    ax.text(0.03, 0.05, f"n={len(d)} items · dots pooled over runs · quintile means",
            transform=ax.transAxes, color=MUTED, fontsize=6.5)
    save(fig, out / "7_item_difficulty")


# --- 8. temporal ---------------------------------------------------------

def fig_temporal(verdicts, out):
    rungs = list(dict.fromkeys((maj, mnr) for _, maj, mnr in COMPARISONS["temporal"]))
    judges = list(dict.fromkeys(j for j, _, _ in COMPARISONS["temporal"]))
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    for j in judges:
        xs, ms, los, his = [], [], [], []
        for i, (maj, mnr) in enumerate(rungs):
            run = verdicts[(verdicts.judge == j) & (verdicts.major == maj) & (verdicts.minor == mnr)]
            if not len(run):
                continue
            m, lo, hi = sem_ci(counterbalanced(run))
            xs.append(i), ms.append(m), los.append(lo), his.append(hi)
        ax.errorbar(xs, ms, yerr=[np.array(ms) - los, np.array(his) - ms],
                    fmt="-o", ms=5, lw=1.3, color=JCOLOR[j], ecolor=JCOLOR[j],
                    elinewidth=0.9, capsize=0, label=j)
    chance(ax)
    ax.set_xticks(range(len(rungs)))
    releases = [MODELS.get(maj, {}).get("release") for maj, _ in rungs]
    ax.set_xticklabels([f"{lab(maj)}\nv {lab(mnr)}" + (f"\n({r[:4]} pair)" if r else "")
                        for (maj, mnr), r in zip(rungs, releases)], fontsize=6)
    ax.set_ylabel("judge score")
    ax.set_ylim(0, 1)
    style(ax)
    legend(ax, loc="lower left", fontsize=6)
    save(fig, out / "8_temporal")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, default=Path("judge_results/judge-pilot"))
    args = ap.parse_args()

    verdicts = pd.read_csv(args.dir / "verdicts.csv")
    models = pd.read_csv(args.dir / "models.csv").set_index("model")
    summary = pd.read_csv(args.dir / "summary.csv")
    fits = pd.read_csv(args.dir / "judge_fits.csv")
    out = args.dir / "figures"
    out.mkdir(parents=True, exist_ok=True)

    fig_run_overview(summary, out)
    fig_gap_curve(summary, fits, out)
    fig_controls(summary, out)
    fig_judge_strength(verdicts, models, out)
    fig_calibration(verdicts, out)
    fig_length(verdicts, summary, out)
    fig_item_difficulty(verdicts, out)
    fig_temporal(verdicts, out)
    print(f"8 figures -> {out}")


if __name__ == "__main__":
    main()
