#!/usr/bin/env python3
"""Judge accuracy vs *signed* reference-score gap, per benchmark.

Like 07, but the gap keeps its sign: better - worse, where "better" is the model
with the higher collection accuracy (the judging ground truth). Negative gap =
the reference score disagrees with collection accuracy about who is stronger.
Pairs that also have reasoning_on runs get a second dot chained to the first,
showing how accuracy moves when reasoning traces are shown.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

INK = "#17202A"
BASE = "#9aa3ab"
BENCHES = ["GPQA", "MBPP"]
# (column, axis label, scale, signed by the empirical winner). Reference scores are
# signed better - worse, so negative means the reference disagrees with the ground
# truth. The bench score defines that ground truth, so signing it that way is
# degenerate: it uses major - minor instead (negative = the designated-stronger
# model actually scored lower on this bench).
METRICS = [("intelligence_index", "Δ Intelligence", 1, True),
           ("coding_index", "Δ Coding index", 1, True),
           ("scicode", "Δ SciCode (pp)", 100, True),
           ("terminalbench_hard", "Δ Terminal-Bench (pp)", 100, True),
           ("accuracy", "Δ Bench score (pp)", 100, False)]
MATCH = ["judge", "benchmark", "pair", "source_repeat", "item", "order"]


def cells(verdicts: pd.DataFrame, aa: pd.DataFrame) -> pd.DataFrame:
    """Per (benchmark, pair): signed gaps, off accuracy, and paired reasoning delta.

    Judge is the replication unit, so accuracies are averaged per judge first.
    """
    valid = verdicts[verdicts.correct.notna()]
    off = valid[valid.condition == "reasoning_off"]
    rows = (off.groupby(["benchmark", "pair", "judge"]).correct.mean()
            .groupby(["benchmark", "pair"]).mean().rename("accuracy").reset_index())

    matched = off[MATCH + ["correct"]].merge(
        valid[valid.condition == "reasoning_on"][MATCH + ["correct"]],
        on=MATCH, suffixes=("_off", "_on"))
    if len(matched):
        delta = (matched.eval("d = correct_on - correct_off")
                 .groupby(["benchmark", "pair", "judge"]).d.mean()
                 .groupby(["benchmark", "pair"]).mean().rename("delta"))
        rows = rows.join(delta, on=["benchmark", "pair"])
    else:
        rows["delta"] = np.nan

    first = off.groupby(["benchmark", "pair"]).first()
    sign = np.where(first.better_model == first.major, 1, -1)
    for metric, _, scale, by_better in METRICS:
        # bench score comes from the collection itself, everything else from AA
        score = first[["major_accuracy", "minor_accuracy"]] if metric == "accuracy" else pd.DataFrame(
            {side: first[side].map(aa[metric]) for side in ("major", "minor")})
        gap = (score.iloc[:, 0] - score.iloc[:, 1]) * scale
        rows[f"{metric}_gap"] = (gap * sign if by_better else gap).to_numpy()
    return rows


def limits(values: pd.Series) -> tuple[float, float]:
    """Round to a nice step so both endpoints carry a tick, with 0 always visible."""
    step = next(s for s in (2, 5, 10, 25, 50) if values.abs().max() / s <= 2)
    return min(0, step * np.floor(values.min() / step)), max(0, step * np.ceil(values.max() / step))


def draw(rows: pd.DataFrame, out: Path) -> None:
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "svg.fonttype": "none"})
    fig, axes = plt.subplots(len(METRICS), len(BENCHES), figsize=(3.0, 4.9),
                             sharey=True, layout="constrained")
    for row, (stem, label, _, by_better) in enumerate(METRICS):
        metric = f"{stem}_gap"
        # reference rows share x so the panels compare; bench scores do not, because
        # MBPP's gaps are an order of magnitude smaller than GPQA's
        shared = limits(rows[metric].dropna()) if by_better else None
        for col, bench in enumerate(BENCHES):
            ax = axes[row, col]
            cell = rows[(rows.benchmark == bench)].dropna(subset=[metric])
            lo, hi = shared or limits(cell[metric])
            ax.axhline(50, color="#777777", linestyle="--", lw=.8, zorder=0)
            ax.axvline(0, color="#777777", linestyle="--", lw=.8, zorder=0)
            chained = cell.dropna(subset=["delta"])
            ax.vlines(chained[metric], chained.accuracy * 100,
                      (chained.accuracy + chained.delta) * 100, color=INK, lw=.8, zorder=2)
            ax.scatter(cell[metric], cell.accuracy * 100, s=9, color=BASE, zorder=3)
            # unfilled so the reasoning-off dot it is chained to stays visible underneath
            ax.scatter(chained[metric], (chained.accuracy + chained.delta) * 100, s=16,
                       facecolor="none", edgecolor=INK, linewidth=.7, zorder=4)
            if len(cell) >= 3 and cell[metric].nunique() > 1:
                rho = spearmanr(cell[metric], cell.accuracy).statistic
                slope, intercept = np.polyfit(cell[metric], cell.accuracy * 100, 1)
                fit_x = np.array([cell[metric].min(), cell[metric].max()])
                ax.plot(fit_x, slope * fit_x + intercept, color=INK, lw=1, alpha=.55, zorder=1)
                ax.annotate(f"ρ={rho:.2f}, n={len(cell)}", (1, 1.02), xycoords="axes fraction",
                            fontsize=6, color="#666666", va="bottom", ha="right")
            if row == 0:
                ax.set_title(bench, loc="left", fontsize=7, color="#666666")
            if col == 0:
                ax.set_ylabel("Accuracy (%)")
            ax.set_xlabel(label)
            ax.set_ylim(0, 100)
            ax.set_yticks([0, 50, 100])
            ax.set_xlim(lo, hi)
            ax.set_xticks(sorted({lo, 0, hi}))
            ax.spines[["top", "right"]].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_position(("outward", 5))
    handles = [plt.Line2D([], [], marker="o", ls="", ms=3, color=BASE, label="reasoning off"),
               plt.Line2D([], [], marker="o", ls="", ms=4, mfc="none", mec=INK, label="reasoning on"),
               plt.Line2D([], [], color=INK, lw=1, alpha=.55, label="least-squares fit")]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False,
               handlelength=1.4, columnspacing=1.2)
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def self_check() -> None:
    base = {"benchmark": "GPQA", "pair": "a vs b", "judge": "j", "source_repeat": "0:0",
            "better_model": "b", "major": "a", "minor": "b",
            "major_accuracy": .3, "minor_accuracy": .5}
    aa = pd.DataFrame({"intelligence_index": {"a": 10.0, "b": 40.0}}).reindex(
        columns=[m for m, *_ in METRICS])
    toy = pd.DataFrame([{**base, "condition": c, "item": i, "order": 1, "correct": v}
                        for c, v in (("reasoning_off", 0.0), ("reasoning_on", 1.0))
                        for i in ("i1", "i2")])
    row = cells(toy, aa).iloc[0]
    assert row.intelligence_index_gap == 30.0, "gap is better - worse, signed"
    assert row.accuracy_gap == -20.0, "bench-score gap is major - minor, in points"
    assert row.accuracy == 0 and row.delta == 1
    print("self-check passed")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verdicts", type=Path, default=Path("analysis/pilot-results/verdicts.csv"))
    parser.add_argument("--aa-scores", type=Path, default=Path("configs/comparisons/aa_scores.csv"))
    parser.add_argument("--out", type=Path, default=Path("analysis/pilot-results/11_reference_gap_signed"))
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return 0
    verdicts = pd.read_csv(args.verdicts, low_memory=False)
    aa = pd.read_csv(args.aa_scores, comment="#").set_index("model")
    unscored = sorted((set(verdicts.major) | set(verdicts.minor)) - set(aa.index))
    if unscored:
        print(f"no reference scores for {unscored} — rerun scripts/judge/fetch_aa_scores.py")
    rows = cells(verdicts[verdicts.benchmark.isin(BENCHES)], aa)
    draw(rows, args.out)
    rows.to_csv(args.out.with_name(args.out.name + "_cells.csv"), index=False)
    print(f"wrote {args.out}.png / .svg ({len(rows)} pairs, "
          f"{rows.delta.notna().sum()} with reasoning_on)")
    for stem, label, *_ in METRICS:  # dots missing from a row = reference table gaps
        have = rows[f"{stem}_gap"].notna()
        print(f"  {label}: {have.sum()}/{len(rows)} pairs scored, "
              f"{(have & rows.delta.notna()).sum()} of them with reasoning_on")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
