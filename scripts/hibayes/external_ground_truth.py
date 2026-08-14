"""Does the analysis hold up under a capability measure other than GPQA?

Every capability claim so far uses aggregate GPQA Diamond accuracy, which is the
benchmark being judged. This re-runs the label-dependent results against the
Artificial Analysis Intelligence Index v4.1.1, a nine-evaluation composite in
which GPQA is one component. Figure 11 compares the orderings. Figure 12 asks
whether the pair-ranking gradient survives the change of label.
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plotting import DARK, HIB, LIGHT, style

OUT = HIB / "readme_figures"
BANDS = [(0, 3, "under 3"), (3, 8, "3 to 8"), (8, 100, "8 and above")]


def finish(fig, n, c):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{n}{'' if c is LIGHT else '_dark'}.png", dpi=150,
                bbox_inches="tight", facecolor=c["surface"])
    if c is LIGHT:
        fig.savefig(OUT / f"{n}.svg", bbox_inches="tight", facecolor=c["surface"])
    plt.close(fig)


def load():
    aa = pd.read_csv(HIB / "artificial_analysis_index.csv").set_index("model")["aa"].to_dict()
    d = pd.read_csv(HIB / "gpqa_correctness.csv", low_memory=False,
                    usecols=["judge", "item", "model_pair", "left_model", "right_model",
                             "left_capability", "right_capability", "is_correct",
                             "self_comparison"])
    d = d[d["self_comparison"] == 0]
    gp = {}
    for m, v in list(zip(d["left_model"], d["left_capability"])) + \
                list(zip(d["right_model"], d["right_capability"])):
        gp[m] = float(v)
    info = {}
    for p in d["model_pair"].unique():
        a, b = p.split("|")
        if a not in aa or b not in aa or aa[a] == aa[b]:
            continue
        agree = (a if gp[a] > gp[b] else b) == (a if aa[a] > aa[b] else b)
        info[p] = (agree, abs(aa[a] - aa[b]), abs(gp[a] - gp[b]))
    d = d[d["model_pair"].isin(info)].copy()
    d["agree"] = d["model_pair"].map(lambda p: info[p][0])
    d["aa_gap"] = d["model_pair"].map(lambda p: info[p][1])
    d["gpqa_gap"] = d["model_pair"].map(lambda p: info[p][2])
    d["correct_aa"] = np.where(d["agree"], d["is_correct"], 1 - d["is_correct"])
    return d, info


def fig11(d, info, c):
    """Do the two capability measures order the pairs the same way?"""
    t = pd.DataFrame([dict(pair=p, agree=v[0], aa_gap=v[1], gpqa_gap=v[2])
                      for p, v in info.items()])
    fig, ax = plt.subplots(figsize=(9.8, 5.4))
    ok, bad = t[t["agree"]], t[~t["agree"]]
    ax.scatter(ok["gpqa_gap"], ok["aa_gap"], s=54, color=c["s1"], zorder=3,
               edgecolor=c["surface"], linewidth=1, label="same ordering")
    ax.scatter(bad["gpqa_gap"], bad["aa_gap"], s=70, color=c["crit"], marker="X",
               zorder=4, edgecolor=c["surface"], linewidth=1,
               label="opposite ordering")
    ax.set_xlabel("capability gap from GPQA Diamond accuracy", color=c["ink2"])
    ax.set_ylabel("capability gap from the AA Intelligence Index", color=c["ink2"])
    ax.set_title(f"The two measures agree on {int(t['agree'].sum())} of {len(t)} pair orderings",
                 color=c["ink"], loc="left", fontsize=11.5)
    ax.annotate("Rank agreement is high. The magnitudes are not proportional,\n"
                "which is what the next figure turns on.",
                (0.02, 0.97), xycoords="axes fraction", va="top", fontsize=8.6,
                color=c["muted"], linespacing=1.6)
    ax.grid(lw=0.8); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, loc="lower right", labelcolor=c["ink2"])
    fig.tight_layout(); finish(fig, 9, c)


def fig12(d, c, sizes=(1, 3, 7, 15, 31, 63, 127, 255, 511), B=400):
    """Does the wide-gap reliability survive the change of label?"""
    rng = np.random.default_rng(0)

    def maj(arr, k):
        # Without replacement, capped at the pot: k is then literally the
        # number of distinct benchmark questions used, and at k = n the value
        # is the pair observed verdict rather than a bootstrap extrapolation.
        n = len(arr)
        if k >= n:
            m = arr.mean()
            return 1.0 if m > 0.5 else (0.5 if m == 0.5 else 0.0)
        wins = 0.0
        for _ in range(B):
            m = arr[rng.choice(n, size=k, replace=False)].mean()
            wins += 1.0 if m > 0.5 else (0.5 if m == 0.5 else 0.0)
        return wins / B

    def votes(col):
        v = d.groupby(["model_pair", "item"])[col].mean().reset_index()
        v["vote"] = np.where(v[col] > 0.5, 1.0, np.where(v[col] < 0.5, 0.0, 0.5))
        return v

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.2), sharey=True)
    # left: GPQA label and GPQA-defined bands
    vg = votes("is_correct")
    vg["gap"] = vg["model_pair"].map(dict(zip(d["model_pair"], d["gpqa_gap"])))
    for (lo, hi, lab), col in zip([(0, .05, "under 0.05"), (.05, .10, "0.05 to 0.10"),
                                   (.10, 1, "0.10 and above")], ["s4", "s3", "s1"]):
        arrs = [g["vote"].to_numpy() for _, g in vg[(vg["gap"] >= lo) & (vg["gap"] < hi)]
                .groupby("model_pair")]
        arrs = [a for a in arrs if len(a) >= 8]
        if arrs:
            vals = [np.mean([maj(a, k) for a in arrs]) for k in sizes]
            if c is LIGHT:
                print("GPQA band", lab, {k: round(v, 3) for k, v in zip(sizes, vals)})
            axes[0].plot(sizes, vals,
                         marker="o", ms=5, lw=2.2, color=c[col], label=f"gap {lab} ({len(arrs)})")
    # right: AA label and AA-defined bands
    va = votes("correct_aa")
    va["gap"] = va["model_pair"].map(dict(zip(d["model_pair"], d["aa_gap"])))
    for (lo, hi, lab), col in zip(BANDS, ["s4", "s3", "s1"]):
        arrs = [g["vote"].to_numpy() for _, g in va[(va["gap"] >= lo) & (va["gap"] < hi)]
                .groupby("model_pair")]
        arrs = [a for a in arrs if len(a) >= 8]
        if arrs:
            vals = [np.mean([maj(a, k) for a in arrs]) for k in sizes]
            if c is LIGHT:
                print("AA band", lab, {k: round(v, 3) for k, v in zip(sizes, vals)})
            axes[1].plot(sizes, vals,
                         marker="s", ms=5, lw=2.2, color=c[col], label=f"gap {lab} ({len(arrs)})")
    for ax, ttl in zip(axes, ["GPQA accuracy as ground truth",
                              "AA Intelligence Index as ground truth"]):
        ax.axhline(0.5, color=c["crit"], lw=1.3, ls=(0, (5, 3)))
        ax.set_xscale("log"); ax.set_ylim(0.42, 1.04)
        ax.set_xlabel("distinct question items pooled per pair", color=c["ink2"])
        ax.set_title(ttl, color=c["ink"], loc="left", fontsize=10.5)
        ax.grid(lw=0.8); ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=8.6, loc="lower right", labelcolor=c["ink2"])
    axes[0].set_ylabel("chance the pooled verdict ranks the pair correctly", color=c["ink2"])
    fig.suptitle("Pooled judgments converge to a pair-specific direction; for some pairs that direction is wrong",
                 color=c["ink"], fontsize=13, fontweight="semibold", x=0.005, ha="left", y=1.03)
    fig.tight_layout(); finish(fig, 2, c)


def main():
    d, info = load()
    print(f"pairs with both AA scores and no tie: {len(info)}")
    print(f"orderings agree: {sum(v[0] for v in info.values())} of {len(info)}")
    for c in (LIGHT, DARK):
        style(c)
        fig11(d, info, c)
        fig12(d, c)
    print("wrote figures 11 and 12")


if __name__ == "__main__":
    main()
