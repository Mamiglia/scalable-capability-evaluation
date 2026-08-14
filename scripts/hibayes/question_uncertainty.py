"""Is a misranked pair systematically inverted, or short of questions?

The pooling curves resample a fixed pot of observed question votes, so their
plateau shows that aggregation converges to each pair's empirical direction. It
cannot by itself show that genuinely new questions would not help: a pair whose
true vote rate is 0.52 can present an observed pot at 0.47 by sampling noise,
and resampling that pot only becomes more confidently wrong.

Two checks separate the readings, both using the unique question as the
sampling unit.

Bootstrap: resample the unique questions with replacement and recompute the
pair share. A misranked pair whose interval sits entirely below 0.5 is
systematically inverted on this question distribution. An interval crossing
0.5 is compatible with finite-question uncertainty.

Split-half: split the unique questions into two disjoint halves at random,
many times. A pair that is independently misranked on both halves is inverted
on disjoint evidence, which resampling noise does not produce.

Writes question_uncertainty.json and readme figure 14.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from plotting import DARK, HIB, LIGHT, style, write_json
from pooling_by_question import load, question_votes

OUT = HIB / "readme_figures"
SEED = 0
B_BOOT = 4000
B_SPLIT = 1000


def per_pair(votes):
    rng = np.random.default_rng(SEED)
    rows = []
    for pair, g in votes.groupby("model_pair"):
        v = g["vote"].to_numpy()
        n = len(v)
        share = float(v.mean())
        idx = rng.integers(0, n, size=(B_BOOT, n))
        means = v[idx].mean(axis=1)
        lo, hi = np.percentile(means, [2.5, 97.5])
        p_below = float((means < 0.5).mean() + 0.5 * (means == 0.5).mean())

        both_wrong = agree = 0
        half = n // 2
        for _ in range(B_SPLIT):
            perm = rng.permutation(n)
            a, b = v[perm[:half]].mean(), v[perm[half:]].mean()
            agree += (a > 0.5) == (b > 0.5)
            both_wrong += (a < 0.5) and (b < 0.5)
        rows.append(dict(pair=pair, n_questions=int(n), share=share,
                         ci_lo=float(lo), ci_hi=float(hi),
                         p_below_half=p_below,
                         split_agreement=agree / B_SPLIT,
                         split_both_wrong=both_wrong / B_SPLIT,
                         gap=float(g["gap"].iloc[0]),
                         correct=bool(share > 0.5)))
    return pd.DataFrame(rows).sort_values("share")


def fig(t, c, n=14):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_, (axl, axr) = plt.subplots(1, 2, figsize=(13.0, 5.8),
                                    gridspec_kw=dict(width_ratios=[1.15, 1]))
    y = np.arange(len(t))
    for i, r in enumerate(t.itertuples()):
        col = c["s1"] if r.correct else c["crit"]
        axl.hlines(i, r.ci_lo, r.ci_hi, color=col, lw=2.0, zorder=2)
        axl.scatter([r.share], [i], s=40, color=col, zorder=3,
                    edgecolor=c["surface"], linewidth=0.9)
    axl.axvline(0.5, color=c["axis"], lw=1.4, zorder=1)
    axl.set_yticks(y)
    axl.set_yticklabels(t["pair"], fontsize=7.2, color=c["ink2"])
    axl.set_xlabel("share of question votes favouring the stronger model,\n"
                   "with a 95% interval from resampling unique questions",
                   color=c["ink2"], fontsize=9.5)
    axl.set_title("Which misrankings could new questions fix?",
                  color=c["ink"], loc="left", fontsize=11)
    axl.grid(axis="x", lw=0.8)
    axl.set_axisbelow(True)

    ok, bad = t[t["correct"]], t[~t["correct"]]
    axr.scatter(ok["share"], ok["split_agreement"], s=46, color=c["s1"], zorder=3,
                edgecolor=c["surface"], linewidth=0.9, label="pair ranked correctly")
    axr.scatter(bad["share"], bad["split_agreement"], s=60, color=c["crit"],
                marker="X", zorder=4, edgecolor=c["surface"], linewidth=0.9,
                label="pair ranked wrong")
    for i, r in enumerate(bad.sort_values("share").itertuples()):
        axr.annotate(f"{r.split_both_wrong:.0%} of splits:\nboth halves wrong",
                     (r.share, r.split_agreement), xytext=(6, -16 - 26 * i),
                     textcoords="offset points", fontsize=7.4, color=c["ink2"],
                     linespacing=1.4,
                     arrowprops=dict(arrowstyle="-", color=c["muted"], lw=0.7))
    axr.axvline(0.5, color=c["axis"], lw=1.2)
    axr.set_xlabel("share of question votes favouring the stronger model",
                   color=c["ink2"], fontsize=9.5)
    axr.set_ylabel("share of random split-halves whose two halves\nagree in direction",
                   color=c["ink2"], fontsize=9.5)
    axr.set_title("Do disjoint question sets agree?", color=c["ink"], loc="left",
                  fontsize=11)
    axr.grid(lw=0.8)
    axr.set_axisbelow(True)
    axr.legend(frameon=False, fontsize=8.6, loc="lower right", labelcolor=c["ink2"])
    fig_.suptitle("Separating systematic inversion from finite-question uncertainty",
                  color=c["ink"], fontsize=13, fontweight="semibold",
                  x=0.005, ha="left", y=1.02)
    fig_.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig_.savefig(OUT / f"{n}{'' if c is LIGHT else '_dark'}.png", dpi=150,
                 bbox_inches="tight", facecolor=c["surface"])
    if c is LIGHT:
        fig_.savefig(OUT / f"{n}.svg", bbox_inches="tight", facecolor=c["surface"])
    plt.close(fig_)


def main():
    votes = question_votes(load())
    t = per_pair(votes)
    print(f"pairs: {len(t)}   median questions per pair: "
          f"{int(t['n_questions'].median())}")
    wrong = t[~t["correct"]]
    print("\nmisranked pairs:")
    for r in wrong.itertuples():
        verdict = ("systematic inversion" if r.ci_hi < 0.5
                   else "compatible with finite-question uncertainty")
        print(f"  {r.pair:<46} share {r.share:.3f}  "
              f"CI [{r.ci_lo:.3f}, {r.ci_hi:.3f}]  "
              f"both-halves-wrong {r.split_both_wrong:.0%}  -> {verdict}")
    right = t[t["correct"]]
    solid = int((right["ci_lo"] > 0.5).sum())
    print(f"\ncorrectly ranked pairs with the whole interval above 0.5: "
          f"{solid} of {len(right)}")
    write_json(OUT / "question_uncertainty.json",
               dict(pairs=t.to_dict("records")))
    for c in (LIGHT, DARK):
        style(c)
        fig(t, c)
    print(f"wrote {OUT / 'question_uncertainty.json'} and figure 14")


if __name__ == "__main__":
    main()
