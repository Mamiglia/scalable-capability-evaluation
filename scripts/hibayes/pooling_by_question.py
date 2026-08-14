"""Pooled pair ranking, resampled at the question level.

The earlier pooling analysis bootstrapped rows. A row is one judge, one
presentation order and one length condition for one question, so a single
question contributes up to 80 rows. Those rows share the same two transcripts
and are not independent draws. Resampling them as if they were inflates the
effective sample size and makes the pooled verdict look like it converges far
faster than it does.

This script fixes that. Rows are first aggregated into one vote per question,
then questions are resampled. The question is the unit that is actually
replicated when an evaluation is scaled up, so `k` here reads directly as "how
many benchmark items do I need", which is the number worth planning against.

Both curves are reported so the size of the correction is visible.
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plotting import DARK, HIB, LIGHT, OUT, finish, style, write_json

CSV = HIB / "gpqa_correctness.csv"
SEED = 0
COLS = ["judge", "item", "model_pair", "left_capability", "right_capability", "major_is_left",
        "capability_difference_major", "is_correct", "self_comparison"]
GAP_BANDS = [(0.00, 0.05, "under 0.05"), (0.05, 0.10, "0.05 to 0.10"),
             (0.10, 0.20, "0.10 to 0.20"), (0.20, 1.00, "0.20 and above")]
SIZES = (1, 3, 7, 15, 31, 63, 127, 255, 511)


def band_of(gap):
    for lo, hi, name in GAP_BANDS:
        if lo <= gap < hi:
            return name
    return GAP_BANDS[-1][2]


def load():
    df = pd.read_csv(CSV, usecols=COLS, low_memory=False)
    df = df[df["self_comparison"] == 0].copy()
    df["gap"] = df["capability_difference_major"].astype(float)
    df["band"] = df["gap"].map(band_of)
    return df


def question_votes(df):
    """One vote per (pair, question), aggregating every row for that question.

    A vote of 1 means the question's rows favoured the more capable model on
    balance. Ties become 0.5 rather than being dropped, so a question that
    splits evenly contributes no evidence in either direction.
    """
    g = (df.groupby(["model_pair", "item"])
           .agg(share=("is_correct", "mean"), rows=("is_correct", "size"))
           .reset_index())
    g["vote"] = np.where(g["share"] > 0.5, 1.0, np.where(g["share"] < 0.5, 0.0, 0.5))
    meta = df.groupby("model_pair").agg(gap=("gap", "first"), band=("band", "first"))
    return g.merge(meta, on="model_pair")


def majority(rng, arr, k, B):
    draw = rng.choice(arr, size=(B, k), replace=True)
    m = draw.mean(axis=1)
    return float(((m > 0.5).sum() + 0.5 * (m == 0.5).sum()) / B)


def curves(votes, rows, B):
    """Question-level and row-level curves, by capability gap band."""
    rng = np.random.default_rng(SEED)
    out = {}
    for lo, hi, band in GAP_BANDS:
        v = votes[(votes["gap"] >= lo) & (votes["gap"] < hi)]
        r = rows[(rows["gap"] >= lo) & (rows["gap"] < hi)]
        if v.empty:
            continue
        by_pair_q = [g["vote"].to_numpy() for _, g in v.groupby("model_pair")]
        by_pair_r = [g["is_correct"].to_numpy() for _, g in r.groupby("model_pair")]
        by_pair_q = [a for a in by_pair_q if len(a) >= 8]
        by_pair_r = [a for a in by_pair_r if len(a) >= 8]
        if not by_pair_q:
            continue
        out[band] = dict(
            n_pairs=len(by_pair_q),
            median_questions=int(np.median([len(a) for a in by_pair_q])),
            question_level=[dict(k=int(k), accuracy=float(np.mean([majority(rng, a, k, B)
                                                                  for a in by_pair_q])))
                            for k in SIZES],
            row_level=[dict(k=int(k), accuracy=float(np.mean([majority(rng, a, k, B)
                                                             for a in by_pair_r])))
                       for k in SIZES],
        )
    return out


def pair_level(votes):
    """Pair ranking from question votes, so every question weighs the same."""
    rows = []
    for pair, g in votes.groupby("model_pair"):
        share = float(g["vote"].mean())
        rows.append(dict(pair=pair, questions=int(len(g)), share=share,
                         gap=float(g["gap"].iloc[0]), band=g["band"].iloc[0],
                         correct=bool(share > 0.5)))
    return pd.DataFrame(rows).sort_values("gap")


def fig_correction(res, c):
    fig, axes = plt.subplots(1, len(res), figsize=(3.4 * len(res), 4.9), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (band, d) in zip(axes, res.items()):
        ax.plot([p["k"] for p in d["row_level"]], [p["accuracy"] for p in d["row_level"]],
                marker="o", ms=4.5, lw=1.8, ls=(0, (4, 3)), color=c["muted"],
                label="rows resampled (old)")
        ax.plot([p["k"] for p in d["question_level"]],
                [p["accuracy"] for p in d["question_level"]],
                marker="s", ms=5, lw=2.2, color=c["s1"], label="questions resampled")
        ax.axhline(0.5, color=c["crit"], lw=1.3, ls=(0, (5, 3)))
        ax.set_xscale("log")
        ax.set_title(f"gap {band}", color=c["ink"], loc="left", fontsize=10.5)
        ax.annotate(f"{d['n_pairs']} pairs\nmedian {d['median_questions']} questions",
                    (0.04, 0.96), xycoords="axes fraction", va="top", fontsize=8.2,
                    color=c["muted"], linespacing=1.5)
        ax.set_xlabel("units pooled per pair", color=c["ink2"], fontsize=9.5)
        ax.grid(lw=0.8); ax.set_axisbelow(True)
    axes[0].set_ylabel("chance the pooled verdict ranks the pair correctly",
                       color=c["ink2"], fontsize=9.5)
    axes[0].set_ylim(0.42, 1.03)
    axes[-1].legend(frameon=False, fontsize=8.6, loc="lower right", labelcolor=c["ink2"])
    fig.suptitle("Resampling questions instead of rows", color=c["ink"], fontsize=13,
                 fontweight="semibold", x=0.005, ha="left", y=1.04)
    fig.tight_layout()
    finish(fig, "fig28_question_level_pooling", c)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--draws", type=int, default=400)
    args = ap.parse_args(argv)

    df = load()
    votes = question_votes(df)
    print(f"rows {len(df)}  questions {len(votes)}  pairs {votes['model_pair'].nunique()}")
    print(f"rows per question: median {votes['rows'].median():.0f}, max {votes['rows'].max()}")

    p = pair_level(votes)
    print(f"\npair ranking from question votes: {int(p['correct'].sum())}/{len(p)} "
          f"= {p['correct'].mean():.3f}")
    for b in p["band"].unique():
        d = p[p["band"] == b]
        print(f"     gap {b:<16} {int(d['correct'].sum())}/{len(d)}")

    res = curves(votes, df, args.draws)
    print("\npooled verdict accuracy, questions resampled (row-level in brackets):")
    for band, d in res.items():
        q = {x["k"]: x["accuracy"] for x in d["question_level"]}
        r = {x["k"]: x["accuracy"] for x in d["row_level"]}
        print(f"  gap {band:<16} k=15 {q[15]:.3f} [{r[15]:.3f}]   "
              f"k=63 {q[63]:.3f} [{r[63]:.3f}]   k=511 {q[511]:.3f} [{r[511]:.3f}]")

    tl = three_levels(df)
    print("\nthree levels, questions resampled:")
    for nm, cur in tl["levels"].items():
        pts = {x["k"]: x["accuracy"] for x in cur}
        print("  %-26s k=15 %.3f  k=63 %.3f  k=511 %.3f" % (nm, pts[15], pts[63], pts[511]))
    jg = judge_by_gap_q(df)
    print("\nper judge at k=63 (narrow / wide):")
    for _, r in jg.iterrows():
        print("  %-24s %.3f / %.3f" % (r["judge"], r["narrow"], r["wide"]))
    write_json(OUT / "pooling_by_question.json",
               dict(pair_level=p.to_dict("records"),
                    n_questions=int(len(votes)),
                    rows_per_question_median=float(votes["rows"].median()),
                    curves=res, three_levels=tl, judge_by_gap=jg.to_dict("records")))
    for c in (LIGHT, DARK):
        style(c)
        fig_correction(res, c)
    print(f"\nwrote {OUT / 'pooling_by_question.json'} and fig28")




def three_levels(df, B=300):
    """Does pooling cancel random error, or does it cancel a shared bias?

    Three ways of pooling the same number of question items. Sample size is held
    fixed, so a gap between the levels is systematic error being cancelled, not
    random error being averaged.

      one judge, one order    only item-to-item noise cancels
      one judge, both orders  that judge's position effect also cancels
      all judges, both orders the judges' differing biases also cancel

    Votes are formed per question inside each cell before resampling, so the
    question remains the sampling unit at every level.
    """
    rng = np.random.default_rng(SEED)

    def votes_of(frame, keys):
        g = frame.groupby(keys + ["item"])["is_correct"].mean().reset_index()
        g["vote"] = np.where(g["is_correct"] > 0.5, 1.0,
                             np.where(g["is_correct"] < 0.5, 0.0, 0.5))
        return g

    df = df.copy()
    df["order"] = df["major_is_left"] if "major_is_left" in df.columns else 0
    a = votes_of(df, ["model_pair", "judge", "order"])
    b = votes_of(df, ["model_pair", "judge"])
    c = votes_of(df, ["model_pair"])

    levels = {"one judge, one order": [], "one judge, both orders": [],
              "all judges, both orders": []}
    cells = {}
    for pair in sorted(df["model_pair"].unique()):
        ja = [g["vote"].to_numpy() for _, g in a[a["model_pair"] == pair].groupby(["judge", "order"])]
        jb = [g["vote"].to_numpy() for _, g in b[b["model_pair"] == pair].groupby("judge")]
        ca = c[c["model_pair"] == pair]["vote"].to_numpy()
        ja = [x for x in ja if len(x) >= 8]
        jb = [x for x in jb if len(x) >= 8]
        if ja and jb and len(ca) >= 8:
            cells[pair] = (ja, jb, ca)
    for k in SIZES:
        acc = {n: [] for n in levels}
        for _, (ja, jb, ca) in cells.items():
            acc["one judge, one order"].append(float(np.mean([majority(rng, x, k, B) for x in ja])))
            acc["one judge, both orders"].append(float(np.mean([majority(rng, x, k, B) for x in jb])))
            acc["all judges, both orders"].append(majority(rng, ca, k, B))
        for n in levels:
            levels[n].append(dict(k=int(k), accuracy=float(np.mean(acc[n]))))
    return dict(n_pairs=len(cells), levels=levels)


def judge_by_gap_q(df, B=300):
    """Per judge, pair-ranking accuracy at narrow and wide gaps, question level."""
    rng = np.random.default_rng(SEED)
    g = (df.groupby(["model_pair", "judge", "item"])["is_correct"].mean().reset_index())
    g["vote"] = np.where(g["is_correct"] > 0.5, 1.0, np.where(g["is_correct"] < 0.5, 0.0, 0.5))
    gap = df.groupby("model_pair")["gap"].first()
    g["gap"] = g["model_pair"].map(gap)
    rows = []
    for j, d in g.groupby("judge"):
        out = {"judge": j}
        for lab, sel in (("narrow", d[d["gap"] < 0.05]), ("wide", d[d["gap"] >= 0.10])):
            arrs = [x["vote"].to_numpy() for _, x in sel.groupby("model_pair")]
            arrs = [x for x in arrs if len(x) >= 8]
            out[lab] = float(np.mean([majority(rng, x, 63, B) for x in arrs])) if arrs else None
            out[lab + "_pairs"] = len(arrs)
        if out.get("narrow") is not None and out.get("wide") is not None:
            rows.append(out)
    return pd.DataFrame(rows).sort_values("narrow")


if __name__ == "__main__":
    main()
