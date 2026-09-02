"""Fig 1 (length bias, summed sentence effects) with the clearer delta-length label."""
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, "/Users/moriohamada/PycharmProjects/ai_safety/small-judge/judge-whitebox")
import mask_summary

S = pathlib.Path("/Users/moriohamada/PycharmProjects/ai_safety/small-judge/tmp/whitebox")
OUT = pathlib.Path("/Users/moriohamada/PycharmProjects/ai_safety/small-judge/judge_results/whitebox_dissection/figures/llama3.2-3B")
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "axes.spines.top": False, "axes.spines.right": False})
BLUE, ORANGE = "#2a78d6", "#eb6834"

scores = pd.read_parquet(S / "llama_sentence_scores.parquet")
slots = scores[scores.group.isin(["transcript_1", "transcript_2"])].copy()
per = slots.groupby(["item", "pair_uid", "order", "group"]).agg(
    total=("score", "sum"), toks=("n_tokens", "sum")).reset_index()
per["towards_own"] = np.where(per.group == "transcript_1", per.total, -per.total)
t1 = per[per.group == "transcript_1"].set_index(["item", "pair_uid", "order"])
t2 = per[per.group == "transcript_2"].set_index(["item", "pair_uid", "order"])
pw = pd.DataFrame({"t1": t1.total, "t2": t2.total, "n1": t1.toks, "n2": t2.toks}).dropna().reset_index()
pw["dlen"], pw["net"] = pw.n1 - pw.n2, pw.t1 + pw.t2


def binned_ci(frame, xcol, ycol, bins=8):
    frame = frame.copy()
    frame["bin"] = pd.qcut(frame[xcol], bins, duplicates="drop")
    rows = []
    for _, g in frame.groupby("bin", observed=True):
        ci = mask_summary.item_bootstrap(g[["item", ycol]].rename(columns={ycol: "v"}),
                                         lambda x: x["v"].mean(), 300)
        rows.append({"x": g[xcol].mean(), **ci})
    return pd.DataFrame(rows)


fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.2))
for grp, name, color in (("transcript_1", "transcript 1", BLUE),
                         ("transcript_2", "transcript 2", ORANGE)):
    g = per[per.group == grp]
    b = binned_ci(g, "toks", "towards_own")
    axes[0].plot(b.x, b.estimate, color=color, lw=1, label=name)
    axes[0].fill_between(b.x, b.lo, b.hi, color=color, alpha=0.18, lw=0)
axes[0].axhline(0, color="0.5", lw=0.8, ls="--")
axes[0].set_xlabel("transcript length (tokens)")
axes[0].set_ylabel("total pull towards\nits own slot (log-odds)")
axes[0].legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
b = binned_ci(pw, "dlen", "net")
axes[1].plot(b.x, b.estimate, color="k", lw=1)
axes[1].fill_between(b.x, b.lo, b.hi, color="0.4", alpha=0.18, lw=0)
axes[1].axhline(0, color="0.5", lw=0.8, ls="--")
axes[1].axvline(0, color="0.5", lw=0.8, ls="--")
axes[1].set_xlabel("Δ transcript length, 1 − 2 (tokens)")
axes[1].set_ylabel("net pull towards\ntranscript 1 (log-odds)")
fig.tight_layout()
fig.savefig(OUT / "fig1_length_bias.png", dpi=300); fig.savefig(OUT / "fig1_length_bias.svg")
print("saved")
