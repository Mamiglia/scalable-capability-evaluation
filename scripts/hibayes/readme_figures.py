"""Figures for ANALYSIS_README.md, numbered in the order they appear.

Written to analysis/hibayes/readme_figures as 1.png through 7.png, each with a
dark variant and an SVG. The numbering is positional, so a figure inserted later
renumbers the ones after it.

  1  four-predictor population effects
  2  what the expanded feature analysis found
  3  judge bias by feature family
  4  coefficient stability across specifications
  5  position bias, population against per judge
  6  out-of-sample accuracy
  7  pair ranking by number of questions
     (8 comes from capability_uncertainty.py)
  8  is the capability ordering itself known
  9  which pooling helps: order against panel
 10  per-judge pair ranking at narrow and wide gaps
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
PRIMARY = HIB / "features_within"


def finish(fig, n, c):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{n}{'' if c is LIGHT else '_dark'}.png", dpi=150,
                bbox_inches="tight", facecolor=c["surface"])
    if c is LIGHT:
        fig.savefig(OUT / f"{n}.svg", bbox_inches="tight", facecolor=c["surface"])
    plt.close(fig)


def nice(s):
    return str(s).replace("_", " ")


def fig1(c):
    """Four-predictor population effects."""
    d = pd.read_csv(HIB / "run" / "communicate" / "population_effects.csv")
    d = d.reindex(d["mean_log_odds"].abs().sort_values().index)
    y = np.arange(len(d))
    cred = (d["hdi_2.5%"] > 0) | (d["hdi_97.5%"] < 0)
    fig, ax = plt.subplots(figsize=(9.2, 3.6))
    ax.hlines(y, d["hdi_2.5%"], d["hdi_97.5%"], color=c["s1"], lw=2.4, zorder=2)
    ax.scatter(d["mean_log_odds"][cred], y[cred.values], s=64, color=c["s1"],
               zorder=3, edgecolor=c["surface"], linewidth=1.2)
    ax.scatter(d["mean_log_odds"][~cred], y[~cred.values], s=64, facecolor=c["surface"],
               zorder=3, edgecolor=c["s1"], linewidth=2)
    ax.axvline(0, color=c["axis"], lw=1.4, zorder=1)
    for yy, m in zip(y, d["mean_log_odds"]):
        ax.annotate(f"{m:+.2f}", (m, yy), xytext=(0, 10), textcoords="offset points",
                    ha="center", fontsize=9, color=c["ink"], family="monospace")
    ax.set_yticks(y); ax.set_yticklabels([nice(e) for e in d["effect"]],
                                         fontsize=10, color=c["ink2"])
    ax.set_xlabel("effect on the log-odds of selecting the more capable model",
                  color=c["ink2"])
    ax.set_title("The four-predictor model", color=c["ink"], loc="left")
    ax.grid(axis="x", lw=0.8); ax.set_axisbelow(True)
    fig.tight_layout(); finish(fig, 6, c)


def fig_pairs(c):
    """The headline result: one point per model pair."""
    import json as _j
    pl=_j.loads((HIB/"figures"/"pooling_by_question.json").read_text())["pair_level"]
    t=pd.DataFrame(pl)
    fig, ax = plt.subplots(figsize=(9.8, 5.0))
    ok, bad = t[t["correct"]], t[~t["correct"]]
    ax.scatter(ok["gap"], ok["share"], s=56, color=c["s1"], zorder=3,
               edgecolor=c["surface"], linewidth=1, label="pair ranked correctly")
    ax.scatter(bad["gap"], bad["share"], s=64, color=c["crit"], marker="X", zorder=4,
               edgecolor=c["surface"], linewidth=1, label="pair ranked wrong")
    ax.axhline(0.5, color=c["axis"], lw=1.4)
    ax.set_xlabel("capability gap (GPQA label)", color=c["ink2"])
    ax.set_ylabel("share of question votes favouring the stronger model", color=c["ink2"])
    ax.set_title("LLM judges recover capability information from correct answers: 21 of 25 pairs ranked correctly",
                 color=c["ink"], loc="left", fontsize=11)
    ax.grid(lw=0.8); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, loc="lower right", labelcolor=c["ink2"])
    fig.tight_layout(); finish(fig, 1, c)


def _marginal():
    b = pd.read_csv(PRIMARY / "bias_map.csv")
    b = b[~b["is_component"].astype(bool)]
    return b[b["spec"] == "marginal"].drop_duplicates(subset="feature")


def fig2(c):
    """What the expanded analysis found, as one bar of 136 features."""
    m = _marginal()
    follows = m["judge_follows"].astype(str).str.lower().eq("true")
    q = m["quadrant"]
    parts = [("no association with judge choices", int((~follows).sum()), "muted"),
             ("judge bias: followed, no capability signal",
              int((q == "JUDGE BIAS").sum()), "s1"),
             ("opposite direction to capability", int((q == "INVERTED").sum()), "s2"),
             ("same direction as capability", int((q == "justified signal").sum()), "s5")]
    total = sum(p[1] for p in parts)
    fig, ax = plt.subplots(figsize=(10.4, 2.9))
    left = 0
    for label, n, col in parts:
        ax.barh([0], [n], left=left, height=.52, color=c[col],
                edgecolor=c["surface"], linewidth=2)
        if n / total > 0.05:
            ax.text(left + n / 2, 0, str(n), ha="center", va="center",
                    fontsize=12, color="#ffffff" if col != "muted" else c["ink"],
                    fontweight="bold")
        left += n
    ax.set_xlim(0, total); ax.set_ylim(-.6, .95)
    ax.set_yticks([])
    ax.set_xlabel(f"features in the within-pair marginal analysis (n = {total})",
                  color=c["ink2"])
    ax.set_title("Most features the judge follows carry no detected capability signal",
                 color=c["ink"], loc="left")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c[col]) for _, _, col in parts]
    ax.legend(handles, [p[0] for p in parts], frameon=False, fontsize=8.8,
              loc="upper left", bbox_to_anchor=(0, -0.42), ncol=2,
              labelcolor=c["ink2"])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout(); finish(fig, 7, c)


def fig3(c):
    """Judge bias as a share of each feature family."""
    m = _marginal()
    t = (m.assign(bias=(m["quadrant"] == "JUDGE BIAS").astype(int))
           .groupby("family").agg(bias=("bias", "sum"), n=("bias", "size")))
    t["share"] = t["bias"] / t["n"]
    t = t.sort_values("share")
    y = np.arange(len(t))
    fig, ax = plt.subplots(figsize=(9.6, 4.4))
    ax.barh(y, t["share"], height=.62, color=c["s1"])
    for i, (b, n) in enumerate(zip(t["bias"], t["n"])):
        ax.annotate(f"{b}/{n}", (t['share'].iloc[i], i), xytext=(7, 0),
                    textcoords="offset points", va="center", fontsize=9,
                    color=c["ink"], family="monospace")
    ax.set_yticks(y); ax.set_yticklabels([nice(f) for f in t.index],
                                         fontsize=10, color=c["ink2"])
    ax.set_xlim(0, 1.12); ax.set_xticks([0, .25, .5, .75, 1])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("share of the family classified as judge bias", color=c["ink2"])
    ax.set_title("Where the bias concentrates", color=c["ink"], loc="left")
    ax.grid(axis="x", lw=0.8); ax.set_axisbelow(True)
    fig.tight_layout(); finish(fig, 12, c)


def fig4(c):
    """Does a coefficient survive its controls?"""
    b = pd.read_csv(PRIMARY / "bias_map.csv")
    b = b[~b["is_component"].astype(bool)]
    w = b.pivot_table(index="feature", columns="spec", values="mean_log_odds",
                      aggfunc="first").dropna()
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.9))
    for ax, spec, lab in zip(axes, ["joint", "adjusted"],
                             ["controlling for the rest of the family",
                              "also controlling for the capability gap"]):
        flip = np.sign(w[spec]) != np.sign(w["marginal"])
        ax.scatter(w["marginal"][~flip], w[spec][~flip], s=30, color=c["s1"],
                   alpha=.75, edgecolor="none", zorder=3, label="sign kept")
        ax.scatter(w["marginal"][flip], w[spec][flip], s=42, color=c["s2"],
                   zorder=4, edgecolor=c["surface"], linewidth=.8,
                   label=f"sign changed ({int(flip.sum())})")
        lim = float(np.abs(w[["marginal", spec]].values).max()) * 1.08
        ax.plot([-lim, lim], [-lim, lim], color=c["muted"], lw=1.2, ls=(0, (4, 3)),
                zorder=2)
        ax.axhline(0, color=c["axis"], lw=1.1, zorder=1)
        ax.axvline(0, color=c["axis"], lw=1.1, zorder=1)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_xlabel("marginal coefficient", color=c["ink2"])
        ax.set_ylabel(f"{spec} coefficient", color=c["ink2"])
        ax.set_title(lab, color=c["ink"], loc="left", fontsize=10.5)
        ax.grid(lw=0.8); ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=8.8, loc="upper left", labelcolor=c["ink2"])
    fig.suptitle("Coefficients hold up under control, apart from one feature in five",
                 color=c["ink"], fontsize=13, fontweight="semibold",
                 x=0.005, ha="left", y=1.03)
    fig.tight_layout(); finish(fig, 8, c)


def fig5(c):
    """Position bias: the population term against the per-judge terms."""
    j = pd.read_csv(HIB / "run" / "communicate" / "judge_feature_effects.csv")
    p = j[j["feature"] == "position_bias_left"].sort_values("mean_log_odds")
    pop = pd.read_csv(HIB / "run" / "communicate" / "population_effects.csv")
    pv = float(pop.loc[pop["effect"] == "position_bias_left", "mean_log_odds"].iloc[0])
    cred = p["credible_nonzero"].astype(str).str.lower().eq("true").values
    fig, ax = plt.subplots(figsize=(10.2, 5.0))
    for i, (m, lo, hi, ok) in enumerate(zip(p["mean_log_odds"], p["hdi_2.5%"],
                                            p["hdi_97.5%"], cred)):
        col = c["s1"] if m > 0 else c["s2"]
        ax.hlines(i, lo, hi, color=col, lw=2.2, alpha=1 if ok else .4, zorder=2)
        ax.scatter([m], [i], s=54, zorder=3, linewidth=1.8,
                   color=col if ok else c["surface"], edgecolor=col)
    ax.axvline(0, color=c["axis"], lw=1.4, zorder=1)
    ax.axvline(pv, color=c["crit"], lw=1.6, ls=(0, (5, 3)), zorder=1)
    ax.annotate(f"population term {pv:+.2f}\ninterval includes zero", (pv, len(p) - 0.6),
                fontsize=8.8, color=c["crit"], ha="center", va="top", linespacing=1.5,
                bbox=dict(boxstyle="round,pad=0.4", facecolor=c["surface"],
                          edgecolor=c["crit"], linewidth=0.9))
    ax.set_yticks(np.arange(len(p)))
    ax.set_yticklabels(p["judge"], fontsize=9.6, color=c["ink2"])
    ax.set_xlabel("favours the second answer  ←        →  favours the first answer",
                  color=c["ink2"], fontsize=9.5)
    ax.set_title(f"{int(cred.sum())} of {len(p)} judges have a position effect, "
                 "in opposing directions", color=c["ink"], loc="left")
    ax.grid(axis="x", lw=0.8); ax.set_axisbelow(True)
    fig.tight_layout(); finish(fig, 3, c)


def fig6(c):
    """Out-of-sample accuracy: single features against the four-predictor model."""
    rows = []
    for d in sorted(PRIMARY.glob("marginal__*/communicate/judge_loo_accuracy.csv")):
        a = pd.read_csv(d)
        rows.append(float(np.average(a["loo_accuracy"], weights=a["n"])))
    fig, ax = plt.subplots(figsize=(9.8, 4.2))
    ax.hist(rows, bins=28, color=c["s1"], edgecolor=c["surface"], linewidth=.6)
    top = ax.get_ylim()[1]
    ax.set_xlabel("leave-one-out accuracy of a one-feature model", color=c["ink2"])
    ax.set_ylabel("features", color=c["ink2"])
    ax.set_title("Leave-one-out accuracy of the 136 one-feature models",
                 color=c["ink"], loc="left", fontsize=11.5)
    ax.grid(axis="y", lw=0.8); ax.set_axisbelow(True)
    fig.tight_layout(); finish(fig, 11, c)


def fig7(c):
    """Pair ranking against the number of questions pooled."""
    j = json.loads((HIB / "figures" / "pooling_by_question.json").read_text())
    cur = j["curves"]
    order = ["0.20 and above", "0.10 to 0.20", "0.05 to 0.10", "under 0.05"]
    cols = ["s2", "s1", "s3", "s4"]
    fig, ax = plt.subplots(figsize=(9.8, 5.4))
    for band, col in zip(order, cols):
        if band not in cur:
            continue
        d = cur[band]["question_level"]
        ax.plot([p["k"] for p in d], [p["accuracy"] for p in d], marker="o", ms=5,
                lw=2.2, color=c[col],
                label=f"gap {band}  ({cur[band]['n_pairs']} pairs)")
    ax.axhline(0.5, color=c["crit"], lw=1.4, ls=(0, (5, 3)))
    ax.annotate("chance", (1.05, .505), color=c["crit"], fontsize=8.8, va="bottom")
    ax.set_xscale("log")
    ax.set_xlabel("question items pooled per model pair", color=c["ink2"])
    ax.set_ylabel("chance the pooled verdict ranks the pair correctly", color=c["ink2"])
    ax.set_title("Under the GPQA label, about 15 questions settle a pair above a 0.10 gap; "
                 "below it the curves flatten", color=c["ink"], loc="left", fontsize=10.5)
    ax.set_ylim(0.42, 1.03)
    ax.grid(lw=0.8); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, loc="lower right", labelcolor=c["ink2"])
    fig.tight_layout(); finish(fig, 13, c)


def main():
    for c in (LIGHT, DARK):
        style(c)
        for f in (fig_pairs, fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9):
            f(c)
    print(f"wrote {len(list(OUT.glob('*')))} files to {OUT}")
    for p in sorted(OUT.glob("*.png")):
        print("  ", p.name)




def fig8(c):
    """Counterbalancing against a panel, at matched sample size."""
    j = json.loads((HIB / "figures" / "pooling_by_question.json").read_text())
    lv = j["three_levels"]["levels"]
    spec = [("one judge, one order", "s4", "o"),
            ("one judge, both orders", "s3", "s"),
            ("all judges, both orders", "s1", "D")]
    fig, ax = plt.subplots(figsize=(9.8, 5.2))
    for name, col, mk in spec:
        d = lv[name]
        ax.plot([p["k"] for p in d], [p["accuracy"] for p in d], marker=mk, ms=5.5,
                lw=2.2, color=c[col], label=name)
    ax.axhline(0.5, color=c["crit"], lw=1.3, ls=(0, (5, 3)))
    ax.set_xscale("log")
    ax.set_xlabel("question items pooled per model pair", color=c["ink2"])
    ax.set_ylabel("chance the pooled verdict ranks the pair correctly", color=c["ink2"])
    ax.set_title("Showing each pair both ways round beats adding nine more judges",
                 color=c["ink"], loc="left", fontsize=11.5)
    ax.annotate("Every level pools the same number of questions, so the gaps are\n"
                "systematic error being cancelled, not random error being averaged.",
                (0.02, 0.97), xycoords="axes fraction", va="top", fontsize=8.6,
                color=c["muted"], linespacing=1.6)
    ax.grid(lw=0.8); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, loc="lower right", labelcolor=c["ink2"])
    fig.tight_layout(); finish(fig, 4, c)


def fig9(c):
    """Per judge, pair ranking at narrow and wide capability gaps."""
    j = json.loads((HIB / "figures" / "pooling_by_question.json").read_text())
    t = pd.DataFrame(j["judge_by_gap"]).sort_values("narrow")
    y = np.arange(len(t))
    fig, ax = plt.subplots(figsize=(10.2, 5.4))
    for i, r in enumerate(t.itertuples()):
        ax.plot([r.narrow, r.wide], [i, i], color=c["grid"], lw=2.4, zorder=1)
    ax.scatter(t["narrow"], y, s=74, color=c["s2"], zorder=3, edgecolor=c["surface"],
               linewidth=1.2, label="capability gap under 0.05")
    ax.scatter(t["wide"], y, s=74, color=c["s1"], zorder=3, edgecolor=c["surface"],
               linewidth=1.2, label="capability gap 0.10 and above")
    ax.axvline(0.5, color=c["crit"], lw=1.4, ls=(0, (5, 3)))
    ax.annotate("chance", (0.5, len(t) - 0.35), color=c["crit"], fontsize=8.8,
                ha="center", va="bottom")
    ax.set_yticks(y); ax.set_yticklabels(t["judge"], fontsize=9.8, color=c["ink2"])
    ax.set_xlim(-0.04, 1.06)
    ax.set_xlabel("share of pairs ranked correctly, pooling 63 questions", color=c["ink2"])
    ax.set_title("Under the GPQA capability label, every judge is perfect on distant pairs; "
                 "on close pairs they range from 0 to 1", color=c["ink"], loc="left", fontsize=10.5)
    ax.grid(axis="x", lw=0.8); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, loc="lower left", labelcolor=c["ink2"],
              bbox_to_anchor=(0, -0.20), ncol=2)
    fig.tight_layout(); finish(fig, 5, c)


if __name__ == "__main__":
    main()
