"""Judge pick-accuracy gap curve, raw vs de-biased, per model pair.

y is P(judge picks the stronger model). Pre uses the observed verdicts; post
removes the judge's fitted presentation biases (length + six style features) by
subtracting their log-odds contribution, keeping the capability signal. A pair
below 0.5 is a "bad case" -- the judge picks the weaker model on average. The
same comparisons are coloured identically in both panels, so you can watch the
bad cases cross back above 0.5 once the biases are removed.

Bias terms are refit per judge (logistic, config_corrected.yaml predictor list)
on the correction's own verdict frame; that fit is only used to isolate the bias
log-odds, so the observed pre rate is untouched. Two figures: pooled over judges,
and every judge x pair comparison with a per-judge mean curve.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "font.family": "sans-serif", "font.sans-serif": ["Arial"], "svg.fonttype": "none",
})
GREEN, ORANGE, PURPLE, GREY = "#1baf7a", "#eb6834", "#4a3aa7", "#b8b8b8"
FATE = {"kept": ("kept right", GREY, 1), "fixed": ("fixed (bad → good)", GREEN, 4),
        "still_bad": ("still bad", ORANGE, 3), "broken": ("broken (good → bad)", PURPLE, 3)}
KEEP = ["capability_difference_sd", "same_model_family", "self_comparison"]
BIAS = ["length_difference_sd", "personal_pronouns_difference_sd", "hedging_difference_sd",
        "backtracking_difference_sd", "uncertainty_difference_sd", "confidence_difference_sd",
        "questions_difference_sd"]
XHI = 0.7
logit = lambda p: np.log(np.clip(p, 1e-3, 1 - 1e-3) / (1 - np.clip(p, 1e-3, 1 - 1e-3)))
sig = lambda x: 1 / (1 + np.exp(-x))


def verdicts(frame: Path) -> pd.DataFrame:
    """Verdict-level frame with pick-stronger outcome and per-judge bias log-odds."""
    cols = ["judge", "left_model", "right_model", "chosen_left",
            "left_capability", "right_capability", *KEEP, *BIAS]
    d = pd.read_csv(frame, usecols=cols)
    d["chosen_left"] = d.chosen_left.astype(int)
    d = d[d.left_capability != d.right_capability].copy()
    d["bias_left"] = 0.0
    for _, idx in d.groupby("judge").groups.items():
        s = d.loc[idx]
        m = LogisticRegression(C=1e4, max_iter=3000).fit(s[KEEP + BIAS].to_numpy(), s.chosen_left)
        d.loc[idx, "bias_left"] = s[BIAS].to_numpy() @ m.coef_[0][len(KEEP):]
    strong_left = d.left_capability > d.right_capability
    d["pick_strong"] = np.where(strong_left, d.chosen_left, 1 - d.chosen_left)
    d["bias_strong"] = np.where(strong_left, d.bias_left, -d.bias_left)
    d["gap"] = (d.left_capability - d.right_capability).abs()
    d["ukey"] = list(zip(np.minimum(d.left_model, d.right_model),
                         np.maximum(d.left_model, d.right_model)))
    return d


def collapse(d: pd.DataFrame, keys: list[str], min_n: int) -> pd.DataFrame:
    """Aggregate verdicts to one row per key group; drop groups below min_n verdicts."""
    g = d.groupby(keys).agg(gap=("gap", "first"), n=("pick_strong", "size"),
                            p_obs=("pick_strong", "mean"), mean_bias=("bias_strong", "mean")).reset_index()
    g = g[g.n >= min_n].copy()
    g["p_debias"] = sig(logit(g.p_obs) - g.mean_bias)
    bad = g.p_obs < 0.5
    g["fate"] = np.select([bad & (g.p_debias >= .5), bad & (g.p_debias < .5), ~bad & (g.p_debias < .5)],
                          ["fixed", "still_bad", "broken"], "kept")
    return g


def curve(ax, g, ycol, groups=None):
    """Binned mean curve; thin grey line per group if groups given, plus a bold pooled line."""
    edges = np.arange(0, XHI + 0.1, 0.1)
    mids = lambda b: [iv.mid for iv in b.index]
    if groups is not None:
        for _, sub in g.groupby(groups):
            b = sub.groupby(pd.cut(sub.gap, edges), observed=True)[ycol].mean().dropna()
            ax.plot(mids(b), b.to_numpy(), lw=0.6, c="#9a9a9a", alpha=0.6, zorder=3)
    b = g.groupby(pd.cut(g.gap, edges), observed=True)[ycol].mean().dropna()
    ax.plot(mids(b), b.to_numpy(), lw=1.4, c="black", zorder=5)


def dress(ax, tag):
    ax.set_title(tag, loc="left", fontsize=7, color="dimgrey")
    ax.set_xticks([0, 0.35, 0.7]); ax.set_yticks([0, 0.5, 1])
    ax.set_xlim(-0.02, XHI + 0.02); ax.set_ylim(-0.02, 1.02)
    for side, (lo, hi) in (("bottom", (0, XHI)), ("left", (0, 1))):
        ax.spines[side].set_position(("outward", 3)); ax.spines[side].set_bounds(lo, hi)


def figure(g, out: Path, name: str, per_judge_curves: bool, unit: str, ms: float,
           kept_alpha: float, legend_y: float):
    fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.5), sharex=True, sharey=True)
    for ax, (col, tag) in zip(axes, [("p_obs", "raw judge (pre)"), ("p_debias", "de-biased (post)")]):
        ax.axhline(0.5, ls="--", lw=0.8, c="grey", zorder=2)
        for fate, (_, colour, z) in FATE.items():
            f = g[g.fate.eq(fate)]
            ax.plot(f.gap, f[col], "o", ms=ms, mec="none", c=colour,
                    alpha=kept_alpha if fate == "kept" else 0.9, zorder=z)
        curve(ax, g, col, groups="judge" if per_judge_curves else None)
        dress(ax, tag)
    axes[0].set_ylabel("P(judge picks stronger)")
    fig.supxlabel("True capability gap  (stronger − weaker)", fontsize=8, y=0.02)

    counts = g.fate.value_counts()
    handles = [plt.Line2D([], [], marker="o", ls="", ms=4, mec="none", c=c,
                          label=f"{lab} (n={counts.get(k, 0)})") for k, (lab, c, _) in FATE.items()]
    if per_judge_curves:
        handles.append(plt.Line2D([], [], lw=0.6, c="#9a9a9a", label="per-judge mean"))
    handles += [plt.Line2D([], [], lw=1.4, c="black", label="pooled mean"),
                plt.Line2D([], [], ls="--", lw=0.8, c="grey", label="chance")]
    fig.legend(handles=handles, frameon=False, ncol=3, fontsize=7,
               loc="lower center", bbox_to_anchor=(0.5, legend_y))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    for suffix in ("png", "svg"):
        fig.savefig(out / f"{name}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"{name}: {len(g)} {unit}, fate {counts.to_dict()}")


# judges coloured by identity; 16 distinct hues so up to 16 judges never collide
JUDGE_PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#d62728", "#8551b5", "#e6a000",
                 "#e87ba4", "#17a2b8", "#8c564b", "#444444", "#7f9a00", "#00a0b0",
                 "#c65ba0", "#5a9e6f", "#b5651d", "#6b6ecf"]


def figure_identity(g: pd.DataFrame, out: Path):
    """Individual comparisons coloured by judge identity, raw | de-biased, with per-judge mean."""
    order = (g.groupby("judge").apply(lambda x: np.average(x.p_obs, weights=x.n), include_groups=False)
             .sort_values().index.tolist())
    colour = {j: JUDGE_PALETTE[i % len(JUDGE_PALETTE)] for i, j in enumerate(order)}
    edges = np.arange(0, XHI + 0.1, 0.1)
    fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.5), sharex=True, sharey=True)
    for ax, (col, tag) in zip(axes, [("p_obs", "raw judge (pre)"), ("p_debias", "de-biased (post)")]):
        ax.axhline(0.5, ls="--", lw=0.8, c="grey", zorder=1)
        for j in order:
            sub = g[g.judge.eq(j)]
            ax.plot(sub.gap, sub[col], "o", ms=2.4, mec="none", c=colour[j], alpha=0.45, zorder=2)
            b = sub.groupby(pd.cut(sub.gap, edges), observed=True)[col].mean().dropna()
            ax.plot([iv.mid for iv in b.index], b.to_numpy(), lw=1.3, c=colour[j],
                    zorder=5, solid_capstyle="round")
        dress(ax, tag)
    axes[0].set_ylabel("P(judge picks stronger)")
    fig.supxlabel("True capability gap  (stronger − weaker)", fontsize=8, y=0.02)
    handles = [plt.Line2D([], [], marker="o", ls="", ms=4, mec="none", c=colour[j], label=j) for j in order]
    handles.append(plt.Line2D([], [], ls="--", lw=0.8, c="grey", label="chance"))
    rows = int(np.ceil((len(order) + 1) / 4))
    fig.legend(handles=handles, frameon=False, ncol=4, fontsize=7,
               loc="lower center", bbox_to_anchor=(0.5, -0.10 - 0.075 * rows))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    for suffix in ("png", "svg"):
        fig.savefig(out / f"gap_curve_debiased_judge_identity.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("judge_identity: weak→strong", order)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parents[2] / "judge_results/glm_correction"
    p.add_argument("--frame", type=Path,
                   default=Path("/private/tmp/small-judge-bias-correction/analysis/hibayes/gpqa_glm_corrected.csv"))
    p.add_argument("--out", type=Path, default=here)
    p.add_argument("--min-judge-pair-n", type=int, default=20)
    args = p.parse_args()

    d = verdicts(args.frame)
    pooled = collapse(d, ["ukey"], min_n=1)
    by_judge = collapse(d, ["judge", "ukey"], min_n=args.min_judge_pair_n)
    pooled.to_csv(args.out / "debiased_pairs.csv", index=False)
    by_judge.to_csv(args.out / "debiased_pairs_by_judge.csv", index=False)

    figure(pooled, args.out, "gap_curve_debiased", per_judge_curves=False,
           unit="pairs", ms=3, kept_alpha=0.5, legend_y=-0.16)
    figure(by_judge, args.out, "gap_curve_debiased_by_judge", per_judge_curves=True,
           unit="judge×pair", ms=2.4, kept_alpha=0.28, legend_y=-0.30)
    figure_identity(by_judge, args.out)
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
