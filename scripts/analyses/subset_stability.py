"""How much judging data is enough? Item-subsampling analysis of the pilot.

For each benchmark, resample n items (with replacement, so the spread at n is
the true sampling noise rather than a finite-population shrinkage) and recompute
every quantity the pilot reports, per judge:

  pair accuracy     RMSE over the judge's pairs vs its full-run value
  judge accuracy    |error| of the judge's balanced accuracy over pairs
  sign flips        share of clear pairs (full |acc-.5| > .1) whose winner flips
  gap correlation   |error| of the Spearman(collection gap, pair accuracy)
  judge ranking     Spearman between full and subset judge orderings (one
                    number per replicate; median and 10-90% band)
  top judge kept    share of replicates whose best judge is the full-run best

n counts distinct benchmark items; a pair only sees the ones both models got
right, so per-run item counts are ~0.7n on GPQA and ~0.95n on MBPP.

Replication unit is the judge: curves are mean +/- t 95% CI across judges.
The dashed line is the binomial worst case 0.5/sqrt(n) for one pair.

Usage:
  uv run python scripts/analyses/subset_stability.py [--verdicts ...] [--out ...]
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, t as tdist

from plots import INK2, MUTED, PALETTE, legend, plt, save, style

warnings.filterwarnings("ignore", "Mean of empty slice")
SIZES = np.array([5, 10, 15, 20, 30, 40, 50, 75, 100, 150, 200])
BENCH_COLOR = {"GPQA": PALETTE[0], "MBPP": PALETTE[1]}
CLEAR = 0.1  # a pair is "clear" when the full run sits this far from chance


def item_scores(verdicts: Path) -> pd.DataFrame:
    v = pd.read_csv(verdicts, dtype={"item": str},
                    usecols=["condition", "benchmark", "judge", "pair", "item", "correct", "collection_accuracy"])
    v = v[(v.condition == "reasoning_off") & v.correct.notna()]
    return (v.groupby(["benchmark", "judge", "pair", "item"])
             .agg(score=("correct", "mean"), gap=("collection_accuracy", "first")).reset_index())


def resample(items: pd.DataFrame, rng, replicates: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(per-judge metrics, per-replicate judge-rank rho) for every bench x n."""
    per_judge, per_rep = [], []
    for bench, g in items.groupby("benchmark"):
        piv = g.pivot_table(index="item", columns=["judge", "pair"], values="score")
        gaps = g.groupby("pair").gap.first()
        judges = piv.columns.get_level_values(0).unique()
        full = piv.mean()
        full_bal = full.groupby(level=0).mean()
        clear = (full - 0.5).abs() > CLEAR
        full_gc = {j: spearmanr(gaps.reindex(full[j].index), full[j]).statistic for j in judges}
        vals = piv.to_numpy()
        for n in SIZES[SIZES <= len(piv)]:
            for r in range(replicates):
                idx = rng.integers(0, len(piv), n)
                sub = pd.Series(np.nanmean(vals[idx], axis=0), index=piv.columns)
                sub_bal = sub.groupby(level=0).mean()
                for j in judges:
                    err = sub[j] - full[j]
                    flip = (np.sign(sub[j] - 0.5) != np.sign(full[j] - 0.5))[clear[j]]
                    ok = sub[j].notna()
                    gc = spearmanr(gaps.reindex(sub[j].index)[ok], sub[j][ok]).statistic if ok.sum() >= 3 else np.nan
                    per_judge.append(dict(benchmark=bench, n=n, replicate=r, judge=j,
                                          pair_rmse=np.sqrt(np.nanmean(err ** 2)),
                                          judge_abs_err=abs(sub_bal[j] - full_bal[j]),
                                          sign_flip=np.nanmean(flip) if len(flip) else np.nan,
                                          gap_corr_abs_err=abs(gc - full_gc[j])))
                per_rep.append(dict(benchmark=bench, n=n, replicate=r,
                                    judge_rank_rho=spearmanr(full_bal, sub_bal).statistic,
                                    top_judge_same=float(sub_bal.idxmax() == full_bal.idxmax())))
        print(f"{bench}: {len(piv)} items, {len(judges)} judges, {piv.shape[1]} runs; "
              f"{int(clear.sum())}/{len(clear)} pairs clear (|acc-.5|>{CLEAR})")
    return pd.DataFrame(per_judge), pd.DataFrame(per_rep)


def across_judges(per_judge: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Per judge, average replicates; then mean +/- t95 across judges."""
    j = per_judge.groupby(["benchmark", "n", "judge"])[metric].mean().reset_index()
    def ci(x):
        x = x.dropna()
        h = tdist.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan
        return pd.Series(dict(mean=x.mean(), lo=x.mean() - h, hi=x.mean() + h, n_judges=len(x)))
    return j.groupby(["benchmark", "n"])[metric].apply(ci).unstack().reset_index()


def figure(per_judge: pd.DataFrame, per_rep: pd.DataFrame, out: Path) -> None:
    panels = [("pair_rmse", "Pair accuracy RMSE", (0, 0.2), [0, .1, .2]),
              ("judge_abs_err", "Judge accuracy |error|", (0, 0.05), [0, .025, .05]),
              ("sign_flip", "Clear pairs flipped", (0, 0.3), [0, .15, .3]),
              ("gap_corr_abs_err", "Gap-corr. |error|", (0, 0.4), [0, .2, .4]),
              ("judge_rank_rho", "Judge-rank ρ", (0, 1), [0, .5, 1]),
              ("top_judge_same", "Top judge kept", (0, 1), [0, .5, 1])]
    fig, axes = plt.subplots(2, 3, figsize=(4.4, 2.9), sharex=True)
    for ax, (metric, label, ylim, yticks) in zip(axes.flat, panels):
        for bench, color in BENCH_COLOR.items():
            if metric in per_judge:
                s = across_judges(per_judge[per_judge.benchmark == bench], metric)
            elif metric == "judge_rank_rho":  # one value per replicate: median, 10-90% band
                s = (per_rep[per_rep.benchmark == bench].groupby("n")[metric]
                     .quantile([.1, .5, .9]).unstack().reset_index())
                s.columns = ["n", "lo", "mean", "hi"]
            else:  # binary per replicate: rate only
                s = per_rep[per_rep.benchmark == bench].groupby("n")[metric].mean().reset_index()
                s["lo"] = s["hi"] = s[metric]
                s = s.rename(columns={metric: "mean"})
            ax.plot(s.n, s["mean"], color=color, lw=1, marker="o", ms=2.5, label=bench)
            ax.fill_between(s.n, s.lo, s.hi, color=color, alpha=0.18, lw=0)
        if metric == "pair_rmse":
            ax.plot(SIZES, 0.5 / np.sqrt(SIZES), color=MUTED, lw=0.8, ls=(0, (4, 3)), label="0.5/√n")
        ax.set_xscale("log")
        ax.set_xticks([5, 20, 50, 200])
        ax.set_xticklabels(["5", "20", "50", "200"])
        ax.minorticks_off()
        ax.set_ylim(*ylim)
        ax.set_yticks(yticks)
        ax.set_ylabel(label)
        style(ax)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_position(("outward", 4))
    for ax in axes[1]:
        ax.set_xlabel("Items")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02),
               handletextpad=0.4, columnspacing=1.2, labelcolor=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=0.6, w_pad=0.6)
    save(fig, out / "subset_stability")


def table(per_judge: pd.DataFrame, per_rep: pd.DataFrame) -> pd.DataFrame:
    rows = [across_judges(per_judge, m).set_index(["benchmark", "n"])["mean"].rename(m)
            for m in ("pair_rmse", "judge_abs_err", "sign_flip", "gap_corr_abs_err")]
    rows += [per_rep.groupby(["benchmark", "n"])[m].mean().rename(m)
             for m in ("judge_rank_rho", "top_judge_same")]
    return pd.concat(rows, axis=1).reset_index()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--verdicts", type=Path, default=Path("analysis/pilot-results/verdicts.csv"))
    p.add_argument("--out", type=Path, default=Path("analysis/subset-stability"))
    p.add_argument("--replicates", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--replot", action="store_true", help="reuse cached per_judge.csv / per_replicate.csv")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.replot:
        per_judge, per_rep = pd.read_csv(a.out / "per_judge.csv"), pd.read_csv(a.out / "per_replicate.csv")
    else:
        per_judge, per_rep = resample(item_scores(a.verdicts), np.random.default_rng(a.seed), a.replicates)
        per_judge.to_csv(a.out / "per_judge.csv", index=False)
        per_rep.to_csv(a.out / "per_replicate.csv", index=False)
    summary = table(per_judge, per_rep)
    summary.to_csv(a.out / "summary.csv", index=False)
    print(summary.round(3).to_string(index=False))
    figure(per_judge, per_rep, a.out)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
