"""Judge-predicted vs observed hard-half accuracy against model release date, one panel per benchmark.

Items of each benchmark are split into an easy and a hard half by their mean accuracy over all
canonical rollouts. One judge's pairwise verdicts on the easy half give Bradley-Terry abilities
(src/ranking.rank). True capability is the ability a perfect judge would give on the same pairs:
one that always picks the model with the higher hard-half accuracy. A model's predicted capability
is a straight line from judge ability to perfect-judge ability fitted on the other models
(leave-one-out). The perfect judge's tournament is separable, so its abilities are only bounded by
the ridge; they are standardised per benchmark (mean 0, SD 1). 95% intervals from an easy-item
bootstrap, abilities and line refitted per draw. AIME years are pooled.

    uv run python scripts/judge/judged_capability_over_time.py --judge o4-mini
"""
import argparse
import glob
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/verdict_dissection"))
from ranking import rank  # noqa: E402
from ranking.scoring import resample_items  # noqa: E402
from build_design_matrices import pool_aime  # noqa: E402

CONFIG = ROOT / "configs/design_matrices.yaml"
OUT = ROOT / "judge_results/judged_capability_over_time"
OBSERVED, PREDICTED = "#666666", "#c8452b"


def slug(name):
    """Lower-case letters, digits and dots only, so YAML names and verdict model names match."""
    return re.sub(r"[^a-z0-9.]", "", name.split("/")[-1].lower())


def release_dates():
    """Model slug -> release date from the model YAMLs (first date of a range)."""
    dates = {}
    for path in glob.glob(str(ROOT / "configs/models/*/*.yml")):
        spec = yaml.safe_load(open(path))
        if spec.get("release_date"):
            dates[slug(spec["name"])] = pd.Timestamp(str(spec["release_date"])[:10])
    return dates


def split_items(rollouts):
    """Per benchmark, the easier half of items (by mean accuracy) is True."""
    difficulty = rollouts.groupby(["bench", "item"]).correct.mean().rename("accuracy").reset_index()
    order = difficulty.groupby("bench").accuracy.rank(method="first", ascending=False)
    difficulty["easy"] = order <= difficulty.groupby("bench").item.transform("size") / 2
    return difficulty.set_index(["bench", "item"]).easy


def perfect_judge(verdicts, accuracy):
    """The same pairs judged by accuracy on the held-out items: the higher-accuracy model wins, equal is a tie."""
    left, right = accuracy[verdicts.model_left].to_numpy(), accuracy[verdicts.model_right].to_numpy()
    return verdicts.assign(pick_left=np.where(left > right, 1.0, np.where(left < right, 0.0, 0.5)))


def predict(verdicts, observed, judge):
    """Leave-one-out predicted capability per model from a line through (judge ability, perfect-judge ability)."""
    ability = rank(verdicts).set_index("model").ability
    models = ability.index.intersection(observed.index).drop(judge, errors="ignore")
    x, y = ability[models].to_numpy(), observed[models].to_numpy()
    predicted = {}
    for i, model in enumerate(models):
        others = np.arange(len(models)) != i
        slope, intercept = np.polyfit(x[others], y[others], 1)
        predicted[model] = slope * x[i] + intercept
    return pd.Series(predicted)


def predictions_with_intervals(verdicts, observed, judge, n_resamples, seed):
    rng = np.random.default_rng(seed)
    point = predict(verdicts, observed, judge)
    draws = pd.concat([predict(resample_items(verdicts, rng), observed, judge) for _ in range(n_resamples)], axis=1)
    return pd.DataFrame({"predicted": point, "observed": observed[point.index],
                         "low": draws.quantile(0.025, axis=1), "high": draws.quantile(0.975, axis=1)})


def plot(table, judge, judge_date, judge_accuracy, bench, path):
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "font.family": "sans-serif", "font.sans-serif": ["Arial"],
                         "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
                         "xtick.major.width": 0.8, "ytick.major.width": 0.8, "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(3.6, 2.4))
    ax.axvline(judge_date, color="k", lw=0.8, label=f"judge released ({judge})")
    for _, r in table.iterrows():
        ax.plot([r.release, r.release], [r.observed, r.predicted], ":", color="k", lw=0.8)
    ax.errorbar(table.release, table.predicted, yerr=[table.predicted - table.low, table.high - table.predicted],
                fmt="o", ms=3, color=PREDICTED, ecolor=PREDICTED, elinewidth=1, capsize=0, label="judge-predicted, 95% CI")
    ax.plot(table.release, table.observed, "o", ms=3, color=OBSERVED, label="observed")
    if judge_accuracy is not None:
        ax.plot([judge_date], [judge_accuracy], "o", ms=5, color="k", label="judge itself")
    ax.set_ylabel("Capability (SD units)")
    low, high = min(table.low.min(), table.observed.min()), max(table.high.max(), table.observed.max())
    ax.set_ylim(low - 0.05 * (high - low), high + 0.05 * (high - low))
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    ticks = [t for t in ax.get_yticks() if ax.get_ylim()[0] <= t <= ax.get_ylim()[1]]
    ax.set_yticks(ticks)
    ax.spines["left"].set_bounds(ticks[0], ticks[-1])
    ax.text(0, 1.02, bench, transform=ax.transAxes, color="grey", fontsize=7, va="bottom")
    first, last = min(table.release.min(), judge_date), max(table.release.max(), judge_date)
    ticks = pd.date_range(first.replace(month=1, day=1), last + pd.DateOffset(months=6), freq="6MS")
    ax.set_xticks(ticks)
    ax.set_xticklabels([t.strftime("%b %Y") for t in ticks])
    ax.set_xlim(ticks[0] - pd.Timedelta(days=30), ticks[-1] + pd.Timedelta(days=30))
    ax.spines["bottom"].set_bounds(*plt.matplotlib.dates.date2num([ticks[0], ticks[-1]]))
    for side in ("left", "bottom"):
        ax.spines[side].set_position(("outward", 3))
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.08), ncol=2, frameon=False, handletextpad=0.3, columnspacing=1)
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge", default="o4-mini")
    parser.add_argument("--resamples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    config = yaml.safe_load(CONFIG.read_text())
    rollouts = pool_aime(pd.read_parquet(Path(config["features_root"]) / "raw/identity.parquet",
                                         columns=["bench", "model", "item", "correct", "is_canonical"]))
    rollouts = rollouts[rollouts.is_canonical]
    verdicts = pd.read_parquet(Path(config["output_root"]) / "pair.parquet",
                               columns=["bench", "judge", "item", "model_left", "model_right", "pick_left"])
    verdicts = verdicts[verdicts.judge == args.judge]
    easy = split_items(rollouts)
    hard_rollouts = rollouts[~easy.reindex(pd.MultiIndex.from_frame(rollouts[["bench", "item"]])).to_numpy()]
    easy_verdicts = verdicts[easy.reindex(pd.MultiIndex.from_frame(verdicts[["bench", "item"]])).to_numpy()]
    dates = release_dates()
    OUT.mkdir(parents=True, exist_ok=True)
    for bench, bench_verdicts in easy_verdicts.groupby("bench"):
        hard_accuracy = hard_rollouts[hard_rollouts.bench == bench].groupby("model").correct.mean()
        observed = rank(perfect_judge(bench_verdicts, hard_accuracy)).set_index("model").ability
        observed = (observed - observed.mean()) / observed.std()
        table = predictions_with_intervals(bench_verdicts, observed, args.judge, args.resamples, args.seed)
        table["release"] = [dates[slug(m)] for m in table.index]
        table.sort_values("release").to_csv(OUT / f"{args.judge}_{bench}.csv", index_label="model")
        covered = ((table.observed >= table.low) & (table.observed <= table.high)).mean()
        print(f"{bench}: {len(table)} models, {len(bench_verdicts)} easy-half verdicts on "
              f"{bench_verdicts.item.nunique()} items, MAE {(table.predicted - table.observed).abs().mean():.3f}, "
              f"CI covers observed {covered:.0%}")
        plot(table, args.judge, dates[slug(args.judge)], observed.get(args.judge), bench, OUT / f"{args.judge}_{bench}")


if __name__ == "__main__":
    main()
