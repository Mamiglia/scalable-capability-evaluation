"""Judge-predicted vs actual hard-half accuracy per model, one panel per benchmark.

Items of each benchmark are split into an easy and a hard half by their mean accuracy over all
canonical rollouts. Three judges see exactly the same easy-half verdict rows (pairs and items):
the LLM judge; an oracle whose pick probability is 0.5 + (easy-half accuracy gap)/2; and an
oracle whose pick probability is 0.5 + (hard-half accuracy gap)/2 (always picking the more accurate
model instead loses the size of the gap, and Bradley-Terry on these sparse pairs then cannot place
the models).
Each judge's Bradley-Terry ability (src/ranking.rank) is turned into a predicted hard-half accuracy by
its own regression accuracy = scale * sigmoid(slope * ability + offset), fitted on the other models
(leave-one-out). 95% intervals from an easy-item bootstrap,
ability and line refitted per draw. AIME years are pooled.

    uv run python scripts/judge/judged_capability_over_time.py --judge o4-mini
    uv run python scripts/judge/judged_capability_over_time.py --judge o4-mini --x capability   # models ordered by hard-half accuracy
    uv run python scripts/judge/judged_capability_over_time.py --judge o4-mini --x frontier     # frozen frontier roster
"""
import argparse
import glob
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
import numpy as np
import pandas as pd
import yaml
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/verdict_dissection"))
from ranking import rank  # noqa: E402
from ranking.scoring import resample_items  # noqa: E402
from build_design_matrices import pool_aime  # noqa: E402

CONFIG = ROOT / "configs/design_matrices.yaml"
OUT = ROOT / "judge_results/judged_capability_over_time"
ACTUAL, PREDICTED, ORACLE, ORACLE_HARD = "#333333", "#2a78d6", "#eb6834", "#1baf7a"
BENCH_LABEL = {"gpqa": "GPQA", "aime": "AIME", "intercode": "InterCode CTF", "swe_bench_mini": "SWE-bench mini"}
EXCLUDED_EFFORT = re.compile(r"(?:^|[-_])(?:low|high|xhigh)$", re.IGNORECASE)
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "font.family": "sans-serif", "font.sans-serif": ["Arial"],
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
                     "xtick.major.width": 0.8, "ytick.major.width": 0.8, "svg.fonttype": "none"})


def slug(name):
    """Lower-case letters, digits and dots only, so YAML names and verdict model names match."""
    return re.sub(r"[^a-z0-9.]", "", name.split("/")[-1].lower())


def short_name(model):
    return re.sub(r"^nvidia-|-bf16$|-it$|-instruct$", "", model)


def load_frontier_roster(path, bench):
    """Load the frozen side-analysis roster and reject excluded effort variants."""
    roster = (yaml.safe_load(Path(path).read_text()) or {}).get(bench, {})
    excluded = [model for model in roster.values() if EXCLUDED_EFFORT.search(model)]
    if excluded:
        raise ValueError(f"frontier roster contains excluded effort variant: {', '.join(excluded)}")
    return roster


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


def scaled_sigmoid(ability, scale, slope, offset):
    return scale / (1 + np.exp(-(slope * ability + offset)))


def oracle_verdicts(verdicts, accuracy):
    """The same verdict rows judged by accuracy: pick_left = 0.5 + (accuracy_left - accuracy_right) / 2."""
    left, right = accuracy.reindex(verdicts.model_left).to_numpy(), accuracy.reindex(verdicts.model_right).to_numpy()
    return verdicts.assign(pick_left=0.5 + (left - right) / 2)


def predict(verdicts, hard_accuracy, judge):
    """Leave-one-out predicted hard-half accuracy per model: scale * sigmoid(slope * ability + offset) fitted on the others."""
    ability = rank(verdicts).set_index("model").ability
    models = ability.index.intersection(hard_accuracy.index).drop(judge, errors="ignore")
    x, y = ability[models].to_numpy(), hard_accuracy[models].to_numpy()
    predicted = {}
    for i, model in enumerate(models):
        others = np.arange(len(models)) != i
        params, _ = curve_fit(scaled_sigmoid, x[others], y[others], p0=(max(y.max(), 0.05), 1.0, 0.0),
                              bounds=([0.01, 0.0, -20.0], [1.0, 50.0, 20.0]), maxfev=5000)
        predicted[model] = scaled_sigmoid(x[i], *params)
    return pd.Series(predicted)


def predictions_with_intervals(verdicts, easy_accuracy, hard_accuracy, judge, n_resamples, seed):
    """Point prediction and bootstrap interval for the LLM judge and the two oracles on the same verdict rows."""
    rng = np.random.default_rng(seed)
    table = pd.DataFrame({"actual": hard_accuracy})
    for name, frame in (("predicted", verdicts), ("oracle", oracle_verdicts(verdicts, easy_accuracy)),
                        ("oracle_hard", oracle_verdicts(verdicts, hard_accuracy))):
        point = predict(frame, hard_accuracy, judge)
        draws = pd.concat([predict(resample_items(frame, rng), hard_accuracy, judge) for _ in range(n_resamples)], axis=1)
        table[name] = point
        table[f"{name}_low"], table[f"{name}_high"] = draws.quantile(0.025, axis=1), draws.quantile(0.975, axis=1)
    return table.dropna(subset=["predicted"])


def draw(ax, x, table, judge):
    """Prediction as a line with its 95% band, actual hard-half accuracy as points."""
    hard_band = ax.fill_between(x, table.oracle_hard_low, table.oracle_hard_high, color=ORACLE_HARD, alpha=0.18, lw=0)
    hard_line, = ax.plot(x, table.oracle_hard, color=ORACLE_HARD, lw=1)
    oracle_band = ax.fill_between(x, table.oracle_low, table.oracle_high, color=ORACLE, alpha=0.18, lw=0)
    oracle_line, = ax.plot(x, table.oracle, color=ORACLE, lw=1)
    band = ax.fill_between(x, table.predicted_low, table.predicted_high, color=PREDICTED, alpha=0.18, lw=0)
    line, = ax.plot(x, table.predicted, color=PREDICTED, lw=1)
    points, = ax.plot(x, table.actual, "o", ms=3, color=ACTUAL, clip_on=False)
    ax.set_ylabel("Hard-half accuracy")
    lows, highs = table[["predicted_low", "oracle_low", "oracle_hard_low"]], table[["predicted_high", "oracle_high", "oracle_hard_high"]]
    low, high = min(0, lows.min().min()), max(1, highs.max().max())
    ax.set_ylim(low - 0.02, high + 0.02)
    ax.set_yticks([0, 0.5, 1])
    ax.spines["left"].set_bounds(0, 1)
    for side in ("left", "bottom"):
        ax.spines[side].set_position(("outward", 3))
    ax.legend([points, (band, line), (oracle_band, oracle_line), (hard_band, hard_line)],
              ["actual", f"{judge} judge, easy half", "oracle picking by easy-half accuracy", "oracle picking by hard-half accuracy"],
              handler_map={tuple: HandlerTuple(ndivide=1)}, loc="lower left", bbox_to_anchor=(0, 1.02), ncol=2,
              frameon=False, handletextpad=0.5, columnspacing=1.5, handlelength=1.8)


def save(fig, path):
    fig.savefig(f"{path}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{path}.svg", bbox_inches="tight")
    plt.close(fig)


def plot_by_capability(table, judge, bench, path):
    """Models along x in order of actual hard-half accuracy."""
    table = table.rename_axis("model").sort_values("actual").reset_index()
    fig, ax = plt.subplots(figsize=(3.6, 2.4))
    x = np.arange(len(table))
    draw(ax, x, table, judge)
    ax.set_xticks(x)
    ax.set_xticklabels(table.model.map(short_name), fontsize=5.5)
    ax.tick_params(axis="x", labelrotation=90)
    ax.set_xlim(-0.5, len(table) - 0.5)
    ax.spines["bottom"].set_bounds(0, len(table) - 1)
    ax.set_xlabel(f"{BENCH_LABEL[bench]}: models by hard-half accuracy")
    save(fig, path)


def date_axis(ax, table, judge, judge_date, bench):
    ax.axvline(judge_date, color="grey", lw=0.8, ls="--")
    ax.text(judge_date, ax.get_ylim()[1], f" {judge} released", color="grey", fontsize=6, va="top", ha="left")
    first, last = min(table.release.min(), judge_date), max(table.release.max(), judge_date)
    every = "12MS" if (last - first).days > 900 else "6MS"   # yearly ticks once the span passes ~2.5 years
    ticks = pd.date_range(first.replace(month=1, day=1), last + pd.DateOffset(months=6), freq=every)
    ax.set_xticks(ticks)
    ax.set_xticklabels([t.strftime("%b %Y") for t in ticks])
    ax.set_xlim(ticks[0] - pd.Timedelta(days=30), ticks[-1] + pd.Timedelta(days=30))
    ax.spines["bottom"].set_bounds(*plt.matplotlib.dates.date2num([ticks[0], ticks[-1]]))
    ax.set_xlabel(f"{BENCH_LABEL[bench]}: model release date")


def plot(table, judge, judge_date, bench, path):
    """Models along x by release date."""
    table = table.sort_values("release")
    fig, ax = plt.subplots(figsize=(3.6, 2.4))
    draw(ax, table.release, table, judge)
    date_axis(ax, table, judge, judge_date, bench)
    save(fig, path)


def frontier(table, min_opponents, roster):
    """The frozen frontier roster, requiring enough judged opponents for every model."""
    eligible = table[table.n_opponents >= min_opponents]
    models = list(roster.values())
    missing = [model for model in models if model not in eligible.index]
    if missing:
        raise ValueError(f"frontier roster models missing or underjudged: {', '.join(missing)}")
    wrong_quarter = [f"{model}: roster {quarter}, release {eligible.loc[model, 'release'].to_period('Q')}"
                     for quarter, model in roster.items()
                     if str(eligible.loc[model, "release"].to_period("Q")) != str(quarter)]
    if wrong_quarter:
        raise ValueError(f"frontier roster has wrong release quarter: {', '.join(wrong_quarter)}")
    return eligible.loc[models].sort_values("release")


def earlier_opponents_only(verdicts, dates, reference_date):
    """Drop verdicts where a model released after the reference judge meets a model not released before it."""
    left = verdicts.model_left.map(lambda m: dates[slug(m)])
    right = verdicts.model_right.map(lambda m: dates[slug(m)])
    later_opponent = ((left > reference_date) & (right >= left)) | ((right > reference_date) & (left >= right))
    return verdicts[~later_opponent.to_numpy()]


def opponents_per_model(verdicts):
    """Distinct models each model was judged against."""
    pairs = pd.concat([verdicts[["model_left", "model_right"]],
                       verdicts[["model_right", "model_left"]].set_axis(["model_left", "model_right"], axis=1)])
    return pairs.drop_duplicates().groupby("model_left").model_right.nunique()


def plot_frontier(table, judge, judge_date, bench, path, style):
    """Frozen frontier models by release date, with predictions as a band or as points."""
    test = table
    fig, ax = plt.subplots(figsize=(3.6, 2.4))
    ax.axvspan(judge_date, table.release.max() + pd.Timedelta(days=400), color="#f2f2f2", lw=0, zorder=0)
    for _, r in test.iterrows():
        ax.plot([r.release, r.release], [r.actual, r.predicted], ":", color="k", lw=0.8)
    if style == "band":
        band = ax.fill_between(test.release, test.predicted_low, test.predicted_high, color=PREDICTED, alpha=0.18, lw=0)
        line, = ax.plot(test.release, test.predicted, color=PREDICTED, lw=1)
        handle = (band, line)
    else:
        handle, = ax.plot(test.release, test.predicted, "o", ms=3, color=PREDICTED, clip_on=False)
    points, = ax.plot(table.release, table.actual, "o", ms=3, color=ACTUAL, clip_on=False)
    for model, r in table.iterrows():   # model names in a row under the axis label, vertical text
        ax.annotate(short_name(model), (r.release, 0), xycoords=("data", "axes fraction"), xytext=(0, -34),
                    textcoords="offset points", fontsize=5, rotation=90, ha="center", va="top",
                    color="#555555", annotation_clip=False)
    ax.set_ylabel("Hard-half accuracy")
    ax.set_ylim(-0.02, 1.02)
    ax.set_yticks([0, 0.5, 1])
    ax.spines["left"].set_bounds(0, 1)
    for side in ("left", "bottom"):
        ax.spines[side].set_position(("outward", 3))
    ax.legend([points, handle], ["actual (frozen frontier roster)", "predicted"],
              handler_map={tuple: HandlerTuple(ndivide=1)}, loc="lower left", bbox_to_anchor=(0, 1.02), ncol=1,
              frameon=False, handletextpad=0.5, handlelength=1.8)
    date_axis(ax, table, judge, judge_date, bench)
    save(fig, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge", default="o4-mini")
    parser.add_argument("--resamples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--x", choices=["release", "capability", "frontier"], default="release", help="what orders the models along x")
    parser.add_argument("--min-opponents", type=int, default=2, help="frontier plot: only models judged against at least this many others")
    parser.add_argument("--frontier-roster", type=Path, default=ROOT / "configs/frontier_roster.yaml",
                        help="frozen benchmark -> quarter -> model roster used by --x frontier")
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--earlier-opponents-only", action="store_true",
                        help="models released after the reference judge keep only verdicts against models released before themselves")
    parser.add_argument("--reference-judge", help="judge whose release date the pairing rule refers to (default: --judge)")
    parser.add_argument("--easy-items", type=Path, help="CSV (bench,item) fixing the easy half; default splits "
                        "the loaded rollouts in two by item accuracy")
    parser.add_argument("--tables", type=Path, help="directory with verdicts.parquet and rollouts.parquet from "
                        "verdict_tables.py; default is the verdict-dissection store")
    args = parser.parse_args()
    config = yaml.safe_load(CONFIG.read_text())
    rollouts_path = args.tables / "rollouts.parquet" if args.tables else Path(config["features_root"]) / "raw/identity.parquet"
    verdicts_path = args.tables / "verdicts.parquet" if args.tables else Path(config["output_root"]) / "pair.parquet"
    rollouts = pool_aime(pd.read_parquet(rollouts_path, columns=["bench", "model", "item", "correct", "is_canonical"]))
    rollouts = rollouts[rollouts.is_canonical]
    verdicts = pd.read_parquet(verdicts_path, columns=["bench", "judge", "item", "model_left", "model_right", "pick_left"])
    verdicts = verdicts[verdicts.judge == args.judge]
    dates = release_dates()
    if args.earlier_opponents_only:
        verdicts = earlier_opponents_only(verdicts, dates, dates[slug(args.reference_judge or args.judge)])
    easy = split_items(rollouts)
    if args.easy_items:   # fixed split; benchmarks absent from the file are left out entirely
        listed = pd.read_csv(args.easy_items, dtype=str).set_index(["bench", "item"]).index
        easy = pd.Series(easy.index.isin(listed), index=easy.index)
        rollouts = rollouts[rollouts.bench.isin(listed.get_level_values("bench"))]
        verdicts = verdicts[verdicts.bench.isin(listed.get_level_values("bench"))]
    rollout_is_easy = easy.reindex(pd.MultiIndex.from_frame(rollouts[["bench", "item"]])).to_numpy()
    easy_rollouts, hard_rollouts = rollouts[rollout_is_easy], rollouts[~rollout_is_easy]
    easy_verdicts = verdicts[easy.reindex(pd.MultiIndex.from_frame(verdicts[["bench", "item"]])).fillna(False).to_numpy(bool)]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for bench, bench_verdicts in easy_verdicts.groupby("bench"):
        easy_accuracy = easy_rollouts[easy_rollouts.bench == bench].groupby("model").correct.mean()
        hard_accuracy = hard_rollouts[hard_rollouts.bench == bench].groupby("model").correct.mean()
        table = predictions_with_intervals(bench_verdicts, easy_accuracy, hard_accuracy, args.judge, args.resamples, args.seed)
        table["release"] = [dates[slug(m)] for m in table.index]
        table["n_opponents"] = opponents_per_model(bench_verdicts).reindex(table.index).fillna(0).astype(int)
        table.sort_values("release").to_csv(args.out_dir / f"{args.judge}_{bench}.csv", index_label="model")
        print(f"{bench}: {len(table)} models, {len(bench_verdicts)} easy-half verdicts on "
              f"{bench_verdicts.item.nunique()} items, MAE {args.judge} {(table.predicted - table.actual).abs().mean():.3f}, "
              f"oracle easy {(table.oracle - table.actual).abs().mean():.3f}, "
              f"oracle hard {(table.oracle_hard - table.actual).abs().mean():.3f}")
        if args.x == "capability":
            plot_by_capability(table, args.judge, bench, args.out_dir / f"{args.judge}_{bench}_by_capability")
        elif args.x == "frontier":
            selected = frontier(table, args.min_opponents, load_frontier_roster(args.frontier_roster, bench))
            selected.to_csv(args.out_dir / f"{args.judge}_{bench}_frontier.csv", index_label="model")
            for style in ("band", "points"):
                plot_frontier(selected, args.judge, dates[slug(args.judge)], bench,
                              args.out_dir / f"{args.judge}_{bench}_frontier_{style}", style)
        else:
            plot(table, args.judge, dates[slug(args.judge)], bench, args.out_dir / f"{args.judge}_{bench}")


if __name__ == "__main__":
    main()
