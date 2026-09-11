#!/usr/bin/env python3
"""Pooled IRT over several benchmarks: one 2PL fit across all of them, so model abilities and
question difficulties share a single scale (the benches are tied together by the models that
ran both).

Uses every successful full-run log of the pilot models (configs/experiments/pairs/models.txt);
each repeat is a separate Bernoulli observation, items are keyed by (benchmark, question id).
Writes analysis/irt/:
  <tag>_capabilities.csv  model, theta (posterior mean, 5/95%), has_all, per-bench accuracy
  <tag>_difficulties.csv  benchmark, item, b (difficulty), a (discrimination), 5/95%
  <tag>_irt.png/svg       top: difficulty density per benchmark; bottom: model abilities
                          with 90% credible intervals, on the same axis. Open markers =
                          models missing a bench (ability anchored by the rest).
tag is "pooled" for the default GPQA+HLE fit, else the bench names joined by "-".

--hle-gold keeps only HLE-Verified's gold subset (668 audited, unmodified items; ids from
inspect_evals) and writes pooled_gold_* instead.

Run:  uv run python scripts/plots/plot_irt_pooled.py [--benches gpqa hle] [--warmup 1000]
      [--samples 1000] [--replot] [--hle-gold]
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from inspect_ai.log import read_eval_log_sample_summaries
from scipy import stats

from configs.model import ModelConfig
from configs.paths import get_eval_path, is_successful_eval
from configs.task import TaskConfig
from irt.data import encode
from irt.fitting import fit
from judge.util import extract_score
from plot_difficulty import BENCHES, MODELS, offset

OUT = Path("analysis/irt")
COLOR = {"gpqa": "#2a78d6", "hle": "#eb6834"}
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]


def color(bench: str, i: int) -> str:
    return COLOR.get(bench, PALETTE[i % len(PALETTE)])


def responses(benches=BENCHES) -> pd.DataFrame:
    rows = []
    for bench in benches:
        task = TaskConfig.from_cli(f"configs/bench/{bench}.yml")
        for yml in MODELS:
            path = get_eval_path(task, ModelConfig.from_cli(yml))
            for f in path.rglob("*.eval"):
                if is_successful_eval(f):
                    rows += [dict(bench=bench, model=Path(yml).stem, item=s.id, correct=extract_score(s),
                                  uid=(s.metadata or {}).get("uid"),  # HLE's HF id, for the gold filter
                                  category=(s.metadata or {}).get("category"))
                             for s in read_eval_log_sample_summaries(str(f))]
    return pd.DataFrame(rows)


def plot(caps: pd.DataFrame, diffs: pd.DataFrame, tag: str, benches=BENCHES):
    caps = caps.sort_values("theta")
    fig, (top, bot) = plt.subplots(2, 1, figsize=(2.2, 3.7), sharex=True,
                                   gridspec_kw=dict(height_ratios=[1, 5], hspace=0.1))
    lo = np.floor(min(diffs.b.min(), caps.theta_lo.min()))
    hi = np.ceil(max(diffs.b.max(), caps.theta_hi.max()))
    bins = np.linspace(lo, hi, 29)
    for i, bench in enumerate(benches):
        top.hist(diffs.b[diffs.bench == bench], bins=bins, density=True, histtype="stepfilled",
                 color=color(bench, i), alpha=0.5, lw=0, label=bench.replace("_", "-"))
    top.set_ylabel("Items"); top.set_yticks([])
    top.spines["left"].set_visible(False); top.spines["bottom"].set_visible(False)
    top.tick_params(bottom=False)
    top.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False,
               handlelength=1, columnspacing=1)
    offset(top)

    y = np.arange(len(caps))
    for full, sub in caps.groupby("has_all"):
        yy = y[caps.has_all.values == full]
        bot.errorbar(sub.theta, yy, xerr=[sub.theta - sub.theta_lo, sub.theta_hi - sub.theta],
                     fmt="o", ms=3, lw=1, capsize=0, color="#333", mfc="#333" if full else "white")
    bot.set_yticks(y, caps.model, fontsize=6); bot.set_ylim(-0.7, len(caps) - 0.3)
    bot.set_xlim(lo, hi); bot.set_xticks(np.arange(lo, hi + 1, 2 if hi - lo > 5 else 1))
    bot.set_xlabel("Ability θ / difficulty b")
    bot.tick_params(length=2)
    offset(bot)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"{tag}_irt.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--benches", nargs="+", default=BENCHES, help="configs/bench/<name>.yml stems")
    ap.add_argument("--tag", default=None, help="output prefix (default: pooled / bench names)")
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--samples", type=int, default=1000)
    ap.add_argument("--replot", action="store_true", help="redraw from the stored CSVs, no fit")
    ap.add_argument("--hle-gold", action="store_true", help="HLE-Verified gold subset only")
    args = ap.parse_args()
    tag = args.tag or ("pooled" if args.benches == BENCHES else "-".join(args.benches))
    if args.hle_gold:
        tag += "_gold"
    if args.replot:
        plot(pd.read_csv(OUT / f"{tag}_capabilities.csv"), pd.read_csv(OUT / f"{tag}_difficulties.csv"),
             tag, args.benches)
        raise SystemExit

    df = responses(args.benches)
    if args.hle_gold:
        from inspect_evals.hle.hle import get_verified_gold_uids
        df = df[(df.bench != "hle") | df.uid.isin(get_verified_gold_uids())]
    print(f"{len(df)} observations, {df.model.nunique()} models, "
          f"{df.groupby('bench').item.nunique().to_dict()} items")
    data = encode(df)
    post = fit(data, model="2pl", warmup=args.warmup, samples=args.samples)

    theta, a, b = (np.asarray(post[k]) for k in ("theta", "a", "b"))
    caps = pd.DataFrame({"model": data.models, "theta": theta.mean(0),
                         "theta_lo": np.percentile(theta, 5, 0), "theta_hi": np.percentile(theta, 95, 0)})
    caps["has_all"] = caps.model.map(df.groupby("model").bench.nunique()) == len(args.benches)
    acc = df.groupby(["model", "bench"]).correct.mean().unstack().add_prefix("acc_")
    caps = caps.merge(acc, left_on="model", right_index=True).sort_values("theta", ascending=False)
    diffs = pd.DataFrame(data.items, columns=["bench", "item"]).assign(
        b=b.mean(0), b_lo=np.percentile(b, 5, 0), b_hi=np.percentile(b, 95, 0),
        a=a.mean(0), a_lo=np.percentile(a, 5, 0), a_hi=np.percentile(a, 95, 0))

    OUT.mkdir(parents=True, exist_ok=True)
    caps.round(3).to_csv(OUT / f"{tag}_capabilities.csv", index=False)
    diffs.round(3).to_csv(OUT / f"{tag}_difficulties.csv", index=False)
    print(caps.round(3).to_string(index=False))
    print(diffs.groupby("bench")[["b", "a"]].agg(["mean", "std"]).round(2))
    for bench in args.benches:
        ok = caps[f"acc_{bench}"].notna()
        rho = stats.spearmanr(caps.theta[ok], caps[f"acc_{bench}"][ok])[0]
        print(f"spearman(theta, {bench} acc) = {rho:.3f}  N={ok.sum()}")
    plot(caps, diffs, tag, args.benches)
