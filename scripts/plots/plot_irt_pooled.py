#!/usr/bin/env python3
"""Pooled GPQA + HLE IRT: one 2PL fit over both benchmarks, so model abilities and question
difficulties share a single scale.

Uses every successful full-run log of the pilot models (configs/experiments/pairs/models.txt);
each repeat is a separate Bernoulli observation, items are keyed by (benchmark, question id).
Writes analysis/irt/:
  pooled_capabilities.csv  model, theta (posterior mean, 5/95%), has_hle, per-bench accuracy
  pooled_difficulties.csv  benchmark, item, b (difficulty), a (discrimination), 5/95%
  pooled_irt.png/svg       top: difficulty density per benchmark; bottom: model abilities
                           with 90% credible intervals, on the same axis. Open markers =
                           models with GPQA logs only (ability anchored by GPQA alone).

--hle-gold keeps only HLE-Verified's gold subset (668 audited, unmodified items; ids from
inspect_evals) and writes pooled_gold_* instead.

Run:  uv run python scripts/plots/plot_irt_pooled.py [--warmup 1000] [--samples 1000] [--replot] [--hle-gold]
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


def responses() -> pd.DataFrame:
    rows = []
    for bench in BENCHES:
        task = TaskConfig.from_cli(f"configs/bench/{bench}.yml")
        for yml in MODELS:
            path = get_eval_path(task, ModelConfig.from_cli(yml))
            for f in path.rglob("*.eval"):
                if is_successful_eval(f):
                    rows += [dict(benchmark=bench, model=Path(yml).stem, item=s.id, correct=extract_score(s),
                                  uid=(s.metadata or {}).get("uid"),  # HLE's HF id, for the gold filter
                                  category=(s.metadata or {}).get("category"))
                             for s in read_eval_log_sample_summaries(str(f))]
    return pd.DataFrame(rows)


def plot(caps: pd.DataFrame, diffs: pd.DataFrame, tag: str):
    caps = caps.sort_values("theta")
    fig, (top, bot) = plt.subplots(2, 1, figsize=(2.2, 3.7), sharex=True,
                                   gridspec_kw=dict(height_ratios=[1, 5], hspace=0.1))
    bins = np.linspace(-4, 3, 29)
    for bench in BENCHES:
        top.hist(diffs.b[diffs.benchmark == bench], bins=bins, density=True, histtype="stepfilled",
                 color=COLOR[bench], alpha=0.5, lw=0, label=bench.upper())
    top.set_ylabel("Items"); top.set_yticks([])
    top.spines["left"].set_visible(False); top.spines["bottom"].set_visible(False)
    top.tick_params(bottom=False)
    top.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False,
               handlelength=1, columnspacing=1)
    offset(top)

    y = np.arange(len(caps))
    for full, sub in caps.groupby("has_hle"):
        yy = y[caps.has_hle.values == full]
        bot.errorbar(sub.theta, yy, xerr=[sub.theta - sub.theta_lo, sub.theta_hi - sub.theta],
                     fmt="o", ms=3, lw=1, capsize=0, color="#333", mfc="#333" if full else "white")
    bot.set_yticks(y, caps.model, fontsize=6); bot.set_ylim(-0.7, len(caps) - 0.3)
    bot.set_xlim(-4, 3); bot.set_xticks([-4, -2, 0, 2])
    bot.set_xlabel("Ability θ / difficulty b")
    bot.tick_params(length=2)
    offset(bot)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"{tag}_irt.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--samples", type=int, default=1000)
    ap.add_argument("--replot", action="store_true", help="redraw from the stored CSVs, no fit")
    ap.add_argument("--hle-gold", action="store_true", help="HLE-Verified gold subset only")
    args = ap.parse_args()
    tag = "pooled_gold" if args.hle_gold else "pooled"
    if args.replot:
        plot(pd.read_csv(OUT / f"{tag}_capabilities.csv"), pd.read_csv(OUT / f"{tag}_difficulties.csv"), tag)
        raise SystemExit

    df = responses()
    if args.hle_gold:
        from inspect_evals.hle.hle import get_verified_gold_uids
        df = df[(df.benchmark != "hle") | df.uid.isin(get_verified_gold_uids())]
    print(f"{len(df)} observations, {df.model.nunique()} models, "
          f"{df.groupby('benchmark').item.nunique().to_dict()} items")
    data = encode(df)
    post = fit(data, model="2pl", warmup=args.warmup, samples=args.samples)

    theta, a, b = (np.asarray(post[k]) for k in ("theta", "a", "b"))
    caps = pd.DataFrame({"model": data.models, "theta": theta.mean(0),
                         "theta_lo": np.percentile(theta, 5, 0), "theta_hi": np.percentile(theta, 95, 0)})
    caps["has_hle"] = caps.model.isin(df.model[df.benchmark == "hle"])
    acc = df.groupby(["model", "benchmark"]).correct.mean().unstack().add_prefix("acc_")
    caps = caps.merge(acc, left_on="model", right_index=True).sort_values("theta", ascending=False)
    diffs = pd.DataFrame(data.items, columns=["benchmark", "item"]).assign(
        b=b.mean(0), b_lo=np.percentile(b, 5, 0), b_hi=np.percentile(b, 95, 0),
        a=a.mean(0), a_lo=np.percentile(a, 5, 0), a_hi=np.percentile(a, 95, 0))

    OUT.mkdir(parents=True, exist_ok=True)
    caps.round(3).to_csv(OUT / f"{tag}_capabilities.csv", index=False)
    diffs.round(3).to_csv(OUT / f"{tag}_difficulties.csv", index=False)
    print(caps.round(3).to_string(index=False))
    print(diffs.groupby("benchmark")[["b", "a"]].agg(["mean", "std"]).round(2))
    for bench in BENCHES:
        ok = caps[f"acc_{bench}"].notna()
        rho = stats.spearmanr(caps.theta[ok], caps[f"acc_{bench}"][ok])[0]
        print(f"spearman(theta, {bench} acc) = {rho:.3f}  N={ok.sum()}")
    plot(caps, diffs, tag)
