#!/usr/bin/env python3
"""Does IRT difficulty add anything over plain P(correct)? Per pool of benchmarks, fit a 2PL
(every repeat an observation) and compare against P(correct) = mean over models of each model's
mean correctness on the item.

Pools (domains kept apart):
  science   gpqa + hle-gold (HLE-Verified gold subset), the pilot models with logs on both (22)
  math      aime24 + aime25 + aime26 + apex_2025 + apex_shortlist, every model with logs on all five (26)
  math+hle  the math pool plus HLE-gold category Math, pilot models with logs on everything (22)
Easy/hard splits built both ways (IRT b vs P) under three schemes: easiest 25% vs hardest 25%,
easiest 25% vs rest, 50/50; for science also the 150 easiest GPQA items -> all HLE.

Writes analysis/irt_vs_p/:
  <pool>_items.csv        benchmark, item, b, a, P
  <pool>_models.csv       model, theta, acc
  splits.csv              per pool x scheme x builder: set sizes, overlap between builders,
                          mean model accuracy on each set, easy-vs-hard r / rho across models
  agreement.{png,svg}     rows = pools: b vs P (items), theta vs accuracy (models), P histogram
  overlap.{png,svg}       per pool, one row per scheme x builder, one cell per item (ordered by
                          P(correct), easiest left): blue = in easy set, orange = in hard set,
                          white = neither; IRT and P rows of a scheme differ only at the boundaries
  curves.{png,svg}        per pool: model accuracy on easy (grey) and hard (black) sets,
                          IRT builder solid vs P builder dashed, models by overall accuracy;
                          band = 95% t-CI over items (IRT builder)
Run:  uv run python scripts/analyses/irt_vs_pcorrect.py [--replot] [--warmup 1000] [--samples 1000]
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from inspect_ai.log import read_eval_log_sample_summaries
from scipy import stats

from configs.model import ModelConfig
from configs.paths import hash_task_args, is_successful_eval
from configs.task import TaskConfig
from irt.data import encode
from irt.fitting import fit
from judge.util import extract_score

sys.path.insert(0, str(Path(__file__).parents[1] / "plots"))
from plot_difficulty import offset  # noqa: E402

OUT = Path("analysis/irt_vs_p")
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "svg.fonttype": "none"})

PILOT = {ModelConfig.from_cli(y).name.split("/")[-1].lower()
         for y in Path("configs/experiments/pairs/models.txt").read_text().split()}
MATH = ["aime24", "aime25", "aime26", "apex", "apex_shortlist"]
POOLS = {  # pool -> ([bench yml], hle filter)
    "science": (["gpqa", "hle"], "gold"),
    "math": (MATH, None),
    "math+hle": (MATH + ["hle"], "gold-math"),
}
SCHEMES = ["25 / 25", "25 / rest", "50 / 50"]
SCHEME_COLOR = dict(zip(SCHEMES, ["#2a78d6", "#eb6834", "#1baf7a"]))
BENCH_COLOR = {"gpqa": "#2a78d6", "hle": "#1baf7a", "aime2024": "#f2a09b", "aime2025": "#d9453d",
               "aime2026": "#8f1a14", "apex_2025": "#eda100", "apex_shortlist": "#eb6834"}
BENCH_LABEL = {"gpqa": "GPQA", "hle": "HLE", "aime2024": "AIME24", "aime2025": "AIME25",
               "aime2026": "AIME26", "apex_2025": "APEX", "apex_shortlist": "APEX-sl"}
N_EASY_GPQA = 150


def read_pool(pool: str) -> pd.DataFrame:
    benches, hle_filter = POOLS[pool]
    models = PILOT if "hle" in benches else None
    df = pd.concat([read_bench(y, hle_filter, models) for y in benches], ignore_index=True)
    full = df.groupby("model").bench.nunique() == len(benches)  # same items for every model
    return df[df.model.isin(full[full].index)]


def read_bench(yml: str, hle_filter, models) -> pd.DataFrame:
    task = TaskConfig.from_cli(f"configs/bench/{yml}.yml")
    root = Path("data") / task.name / hash_task_args(task)
    rows = []
    for mdir in sorted(p for p in root.iterdir() if p.is_dir() and (models is None or p.name in models)):
        for f in mdir.rglob("*.eval"):
            if is_successful_eval(f):
                for s in read_eval_log_sample_summaries(str(f)):
                    md = s.metadata or {}
                    rows.append(dict(bench=task.name, model=mdir.name, item=str(s.id), correct=extract_score(s),
                                     uid=md.get("uid"), category=md.get("category")))
    df = pd.DataFrame(rows)
    if yml == "hle":
        from inspect_evals.hle.hle import get_verified_gold_uids
        df = df[df.uid.isin(get_verified_gold_uids())]
        if hle_filter == "gold-math":
            df = df[df.category == "Math"]
    return df.drop(columns=["uid", "category"])


def analyse(pool: str, warmup: int, samples: int, replot: bool):
    items_csv, models_csv = OUT / f"{pool}_items.csv", OUT / f"{pool}_models.csv"
    df = read_pool(pool)
    M = df.groupby(["bench", "item", "model"]).correct.mean().unstack("model")  # item x model
    if replot and items_csv.exists():
        items = pd.read_csv(items_csv, dtype={"item": str}).set_index(["bench", "item"])
        models = pd.read_csv(models_csv).set_index("model")
    else:
        data = encode(df)
        post = fit(data, model="2pl", warmup=warmup, samples=samples)
        items = pd.DataFrame(data.items, columns=["bench", "item"]).assign(
            b=np.asarray(post["b"]).mean(0), a=np.asarray(post["a"]).mean(0)).set_index(["bench", "item"])
        models = pd.DataFrame({"theta": np.asarray(post["theta"]).mean(0)}, index=pd.Index(data.models, name="model"))
    items["P"] = M.mean(axis=1)
    models["acc"] = M.mean(axis=0)
    items.to_csv(items_csv); models.to_csv(models_csv)
    return M, items, models


def split(M, items, builder, scheme):
    r = items.b.rank() if builder == "IRT" else (-items.P).rank(method="first")  # 1 = easiest
    n = len(items)
    if scheme == "GPQA-150 / HLE":
        g = r[items.index.get_level_values(0) == "gpqa"].rank()
        return set(g[g <= N_EASY_GPQA].index), set(items.index[items.index.get_level_values(0) == "hle"])
    q = round(n / 4) if scheme.startswith("25") else round(n / 2)
    easy = set(r[r <= q].index)
    hard = set(r[r > n - q].index) if scheme == "25 / 25" else set(items.index) - easy
    return easy, hard


def set_acc(M, ids):
    sub = M.loc[list(ids)]
    n = sub.notna().sum()
    return sub.mean(), stats.t.ppf(0.975, n - 1) * sub.std() / np.sqrt(n)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--replot", action="store_true")
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--samples", type=int, default=1000)
    args = ap.parse_args()

    results, rows = {}, []
    for pool in POOLS:
        M, items, models = analyse(pool, args.warmup, args.samples, args.replot)
        results[pool] = (M, items, models)
        schemes = SCHEMES + (["GPQA-150 / HLE"] if pool == "science" else [])
        for scheme in schemes:
            sets = {k: split(M, items, k, scheme) for k in ("IRT", "P")}
            for k, (easy, hard) in sets.items():
                ae, ah = M.loc[list(easy)].mean(), M.loc[list(hard)].mean()
                ok = ae.notna() & ah.notna()
                rows.append(dict(pool=pool, scheme=scheme, builder=k, n_easy=len(easy), n_hard=len(hard),
                                 easy_overlap=len(sets["IRT"][0] & sets["P"][0]) / len(easy),
                                 hard_overlap=len(sets["IRT"][1] & sets["P"][1]) / len(hard),
                                 acc_easy=ae.mean(), acc_hard=ah.mean(), n_models=int(ok.sum()),
                                 r=stats.pearsonr(ae[ok], ah[ok])[0], rho=stats.spearmanr(ae[ok], ah[ok])[0]))
        print(f"[{pool}] {len(items)} items, {len(models)} models, "
              f"spearman(b, P) = {stats.spearmanr(items.b, items.P)[0]:.3f}, "
              f"spearman(theta, acc) = {stats.spearmanr(models.theta, models.acc)[0]:.3f}")
    splits = pd.DataFrame(rows)
    splits.to_csv(OUT / "splits.csv", index=False)
    print(splits.round(3).to_string(index=False))

    # --- agreement: rows = pools; b vs P, theta vs acc, P histogram with quartile cuts
    n = len(POOLS)
    fig, axs = plt.subplots(n, 3, figsize=(3.6, 0.95 * n), gridspec_kw=dict(wspace=0.8, hspace=0.7))
    for i, (pool, (M, items, models)) in enumerate(results.items()):
        benches = [b for b in BENCH_COLOR if b in items.index.get_level_values(0)]
        by_bench = [items.xs(b, level=0) for b in benches]
        ax = axs[i, 0]
        for b, sub in zip(benches, by_bench):
            ax.scatter(sub.P, sub.b, s=3, color=BENCH_COLOR[b], alpha=0.6, lw=0, label=BENCH_LABEL[b])
        ax.text(0.97, 0.95, f"ρ = {stats.spearmanr(items.b, items.P)[0]:.2f}\nN = {len(items)}",
                transform=ax.transAxes, ha="right", va="top", fontsize=6, color="grey")
        ax.set_xlim(0, 1); ax.set_xticks([0, 0.5, 1], ["0", ".5", "1"])
        lim = np.ceil(items.b.abs().max()); ax.set_ylim(-lim, lim); ax.set_yticks([-lim, 0, lim])
        ax.set_ylabel("IRT b"); ax.text(0, 1.12, pool, transform=ax.transAxes, fontsize=6, color="grey")
        ax = axs[i, 1]
        ax.scatter(models.acc, models.theta, s=5, color="#333", lw=0)
        ax.text(0.03, 0.95, f"ρ = {stats.spearmanr(models.theta, models.acc)[0]:.2f}\nN = {len(models)}",
                transform=ax.transAxes, ha="left", va="top", fontsize=6, color="grey")
        ax.set_xlim(0, 1); ax.set_xticks([0, 0.5, 1], ["0", ".5", "1"])
        lim = np.ceil(models.theta.abs().max()); ax.set_ylim(-lim, lim); ax.set_yticks([-lim, 0, lim])
        ax.set_ylabel("IRT θ")
        ax = axs[i, 2]
        ax.hist([sub.P for sub in by_bench], bins=np.linspace(0, 1, 21), stacked=True,
                color=[BENCH_COLOR[b] for b in benches], lw=0)
        for q in (0.25, 0.5, 0.75):
            ax.axvline(items.P.quantile(q), color="#333", lw=0.6, ls="--")
        ax.set_xlim(0, 1); ax.set_xticks([0, 0.5, 1], ["0", ".5", "1"]); ax.set_yticks([])
        ax.spines["left"].set_visible(False); ax.set_ylabel("Items")
        for ax in axs[i]:
            offset(ax); ax.tick_params(length=2)
    axs[-1, 0].set_xlabel("P(correct), item"); axs[-1, 1].set_xlabel("Accuracy, model"); axs[-1, 2].set_xlabel("P(correct), item")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, lw=0) for c in BENCH_COLOR.values()]
    fig.legend(handles, list(BENCH_LABEL.values()), loc="lower center", bbox_to_anchor=(0.5, 0.92), ncol=4,
               frameon=False, handlelength=1, columnspacing=1)
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"agreement.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --- overlap: per pool, rows = scheme x builder, cells = items ordered by P(correct)
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["#2a78d6", "white", "#eb6834"])  # easy, neither, hard
    fig, axs = plt.subplots(n, 1, figsize=(3.4, 0.36 * sum(2 * (len(SCHEMES) + (p == "science")) for p in POOLS) / 2.2 + 0.4),
                            gridspec_kw=dict(hspace=0.9, height_ratios=[len(SCHEMES) + (p == "science") for p in POOLS]))
    for ax, (pool, (M, items, models)) in zip(axs, results.items()):
        order = items.P.sort_values(ascending=False).index  # easiest left
        schemes = SCHEMES + (["GPQA-150 / HLE"] if pool == "science" else [])
        grid, labels = [], []
        for scheme in schemes:
            for builder in ("IRT", "P"):
                easy, hard = split(M, items, builder, scheme)
                grid.append([-1 if i in easy else 1 if i in hard else 0 for i in order])
                labels.append(f"{scheme}  {builder}")
        ax.imshow(np.array(grid), aspect="auto", cmap=cmap, vmin=-1, vmax=1, interpolation="nearest")
        ax.set_yticks(range(len(labels)), labels, fontsize=6)
        ax.set_xticks([0, len(order) - 1], [1, len(order)])
        ax.tick_params(length=2)
        ax.text(0, 1.08, pool, transform=ax.transAxes, fontsize=6, color="grey")
        for k in range(1, len(schemes)):
            ax.axhline(2 * k - 0.5, color="#333", lw=0.5)
    axs[-1].set_xlabel("Item, by P(correct) (easiest → hardest)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, lw=0) for c in ("#2a78d6", "#eb6834")]
    fig.legend(handles, ["easy set", "hard set"], loc="lower center", bbox_to_anchor=(0.5, 0.95), ncol=2,
               frameon=False, handlelength=1, columnspacing=1)
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"overlap.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --- curves: per pool, easy/hard accuracy per model (25 / rest), IRT solid vs P dashed
    fig, axs = plt.subplots(1, 3, figsize=(3.6, 1.3), gridspec_kw=dict(wspace=0.35))
    for ax, (pool, (M, items, models)) in zip(axs.flat, results.items()):
        order = models.acc.sort_values(ascending=False).index
        xx = np.arange(len(order))
        offset(ax); ax.tick_params(length=2)
        for builder, ls in (("IRT", "-"), ("P", "--")):
            easy, hard = split(M, items, builder, "25 / rest")
            for ids, color, name in ((easy, "#999", "easy"), (hard, "#222", "hard")):
                m, ci = set_acc(M, ids)
                ax.plot(xx, m[order], lw=1, ls=ls, color=color, label=f"{name}, {builder}")
                if builder == "IRT":
                    ax.fill_between(xx, (m - ci)[order], (m + ci)[order], color=color, alpha=0.15, lw=0)
        ax.set_xlim(-0.5, len(order) - 0.5); ax.set_xticks([0, len(order) - 1], [1, len(order)])
        ax.set_ylim(0, 1); ax.set_yticks([0, 0.5, 1], ["0", ".5", "1"])
        ax.text(0, 1.05, pool, transform=ax.transAxes, fontsize=6, color="grey")
    axs[0].set_ylabel("Accuracy")
    for ax in axs:
        ax.set_xlabel("Model rank")
    axs[1].legend(loc="lower center", bbox_to_anchor=(0.5, 1.15), ncol=4, frameon=False, handlelength=1.5, columnspacing=1)
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"curves.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
