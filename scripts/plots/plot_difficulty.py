#!/usr/bin/env python3
"""Question difficulty and per-model accuracy on GPQA / HLE for the pilot models.

Reads every successful full-run eval log of the 26 models in
configs/experiments/pairs/models.txt, averages correctness over repeats, and writes to
analysis/difficulty/:
  <bench>_matrix.csv       question x model mean correctness (NaN = not collected)
  <bench>_difficulty.png   heatmap (questions hardest->easiest, models weakest->strongest)
                           with marginals: mean over models per question (top), mean
                           over questions per model +-95% t-CI, N=questions (right)
  hle_vs_gpqa.png          per-model GPQA vs HLE accuracy with Pearson r / Spearman rho
Models with no logs yet are skipped and listed on stdout; re-run after downloads finish.

Run:  uv run python scripts/plots/plot_difficulty.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from inspect_ai.log import read_eval_log_sample_summaries
from scipy import stats

from configs.model import ModelConfig
from configs.paths import get_eval_path, is_successful_eval
from configs.task import TaskConfig
from judge.util import extract_score

BENCHES = ["gpqa", "hle"]
MODELS = Path("configs/experiments/pairs/models.txt").read_text().split()
OUT = Path("analysis/difficulty")
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "svg.fonttype": "none"})


def offset(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_position(("outward", 4))


def matrix(bench: str) -> pd.DataFrame:
    task = TaskConfig.from_cli(f"configs/bench/{bench}.yml")
    cols = {}
    for yml in MODELS:
        label = Path(yml).stem
        path = get_eval_path(task, ModelConfig.from_cli(yml))
        logs = [f for f in path.rglob("*.eval") if is_successful_eval(f)]
        if not logs:
            print(f"[{bench}] missing: {label}")
            continue
        runs = [pd.Series({s.id: float(extract_score(s)) for s in read_eval_log_sample_summaries(str(f))})
                for f in logs]
        cols[label] = pd.concat(runs, axis=1).mean(axis=1)
    df = pd.DataFrame(cols)
    df.to_csv(OUT / f"{bench}_matrix.csv")
    return df


def plot_bench(bench: str, df: pd.DataFrame):
    q = df.mean(axis=1).sort_values()                # hardest -> easiest
    m = df.mean(axis=0).sort_values()                # weakest -> strongest
    n = df.notna().sum(axis=0)[m.index]
    ci = stats.t.ppf(0.975, n - 1) * df.std(axis=0)[m.index] / np.sqrt(n)
    grid = df.loc[q.index, m.index].T.values

    fig, axs = plt.subplots(2, 2, figsize=(3.4, 4.2), gridspec_kw=dict(
        width_ratios=[4, 1.2], height_ratios=[1, 5], wspace=0.08, hspace=0.08))
    axs[0, 1].axis("off")

    ax = axs[1, 0]
    ax.imshow(grid, aspect="auto", cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
    ax.set_yticks(range(len(m)), m.index)
    ax.set_xlabel("Question (hardest → easiest)")
    ax.set_xticks([0, len(q) - 1], [1, len(q)])
    ax.tick_params(length=2)

    ax = axs[0, 0]
    ax.fill_between(range(len(q)), q.values, color="#2a78d6", alpha=0.6, lw=0)
    ax.set_xlim(-0.5, len(q) - 0.5)
    ax.set_ylim(0, 1); ax.set_yticks([0, 1]); ax.set_xticks([])
    ax.set_ylabel("P(correct)")
    ax.spines["bottom"].set_visible(False)
    offset(ax)

    ax = axs[1, 1]
    ax.errorbar(m.values, range(len(m)), xerr=ci.values, fmt="o", ms=3, lw=1, capsize=0,
                color="#2a78d6")
    ax.set_ylim(len(m) - 0.5, -0.5); ax.set_yticks([])
    ax.set_xlim(0, 1); ax.set_xticks([0, 0.5, 1], ["0", ".5", "1"])
    ax.set_xlabel("Accuracy")
    ax.spines["left"].set_visible(False)
    offset(ax)
    ax.text(0, -1.2, bench.upper(), color="grey", fontsize=7, ha="left")

    for ext in ("png", "svg"):
        fig.savefig(OUT / f"{bench}_difficulty.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_corr(acc: pd.DataFrame):
    acc = acc.dropna().sort_values("gpqa", ascending=False)
    r, rho = stats.pearsonr(acc.gpqa, acc.hle)[0], stats.spearmanr(acc.gpqa, acc.hle)[0]
    fig, ax = plt.subplots(figsize=(2.2, 2.2))
    ax.scatter(acc.gpqa, acc.hle, s=12, color="#2a78d6", zorder=3)
    for i, (name, row) in enumerate(acc.iterrows(), 1):
        ax.annotate(i, (row.gpqa, row.hle), xytext=(2, 1), textcoords="offset points",
                    fontsize=5, color="#444")
        ax.plot([], [], " ", label=f"{i} {name}")   # numbered key below the axes
    ax.legend(loc="upper center", bbox_to_anchor=(0.4, -0.35), ncol=2, frameon=False,
              fontsize=5, handlelength=0, handletextpad=0, columnspacing=1)
    ax.text(0.02, 0.97, f"r = {r:.2f}\nρ = {rho:.2f}\nN = {len(acc)}", transform=ax.transAxes,
            va="top", fontsize=7, color="grey")
    ax.set_xlabel("GPQA accuracy"); ax.set_ylabel("HLE accuracy")
    ax.set_xlim(0, 1); ax.set_xticks([0, 0.5, 1], ["0", ".5", "1"])
    top = np.ceil(acc.hle.max() * 10) / 10
    ax.set_ylim(0, top); ax.set_yticks([0, top])
    offset(ax)
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"hle_vs_gpqa.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"GPQA vs HLE: pearson r={r:.3f}, spearman rho={rho:.3f}, N={len(acc)}")


if __name__ == "__main__":
    acc = {}
    for bench in BENCHES:
        df = matrix(bench)
        plot_bench(bench, df)
        acc[bench] = df.mean(axis=0)
        print(f"[{bench}] {df.shape[1]} models, {df.shape[0]} questions")
    acc = pd.DataFrame(acc)
    acc.round(3).sort_values("gpqa", ascending=False).to_csv(OUT / "model_accuracy.csv")
    print(acc.round(3).sort_values("gpqa", ascending=False).to_string())
    plot_corr(acc)
