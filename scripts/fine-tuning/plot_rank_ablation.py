"""Rank-ablation summary: accuracy vs adapter rank, per size. Two panels
(0.6B, 8B); lines = diagonal (own/LOO-target) vs transfer cells, solid = ECI
labels, dashed = in-domain. 95% t-CI across cells (the replication unit).

  python plot_rank_ablation.py
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

RANKS = {16: "cross", 4: "cross_r4", 2: "cross_r2", 1: "cross_r1"}
COLORS = {"diag": "#2a78d6", "transfer": "#eb6834"}
STYLES = {"eci": "-", "indomain": "--"}
NICE = {"Qwen3-0.6B": "Qwen3-0.6B", "Qwen3-8B": "Qwen3-8B",
        "Llama-3.2-3B-Instruct": "Llama-3.2-3B"}


def load(results_dir, size):
    dfs = []
    for rank, root in RANKS.items():
        d = pd.read_csv(Path(results_dir) / root / f"accuracy_long_{size}.csv")
        d["rank"] = rank
        dfs.append(d)
    df = pd.concat(dfs)
    df = df[df.train != "base"]
    df["kind"] = df.apply(
        lambda r: "diag" if r["train"] == r["test"] or r["train"] == f"loo_{r['test']}"
        else "transfer", axis=1)
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default="judge_results/judge_ft")
    ap.add_argument("--out-dir", default="judge_results/judge_ft/cross_ranks")
    args = ap.parse_args()

    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                         "xtick.labelsize": 7, "ytick.labelsize": 7})
    sizes = ["Qwen3-0.6B", "Llama-3.2-3B-Instruct", "Qwen3-8B"]
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.1), sharey=True)
    for ax, size in zip(axes, sizes):
        df = load(args.results_dir, size)
        ax.axhline(0.5, color="#999", lw=0.8, ls="--")
        for kind in ("diag", "transfer"):
            for label in ("eci", "indomain"):
                sub = df[(df.kind == kind) & (df.label == label)]
                g = sub.groupby("rank").acc
                m, sd, n = g.mean(), g.std(), g.size()
                ci = stats.t.ppf(0.975, n - 1) * sd / n.pow(0.5)
                x = m.index
                ax.errorbar(x, m, yerr=ci, color=COLORS[kind], ls=STYLES[label],
                            lw=1, marker="o", ms=3, capsize=0,
                            label=f"{'own/LOO' if kind == 'diag' else 'transfer'}, {label}")
        ax.set_xscale("log", base=2)
        ax.set_xticks(list(RANKS), [str(r) for r in RANKS])
        ax.minorticks_off()
        ax.set_xlabel("adapter rank")
        ax.text(0.02, 0.95, NICE[size], transform=ax.transAxes, fontsize=7,
                color="#777", va="top")  # panel tag
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("verdict accuracy")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, frameon=False, loc="lower center",
               bbox_to_anchor=(0.5, 0.98))
    fig.tight_layout()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(out / f"rank_ablation.{ext}", dpi=300, bbox_inches="tight")
    print(f"wrote {out}/rank_ablation.png + .svg")

    # paired deltas vs rank 16, same cells: the actual test of "low rank matches"
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.1), sharey=True)
    for ax, size in zip(axes, sizes):
        df = load(args.results_dir, size)
        wide = df.pivot_table(index=["label", "train", "test", "kind"],
                              columns="rank", values="acc")
        ax.axhline(0, color="#999", lw=0.8, ls="--")
        for kind in ("diag", "transfer"):
            for label in ("eci", "indomain"):
                sub = wide.xs((label, kind), level=("label", "kind"))
                for r in (1, 2, 4):
                    d = sub[r] - sub[16]
                    m, ci = d.mean(), stats.t.ppf(0.975, len(d) - 1) * d.sem()
                    ax.errorbar([r], [m], yerr=[ci], color=COLORS[kind],
                                ls=STYLES[label], lw=1, marker="o", ms=3, capsize=0)
        for kind in ("diag", "transfer"):          # connect means for readability
            for label in ("eci", "indomain"):
                sub = wide.xs((label, kind), level=("label", "kind"))
                ms_ = [(sub[r] - sub[16]).mean() for r in (1, 2, 4)]
                ax.plot([1, 2, 4], ms_, color=COLORS[kind], ls=STYLES[label], lw=1,
                        label=f"{'own/LOO' if kind == 'diag' else 'transfer'}, {label}")
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4], ["1", "2", "4"])
        ax.minorticks_off()
        ax.set_xlabel("adapter rank")
        ax.set_ylim(-0.08, 0.08)
        ax.text(0.02, 0.95, NICE[size], transform=ax.transAxes, fontsize=7,
                color="#777", va="top")  # panel tag
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("Δ accuracy vs rank 16")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, frameon=False, loc="lower center",
               bbox_to_anchor=(0.5, 0.98))
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(out / f"rank_ablation_delta.{ext}", dpi=300, bbox_inches="tight")
    print(f"wrote {out}/rank_ablation_delta.png + .svg")


if __name__ == "__main__":
    main()
