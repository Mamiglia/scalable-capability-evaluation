"""Do the three surprisal scorers (llama, gemma, qwen) agree, and which should the GLMs use?

    uv run python scripts/verdict_dissection/scorer_agreement.py

Reads the transcript design matrix (surprisal columns already log1p'd and z-scaled per
benchmark). Writes, under judge_results/verdict_dissection/scorer_agreement/:
- pairwise.csv: per quantity, benchmark (and pooled), scorer pair: Spearman and Pearson
- target.csv: per quantity, benchmark, scorer (and scorer mean): Spearman between the
  model-level feature mean and the model's benchmark accuracy (canonical rollouts, AIME pooled)
- summary.csv: one row per quantity with the worst pairwise agreement and target correlations
- figures for the GLM quantities and an overview of all quantities
"""
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

ROOT = Path("/Volumes/T7/judge-dissection")
OUT = Path("judge_results/verdict_dissection/scorer_agreement")
SCORERS = ["llama", "gemma", "qwen"]
PAIRS = list(itertools.combinations(SCORERS, 2))
BENCHES = ["aime", "gpqa", "intercode", "swe_bench_mini"]


def family(quantity):
    """Plain-word family of a quantity, from its name."""
    if "tail" in quantity:
        return "rolling tail gain"
    if "gain" in quantity:
        return "rolling gain" if "rolling" in quantity else "gain"
    if quantity.endswith(("_count", "_chars", "_available")):
        return "count"
    return "rolling level" if "rolling" in quantity else "level"


def load():
    transcript = pd.read_parquet(ROOT / "design-matrices/transcript.parquet")
    columns = [c for c in transcript.columns if c.startswith("surprisal_")]
    quantities = sorted({c.rsplit("__", 1)[0] for c in columns})
    return transcript[["bench", "model", "split"] + columns], quantities


def model_accuracy():
    identity = pd.read_parquet(ROOT / "transcript-features/raw/identity.parquet",
                               columns=["bench", "model", "correct", "is_canonical"])
    identity = identity[identity.is_canonical]
    identity["bench"] = identity.bench.where(~identity.bench.str.startswith("aime"), "aime")
    return identity.groupby(["bench", "model"]).correct.mean().rename("accuracy")


def pairwise(transcript, quantities):
    rows = []
    for bench, frame in [("pooled", transcript)] + [(b, transcript[transcript.bench == b]) for b in BENCHES]:
        for quantity in quantities:
            values = frame[[f"{quantity}__{s}" for s in SCORERS]].to_numpy(np.float64)
            ranks = pd.DataFrame(values).rank().to_numpy()
            for i, j in itertools.combinations(range(3), 2):
                rows.append({"quantity": quantity, "bench": bench, "pair": f"{SCORERS[i]}-{SCORERS[j]}",
                             "spearman": np.corrcoef(ranks[:, i], ranks[:, j])[0, 1],
                             "pearson": np.corrcoef(values[:, i], values[:, j])[0, 1]})
    return pd.DataFrame(rows)


def target(transcript, quantities, accuracy):
    """Model-level Spearman between each scorer's feature (and the scorer mean) and accuracy."""
    columns = [f"{q}__{s}" for q in quantities for s in SCORERS]
    by_model = transcript.groupby(["bench", "model"])[columns].mean()
    means = {f"{q}__mean": by_model[[f"{q}__{s}" for s in SCORERS]].mean(axis=1) for q in quantities}
    by_model = pd.concat([by_model, pd.DataFrame(means)], axis=1)   # z-scored per benchmark already, so a plain mean
    by_model = by_model.join(accuracy, how="inner")
    rows = []
    for bench, frame in by_model.groupby(level="bench"):
        for quantity in quantities:
            for scorer in SCORERS + ["mean"]:
                rho = spearmanr(frame[f"{quantity}__{scorer}"], frame.accuracy).statistic
                rows.append({"quantity": quantity, "bench": bench, "scorer": scorer,
                             "spearman_with_accuracy": rho, "n_models": len(frame)})
    return pd.DataFrame(rows)


def summarise(pair_table, target_table):
    worst = (pair_table[pair_table.bench != "pooled"].groupby("quantity").spearman.min()
             .rename("worst_pair_spearman_any_bench"))
    pooled = (pair_table[pair_table.bench == "pooled"].groupby("quantity").spearman
              .agg(pooled_min_pair_spearman="min", pooled_mean_pair_spearman="mean"))
    per_bench = (pair_table[pair_table.bench != "pooled"].groupby(["quantity", "bench"]).spearman.min()
                 .unstack().add_prefix("min_pair_spearman_"))
    rho = target_table.pivot_table(index="quantity", columns=["scorer"], values="spearman_with_accuracy",
                                   aggfunc=lambda r: np.mean(np.abs(r)))
    rho = rho[SCORERS + ["mean"]].add_prefix("mean_abs_accuracy_spearman_")
    spread = (target_table[target_table.scorer != "mean"].groupby(["quantity", "bench"])
              .spearman_with_accuracy.agg(lambda r: r.max() - r.min()).groupby("quantity").max()
              .rename("max_accuracy_spearman_spread_between_scorers"))
    summary = pd.concat([pooled, worst, per_bench, rho, spread], axis=1).reset_index()
    summary.insert(1, "family", summary.quantity.map(family))
    return summary.sort_values("worst_pair_spearman_any_bench")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    transcript, quantities = load()
    pair_table = pairwise(transcript, quantities)
    target_table = target(transcript, quantities, model_accuracy())
    summary = summarise(pair_table, target_table)
    pair_table.to_csv(OUT / "pairwise.csv", index=False)
    target_table.to_csv(OUT / "target.csv", index=False)
    summary.to_csv(OUT / "summary.csv", index=False)
    print(summary.round(2).to_string(index=False))


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------- figures
GLM_QUANTITIES = [f"surprisal_{q}" for q in [
    "local_mean", "full_bpc", "question_gain_bpc", "total_context_gain_bpc", "long_context_gain_bpc",
    "rolling_short_position_adjusted_mean", "rolling_long_position_adjusted_mean",
    "rolling_long_question_position_adjusted_mean", "rolling_context_gain_position_adjusted_mean",
    "rolling_question_gain_position_adjusted_mean"]]
BENCH_LABEL = {"aime": "AIME", "gpqa": "GPQA", "intercode": "InterCode", "swe_bench_mini": "SWE-bench mini"}
PAIR_COLOUR = {"llama-gemma": "#2a78d6", "llama-qwen": "#eb6834", "gemma-qwen": "#1baf7a"}
SCORER_COLOUR = {"llama": "#2a78d6", "gemma": "#eb6834", "qwen": "#1baf7a", "mean": "black"}
BENCH_COLOUR = {"aime": "#2a78d6", "gpqa": "#eb6834", "intercode": "#1baf7a", "swe_bench_mini": "#eda100"}


def style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
                         "font.family": "sans-serif", "font.sans-serif": ["Arial"], "svg.fonttype": "none"})
    return plt


def trim(ax, xlo, xhi, ylo, yhi):
    for side in ("left", "bottom"):
        ax.spines[side].set_position(("outward", 3))
    ax.spines["bottom"].set_bounds(xlo, xhi)
    ax.spines["left"].set_bounds(ylo, yhi)


def short(quantity):
    return quantity.replace("surprisal_", "").replace("_position_adjusted_mean", " (pos. adj.)").replace("_", " ")


def figure_glm_quantities(pair_table, target_table):
    """Rows: GLM quantities. Top: pairwise Spearman per scorer pair. Bottom: model-level Spearman
    with benchmark accuracy per scorer and for the scorer mean. One column per benchmark."""
    plt = style()
    fig, axes = plt.subplots(2, 4, figsize=(6.6, 4.4), sharey=True)
    y = np.arange(len(GLM_QUANTITIES))[::-1]
    for column, bench in enumerate(BENCHES):
        ax = axes[0, column]
        for pair, colour in PAIR_COLOUR.items():
            sub = pair_table[(pair_table.bench == bench) & (pair_table.pair == pair)].set_index("quantity")
            ax.plot(sub.loc[GLM_QUANTITIES, "spearman"], y, "o", ms=3, color=colour, clip_on=False, label=pair)
        ax.set_xlim(0.55, 1.02); ax.set_xticks([0.6, 0.8, 1.0]); trim(ax, 0.6, 1.0, y.min(), y.max())
        ax.set_title(BENCH_LABEL[bench], loc="left", fontsize=7, color="grey")
        ax = axes[1, column]
        ax.axvline(0, lw=0.8, ls="--", color="grey")
        for scorer, colour in SCORER_COLOUR.items():
            sub = target_table[(target_table.bench == bench) & (target_table.scorer == scorer)].set_index("quantity")
            marker = "s" if scorer == "mean" else "o"
            ax.plot(sub.loc[GLM_QUANTITIES, "spearman_with_accuracy"], y, marker, ms=3, color=colour,
                    mfc="none" if scorer == "mean" else colour, clip_on=False, label=scorer)
        ax.set_xlim(-0.75, 0.75); ax.set_xticks([-0.6, 0, 0.6]); trim(ax, -0.6, 0.6, y.min(), y.max())
    axes[0, 0].set_yticks(y); axes[0, 0].set_yticklabels([short(q) for q in GLM_QUANTITIES])
    axes[0, 0].set_ylim(y.min() - 0.5, y.max() + 0.5)
    fig.subplots_adjust(left=0.31, right=0.99, top=0.9, bottom=0.08, wspace=0.25, hspace=0.75)
    fig.text(0.65, 0.515, "Spearman between scorers", ha="center", va="top")
    fig.text(0.65, 0.0, "Spearman with accuracy", ha="center", va="bottom")
    axes[0, 0].legend(loc="lower left", bbox_to_anchor=(0, 1.08), ncol=3, frameon=False, handletextpad=0.2)
    axes[1, 0].legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=4, frameon=False, handletextpad=0.2)
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"glm_quantities.{ext}", dpi=300, bbox_inches="tight")


def figure_all_quantities(pair_table):
    """Every quantity: worst pairwise Spearman on each benchmark, grouped by family."""
    plt = style()
    worst = pair_table[pair_table.bench != "pooled"].groupby(["quantity", "bench"]).spearman.min().reset_index()
    worst["family"] = worst.quantity.map(family)
    families = ["count", "level", "rolling level", "gain", "rolling gain", "rolling tail gain"]
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    rng = np.random.default_rng(0)
    for i, fam in enumerate(families):
        for j, bench in enumerate(BENCHES):
            sub = worst[(worst.family == fam) & (worst.bench == bench)]
            jitter = (j - 1.5) * 0.17 + rng.uniform(-0.05, 0.05, len(sub))
            ax.plot(i + jitter, sub.spearman, "o", ms=2.5, color=BENCH_COLOUR[bench], alpha=0.8, clip_on=False,
                    label=BENCH_LABEL[bench] if i == 0 else None)
    ax.axhline(0.8, lw=0.8, ls="--", color="grey")
    ax.set_xticks(range(len(families))); ax.set_xticklabels(families); ax.tick_params(axis="x", rotation=90)
    ax.set_yticks([-1, -0.5, 0, 0.5, 1]); ax.set_ylim(-1.05, 1.05); trim(ax, 0, len(families) - 1, -1, 1)
    ax.set_ylabel("worst pair Spearman")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=4, frameon=False, handletextpad=0.2, columnspacing=0.8)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"all_quantities.{ext}", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    pair_table = pd.read_csv(OUT / "pairwise.csv")
    target_table = pd.read_csv(OUT / "target.csv")
    figure_glm_quantities(pair_table, target_table)
    figure_all_quantities(pair_table)
