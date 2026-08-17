#!/usr/bin/env python3
"""Nature-style counterparts of the main HiBayes GLM figures."""

import argparse
from pathlib import Path

import arviz as az
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit, logsumexp

PREDICTORS = ["length_difference_sd", "capability_difference_sd", "same_model_family"]
FIGURE_NAMES = (
    "judge_feature_heatmap", "judge_feature_forest", "overall_predictor_importance",
    "judge_effect_interactions", "judge_loo_accuracy", "trace_diagnostics",
)
LABELS = {
    "position_bias_left": "Left position",
    "length_difference_sd": "Length",
    "capability_difference_sd": "Capability",
    "same_model_family": "Same family",
    "condition_mbpp": "MBPP vs GPQA",
}
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7"]
DISPLAY = {
    "deepseek-v4-flash": "DeepSeek", "gemma4-31b": "Gemma", "gpt-oss-120b": "GPT-OSS",
    "ling": "Ling", "llama3.1-8b": "Llama",
}


def load_draws(frame, path):
    raw = np.load(path)
    judges = sorted(frame["judge"].unique())
    conditions = sorted(frame["transcript_condition"].unique())
    item_count = frame["item"].nunique()

    def flat(name):
        values = raw[name]
        return values.reshape((-1,) + values.shape[2:])

    population = flat("predictor_effect_mean")
    effects = population[:, None, :] + flat("predictor_effect_scale")[:, None, :] * flat(
        "judge_predictor_effects_raw"
    )
    position_mean = flat("position_effect_mean")
    intercepts = position_mean[:, None] + flat("position_effect_scale")[:, None] * flat(
        "judge_position_effects_raw"
    )
    conditions_draws = np.concatenate(
        [np.zeros((len(population), len(judges), 1)), flat("judge_condition_effects_free")], axis=2
    )
    items = flat("item_sigma")[:, None] * flat("item_effects_raw")
    if items.shape[1] != item_count:
        raise ValueError(f"posterior has {items.shape[1]} items, data has {item_count}")
    return raw, judges, conditions, population, effects, position_mean, intercepts, conditions_draws, items


def interval_row(judge, feature, values):
    lo, hi = np.quantile(values, [0.025, 0.975])
    return {"judge": judge, "feature": feature, "mean_log_odds": values.mean(),
            "hdi_2.5%": lo, "hdi_97.5%": hi, "p_effect_gt_0": (values > 0).mean()}


def offset_axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_position(("outward", 5))
    ax.spines["bottom"].set_position(("outward", 5))


def save(fig, out, name):
    for suffix, options in (("png", {"dpi": 300}), ("svg", {})):
        target = out / f"{name}.{suffix}"
        if target.exists():
            continue
        tmp = out / f"{name}.tmp.{suffix}"
        fig.savefig(tmp, **options)
        tmp.replace(target)
    plt.close(fig)


def atomic_csv(frame, path):
    if path.exists():
        return
    tmp = path.with_suffix(".tmp.csv")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def loo_accuracy(frame, judges, effects, intercepts, condition_effects, item_effects, batch=256):
    judge_index = pd.Categorical(frame["judge"], categories=judges).codes
    condition_names = sorted(frame["transcript_condition"].unique())
    condition_index = pd.Categorical(frame["transcript_condition"], categories=condition_names).codes
    item_index = pd.factorize(frame["item"], sort=True)[0]
    design = frame[PREDICTORS].to_numpy(float)
    observed = frame["chosen_left"].to_numpy(bool)
    loo_probability, pareto_k = np.empty(len(frame)), np.empty(len(frame))
    for start in range(0, len(frame), batch):
        stop = min(start + batch, len(frame))
        j, c, item = judge_index[start:stop], condition_index[start:stop], item_index[start:stop]
        logits = intercepts[:, j]
        logits += np.einsum("dbp,bp->db", effects[:, j, :], design[start:stop])
        logits += condition_effects[:, j, c] + item_effects[:, item]
        probability_one = expit(logits).T
        log_likelihood = np.where(
            observed[start:stop, None], -np.logaddexp(0, -logits.T), -np.logaddexp(0, logits.T)
        )
        log_weights, pareto_k[start:stop] = az.psislw(-log_likelihood)
        weights = np.exp(log_weights - logsumexp(log_weights, axis=1, keepdims=True))
        loo_probability[start:stop] = np.sum(weights * probability_one, axis=1)
    predicted = loo_probability >= 0.5
    return pd.DataFrame([
        {"judge": judge, "n": int((judge_index == index).sum()),
         "loo_accuracy": float((predicted[judge_index == index] == observed[judge_index == index]).mean()),
         "mean_loo_probability": float(loo_probability[judge_index == index].mean()),
         "pareto_k_over_0.7": int((pareto_k[judge_index == index] > 0.7).sum())}
        for index, judge in enumerate(judges)
    ])


def make_plots(data_path, posterior_path, out, loo_batch=256):
    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "svg.fonttype": "none",
    })
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(data_path)
    (raw, judges, conditions, population, effects, position_mean, intercepts,
     condition_effects, item_effects) = load_draws(frame, posterior_path)
    if conditions != ["gpqa", "mbpp"]:
        raise ValueError(f"expected GPQA/MBPP conditions, got {conditions}")

    effect_rows = []
    for index, judge in enumerate(judges):
        effect_rows.append(interval_row(judge, "position_bias_left", intercepts[:, index]))
        effect_rows += [interval_row(judge, name, effects[:, index, p]) for p, name in enumerate(PREDICTORS)]
        effect_rows.append(interval_row(judge, "condition_mbpp", condition_effects[:, index, 1]))
    effect_table = pd.DataFrame(effect_rows)
    atomic_csv(effect_table, out / "judge_feature_effects.csv")
    order = ["position_bias_left", *PREDICTORS, "condition_mbpp"]

    matrix = effect_table.pivot(index="judge", columns="feature", values="mean_log_odds").loc[judges, order]
    fig, ax = plt.subplots(figsize=(3.35, 2.45))
    limit = max(0.1, np.abs(matrix.to_numpy()).max())
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(len(order)), [LABELS[name] for name in order], rotation=90)
    ax.set_yticks(range(len(judges)), [DISPLAY.get(name, name) for name in judges])
    for (row, col), value in np.ndenumerate(matrix.to_numpy()):
        ax.text(col, row, f"{value:+.2f}", ha="center", va="center", fontsize=6,
                color="white" if abs(value) > limit * 0.55 else "black")
    fig.colorbar(image, ax=ax, label="Log-odds", fraction=0.045, pad=0.03)
    fig.subplots_adjust(left=0.26, right=0.82, bottom=0.34, top=0.98)
    save(fig, out, "judge_feature_heatmap")

    fig, axes = plt.subplots(len(judges), 1, figsize=(3.35, 3.55), sharex=True)
    xmin, xmax = effect_table["hdi_2.5%"].min(), effect_table["hdi_97.5%"].max()
    margin = 0.08 * (xmax - xmin)
    for ax, judge, color in zip(axes, judges, COLORS):
        block = effect_table[effect_table["judge"] == judge].set_index("feature").loc[order]
        y = np.arange(len(order))
        lower = np.maximum(0, block["mean_log_odds"].to_numpy() - block["hdi_2.5%"].to_numpy())
        upper = np.maximum(0, block["hdi_97.5%"].to_numpy() - block["mean_log_odds"].to_numpy())
        ax.errorbar(block["mean_log_odds"], y,
                    xerr=[lower, upper],
                    fmt="o", ms=3, lw=1, color=color, capsize=0)
        ax.axvline(0, color="#8a8a86", lw=0.8, ls="--")
        ax.set_yticks(y, [LABELS[name] for name in order])
        ax.set_xlim(xmin - margin, xmax + margin)
        ax.text(0.01, 0.86, DISPLAY.get(judge, judge), transform=ax.transAxes,
                color="#66645f", fontsize=7)
        offset_axes(ax)
    axes[-1].set_xlabel("Choice effect (log-odds)")
    fig.subplots_adjust(left=0.29, right=0.97, bottom=0.12, top=0.99, hspace=0.35)
    save(fig, out, "judge_feature_forest")

    pop_rows = [interval_row("population", "position_bias_left", position_mean)]
    pop_rows += [interval_row("population", name, population[:, p]) for p, name in enumerate(PREDICTORS)]
    pop = pd.DataFrame(pop_rows).sort_values("mean_log_odds").reset_index(drop=True)
    atomic_csv(pop, out / "population_effects.csv")
    fig, ax = plt.subplots(figsize=(3.25, 2.45))
    y = np.arange(len(pop))
    for j, (judge, color) in enumerate(zip(judges, COLORS)):
        judge_values = {"position_bias_left": intercepts[:, j].mean(),
                        **{name: effects[:, j, p].mean() for p, name in enumerate(PREDICTORS)}}
        ax.scatter([judge_values[name] for name in pop["feature"]], y + np.linspace(-0.14, 0.14, len(judges))[j],
                   s=12, color=color, label=DISPLAY.get(judge, judge), zorder=3)
    ax.errorbar(pop["mean_log_odds"], y,
                xerr=[np.maximum(0, pop["mean_log_odds"] - pop["hdi_2.5%"]).to_numpy(),
                      np.maximum(0, pop["hdi_97.5%"] - pop["mean_log_odds"]).to_numpy()],
                fmt="D", ms=3, color="#111111", lw=1, capsize=0, label="Population", zorder=4)
    ax.axvline(0, color="#8a8a86", lw=0.8, ls="--")
    ax.set_yticks(y, [LABELS[name] for name in pop["feature"]])
    ax.set_xlabel("Choice effect (log-odds)")
    offset_axes(ax)
    handles, legend_labels = ax.get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", bbox_to_anchor=(0.5, 0.01),
               ncol=3, frameon=False, fontsize=6, handletextpad=0.3, columnspacing=0.8)
    ax.legend().remove()
    fig.subplots_adjust(left=0.29, right=0.97, bottom=0.31, top=0.98)
    save(fig, out, "overall_predictor_importance")

    interaction_names = ["position_bias_left", *PREDICTORS]
    interaction_values = np.column_stack(
        [(intercepts - position_mean[:, None]).mean(axis=0)]
        + [(effects[:, :, p] - population[:, None, p]).mean(axis=0) for p in range(len(PREDICTORS))]
    )
    fig, ax = plt.subplots(figsize=(3.15, 2.1))
    limit = max(0.1, np.abs(interaction_values).max())
    image = ax.imshow(interaction_values, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(len(interaction_names)), [LABELS[name] for name in interaction_names], rotation=90)
    ax.set_yticks(range(len(judges)), [DISPLAY.get(name, name) for name in judges])
    for (row, col), value in np.ndenumerate(interaction_values):
        ax.text(col, row, f"{value:+.2f}", ha="center", va="center", fontsize=6,
                color="white" if abs(value) > limit * 0.55 else "black")
    fig.colorbar(image, ax=ax, label="Deviation (log-odds)", fraction=0.05, pad=0.03)
    fig.subplots_adjust(left=0.28, right=0.81, bottom=0.36, top=0.98)
    save(fig, out, "judge_effect_interactions")

    accuracy = loo_accuracy(frame, judges, effects, intercepts, condition_effects, item_effects, loo_batch)
    atomic_csv(accuracy, out / "judge_loo_accuracy.csv")
    fig, ax = plt.subplots(figsize=(3.0, 1.85))
    ax.bar(np.arange(len(judges)), accuracy["loo_accuracy"], color="#8a8a86", width=0.68)
    ax.axhline(0.5, color="#555555", lw=0.8, ls="--", label="Chance")
    ax.set_xticks(range(len(judges)), [DISPLAY.get(name, name) for name in judges], rotation=90)
    for label in ax.get_xticklabels():
        label.set_rotation(90)
        label.set_ha("center")
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.5, 1])
    ax.set_ylabel("PSIS-LOO accuracy")
    offset_axes(ax)
    handles, legend_labels = ax.get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", bbox_to_anchor=(0.5, 0.02), frameon=False)
    ax.legend().remove()
    fig.subplots_adjust(left=0.2, right=0.97, bottom=0.48, top=0.98)
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center")
    save(fig, out, "judge_loo_accuracy")

    trace_names = ["position_effect_mean", "length_difference_sd", "capability_difference_sd", "same_model_family"]
    trace_values = [raw["position_effect_mean"], *[raw["predictor_effect_mean"][:, :, p] for p in range(3)]]
    fig, axes = plt.subplots(2, 2, figsize=(3.35, 2.9), sharex=True)
    for ax, name, values in zip(axes.flat, trace_names, trace_values):
        for chain in range(values.shape[0]):
            ax.plot(np.arange(1, values.shape[1] + 1), values[chain], lw=0.45,
                    color=COLORS[chain], alpha=0.75, label=f"Chain {chain + 1}")
        ax.text(0.02, 0.9, LABELS.get(name, name.replace("_", " ").title()),
                transform=ax.transAxes, color="#66645f", fontsize=7)
        ax.set_xlim(1, values.shape[1])
        ax.set_xticks([1, values.shape[1]])
        offset_axes(ax)
    axes[1, 0].set_xlabel("Posterior draw")
    axes[1, 1].set_xlabel("Posterior draw")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01),
               ncol=4, frameon=False, fontsize=6)
    fig.subplots_adjust(left=0.14, right=0.94, bottom=0.25, top=0.98, wspace=0.36, hspace=0.38)
    save(fig, out, "trace_diagnostics")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--posterior", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--loo-batch", type=int, default=256)
    args = parser.parse_args()
    make_plots(args.data, args.posterior, args.out, args.loo_batch)


if __name__ == "__main__":
    main()
