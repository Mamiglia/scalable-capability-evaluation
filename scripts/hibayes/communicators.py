"""Compact tables and plots for the GPQA judge-feature model."""

from collections.abc import Sequence
from typing import Tuple

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import logsumexp

from hibayes.analysis_state import AnalysisState
from hibayes.communicate import CommunicateResult, Communicator, communicate
from hibayes.ui import ModellingDisplay


PLOT_LABELS = {
    "capability_difference_sd": "Capability difference (1 SD)",
    "length_difference_sd": "Length difference (1 SD)",
    "position_bias_left": "Left-position bias",
    "same_model_family": "Same model family",
}


def _effect_table(model, predictors: Sequence[str]) -> pd.DataFrame:
    posterior = model.inference_data.posterior
    judges = model.coords["judge"]
    conditions = model.coords["transcript_condition"]
    draws = posterior["judge_predictor_effects"].values.reshape(-1, len(judges), len(predictors))
    condition_draws = posterior["judge_condition_effects"].values.reshape(-1, len(judges), len(conditions))
    rows = []
    for judge_index, judge in enumerate(judges):
        intercept_draws = posterior["judge_intercepts"].values.reshape(-1, len(judges))
        values = [("position_bias_left", intercept_draws[:, judge_index])]
        values += [(name, draws[:, judge_index, i]) for i, name in enumerate(predictors)]
        values += [
            (f"transcript_condition={condition} vs {conditions[0]}", condition_draws[:, judge_index, i])
            for i, condition in enumerate(conditions)
            if condition != conditions[0]
        ]
        judge_rows = np.asarray(model.features["judge_index"]) == judge_index
        for feature, samples in values:
            lo, hi = np.quantile(samples, [0.025, 0.975])
            raw_feature = feature.split("=", 1)[0] if feature.startswith("transcript_condition=") else feature
            identified = True
            if raw_feature in model.features:
                identified = np.unique(np.asarray(model.features[raw_feature])[judge_rows]).size > 1
            rows.append(
                {
                    "judge": judge,
                    "feature": feature,
                    "mean_log_odds": samples.mean(),
                    "hdi_2.5%": lo,
                    "hdi_97.5%": hi,
                    "p_effect_gt_0": (samples > 0).mean(),
                    "credible_nonzero": bool(lo > 0 or hi < 0),
                    "identified": identified,
                }
            )
    return pd.DataFrame(rows)


def _loo_accuracy(model) -> pd.DataFrame:
    log_likelihood = model.inference_data.log_likelihood["obs"].values
    samples = log_likelihood.reshape(-1, log_likelihood.shape[-1]).T
    observed = np.asarray(model.features["obs"]).astype(bool)
    loo_probability = np.empty(len(observed))
    pareto_k = np.empty(len(observed))
    for start in range(0, len(observed), 256):
        stop = min(start + 256, len(observed))
        batch = samples[start:stop]
        log_weights, pareto_k[start:stop] = az.psislw(-batch)
        weights = np.exp(log_weights - logsumexp(log_weights, axis=1, keepdims=True))
        probability_observed = np.exp(batch)
        probability_one = np.where(
            observed[start:stop, None], probability_observed, 1 - probability_observed
        )
        loo_probability[start:stop] = np.sum(weights * probability_one, axis=1)
    predicted = loo_probability >= 0.5

    rows = []
    for index, judge in enumerate(model.coords["judge"]):
        mask = np.asarray(model.features["judge_index"]) == index
        rows.append(
            {
                "judge": judge,
                "n": int(mask.sum()),
                "loo_accuracy": float((predicted[mask] == observed[mask]).mean()),
                "mean_loo_probability": float(loo_probability[mask].mean()),
                "pareto_k_over_0.7": int((pareto_k[mask] > 0.7).sum()),
            }
        )
    return pd.DataFrame(rows)


def _summary_row(effect: str, samples: np.ndarray) -> dict:
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return {"effect": effect, "mean_log_odds": samples.mean(), "hdi_2.5%": lo, "hdi_97.5%": hi}


@communicate
def judge_feature_report(predictors: Sequence[str], plot_exclude: Sequence[str] = ()) -> Communicator:
    def build(
        state: AnalysisState, display: ModellingDisplay | None = None
    ) -> Tuple[AnalysisState, CommunicateResult]:
        model = next(model for model in state.models if model.is_fitted)
        effects = _effect_table(model, predictors)
        accuracy = _loo_accuracy(model)
        state.add_table(effects, "judge_feature_effects")
        state.add_table(accuracy, "judge_loo_accuracy")

        plot_effects = effects[~effects["feature"].isin(plot_exclude)].copy()
        heatmap = plot_effects.pivot(index="judge", columns="feature", values="mean_log_odds")
        heatmap.columns = [PLOT_LABELS.get(column, column) for column in heatmap.columns]
        fig, ax = plt.subplots(figsize=(max(9, 1.3 * len(heatmap.columns)), 6))
        limit = np.nanmax(np.abs(heatmap.values))
        image = ax.imshow(heatmap, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
        ax.set_xticks(range(len(heatmap.columns)), heatmap.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(heatmap.index)), heatmap.index)
        ax.set_title("Judge-specific effects on choosing the left transcript", weight="bold")
        for (row, col), value in np.ndenumerate(heatmap.values):
            ax.text(col, row, f"{value:+.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(value) > limit * 0.55 else "black")
        fig.colorbar(image, ax=ax, label="log-odds")
        fig.tight_layout()
        state.add_plot(fig, "judge_feature_heatmap")

        fig, axes = plt.subplots(len(heatmap.index), 1, figsize=(10, 1.8 * len(heatmap.index)), squeeze=False)
        for ax, judge in zip(axes[:, 0], heatmap.index):
            subset = plot_effects[plot_effects["judge"] == judge].reset_index(drop=True)
            y = np.arange(len(subset))
            ax.errorbar(
                subset["mean_log_odds"],
                y,
                xerr=[subset["mean_log_odds"] - subset["hdi_2.5%"], subset["hdi_97.5%"] - subset["mean_log_odds"]],
                fmt="o", color="#225ea8", ecolor="#225ea8", capsize=3,
            )
            ax.axvline(0, color="#555555", linewidth=1)
            ax.set_yticks(y, [PLOT_LABELS.get(feature, feature) for feature in subset["feature"]])
            ax.set_title(judge, loc="left", weight="bold")
        fig.tight_layout()
        state.add_plot(fig, "judge_feature_forest")

        posterior = model.inference_data.posterior
        predictor_means = posterior["predictor_effect_mean"].values.reshape(-1, len(predictors))
        rows = [_summary_row("position_bias_left", posterior["position_effect_mean"].values.reshape(-1))]
        for index, predictor in enumerate(predictors):
            if predictor in plot_exclude:
                continue
            rows.append(_summary_row(predictor, predictor_means[:, index]))
        importance = pd.DataFrame(rows).sort_values("mean_log_odds")
        state.add_table(importance, "population_effects")
        fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
        ax.set_facecolor("#f9f9fb")
        fig.patch.set_facecolor("white")
        ax.grid(axis="x", linestyle="--", alpha=0.5, color="#cccccc", zorder=0)
        ax.axvline(0, color="#666666", linestyle="-", linewidth=1.2, alpha=0.7, zorder=1)

        judges = list(model.coords["judge"])
        num_judges = len(judges)
        cmap = plt.get_cmap("tab10")
        judge_colors = {judge: cmap(i % 10) for i, judge in enumerate(judges)}

        y_positions = np.arange(len(importance))
        offsets = np.linspace(-0.2, 0.2, num_judges)

        # Plot individual judge model datapoints alongside the population estimate
        added_judge_labels = set()
        for y_idx, (_, pop_row) in enumerate(importance.iterrows()):
            effect_name = pop_row["effect"]
            feature_judges = plot_effects[plot_effects["feature"] == effect_name]

            for j_idx, judge in enumerate(judges):
                j_row = feature_judges[feature_judges["judge"] == judge]
                if not j_row.empty:
                    val = j_row["mean_log_odds"].values[0]
                    label = judge if judge not in added_judge_labels else None
                    ax.scatter(
                        val,
                        y_idx + offsets[j_idx],
                        color=judge_colors[judge],
                        s=45,
                        alpha=0.85,
                        edgecolor="white",
                        linewidth=0.5,
                        zorder=3,
                        label=label,
                    )
                    added_judge_labels.add(judge)

        # Plot population mean & 95% HDI interval centered on each row
        for y_idx, (_, pop_row) in enumerate(importance.iterrows()):
            mean_val = pop_row["mean_log_odds"]
            hdi_lo = pop_row["hdi_2.5%"]
            hdi_hi = pop_row["hdi_97.5%"]

            ax.errorbar(
                mean_val,
                y_idx,
                xerr=[[mean_val - hdi_lo], [hdi_hi - mean_val]],
                fmt="none",
                ecolor="#111111",
                elinewidth=2.5,
                capsize=5,
                capthick=1.8,
                zorder=4,
            )
            label = "Population Mean (95% HDI)" if y_idx == 0 else None
            ax.scatter(
                mean_val,
                y_idx,
                s=120,
                color="#2b2b2b",
                marker="D",
                edgecolor="white",
                linewidth=1.2,
                zorder=5,
                label=label,
            )

        y_labels = [PLOT_LABELS.get(effect, effect) for effect in importance["effect"]]
        ax.set_yticks(y_positions)
        ax.set_yticklabels(y_labels, fontsize=11, fontweight="medium", color="#222222")

        ax.set_xlabel("Effect on choice log-odds (95% HDI)", fontsize=11, fontweight="medium", labelpad=8, color="#222222")
        ax.set_title("Capability is the strongest cross-judge predictor", fontsize=13, fontweight="bold", pad=12, color="#111111")

        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color("#888888")
            ax.spines[spine].set_linewidth(0.8)

        handles, legend_labels = ax.get_legend_handles_labels()
        pop_idx = [i for i, l in enumerate(legend_labels) if "Population" in l]
        other_idxs = [i for i, l in enumerate(legend_labels) if "Population" not in l]
        reordered_handles = [handles[i] for i in pop_idx + other_idxs]
        reordered_labels = [legend_labels[i] for i in pop_idx + other_idxs]

        ax.legend(
            reordered_handles,
            reordered_labels,
            bbox_to_anchor=(1.02, 0.5),
            loc="center left",
            frameon=True,
            facecolor="#ffffff",
            edgecolor="#dddddd",
            fontsize=9.5,
            title="Estimates",
            title_fontsize=10,
        )

        fig.tight_layout()
        state.add_plot(fig, "overall_predictor_importance")

        interaction_rows = []
        judge_effects = posterior["judge_predictor_effects"].values.reshape(-1, len(model.coords["judge"]), len(predictors))
        for index, predictor in enumerate(predictors):
            if predictor not in plot_exclude:
                interaction_rows.append((predictor, judge_effects[:, :, index] - predictor_means[:, None, index]))
        interaction_rows.append(("position_bias_left", posterior["judge_intercepts"].values.reshape(-1, len(model.coords["judge"])) - posterior["position_effect_mean"].values.reshape(-1, 1)))
        interactions = pd.DataFrame(
            {PLOT_LABELS.get(feature, feature): draws.mean(axis=0) for feature, draws in interaction_rows},
            index=model.coords["judge"],
        )
        fig, ax = plt.subplots(figsize=(max(9, 1.3 * len(interactions.columns)), 6))
        limit = np.nanmax(np.abs(interactions.values))
        image = ax.imshow(interactions, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
        ax.set_xticks(range(len(interactions.columns)), interactions.columns, rotation=35, ha="right")
        ax.set_yticks(range(len(interactions.index)), interactions.index)
        ax.set_title("Judge deviations from the population effects", weight="bold")
        for (row, col), value in np.ndenumerate(interactions.values):
            ax.text(col, row, f"{value:+.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(value) > limit * 0.55 else "black")
        fig.colorbar(image, ax=ax, label="judge interaction (log-odds)")
        fig.tight_layout()
        state.add_plot(fig, "judge_effect_interactions")

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(accuracy["judge"], accuracy["loo_accuracy"])
        ax.axhline(0.5, color="black", linestyle="--", linewidth=1)
        ax.set_ylim(0, 1)
        ax.set_ylabel("PSIS-LOO accuracy")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        state.add_plot(fig, "judge_loo_accuracy")
        return state, "pass"

    return build
