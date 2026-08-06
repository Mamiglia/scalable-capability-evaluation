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
            (f"transcript_condition={condition} vs raw", condition_draws[:, judge_index, i])
            for i, condition in enumerate(conditions)
            if condition != "raw"
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


@communicate
def judge_feature_report(predictors: Sequence[str]) -> Communicator:
    def build(
        state: AnalysisState, display: ModellingDisplay | None = None
    ) -> Tuple[AnalysisState, CommunicateResult]:
        model = next(model for model in state.models if model.is_fitted)
        effects = _effect_table(model, predictors)
        accuracy = _loo_accuracy(model)
        state.add_table(effects, "judge_feature_effects")
        state.add_table(accuracy, "judge_loo_accuracy")

        heatmap = effects.pivot(index="judge", columns="feature", values="mean_log_odds")
        fig, ax = plt.subplots(figsize=(max(9, 1.2 * len(heatmap.columns)), 5))
        limit = np.nanmax(np.abs(heatmap.values))
        image = ax.imshow(heatmap, cmap="coolwarm", vmin=-limit, vmax=limit, aspect="auto")
        ax.set_xticks(range(len(heatmap.columns)), heatmap.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(heatmap.index)), heatmap.index)
        ax.set_title("Judge feature effects (posterior mean log-odds)")
        fig.colorbar(image, ax=ax, label="log-odds")
        fig.tight_layout()
        state.add_plot(fig, "judge_feature_heatmap")

        fig, axes = plt.subplots(len(heatmap.index), 1, figsize=(10, 3 * len(heatmap.index)), squeeze=False)
        for ax, judge in zip(axes[:, 0], heatmap.index):
            subset = effects[effects["judge"] == judge].reset_index(drop=True)
            y = np.arange(len(subset))
            ax.errorbar(
                subset["mean_log_odds"],
                y,
                xerr=[subset["mean_log_odds"] - subset["hdi_2.5%"], subset["hdi_97.5%"] - subset["mean_log_odds"]],
                fmt="o",
            )
            ax.axvline(0, color="black", linewidth=1)
            ax.set_yticks(y, subset["feature"])
            ax.set_title(judge)
        fig.tight_layout()
        state.add_plot(fig, "judge_feature_forest")

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
