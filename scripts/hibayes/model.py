"""HiBayes Bernoulli model for judge-specific GPQA feature effects."""

from collections.abc import Sequence

import jax.numpy as jnp
import numpyro
from numpyro import distributions as dist

from hibayes.model import Model, check_features, model
from hibayes.process import Features


@model
def judge_feature_logistic_model(
    predictors: Sequence[str],
    pool_judge_effects: bool = True,
    prior_intercept_scale: float = 1.5,
    prior_effect_scale: float = 0.5,
    prior_item_scale: float = 0.5,
) -> Model:
    """Bernoulli GLM with population effects and pooled judge deviations."""

    def statistical_model(features: Features) -> None:
        required = [
            "obs",
            "judge_index",
            "num_judge",
            "item_index",
            "num_item",
            "transcript_condition_index",
            "num_transcript_condition",
            *predictors,
        ]
        check_features(features, required)

        num_judges = features["num_judge"]
        num_conditions = features["num_transcript_condition"]
        judge = features["judge_index"]

        position_mean = numpyro.sample("position_effect_mean", dist.Normal(0, prior_intercept_scale))
        position_scale = numpyro.sample("position_effect_scale", dist.HalfNormal(prior_effect_scale))
        position_raw = numpyro.sample("judge_position_effects_raw", dist.Normal(0, 1).expand([num_judges]).to_event(1))
        intercepts = position_mean + position_scale * position_raw
        numpyro.deterministic("judge_intercepts", intercepts)
        if not pool_judge_effects:
            raise ValueError("judge_feature_logistic_model requires pooled judge effects")
        effect_mean = numpyro.sample(
            "predictor_effect_mean", dist.Normal(0, prior_effect_scale).expand([len(predictors)]).to_event(1)
        )
        effect_scale = numpyro.sample(
            "predictor_effect_scale", dist.HalfNormal(prior_effect_scale).expand([len(predictors)]).to_event(1)
        )
        effect_raw = numpyro.sample(
            "judge_predictor_effects_raw", dist.Normal(0, 1).expand([num_judges, len(predictors)]).to_event(2)
        )
        effects = effect_mean + effect_scale * effect_raw
        numpyro.deterministic("judge_predictor_effects", effects)
        condition_free = numpyro.sample(
            "judge_condition_effects_free",
            dist.Normal(0, prior_effect_scale).expand([num_judges, num_conditions - 1]).to_event(2),
        )
        condition_effects = jnp.concatenate(
            [jnp.zeros((num_judges, 1)), condition_free], axis=1
        )
        # Record only the free contrasts. The reference column is identically
        # zero, so its R-hat is NaN, and `np.all(r_hat < threshold)` is False
        # whenever a NaN is present: recording it made the R-hat checker fail on
        # every run regardless of convergence, which hid real failures.
        # The first category is the reference, so callers must order the
        # transcript_condition levels with `raw` first.
        numpyro.deterministic("judge_condition_effects", condition_free)

        item_sigma = numpyro.sample("item_sigma", dist.HalfNormal(prior_item_scale))
        item_raw = numpyro.sample(
            "item_effects_raw", dist.Normal(0, 1).expand([features["num_item"]]).to_event(1)
        )
        item_effects = item_sigma * item_raw
        numpyro.deterministic("item_effects", item_effects)

        design = jnp.stack([features[name] for name in predictors], axis=1)
        logits = (
            intercepts[judge]
            + jnp.sum(effects[judge] * design, axis=1)
            + condition_effects[judge, features["transcript_condition_index"]]
            + item_effects[features["item_index"]]
        )
        numpyro.sample("obs", dist.Bernoulli(logits=logits), obs=features["obs"])

    return statistical_model
