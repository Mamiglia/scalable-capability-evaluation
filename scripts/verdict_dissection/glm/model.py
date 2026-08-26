"""The three hibayes models. Every slope block is partially pooled: each group (benchmark or
judge) gets its own slope per predictor, shrunk towards a shared population slope."""
from collections.abc import Sequence

import jax.numpy as jnp
import numpyro
from numpyro import distributions as dist

from hibayes.model import Model, check_features, model
from hibayes.process import Features


def partially_pooled_slopes(name, num_groups, num_predictors, prior_scale):
    population = numpyro.sample(f"{name}_population",
                                dist.Normal(0, prior_scale).expand([num_predictors]).to_event(1))
    spread = numpyro.sample(f"{name}_spread",
                            dist.HalfNormal(prior_scale).expand([num_predictors]).to_event(1))
    offsets = numpyro.sample(f"{name}_offsets",
                             dist.Normal(0, 1).expand([num_groups, num_predictors]).to_event(2))
    return numpyro.deterministic(name, population + spread * offsets)


def design_matrix(features, predictors):
    return jnp.stack([features[name] for name in predictors], axis=1)


@model
def capability_from_transcript(predictors: Sequence[str], prior_scale: float = 0.5) -> Model:
    """Logit model accuracy ~ benchmark intercept + benchmark slopes · features + item effect."""

    def statistical_model(features: Features) -> None:
        check_features(features, ["obs", "bench_index", "num_bench", "item_index", "num_item", *predictors])
        bench, item = features["bench_index"], features["item_index"]
        intercepts = numpyro.sample("benchmark_intercepts",
                                    dist.Normal(0, 2).expand([features["num_bench"]]).to_event(1))
        slopes = partially_pooled_slopes("benchmark_slopes", features["num_bench"], len(predictors), prior_scale)
        item_spread = numpyro.sample("item_spread", dist.HalfNormal(1))
        item_offsets = numpyro.sample("item_offsets", dist.Normal(0, 1).expand([features["num_item"]]).to_event(1))
        noise = numpyro.sample("noise", dist.HalfNormal(2))
        mean = (intercepts[bench] + jnp.sum(slopes[bench] * design_matrix(features, predictors), axis=1)
                + item_spread * item_offsets[item])
        numpyro.sample("obs", dist.Normal(mean, noise), obs=features["obs"])

    return statistical_model


@model
def stronger_model_from_pair(predictors: Sequence[str], prior_scale: float = 0.5) -> Model:
    """P(left model is stronger) ~ benchmark slopes · (left − right features).
    No intercept and no item effect: the differences are antisymmetric, so both would cancel."""

    def statistical_model(features: Features) -> None:
        check_features(features, ["obs", "bench_index", "num_bench", *predictors])
        bench = features["bench_index"]
        slopes = partially_pooled_slopes("benchmark_slopes", features["num_bench"], len(predictors), prior_scale)
        logits = jnp.sum(slopes[bench] * design_matrix(features, predictors), axis=1)
        numpyro.sample("obs", dist.Bernoulli(logits=logits), obs=features["obs"])

    return statistical_model


@model
def judge_verdict_from_pair(predictors: Sequence[str], use_accuracy_gap: bool = False,
                            prior_scale: float = 0.5) -> Model:
    """P(judge picks left) ~ judge position bias + judge slopes · (left − right features).
    The base specification sees only the transcripts; with use_accuracy_gap the true accuracy
    difference enters as one more per-judge regressor, to test whether judges detect more."""

    def statistical_model(features: Features) -> None:
        required = ["obs", "judge_index", "num_judge", *predictors]
        check_features(features, required + (["accuracy_difference"] if use_accuracy_gap else []))
        judge, num_judges = features["judge_index"], features["num_judge"]
        position_bias = numpyro.sample("judge_position_bias", dist.Normal(0, 1).expand([num_judges]).to_event(1))
        slopes = partially_pooled_slopes("judge_slopes", num_judges, len(predictors), prior_scale)
        logits = position_bias[judge] + jnp.sum(slopes[judge] * design_matrix(features, predictors), axis=1)
        if use_accuracy_gap:
            gap_slopes = numpyro.sample("judge_accuracy_gap_slopes",
                                        dist.Normal(0, 2).expand([num_judges]).to_event(1))
            logits = logits + gap_slopes[judge] * features["accuracy_difference"]
        numpyro.sample("obs", dist.Bernoulli(logits=logits), obs=features["obs"])

    return statistical_model
