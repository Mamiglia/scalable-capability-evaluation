"""The SWC standalone fit is the intended hierarchical logistic density.

model.py needs hibayes (cluster-absent, and not in this venv), so instead of
comparing objects we pin the standalone model's log density against an
independent closed-form of the same specification. A wrong prior scale or a
mis-assembled logit fails here.
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import yaml
from numpyro import distributions as dist
from numpyro.handlers import seed, substitute, trace

from run_hf_swc import (
    PREDICTORS,
    PRIOR_EFFECT_SCALE,
    PRIOR_INTERCEPT_SCALE,
    PRIOR_ITEM_SCALE,
    judge_feature_model,
)


def test_predictors_match_config_and_preprocessing():
    from prepare_hf import PREDICTORS as PREP

    config = yaml.safe_load(Path(__file__).with_name("config_hf.yaml").read_text())
    assert tuple(PREP) == PREDICTORS
    assert config["model"]["models"][0]["config"]["predictors"] == list(PREDICTORS)


def test_log_density_matches_closed_form():
    predictors = ("f0", "f1")
    num_judges, num_items, num_rows = 2, 3, 4
    rng = np.random.default_rng(0)
    features = {
        "obs": jnp.asarray([1, 0, 1, 0], np.int8),
        "judge_index": jnp.asarray([0, 1, 0, 1]),
        "item_index": jnp.asarray([0, 1, 2, 0]),
        "num_judges": num_judges,
        "num_items": num_items,
        "f0": jnp.asarray(rng.normal(size=num_rows)),
        "f1": jnp.asarray(rng.normal(size=num_rows)),
    }
    latents = {
        "position_effect_mean": jnp.asarray(0.3),
        "position_effect_scale": jnp.asarray(0.4),
        "judge_position_effects_raw": jnp.asarray(rng.normal(size=num_judges)),
        "predictor_effect_mean": jnp.asarray(rng.normal(size=2)),
        "predictor_effect_scale": jnp.asarray(np.abs(rng.normal(size=2)) + 0.1),
        "judge_predictor_effects_raw": jnp.asarray(rng.normal(size=(num_judges, 2))),
        "item_sigma": jnp.asarray(0.5),
        "item_effects_raw": jnp.asarray(rng.normal(size=num_items)),
    }
    model = substitute(seed(lambda f: judge_feature_model(f, predictors), rng_seed=0), latents)
    sites = trace(model).get_trace(features)
    model_logp = sum(
        site["fn"].log_prob(site["value"]).sum() for site in sites.values() if site["type"] == "sample"
    )

    # Independent reference for the same specification.
    intercepts = latents["position_effect_mean"] + latents["position_effect_scale"] * latents[
        "judge_position_effects_raw"
    ]
    effects = latents["predictor_effect_mean"] + latents["predictor_effect_scale"] * latents[
        "judge_predictor_effects_raw"
    ]
    item_effects = latents["item_sigma"] * latents["item_effects_raw"]
    design = jnp.stack([features["f0"], features["f1"]], axis=1)
    judge = np.asarray(features["judge_index"])
    logits = (
        intercepts[judge]
        + jnp.sum(effects[judge] * design, axis=1)
        + item_effects[np.asarray(features["item_index"])]
    )
    reference = (
        dist.Normal(0, PRIOR_INTERCEPT_SCALE).log_prob(latents["position_effect_mean"])
        + dist.HalfNormal(PRIOR_EFFECT_SCALE).log_prob(latents["position_effect_scale"])
        + dist.Normal(0, 1).log_prob(latents["judge_position_effects_raw"]).sum()
        + dist.Normal(0, PRIOR_EFFECT_SCALE).log_prob(latents["predictor_effect_mean"]).sum()
        + dist.HalfNormal(PRIOR_EFFECT_SCALE).log_prob(latents["predictor_effect_scale"]).sum()
        + dist.Normal(0, 1).log_prob(latents["judge_predictor_effects_raw"]).sum()
        + dist.HalfNormal(PRIOR_ITEM_SCALE).log_prob(latents["item_sigma"])
        + dist.Normal(0, 1).log_prob(latents["item_effects_raw"]).sum()
        + dist.Bernoulli(logits=logits).log_prob(features["obs"]).sum()
    )
    assert np.isclose(float(model_logp), float(reference), rtol=1e-5)
