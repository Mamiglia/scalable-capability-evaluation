"""The SWC runner retains the fitted HiBayes GLM's predictor ordering."""

import jax.numpy as jnp
from numpyro.handlers import seed, trace

try:
    from run_hf_swc import atomic_npz, judge_feature_model
except ImportError:
    atomic_npz = None
    judge_feature_model = None


def test_model_has_one_population_effect_per_ordered_predictor():
    assert judge_feature_model is not None, "run_hf_swc.judge_feature_model is not implemented"
    features = {
        "obs": jnp.array([1, 0]),
        "judge_index": jnp.array([0, 0]),
        "item_index": jnp.array([0, 1]),
        "condition_index": jnp.array([0, 1]),
        "num_judges": 1,
        "num_items": 2,
        "num_conditions": 2,
        "capability_difference_sd": jnp.array([1.0, -1.0]),
        "length_difference_sd": jnp.array([0.5, -0.5]),
        "same_model_family": jnp.array([0.0, 1.0]),
    }
    predictors = ["length_difference_sd", "capability_difference_sd", "same_model_family"]

    sites = trace(seed(judge_feature_model, 0)).get_trace(features, predictors)

    assert sites["predictor_effect_mean"]["value"].shape == (3,)
    assert sites["judge_predictor_effects"]["value"].shape == (1, 3)
    assert sites["obs"]["value"].tolist() == [1, 0]


def test_posterior_is_published_atomically(tmp_path):
    assert atomic_npz is not None, "run_hf_swc.atomic_npz is not implemented"
    path = tmp_path / "posterior.npz"

    atomic_npz({"effect": jnp.array([[1.0, 2.0]])}, path)

    assert path.exists()
    assert not (tmp_path / "posterior.tmp.npz").exists()
