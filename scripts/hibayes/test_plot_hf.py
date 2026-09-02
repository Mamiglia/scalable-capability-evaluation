"""HF posterior plotting: predictor list and posterior-reshaping are consistent."""

from pathlib import Path

import numpy as np
import pandas as pd

from plot_hf import PREDICTORS, load_draws
from run_hf_swc import PREDICTORS as FIT_PREDICTORS


def test_predictors_match_the_fitted_model():
    assert PREDICTORS == list(FIT_PREDICTORS)
    assert set(PREDICTORS).issubset(load_draws.__globals__["LABELS"])


def test_load_draws_reshapes_chains_and_checks_shapes(tmp_path):
    chains, draws, judges, items, predictors = 4, 5, 2, 3, len(PREDICTORS)
    rng = np.random.default_rng(0)

    def block(*shape):
        return rng.normal(size=(chains, draws, *shape)).astype(np.float32)

    posterior = tmp_path / "posterior.npz"
    np.savez(
        posterior,
        predictor_effect_mean=block(predictors),
        predictor_effect_scale=np.abs(block(predictors)),
        judge_predictor_effects_raw=block(judges, predictors),
        position_effect_mean=block(),
        position_effect_scale=np.abs(block()),
        judge_position_effects_raw=block(judges),
        item_sigma=np.abs(block()),
        item_effects_raw=block(items),
    )
    frame = pd.DataFrame({
        "judge": ["a", "b"] * 3,
        "item": [f"gpqa:{i}" for i in range(items)] * 2,
        "chosen_left": [0, 1] * 3,
    })

    raw, got_judges, population, effects, position_mean, intercepts, item_effects = load_draws(frame, posterior)
    assert got_judges == ["a", "b"]
    assert population.shape == (chains * draws, predictors)
    assert effects.shape == (chains * draws, judges, predictors)
    assert intercepts.shape == (chains * draws, judges)
    assert item_effects.shape == (chains * draws, items)
