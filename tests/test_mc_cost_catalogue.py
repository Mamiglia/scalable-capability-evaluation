import json
import numpy as np
import pandas as pd
import pytest

from mc_cost.catalogue import (
    build_capability_draws, build_catalogue, theta_draws_from_accuracy, validate_catalogue,
)


def test_catalogue_joins_normalized_names_and_keeps_missing_metadata_visible(tmp_path):
    canonical = tmp_path / "canonical.md"
    canonical.write_text("- model-a\n- model-missing\n")
    pool = tmp_path / "pool.json"
    pool.write_text(json.dumps([{
        "name": "Model A", "release": "2025-01-02", "price_in": 1.0,
        "price_out": 4.0, "reasoning": False,
    }]))
    capabilities = pd.DataFrame([{
        "irt_model": "fixed_3pl", "fit": "gpqa", "model": "model-a",
        "theta": 1.2, "theta_p5": 1.0, "theta_p95": 1.4,
    }])

    got = build_catalogue(canonical, pool, capabilities)

    assert got.loc[got.model_id == "model-a", "release"].item() == "2025-01-02"
    assert got.loc[got.model_id == "model-a", "theta"].item() == 1.2
    assert pd.isna(got.loc[got.model_id == "model-missing", "release"]).item()


def test_accuracy_to_theta_inverts_each_fixed_3pl_draw():
    draws = {"a": np.ones((3, 2)), "b": np.zeros((3, 2)), "floor": np.full(2, 0.25)}
    got = theta_draws_from_accuracy(75, 100, draws, np.random.default_rng(3))
    assert got.shape == (3,)
    assert np.all(np.isfinite(got))


def test_capability_draws_reuse_irt_and_do_not_impute_bare_point_estimate():
    catalogue = pd.DataFrame([
        {"model_id": "fitted", "theta": 1.0, "theta_source": "fixed_3pl"},
        {"model_id": "counted", "gpqa_correct": 7, "gpqa_total": 10,
         "theta_source": "gpqa_counts"},
        {"model_id": "bare", "theta": 0.0, "theta_source": "manual_point"},
    ])
    irt = {
        "models": ["fitted"], "theta": np.array([[0.9], [1.0], [1.1]]),
        "a": np.ones((3, 2)), "b": np.zeros((3, 2)), "floor": np.full(2, 0.25),
    }
    models, draws = build_capability_draws(catalogue, irt, np.random.default_rng(5))
    assert models == ["fitted", "counted"]
    assert draws.shape == (3, 2)
    assert np.allclose(draws[:, 0], [0.9, 1.0, 1.1])


def test_catalogue_rejects_duplicate_ids_and_bad_reasoning_class():
    frame = pd.DataFrame([
        {"model_id": "m", "reasoning_class": "reasoning", "release": "2025-01-01", "theta": 0.0},
        {"model_id": "m", "reasoning_class": "sometimes", "release": "2025-01-01", "theta": 0.0},
    ])
    with pytest.raises(ValueError, match="duplicate model_id|reasoning_class"):
        validate_catalogue(frame)
