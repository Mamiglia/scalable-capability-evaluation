from itertools import combinations

import numpy as np
import pandas as pd


def test_monte_carlo_is_seeded_and_uses_observed_budgets():
    from simulation import monte_carlo_budgets

    models = ["weak", "a", "b", "c", "d"]
    rounds = pd.DataFrame([
        {
            "source": "s",
            "benchmark": "gpqa",
            "item_id": f"i{item}",
            "vote_id": f"s:i{item}",
            "judge": "j",
            "major": major,
            "minor": minor,
            "picked_major": float(models.index(major) < models.index(minor)),
            "n_calls": 2,
        }
        for item in range(6)
        for major, minor in combinations(models, 2)
    ])
    splits = pd.DataFrame([
        {
            "irt_model": "fixed_3pl",
            "fit": "gpqa",
            "benchmark": "gpqa",
            "item": f"i{item}",
            "half": "easy",
            "quartile": "Q1",
        }
        for item in range(6)
    ])
    easy_accuracy = {"weak": 0.70, "a": 0.80, "b": 0.85, "c": 0.90, "d": 0.95}
    hard_accuracy = {"weak": 0.10, "a": 0.20, "b": 0.30, "c": 0.40, "d": 0.50}
    accuracies = pd.DataFrame([
        {
            "irt_model": "fixed_3pl",
            "fit": "gpqa",
            "subset": subset,
            "model": model,
            "accuracy": values[model],
        }
        for subset, values in (("easy", easy_accuracy), ("hard", hard_accuracy))
        for model in models
    ])

    raw1, summary1 = monte_carlo_budgets(
        rounds, splits, accuracies, resamples=3, seed=7,
    )
    raw2, summary2 = monte_carlo_budgets(
        rounds, splits, accuracies, resamples=3, seed=7,
    )

    pd.testing.assert_frame_equal(raw1, raw2)
    pd.testing.assert_frame_equal(summary1, summary2)
    assert set(raw1["axis"]) == {"models", "items", "rounds"}
    assert raw1["component"].notna().all()
    assert raw1["valid"].isin([True, False]).all()
    assert (raw1["calls"] >= raw1["rounds"]).all()
    assert not raw1["full_component_models"].str.contains("weak").any()


def test_monte_carlo_summary_counts_disconnected_draws_as_invalid():
    from simulation import _summarize

    common = {
        "irt_model": "fixed_3pl",
        "fit": "gpqa",
        "judge": "consensus",
        "full_component": 0,
        "full_component_models": "a|b|c|d",
        "axis": "items",
        "budget": 5,
        "baseline_tau": 0.4,
        "baseline_mae": 0.2,
    }
    raw = pd.DataFrame([
        {
            **common,
            "component": "0.0",
            "component_models": "a|b|c|d",
            "replicate": 0,
            "covers_sampled_models": True,
            "valid": True,
            "stable": True,
            "tau": 0.5,
            "mae": 0.1,
        },
        {
            **common,
            "component": "0.0",
            "component_models": "a|b",
            "replicate": 1,
            "covers_sampled_models": False,
            "valid": True,
            "stable": False,
            "tau": 0.5,
            "mae": 0.1,
        },
        {
            **common,
            "component": "0.1",
            "component_models": "c|d",
            "replicate": 1,
            "covers_sampled_models": False,
            "valid": False,
            "stable": False,
            "tau": np.nan,
            "mae": np.nan,
        },
    ])

    got = _summarize(raw, resamples=2)

    assert len(got) == 1
    assert got.iloc[0]["valid_fraction"] == 0.5
    assert got.iloc[0]["stable_fraction"] == 0.5
    assert got.iloc[0]["tau_median"] == 0.5
