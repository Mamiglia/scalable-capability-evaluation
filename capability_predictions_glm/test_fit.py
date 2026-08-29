from dataclasses import replace

import numpy as np
import pandas as pd

from capability_predictions_glm.config import CONFIG
from capability_predictions_glm.fit import (
    fit_direct,
    fit_direct_fixed,
    fit_pairwise,
    fit_pairwise_fixed,
    model_group_splits,
    predict_pairwise,
)


def test_direct_fit_groups_inner_validation_by_model() -> None:
    rows = []
    for index in range(6):
        for item in ("easy-1", "easy-2"):
            rows.append({
                "bench": "b", "item": item, "model": f"m{index}",
                "signal": float(index), "target_accuracy": 0.2 + 0.1 * index,
            })
    train = pd.DataFrame(rows)
    test = pd.DataFrame([
        {"bench": "b", "item": item, "model": model,
         "signal": signal, "target_accuracy": target}
        for model, signal, target in (("held-low", 1.0, 0.3), ("held-high", 4.0, 0.6))
        for item in ("easy-1", "easy-2")
    ])
    config = replace(
        CONFIG, ridge_alphas=(1.0,), direct_elastic_alphas=(0.001,),
        elastic_l1_ratios=(0.5,), inner_folds=3)

    for fit_rows, check_rows in model_group_splits(train, config):
        assert set(train.iloc[fit_rows].model).isdisjoint(train.iloc[check_rows].model)

    result = fit_direct(train, test, ["signal"], "Q2", config)

    predictions = result.model_predictions.set_index("model")
    assert predictions.index.tolist() == ["held-high", "held-low"]
    assert predictions.predicted_accuracy.between(0, 1).all()
    assert predictions.loc["held-high", "predicted_accuracy"] > predictions.loc["held-low", "predicted_accuracy"]
    assert result.target_quartile == "Q2"
    assert result.estimator.fit_intercept is False
    assert "signal" in result.selected_features
    assert set(result.tuning.stage) == {"elastic_net", "ridge"}
    assert result.diagnostics["elastic_net_alpha"] == 0.001
    assert result.diagnostics["ridge_alpha"] == 1.0


def test_pairwise_fit_is_antisymmetric() -> None:
    train = pd.DataFrame([
        {
            "bench": "b", "item": f"i{index}",
            "model_left": f"left-{index}", "model_right": f"right-{index}",
            "d_signal": signal, "left_stronger": int(signal > 0),
        }
        for index, signal in enumerate((-2.0, -1.5, -1.0, 1.0, 1.5, 2.0))
    ])
    test = pd.DataFrame([
        {"bench": "b", "item": "new", "model_left": "a", "model_right": "b",
         "d_signal": 0.7, "left_stronger": 1},
        {"bench": "b", "item": "new", "model_left": "b", "model_right": "a",
         "d_signal": -0.7, "left_stronger": 0},
    ])
    config = replace(
        CONFIG, logistic_cs=(1.0,), pair_elastic_cs=(1.0,),
        elastic_l1_ratios=(0.5,), inner_folds=3)

    result = fit_pairwise(train, test, ["signal"], "Q4", config)
    probability = predict_pairwise(result, test.drop(columns="left_stronger"))

    assert np.isclose(probability[0], 1 - probability[1])
    assert result.estimator.fit_intercept is False
    assert result.target_quartile == "Q4"
    assert set(result.metrics.metric) == {"accuracy", "log_loss", "auc"}
    assert "signal" in result.selected_features
    assert set(result.tuning.stage) == {"elastic_net", "ridge"}
    assert result.diagnostics["elastic_net_C"] == 1.0
    assert result.diagnostics["ridge_C"] == 1.0
    population = result.coefficients[result.coefficients.term == "population_slope"].set_index("feature")
    deviation = result.coefficients[result.coefficients.term == "benchmark_deviation"].set_index("feature")
    assert deviation.loc["signal", "effective"] == (
        population.loc["signal", "natural"] + deviation.loc["signal", "natural"])


def test_fixed_refits_use_supplied_penalties_without_tuning() -> None:
    direct_train = pd.DataFrame([
        {"bench": "b", "item": "i", "model": f"m{i}", "signal": float(i),
         "target_accuracy": 0.2 + i / 10}
        for i in range(6)
    ])
    direct_test = pd.DataFrame([
        {"bench": "b", "item": "i", "model": "held", "signal": 2.5,
         "target_accuracy": 0.5}
    ])
    pair_train = pd.DataFrame([
        {"bench": "b", "item": f"i{i}", "model_left": f"a{i}", "model_right": f"b{i}",
         "d_signal": value, "left_stronger": int(value > 0)}
        for i, value in enumerate((-2.0, -1.0, 1.0, 2.0))
    ])
    pair_test = pd.DataFrame([
        {"bench": "b", "item": "new", "model_left": "a", "model_right": "b",
         "d_signal": 0.5, "left_stronger": 1}
    ])

    direct = fit_direct_fixed(direct_train, direct_test, ["signal"], "Q2", CONFIG, ridge_alpha=7.0)
    pair = fit_pairwise_fixed(pair_train, pair_test, ["signal"], "Q2", CONFIG, ridge_c=0.07)

    assert direct.diagnostics == {"ridge_alpha": 7.0}
    assert pair.diagnostics == {"ridge_C": 0.07}
    assert direct.tuning.empty and pair.tuning.empty
