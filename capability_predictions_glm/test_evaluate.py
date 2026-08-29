import pandas as pd
import pytest

from capability_predictions_glm.evaluate import (
    ability_table,
    calibrate_scores,
    matched_prediction_metrics,
)


def test_ability_table_recovers_order_from_pair_probabilities() -> None:
    pairs = pd.DataFrame([
        {"bench": "b", "item": "i1", "model_left": "strong", "model_right": "middle", "probability": 0.9},
        {"bench": "b", "item": "i2", "model_left": "middle", "model_right": "weak", "probability": 0.8},
        {"bench": "b", "item": "i3", "model_left": "strong", "model_right": "weak", "probability": 0.95},
    ])

    scores = ability_table(pairs, "probability", "pair_glm").set_index("model")

    assert scores.loc["strong", "score"] > scores.loc["middle", "score"] > scores.loc["weak", "score"]


def test_ability_table_uses_requested_probability_when_verdict_column_is_present() -> None:
    pairs = pd.DataFrame([
        {"bench": "b", "judge": "j", "item": "i", "model_left": "a", "model_right": "b",
         "pick_left": 0, "probability": 0.9},
    ])

    scores = ability_table(pairs, "probability", "pair_glm").set_index("model")

    assert scores.loc["a", "score"] > scores.loc["b", "score"]


def test_calibration_uses_seen_models_only() -> None:
    scores = pd.DataFrame({
        "bench": ["b", "b", "b"], "model": ["seen-low", "seen-high", "held"],
        "score": [0.0, 1.0, 0.5], "method": ["judge"] * 3,
    })
    accuracy = pd.DataFrame({
        "bench": ["b", "b", "b"], "model": ["seen-low", "seen-high", "held"],
        "quartile": ["Q3"] * 3, "accuracy": [0.2, 0.8, 0.9],
    })

    first, calibration = calibrate_scores(scores, accuracy, {"held"}, "Q3", "fold-0")
    changed = accuracy.copy()
    changed.loc[changed.model == "held", "accuracy"] = 0.0
    second, _ = calibrate_scores(scores, changed, {"held"}, "Q3", "fold-0")

    assert first.predicted_accuracy.tolist() == pytest.approx([0.5])
    assert second.predicted_accuracy.tolist() == pytest.approx([0.5])
    assert calibration.loc[0, "n_training_models"] == 2


def test_matched_metrics_use_identical_models() -> None:
    predictions = pd.DataFrame([
        {"method": "a", "bench": "b", "model": "m1", "predicted_accuracy": 0.0, "observed_accuracy": 0.0},
        {"method": "a", "bench": "b", "model": "m2", "predicted_accuracy": 0.4, "observed_accuracy": 0.5},
        {"method": "a", "bench": "b", "model": "m3", "predicted_accuracy": 0.9, "observed_accuracy": 1.0},
        {"method": "b", "bench": "b", "model": "m2", "predicted_accuracy": 0.6, "observed_accuracy": 0.5},
        {"method": "b", "bench": "b", "model": "m3", "predicted_accuracy": 0.8, "observed_accuracy": 1.0},
    ])

    metrics = matched_prediction_metrics(predictions, ["a", "b"], comparison="a_vs_b")
    values = metrics.pivot(index="method", columns="metric", values="value")

    assert set(metrics.model_count) == {2}
    assert values.loc["a", "mae"] == pytest.approx(0.1)
    assert values.loc["b", "mae"] == pytest.approx(0.15)
    assert values.loc["a", "rmse"] == pytest.approx(0.1)
    assert values.loc["a", "spearman"] == pytest.approx(1.0)
