"""Model rankings, training-only calibration, and matched prediction metrics."""

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

from ranking import rank


def ability_table(pairs: pd.DataFrame, pick_column: str, method: str) -> pd.DataFrame:
    """Bradley–Terry abilities by benchmark from probabilities or verdicts."""
    required = {"bench", "item", "model_left", "model_right", pick_column}
    if missing := required - set(pairs):
        raise ValueError(f"pair table missing columns: {sorted(missing)}")
    rows = []
    for bench, frame in pairs.groupby("bench", sort=False):
        verdicts = frame.copy()
        verdicts["pick_left"] = verdicts[pick_column].to_numpy(float)
        if "judge" not in verdicts:
            verdicts["judge"] = method
        abilities = rank(verdicts)
        rows.append(abilities[["model", "ability", "n_comparisons"]].rename(
            columns={"ability": "score"}).assign(bench=bench, method=method))
    columns = ["bench", "model", "score", "method", "n_comparisons"]
    return (pd.concat(rows, ignore_index=True)[columns] if rows
            else pd.DataFrame(columns=columns))


def calibrate_scores(
    scores: pd.DataFrame,
    accuracies: pd.DataFrame,
    held_out_models: set[str],
    target_quartile: str,
    fold: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit score→target accuracy on seen models, then predict held-out models."""
    target = accuracies[accuracies.quartile == target_quartile]
    joined = scores.merge(target[["bench", "model", "accuracy"]],
                          on=["bench", "model"], how="inner", validate="many_to_one")
    prediction_rows, calibration_rows = [], []
    for (bench, method), frame in joined.groupby(["bench", "method"], sort=False):
        train = frame[~frame.model.isin(held_out_models)]
        test = frame[frame.model.isin(held_out_models)]
        if len(train) < 2 or test.empty:
            continue
        x, y = train[["score"]].to_numpy(float), train.accuracy.to_numpy(float)
        if np.ptp(x) == 0:
            slope, intercept = 0.0, float(y.mean())
            predicted = np.full(len(test), intercept)
        else:
            model = LinearRegression().fit(x, y)
            slope, intercept = float(model.coef_[0]), float(model.intercept_)
            predicted = model.predict(test[["score"]].to_numpy(float))
        output = test[["model", "score", "accuracy"]].rename(columns={"accuracy": "observed_accuracy"})
        output["predicted_accuracy"] = np.clip(predicted, 0, 1)
        output = output.assign(fold=fold, bench=bench, target_quartile=target_quartile, method=method)
        prediction_rows.append(output)
        calibration_rows.append({
            "fold": fold, "bench": bench, "target_quartile": target_quartile,
            "method": method, "slope": slope, "intercept": intercept,
            "n_training_models": len(train), "n_held_out_models": len(test),
        })
    prediction_columns = ["fold", "bench", "target_quartile", "method", "model", "score",
                          "predicted_accuracy", "observed_accuracy"]
    return (
        pd.concat(prediction_rows, ignore_index=True)[prediction_columns]
        if prediction_rows else pd.DataFrame(columns=prediction_columns),
        pd.DataFrame(calibration_rows),
    )


def _correlation(function, observed: np.ndarray, predicted: np.ndarray) -> float:
    if len(observed) < 2 or len(set(observed)) < 2 or len(set(predicted)) < 2:
        return np.nan
    return float(function(observed, predicted).statistic)


def matched_prediction_metrics(
    predictions: pd.DataFrame,
    methods: list[str],
    *,
    comparison: str,
) -> pd.DataFrame:
    """Score methods on their within-benchmark intersection of held-out models."""
    rows = []
    selected = predictions[predictions.method.isin(methods)]
    for bench, frame in selected.groupby("bench", sort=False):
        model_sets = [set(frame.loc[frame.method == method, "model"]) for method in methods]
        common = set.intersection(*model_sets) if model_sets else set()
        for method in methods:
            values = frame[(frame.method == method) & frame.model.isin(common)].sort_values("model")
            if not len(values):
                continue
            observed = values.observed_accuracy.to_numpy(float)
            predicted = values.predicted_accuracy.to_numpy(float)
            metrics = {
                "mae": mean_absolute_error(observed, predicted),
                "rmse": np.sqrt(mean_squared_error(observed, predicted)),
                "spearman": _correlation(spearmanr, observed, predicted),
                "kendall": _correlation(kendalltau, observed, predicted),
            }
            rows.extend({
                "comparison": comparison, "bench": bench, "method": method,
                "model_count": len(common), "metric": metric, "value": float(value),
            } for metric, value in metrics.items())
    return pd.DataFrame(rows)
