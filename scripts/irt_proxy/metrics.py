"""Evaluate easy-item predictors against hard-item outcomes.

Every endpoint is calculated on a matched candidate set within one comparison
component. Undefined small-sample results stay in the tables with a reason
instead of silently disappearing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mc_cost.ranking import _coverage_fields, _metric_keys, lomo_predictions

BOOTSTRAP_RESAMPLES = 5000
CORRECTION_GROUP_COLUMNS = ["benchmark", "irt_model"]
GAP_BIN_WIDTH = 0.05
GAP_BIN_EDGES = np.round(np.arange(0, 1 + GAP_BIN_WIDTH, GAP_BIN_WIDTH), 10)


def kendall_tau_a(predictor, outcome, weights=None) -> float:
    """Kendall tau-a, with every tied model pair contributing zero.

    Unlike tau-b, the denominator remains the total number of model pairs.
    A predictor tie therefore lowers the score instead of being corrected away.
    """
    predictor = np.asarray(predictor, dtype=float)
    outcome = np.asarray(outcome, dtype=float)
    if len(predictor) < 2:
        return np.nan
    left, right = np.triu_indices(len(predictor), k=1)
    pair_scores = np.sign(predictor[left] - predictor[right]) * np.sign(
        outcome[left] - outcome[right]
    )
    if weights is None:
        return float(pair_scores.mean())
    weights = np.asarray(weights, dtype=float)
    pair_weights = weights[left] * weights[right]
    denominator = pair_weights.sum()
    return (
        float(np.dot(pair_scores, pair_weights) / denominator)
        if denominator
        else np.nan
    )


def _correction_metric_samples(prediction, observed, lower, upper, weights):
    """Calculate prediction metrics for rows of model-bootstrap weights."""
    prediction = np.asarray(prediction, dtype=float)
    observed = np.asarray(observed, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    weights = np.atleast_2d(np.asarray(weights, dtype=float))
    if not len(prediction):
        return {
            name: np.full(len(weights), np.nan)
            for name in (
                "tau_a", "pair_accuracy", "mae", "rmse",
                "calibration_slope", "calibration_intercept",
                "interval_coverage",
            )
        }
    total = weights.sum(axis=1)
    error = prediction - observed

    mean_prediction = weights @ prediction / total
    mean_observed = weights @ observed / total
    variance = weights @ prediction**2 / total - mean_prediction**2
    covariance = weights @ (prediction * observed) / total - mean_prediction * mean_observed
    slope = np.divide(
        covariance, variance, out=np.full(len(weights), np.nan), where=variance > 0,
    )

    left, right = np.triu_indices(len(prediction), k=1)
    pair_weights = weights[:, left] * weights[:, right]
    true_sign = np.sign(observed[left] - observed[right])
    estimated_sign = np.sign(prediction[left] - prediction[right])
    pair_denominator = pair_weights.sum(axis=1)
    pair_accuracy = np.divide(
        pair_weights @ ((estimated_sign * true_sign) > 0),
        pair_denominator,
        out=np.full(len(weights), np.nan),
        where=pair_denominator > 0,
    )
    tau_denominator = pair_weights.sum(axis=1)
    tau = np.divide(
        pair_weights @ (estimated_sign * true_sign),
        tau_denominator,
        out=np.full(len(weights), np.nan),
        where=tau_denominator > 0,
    )
    return {
        "tau_a": tau,
        "pair_accuracy": pair_accuracy,
        "mae": weights @ np.abs(error) / total,
        "rmse": np.sqrt(weights @ error**2 / total),
        "calibration_slope": slope,
        "calibration_intercept": mean_observed - slope * mean_prediction,
        "interval_coverage": weights @ ((lower <= observed) & (observed <= upper)) / total,
    }


def correction_metrics(
    predictions: pd.DataFrame,
    *,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = 0,
) -> pd.DataFrame:
    """Summarize GPQA out-of-fold predictions with models as replicates."""
    data = predictions[predictions["benchmark"].eq("gpqa")].copy()
    group_columns = [column for column in CORRECTION_GROUP_COLUMNS if column in data]
    rows = []
    grouped_predictions = {}
    for values, group in data.groupby(group_columns + ["method"], dropna=False, sort=True):
        values = values if isinstance(values, tuple) else (values,)
        identity = dict(zip(group_columns + ["method"], values))
        valid = group.dropna(subset=["prediction", "observed"]).sort_values("model")
        if valid["model"].duplicated().any():
            raise ValueError("correction predictions must have one out-of-fold row per model/method")
        metrics = _correction_metric_samples(
            valid["prediction"], valid["observed"],
            valid["prediction_p5"], valid["prediction_p95"],
            np.ones(len(valid)),
        )
        metrics["tau_a"][0] = kendall_tau_a(valid["prediction"], valid["observed"])
        rows.append({
            **identity,
            **{name: sample[0] for name, sample in metrics.items()},
            "n_models": valid["model"].nunique(),
            "n_pairs": len(valid) * (len(valid) - 1) // 2,
        })
        grouped_predictions[tuple(identity.get(column) for column in group_columns) + (identity["method"],)] = valid

    result = pd.DataFrame(rows)
    metric_names = list(_correction_metric_samples([0, 1], [0, 1], [0, 1], [0, 1], [1, 1]))
    for index, row in result.iterrows():
        key = tuple(row[column] for column in group_columns)
        method = row["method"]
        method_data = grouped_predictions[key + (method,)]
        corrected = grouped_predictions.get(key + ("corrected",))
        if corrected is None:
            continue
        matched = method_data.merge(
            corrected,
            on="model",
            suffixes=("", "_corrected"),
            validate="one_to_one",
        ).sort_values("model")
        count = len(matched)
        if count < 2:
            continue
        weights = np.random.default_rng(seed).multinomial(
            count, np.repeat(1 / count, count), size=n_resamples,
        )
        samples = _correction_metric_samples(
            matched["prediction"], matched["observed"],
            matched["prediction_p5"], matched["prediction_p95"], weights,
        )
        corrected_samples = _correction_metric_samples(
            matched["prediction_corrected"], matched["observed_corrected"],
            matched["prediction_p5_corrected"], matched["prediction_p95_corrected"],
            weights,
        )
        point_weights = np.ones(count)
        matched_point = _correction_metric_samples(
            matched["prediction"], matched["observed"],
            matched["prediction_p5"], matched["prediction_p95"], point_weights,
        )
        corrected_point = _correction_metric_samples(
            matched["prediction_corrected"], matched["observed_corrected"],
            matched["prediction_p5_corrected"], matched["prediction_p95_corrected"],
            point_weights,
        )
        for name in metric_names:
            difference = samples[name] - corrected_samples[name]
            finite = difference[np.isfinite(difference)]
            result.loc[index, f"{name}_difference_vs_corrected"] = (
                matched_point[name][0] - corrected_point[name][0]
            )
            if len(finite):
                result.loc[index, f"{name}_difference_ci_low"] = np.percentile(finite, 2.5)
                result.loc[index, f"{name}_difference_ci_high"] = np.percentile(finite, 97.5)
        result.loc[index, "n_bootstrap_models"] = count
        result.loc[index, "n_bootstrap_resamples"] = n_resamples
        result.loc[index, "ci_method"] = "paired model bootstrap percentile 95%"
    return result


def capability_gap_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    """Return every within-analysis model pair on the hard-accuracy scale."""
    data = predictions[predictions["benchmark"].eq("gpqa")].copy()
    group_columns = [column for column in CORRECTION_GROUP_COLUMNS if column in data]
    rows = []
    for values, group in data.groupby(group_columns + ["method"], dropna=False, sort=True):
        values = values if isinstance(values, tuple) else (values,)
        identity = dict(zip(group_columns + ["method"], values))
        valid = group.dropna(subset=["prediction", "observed"]).sort_values("model")
        if valid["model"].duplicated().any():
            raise ValueError("correction predictions must have one out-of-fold row per model/method")
        for left, right in zip(*np.triu_indices(len(valid), k=1)):
            left_row, right_row = valid.iloc[left], valid.iloc[right]
            true_gap = left_row["observed"] - right_row["observed"]
            estimated_gap = left_row["prediction"] - right_row["prediction"]
            absolute_gap = round(abs(float(true_gap)), 10)
            bin_index = int(np.searchsorted(
                GAP_BIN_EDGES, absolute_gap, side="right",
            ) - 1)
            bin_index = int(np.clip(bin_index, 0, len(GAP_BIN_EDGES) - 2))
            rows.append({
                **identity,
                "model_left": left_row["model"],
                "model_right": right_row["model"],
                "fold_left": left_row["fold"],
                "fold_right": right_row["fold"],
                "prediction_left": left_row["prediction"],
                "prediction_right": right_row["prediction"],
                "observed_left": left_row["observed"],
                "observed_right": right_row["observed"],
                "estimated_gap": estimated_gap,
                "true_gap": true_gap,
                "absolute_true_gap": absolute_gap,
                "ordering_correct": int(np.sign(estimated_gap) * np.sign(true_gap) > 0),
                "gap_bin_left": GAP_BIN_EDGES[bin_index],
                "gap_bin_right": GAP_BIN_EDGES[bin_index + 1],
                "gap_bin_center": (
                    GAP_BIN_EDGES[bin_index] + GAP_BIN_EDGES[bin_index + 1]
                ) / 2,
            })
    result = pd.DataFrame(rows)
    if not result.empty:
        result["gap_bin_count"] = result.groupby(
            group_columns + ["method", "gap_bin_left"], dropna=False,
        )["model_left"].transform("size")
    return result


def model_bootstrap_intervals(
    predictions: pd.DataFrame,
    *,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = 0,
) -> pd.DataFrame:
    """Paired percentile intervals from resampling evaluated models.

    Models—not items, judgments, or model/item cells—are the sampling unit.
    Reinitializing the RNG for each method gives Elo and accuracy the same
    resampled model indices when their matched set has the same size.
    """
    keys = _metric_keys(predictions) + ["method"]
    rows = []
    data = predictions[predictions["method"].isin(["elo", "easy_accuracy"])]
    for values, group in data.groupby(keys, dropna=False, sort=True):
        values = values if isinstance(values, tuple) else (values,)
        identity = dict(zip(keys, values))
        matched = (
            group.dropna(subset=["predictor", "prediction", "observed"])
            .drop_duplicates("model")
            .sort_values("model")
        )
        n_models = len(matched)
        if n_models < 3:
            continue

        indices = np.random.default_rng(seed).integers(
            0,
            n_models,
            size=(n_resamples, n_models),
        )
        predictor = matched["predictor"].to_numpy(float)
        prediction = matched["prediction"].to_numpy(float)
        observed = matched["observed"].to_numpy(float)

        # Bootstrap multiplicities weight pairs of distinct evaluated models.
        # Two copies of the same model are not a new tied pair.
        weights = np.zeros((n_resamples, n_models), dtype=np.int16)
        np.add.at(
            weights,
            (np.repeat(np.arange(n_resamples), n_models), indices.ravel()),
            1,
        )
        left, right = np.triu_indices(n_models, k=1)
        pair_scores = np.sign(predictor[left] - predictor[right]) * np.sign(
            observed[left] - observed[right]
        )
        pair_weights = weights[:, left] * weights[:, right]
        denominators = pair_weights.sum(axis=1)
        tau_samples = np.divide(
            pair_weights @ pair_scores,
            denominators,
            out=np.full(n_resamples, np.nan),
            where=denominators > 0,
        )
        error = np.abs(prediction - observed)
        samples = {
            "tau_a": tau_samples[np.isfinite(tau_samples)],
            "mae": error[indices].mean(axis=1),
        }
        for metric, estimates in samples.items():
            low, high = (
                np.percentile(estimates, [2.5, 97.5])
                if len(estimates)
                else (np.nan, np.nan)
            )
            rows.append({
                **identity,
                "metric": metric,
                "ci_low": low,
                "ci_high": high,
                "n_bootstrap_models": n_models,
                "n_bootstrap_resamples": n_resamples,
                "ci_method": "paired model bootstrap percentile 95%",
            })
    return pd.DataFrame(rows)


def paired_method_randomization(
    predictions: pd.DataFrame,
    *,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = 0,
) -> pd.DataFrame:
    """Compare Elo and accuracy on the same evaluated models.

    Under the null, either method label is equally plausible for each model.
    Swapping those labels preserves the paired design and gives a two-sided
    randomization test for both rank correlation and held-out error.
    """
    random = np.random.default_rng(seed)
    keys = _metric_keys(predictions)
    rows = []
    selected = predictions[
        predictions["method"].isin(("elo", "easy_accuracy"))
    ]
    for values, group in selected.groupby(keys, dropna=False, sort=True):
        values = values if isinstance(values, tuple) else (values,)
        identity = dict(zip(keys, values))
        wide = group.pivot(
            index="model",
            columns="method",
            values=["predictor", "prediction", "observed"],
        ).dropna().sort_index()
        n_models = len(wide)
        if n_models < 3:
            continue

        observed = wide[("observed", "elo")].to_numpy(float)
        elo_predictor = wide[("predictor", "elo")].to_numpy(float)
        accuracy_predictor = wide[("predictor", "easy_accuracy")].to_numpy(float)
        swaps = random.integers(0, 2, size=(n_resamples, n_models), dtype=bool)
        tau_difference = (
            kendall_tau_a(elo_predictor, observed)
            - kendall_tau_a(accuracy_predictor, observed)
        )
        randomized_elo = np.where(swaps, accuracy_predictor, elo_predictor)
        randomized_accuracy = np.where(swaps, elo_predictor, accuracy_predictor)
        left, right = np.triu_indices(n_models, k=1)
        outcome_pairs = np.sign(observed[left] - observed[right])
        randomized_elo_tau = (
            np.sign(randomized_elo[:, left] - randomized_elo[:, right])
            * outcome_pairs
        ).mean(axis=1)
        randomized_accuracy_tau = (
            np.sign(randomized_accuracy[:, left] - randomized_accuracy[:, right])
            * outcome_pairs
        ).mean(axis=1)
        tau_null = randomized_elo_tau - randomized_accuracy_tau

        elo_error = np.abs(
            wide[("prediction", "elo")].to_numpy(float) - observed,
        )
        accuracy_error = np.abs(
            wide[("prediction", "easy_accuracy")].to_numpy(float) - observed,
        )
        mae_difference = elo_error.mean() - accuracy_error.mean()
        mae_null = (
            np.where(swaps, accuracy_error, elo_error).mean(axis=1)
            - np.where(swaps, elo_error, accuracy_error).mean(axis=1)
        )

        for metric, difference, null in (
            ("tau_a", tau_difference, tau_null),
            ("mae", mae_difference, mae_null),
        ):
            rows.append({
                **identity,
                "metric": metric,
                "elo_minus_accuracy": difference,
                "comparison_p_value": (
                    (np.abs(null) >= abs(difference)).sum() + 1
                ) / (n_resamples + 1),
                "n_comparison_models": n_models,
                "n_comparison_randomizations": n_resamples,
                "comparison_test": "paired model-label randomization, two-sided",
            })
    return pd.DataFrame(rows)


def rank_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute tie-penalized tau-a on one shared model set."""
    rows = []
    keys = _metric_keys(frame)
    # Each group is one Elo scale. In particular, `component` is part of the
    # key, so disconnected GPQA strengths can never meet in one correlation.
    for values, group in frame.groupby(keys, dropna=False, sort=True):
        values = values if isinstance(values, tuple) else (values,)
        identity = dict(zip(keys, values))
        matched = group.dropna(
            subset=["elo", "easy_accuracy", "hard_accuracy"],
        ).drop_duplicates("model")
        coverage = _coverage_fields(group)
        n_models = len(matched)
        reason = "" if n_models >= 3 else "fewer than 3 matched models"

        for method, predictor in (
            ("elo", "elo"),
            ("easy_accuracy", "easy_accuracy"),
        ):
            tau = (
                kendall_tau_a(matched[predictor], matched["hard_accuracy"])
                if n_models >= 3
                else np.nan
            )
            rows.append({
                **identity,
                **coverage,
                "method": method,
                "tau_a": tau,
                "n_models": n_models,
                "reason": reason,
            })
    return pd.DataFrame(rows)


def value_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Summarize held-out absolute and squared error within each component."""
    rows = []
    keys = _metric_keys(predictions) + ["method"]
    for values, group in predictions.groupby(keys, dropna=False, sort=True):
        values = values if isinstance(values, tuple) else (values,)
        identity = dict(zip(keys, values))
        valid = group.dropna(subset=["prediction", "observed"])
        coverage = _coverage_fields(group)
        if valid.empty:
            rows.append({
                **identity,
                **coverage,
                "mae": np.nan,
                "rmse": np.nan,
                "n_models": int(group["n_models"].max()),
                "reason": group["reason"].iloc[0],
            })
            continue

        error = valid["prediction"] - valid["observed"]
        rows.append({
            **identity,
            **coverage,
            "mae": error.abs().mean(),
            "rmse": np.sqrt(np.mean(error**2)),
            "n_models": valid["model"].nunique(),
            "reason": "",
        })
    return pd.DataFrame(rows)


def _endpoint_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Combine the two requested endpoints in one plotting-friendly table."""
    ranks = rank_metrics(frame).rename(columns={"tau_a": "value"})
    ranks["metric"] = "tau_a"

    errors = value_metrics(lomo_predictions(frame))
    errors = errors[errors["method"].isin(["elo", "easy_accuracy"])].rename(
        columns={"mae": "value"},
    )
    errors["metric"] = "mae"

    # Reindexing gives rank and value rows one stable schema without a bespoke
    # result class or another output table.
    columns = list(dict.fromkeys([*ranks.columns, *errors.columns]))
    return pd.concat(
        [ranks.reindex(columns=columns), errors.reindex(columns=columns)],
        ignore_index=True,
    )


def quartile_analysis(
    q1_elos: pd.DataFrame,
    quartile_accuracies: pd.DataFrame,
    coverage: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Use Q1 to predict each later quartile and their pooled accuracy."""
    metrics, _ = partition_analysis(
        q1_elos,
        quartile_accuracies,
        coverage,
        predictor_bin="Q1",
        target_bins=("Q2", "Q3", "Q4", "Q2–Q4"),
    )
    return metrics.rename(columns={"target_bin": "target_quartile"})


def partition_analysis(
    predictor_elos: pd.DataFrame,
    accuracies: pd.DataFrame,
    coverage: pd.DataFrame | None = None,
    *,
    predictor_bin: str,
    target_bins: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate one easiest-bin predictor against each later bin separately."""
    from scores import analysis_frame

    partition = (
        "halves"
        if predictor_bin == "easy"
        else {
            "T": "tertile",
            "Q": "quartile",
            "P": "quintile",
        }.get(predictor_bin[:1])
    )
    if partition is None:
        raise ValueError(f"unknown predictor bin: {predictor_bin}")

    outputs = []
    prediction_outputs = []
    for target_bin in target_bins:
        frame = analysis_frame(
            predictor_elos,
            accuracies,
            coverage,
            predictor_subset=predictor_bin,
            target_subset=target_bin,
        )
        frame["partition"] = partition
        frame["predictor_bin"] = predictor_bin
        frame["target_bin"] = target_bin
        outputs.append(_endpoint_metrics(frame))
        prediction_outputs.append(lomo_predictions(frame))
    metrics = pd.concat(outputs, ignore_index=True)
    predictions = pd.concat(prediction_outputs, ignore_index=True)

    # Models are the replication unit for every judge-specific correlation.
    # Keeping these intervals in the table also makes the figures reproducible.
    intervals = model_bootstrap_intervals(predictions)
    if not intervals.empty:
        interval_keys = _metric_keys(predictions) + ["method", "metric"]
        metrics = metrics.merge(
            intervals,
            on=interval_keys,
            how="left",
            validate="many_to_one",
        )
    comparisons = paired_method_randomization(predictions)
    if not comparisons.empty:
        comparison_keys = _metric_keys(predictions) + ["metric"]
        metrics = metrics.merge(
            comparisons,
            on=comparison_keys,
            how="left",
            validate="many_to_one",
        )
    return metrics, predictions


def cross_benchmark_analysis(
    elos: pd.DataFrame,
    accuracies: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate GPQA→AIME and AIME→GPQA within the source Elo component."""
    outputs = []
    for source_fit, target_fit in (("gpqa", "aime"), ("aime", "gpqa")):
        source_elos = elos[elos["fit"] == source_fit]
        # The source bank supplies both candidate scores; only the hard outcome
        # crosses benchmarks. This keeps the accuracy baseline exactly matched.
        source_accuracy = accuracies[
            (accuracies["fit"] == source_fit)
            & (accuracies["subset"] == "easy")
        ].rename(columns={"accuracy": "easy_accuracy"})
        target_accuracy = accuracies[
            (accuracies["fit"] == target_fit)
            & (accuracies["subset"] == "hard")
        ].rename(columns={"accuracy": "hard_accuracy"})

        # `fit` and `component` remain those of the source predictor. The target
        # contributes outcomes only; its own Elo graph is irrelevant here.
        model_key = ["irt_model", "model"]
        frame = source_elos.merge(
            source_accuracy[model_key + ["easy_accuracy"]],
            on=model_key,
        ).merge(
            target_accuracy[model_key + ["hard_accuracy"]],
            on=model_key,
        )
        frame["direction"] = f"{source_fit}_to_{target_fit}"
        comparison_key = ["irt_model", "fit", "bank", "judge", "component"]
        frame["n_matched_models"] = frame.groupby(comparison_key)["model"].transform(
            "nunique",
        )
        outputs.append(_endpoint_metrics(frame))
    return pd.concat(outputs, ignore_index=True)


def newer_stronger_analysis(
    frame: pd.DataFrame,
    model_info: pd.DataFrame,
) -> pd.DataFrame:
    """Restrict reporting to models released later and scoring above the judge.

    Elo is fitted before this restriction. The stress test changes which
    candidates are reported, not the comparison graph used to estimate them.
    Missing dates or accuracies are counted and excluded; they are never
    imputed from model names or announcement chronology.
    """
    if model_info.duplicated(["fit", "model"]).any():
        raise ValueError("model_info must have one row per fit/model")

    outputs = []
    keys = _metric_keys(frame)
    for values, group in frame.groupby(keys, dropna=False, sort=True):
        values = values if isinstance(values, tuple) else (values,)
        identity = dict(zip(keys, values))
        fit_name = identity["fit"]
        judge = identity["judge"]
        judge_info = model_info[
            (model_info["fit"] == fit_name) & (model_info["model"] == judge)
        ]
        candidates = group.merge(
            model_info.rename(columns={
                "release": "candidate_release",
                "full_accuracy": "candidate_full_accuracy",
            }),
            on=["fit", "model"],
            how="left",
        )

        # Missing metadata is an exclusion category, never something inferred
        # from a model name or silently filled from another endpoint.
        judge_metadata_missing = len(judge_info) != 1
        if judge_metadata_missing:
            judge_release = pd.NaT
            judge_accuracy = np.nan
        else:
            judge_release = pd.to_datetime(
                judge_info.iloc[0]["release"],
                errors="coerce",
            )
            judge_accuracy = judge_info.iloc[0]["full_accuracy"]

        candidate_release = pd.to_datetime(
            candidates["candidate_release"],
            errors="coerce",
        )
        missing_release = candidate_release.isna()
        missing_accuracy = candidates["candidate_full_accuracy"].isna()
        not_newer = ~missing_release & candidate_release.le(judge_release)
        not_stronger = (
            ~missing_accuracy
            & candidates["candidate_full_accuracy"].le(judge_accuracy)
        )
        keep = ~(missing_release | missing_accuracy | not_newer | not_stronger)
        if (
            judge_metadata_missing
            or pd.isna(judge_release)
            or pd.isna(judge_accuracy)
        ):
            keep[:] = False

        filtered = candidates[keep][group.columns]
        if filtered.empty:
            # Preserve an explicit ineligible metric row rather than dropping
            # the judge/component from the output without explanation.
            filtered = pd.DataFrame([{
                **identity,
                "model": "",
                "elo": np.nan,
                "easy_accuracy": np.nan,
                "hard_accuracy": np.nan,
            }])

        result = _endpoint_metrics(filtered)
        result["included_models"] = "|".join(sorted(candidates.loc[keep, "model"]))
        result["n_candidates_before"] = len(candidates)
        result["n_candidates_after"] = int(keep.sum())
        result["excluded_missing_release"] = int(missing_release.sum())
        result["excluded_missing_accuracy"] = int(missing_accuracy.sum())
        result["excluded_not_newer"] = int(not_newer.sum())
        result["excluded_not_stronger"] = int(not_stronger.sum())
        result["excluded_missing_judge_metadata"] = int(judge_metadata_missing)
        outputs.append(result)
    return pd.concat(outputs, ignore_index=True)
