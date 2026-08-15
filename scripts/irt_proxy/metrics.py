"""Evaluate easy-item predictors against hard-item outcomes.

Every endpoint is calculated on a matched candidate set within one comparison
component. Undefined small-sample results stay in the tables with a reason
instead of silently disappearing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

BOOTSTRAP_RESAMPLES = 5000


def _metric_keys(table: pd.DataFrame) -> list[str]:
    """Return the columns that identify one independent comparison problem."""
    candidates = [
        "analysis",
        "irt_model",
        "fit",
        "bank",
        "partition",
        "predictor_bin",
        "target_bin",
        "direction",
        "target_quartile",
        "judge",
        "component",
    ]
    return [column for column in candidates if column in table]


def _coverage_fields(group: pd.DataFrame) -> dict:
    """Carry constant provenance fields into each derived endpoint row."""
    candidates = [
        "n_response_models",
        "n_graph_models",
        "n_component_models",
        "component_models",
        "n_rounds",
        "n_calls",
        "n_unique_items",
        "n_judges",
        "n_matched_models",
    ]
    return {
        column: group[column].iloc[0]
        for column in candidates
        if column in group
    }


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

        tau_samples = np.array([
            kendalltau(predictor[index], observed[index]).statistic
            for index in indices
        ])
        error = np.abs(prediction - observed)
        samples = {
            "tau_b": tau_samples[np.isfinite(tau_samples)],
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


def rank_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute tau-b for Elo and easy accuracy on one shared model set."""
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
            # scipy's default variant is tau-b, which corrects for the accuracy
            # ties common in the small AIME candidate set.
            tau = (
                kendalltau(matched[predictor], matched["hard_accuracy"]).statistic
                if n_models >= 3
                else np.nan
            )
            rows.append({
                **identity,
                **coverage,
                "method": method,
                "tau_b": tau,
                "n_models": n_models,
                "reason": reason,
            })
    return pd.DataFrame(rows)


def _linear_predict(x_train, y_train, x_test) -> float:
    """Least-squares line with the probability-scale prediction clipped."""
    design = np.column_stack([np.ones(len(x_train)), np.asarray(x_train, float)])
    intercept, slope = np.linalg.lstsq(
        design,
        np.asarray(y_train, float),
        rcond=None,
    )[0]
    prediction = np.clip(intercept + slope * x_test, 0.0, 1.0)
    return round(float(prediction), 12)


def lomo_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    """Calibrate Elo and easy accuracy without seeing the held-out hard target.

    The IRT split itself is fixed; "leave one model out" applies only to the
    scalar calibration line. Mean and identity rows are diagnostics, not
    alternative learned models.
    """
    rows = []
    keys = _metric_keys(frame)
    methods = ("elo", "easy_accuracy", "mean", "identity")
    for values, group in frame.groupby(keys, dropna=False, sort=True):
        values = values if isinstance(values, tuple) else (values,)
        identity = dict(zip(keys, values))
        matched = group.dropna(
            subset=["elo", "easy_accuracy", "hard_accuracy"],
        ).drop_duplicates("model").reset_index(drop=True)
        coverage = _coverage_fields(group)
        n_models = len(matched)
        exploratory = identity.get("fit") == "aime"

        if n_models < 4:
            # LOMO needs at least three training models after holding one out.
            # Emit explicit rows so small components remain visible in reports.
            for method in methods:
                rows.append({
                    **identity,
                    **coverage,
                    "model": "",
                    "method": method,
                    "predictor": np.nan,
                    "prediction": np.nan,
                    "observed": np.nan,
                    "n_models": n_models,
                    "exploratory": exploratory,
                    "reason": "fewer than 4 matched models",
                })
            continue

        for held_index, held_out in matched.iterrows():
            # Drop the whole held-out row before either calibration is fitted;
            # its hard accuracy is never available to the regression.
            training = matched.drop(index=held_index)
            predictions = {
                "elo": _linear_predict(
                    training["elo"],
                    training["hard_accuracy"],
                    held_out["elo"],
                ),
                "easy_accuracy": _linear_predict(
                    training["easy_accuracy"],
                    training["hard_accuracy"],
                    held_out["easy_accuracy"],
                ),
                # These two simple references make it easier to tell whether a
                # fitted line is helping in a five-model component.
                "mean": round(float(training["hard_accuracy"].mean()), 12),
                "identity": float(np.clip(held_out["easy_accuracy"], 0.0, 1.0)),
            }
            predictor_values = {
                "elo": held_out["elo"],
                "easy_accuracy": held_out["easy_accuracy"],
                "mean": np.nan,
                "identity": held_out["easy_accuracy"],
            }
            for method, prediction in predictions.items():
                rows.append({
                    **identity,
                    **coverage,
                    "model": held_out["model"],
                    "method": method,
                    "predictor": predictor_values[method],
                    "prediction": prediction,
                    "observed": held_out["hard_accuracy"],
                    "n_models": n_models,
                    "exploratory": exploratory,
                    "reason": "",
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
    ranks = rank_metrics(frame).rename(columns={"tau_b": "value"})
    ranks["metric"] = "tau_b"

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
    """Use the Q1 predictor bank against Q2, Q3, and Q4 accuracy separately."""
    metrics, _ = partition_analysis(
        q1_elos,
        quartile_accuracies,
        coverage,
        predictor_bin="Q1",
        target_bins=("Q2", "Q3", "Q4"),
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

    # The plotted endpoint is the pooled consensus estimate. Bootstrap models
    # inside each matched comparison component and save the intervals beside
    # that estimate, so the figure is reproducible from the result table.
    consensus = predictions[predictions["judge"] == "consensus"]
    intervals = model_bootstrap_intervals(consensus)
    if not intervals.empty:
        interval_keys = _metric_keys(predictions) + ["method", "metric"]
        metrics = metrics.merge(
            intervals,
            on=interval_keys,
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
