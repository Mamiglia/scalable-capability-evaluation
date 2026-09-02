"""Design-matched chance distributions for the prediction endpoints."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm, t

from mc_cost.ranking import _metric_keys, lomo_predictions
from metrics import kendall_tau_a, rank_metrics, value_metrics
from scores import fit_easy_elos


def _paired_t_less(differences: np.ndarray) -> float:
    """One-sided paired t-test, including the zero-variance edge case."""
    if len(differences) < 2:
        return np.nan
    standard_deviation = differences.std(ddof=1)
    if standard_deviation == 0:
        return 0.0 if differences.mean() < 0 else 1.0
    statistic = differences.mean() / (standard_deviation / np.sqrt(len(differences)))
    return float(t.cdf(statistic, len(differences) - 1))


def _analytic_group(group: pd.DataFrame, identity: dict) -> list[dict]:
    """Approximate no-signal rank and error baselines for one matched frame."""
    join_keys = ["model"]
    if identity.get("component") == "pooled":
        join_keys.insert(0, "component")
    mean_rows = group[group["method"] == "mean"][
        join_keys + ["prediction"]
    ].drop_duplicates(join_keys).rename(columns={"prediction": "null_prediction"})
    rows = []
    for method in ("elo", "easy_accuracy"):
        method_rows = group[group["method"] == method][
            join_keys + ["predictor", "prediction", "observed"]
        ].drop_duplicates(join_keys)
        matched = method_rows.merge(
            mean_rows,
            on=join_keys,
            how="inner",
            validate="one_to_one",
        ).dropna()
        if len(matched) < 3:
            continue

        pair_total = 0
        tau_total = 0.0
        variance_terms = []
        components = (
            matched.groupby("component", sort=False)
            if "component" in matched
            else [(identity.get("component"), matched)]
        )
        for _, component in components:
            count = len(component)
            pair_count = count * (count - 1) // 2
            if not pair_count:
                continue
            tau_total += kendall_tau_a(
                component["predictor"], component["observed"],
            ) * pair_count
            pair_total += pair_count
            variance_terms.append((
                pair_count,
                2 * (2 * count + 5) / (9 * count * (count - 1)),
            ))
        if pair_total:
            observed_tau = tau_total / pair_total
            tau_variance = sum(
                (pair_count / pair_total) ** 2 * variance
                for pair_count, variance in variance_terms
            )
            tau_standard_error = np.sqrt(tau_variance)
            rows.append({
                **identity,
                "method": method,
                "metric": "tau_a",
                "null_type": "asymptotic Kendall tau-a",
                "observed_value": observed_tau,
                "null_median": 0.0,
                "ci_low": max(-1.0, -1.96 * tau_standard_error),
                "ci_high": min(1.0, 1.96 * tau_standard_error),
                "p_value": float(norm.sf(observed_tau / tau_standard_error)),
                "n_randomizations": 0,
            })

        method_error = np.abs(matched["prediction"] - matched["observed"])
        null_error = np.abs(matched["null_prediction"] - matched["observed"])
        null_mean = null_error.mean()
        null_standard_error = null_error.std(ddof=1) / np.sqrt(len(null_error))
        margin = t.ppf(0.975, len(null_error) - 1) * null_standard_error
        rows.append({
            **identity,
            "method": method,
            "metric": "mae",
            "null_type": "intercept-only LOMO",
            "observed_value": method_error.mean(),
            "null_median": null_mean,
            "ci_low": max(0.0, null_mean - margin),
            "ci_high": null_mean + margin,
            "p_value": _paired_t_less((method_error - null_error).to_numpy()),
            "n_randomizations": 0,
        })
    return rows


def analytic_null_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Fast provisional nulls that avoid repeated graph refits.

    Tau uses the standard large-sample no-association variance within each Elo
    component. MAE uses the already-computed intercept-only LOMO predictor.
    """
    keys = _metric_keys(predictions)
    rows = []
    for values, group in predictions.groupby(keys, dropna=False, sort=True):
        values = values if isinstance(values, tuple) else (values,)
        rows.extend(_analytic_group(group, dict(zip(keys, values))))

    pooled_keys = [key for key in keys if key != "component"]
    for values, group in predictions.groupby(pooled_keys, dropna=False, sort=True):
        values = values if isinstance(values, tuple) else (values,)
        identity = {**dict(zip(pooled_keys, values)), "component": "pooled"}
        rows.extend(_analytic_group(group, identity))
    return pd.DataFrame(rows)


def coin_flip_rounds(rounds: pd.DataFrame, random: np.random.Generator) -> pd.DataFrame:
    """Replace each recorded presentation choice with a fair coin flip.

    A normal round has one or two successfully parsed presentation orders.
    Flipping each order separately preserves the observed counterbalancing and
    makes a two-order disagreement a half-win, exactly as in the real data.
    """
    randomized = rounds.copy()
    calls = randomized["n_calls"].fillna(1).astype(int)
    if not calls.isin((1, 2)).all():
        raise ValueError("chance analysis expects one or two calls per item round")
    randomized["picked_major"] = random.binomial(calls, 0.5) / calls
    return randomized


def _frame_metrics(frame: pd.DataFrame, method: str, replicate: int) -> list[dict]:
    """Compute tau-a and LOMO MAE for one randomized predictor."""
    keys = _metric_keys(frame)
    ranks = rank_metrics(frame)
    ranks = ranks[ranks["method"] == method]
    predictions = lomo_predictions(frame)
    errors = value_metrics(predictions)
    errors = errors[errors["method"] == method]

    rows = []
    for _, row in ranks.iterrows():
        rows.append({
            **{key: row[key] for key in keys},
            "method": method,
            "metric": "tau_a",
            "value": row["tau_a"],
            "replicate": replicate,
        })
    for _, row in errors.iterrows():
        rows.append({
            **{key: row[key] for key in keys},
            "method": method,
            "metric": "mae",
            "value": row["mae"],
            "replicate": replicate,
        })

    # The per-judge figure summarizes disconnected components without ever
    # comparing Elo values across them. Save that pooled null explicitly.
    pooled_keys = [key for key in keys if key != "component"]
    predictor = "elo" if method == "elo" else "easy_accuracy"
    for values, group in frame.groupby(pooled_keys, dropna=False, sort=True):
        values = values if isinstance(values, tuple) else (values,)
        identity = dict(zip(pooled_keys, values))
        pair_score = pair_count = 0
        for _, component in group.groupby("component", sort=False):
            valid = component.dropna(subset=[predictor, "hard_accuracy"])
            count = len(valid) * (len(valid) - 1) // 2
            if count:
                pair_score += kendall_tau_a(
                    valid[predictor], valid["hard_accuracy"],
                ) * count
                pair_count += count
        rows.append({
            **identity,
            "component": "pooled",
            "method": method,
            "metric": "tau_a",
            "value": pair_score / pair_count if pair_count else np.nan,
            "replicate": replicate,
        })

    selected = predictions[predictions["method"] == method]
    for values, group in selected.groupby(pooled_keys, dropna=False, sort=True):
        values = values if isinstance(values, tuple) else (values,)
        identity = dict(zip(pooled_keys, values))
        valid = group.dropna(subset=["prediction", "observed"])
        rows.append({
            **identity,
            "component": "pooled",
            "method": method,
            "metric": "mae",
            "value": (valid["prediction"] - valid["observed"]).abs().mean(),
            "replicate": replicate,
        })
    return rows


def _observed_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Recover the plotted endpoint values from saved model-level predictions."""
    keys = _metric_keys(predictions) + ["method"]
    rows = []
    data = predictions[predictions["method"].isin(("elo", "easy_accuracy"))]
    for values, group in data.groupby(keys, dropna=False, sort=True):
        values = values if isinstance(values, tuple) else (values,)
        identity = dict(zip(keys, values))
        valid = group.dropna(subset=["predictor", "prediction", "observed"])
        valid = valid.drop_duplicates("model")
        if len(valid) < 3:
            continue
        rows.extend((
            {
                **identity,
                "metric": "tau_a",
                "observed_value": kendall_tau_a(valid["predictor"], valid["observed"]),
            },
            {
                **identity,
                "metric": "mae",
                "observed_value": (valid["prediction"] - valid["observed"]).abs().mean(),
            },
        ))

    pooled_keys = [key for key in keys if key != "component"]
    for values, group in data.groupby(pooled_keys, dropna=False, sort=True):
        values = values if isinstance(values, tuple) else (values,)
        identity = dict(zip(pooled_keys, values))
        pair_score = pair_count = 0
        for _, component in group.groupby("component", sort=False):
            valid = component.dropna(subset=["predictor", "observed"])
            valid = valid.drop_duplicates("model")
            count = len(valid) * (len(valid) - 1) // 2
            if count:
                pair_score += kendall_tau_a(
                    valid["predictor"], valid["observed"],
                ) * count
                pair_count += count
        valid = group.dropna(subset=["prediction", "observed"])
        valid = valid.drop_duplicates(["component", "model"])
        rows.extend((
            {
                **identity,
                "component": "pooled",
                "metric": "tau_a",
                "observed_value": pair_score / pair_count if pair_count else np.nan,
            },
            {
                **identity,
                "component": "pooled",
                "metric": "mae",
                "observed_value": (valid["prediction"] - valid["observed"]).abs().mean(),
            },
        ))
    return pd.DataFrame(rows)


def _permute_accuracy(frame: pd.DataFrame, random: np.random.Generator) -> pd.DataFrame:
    """Break the model-to-easy-accuracy link within each matched component."""
    permuted = frame.copy()
    for _, group in permuted.groupby(_metric_keys(permuted), sort=False):
        permuted.loc[group.index, "easy_accuracy"] = random.permutation(
            group["easy_accuracy"].to_numpy(),
        )
    return permuted


def randomized_null_metrics(
    rounds: pd.DataFrame,
    splits: pd.DataFrame,
    observed_elos: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    resamples: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    """Run the full prediction pipeline under two matched no-signal designs.

    Elo uses fair-coin judgments on the observed graph. Easy accuracy is
    permuted across model identities. Both retain the same matched models,
    components, target accuracies, and leave-one-model-out calibration.
    """
    random = np.random.default_rng(seed)
    base = predictions[predictions["method"] == "easy_accuracy"].rename(
        columns={"predictor": "easy_accuracy", "observed": "hard_accuracy"},
    )
    frame_columns = [
        *_metric_keys(base), "model", "easy_accuracy", "hard_accuracy",
    ]
    base = base[frame_columns].drop_duplicates()

    bank_specs = []
    for values, group in observed_elos.groupby(
        ["irt_model", "fit", "bank"],
        sort=True,
    ):
        irt_model, fit_name, bank = values
        eligible_models = group[group["judge"] == "consensus"][[
            "irt_model", "fit", "model",
        ]].drop_duplicates()
        eligible_judges = group[group["judge"] != "consensus"][[
            "irt_model", "fit", "judge",
        ]].drop_duplicates()
        bank_base = base[
            (base["irt_model"] == irt_model)
            & (base["fit"] == fit_name)
            & (base["bank"] == bank)
        ]
        if not eligible_models.empty and not eligible_judges.empty and not bank_base.empty:
            bank_specs.append((bank, eligible_models, eligible_judges, bank_base))

    raw_rows = []
    join_keys = ["irt_model", "fit", "bank", "judge", "component", "model"]
    for replicate in range(resamples):
        randomized_rounds = coin_flip_rounds(rounds, random)
        for bank, eligible_models, eligible_judges, bank_base in bank_specs:
            randomized_elos, _ = fit_easy_elos(
                randomized_rounds,
                splits,
                bank=bank,
                eligible_models=eligible_models,
                eligible_judges=eligible_judges,
            )
            frame = randomized_elos[join_keys + ["elo"]].merge(
                bank_base,
                on=join_keys,
                how="inner",
                validate="one_to_many",
            )
            raw_rows.extend(_frame_metrics(frame, "elo", replicate))
            raw_rows.extend(_frame_metrics(
                _permute_accuracy(frame, random),
                "easy_accuracy",
                replicate,
            ))

    raw = pd.DataFrame(raw_rows)
    observed = _observed_metrics(predictions)
    keys = _metric_keys(raw) + ["method", "metric"]
    summary_rows = []
    for values, group in raw.groupby(keys, dropna=False, sort=True):
        values = values if isinstance(values, tuple) else (values,)
        identity = dict(zip(keys, values))
        null_values = group["value"].dropna().to_numpy()
        match = observed
        for key, value in identity.items():
            match = match[match[key].eq(value)]
        observed_value = match["observed_value"].iloc[0]
        extreme = (
            (null_values >= observed_value).sum()
            if identity["metric"] == "tau_a"
            else (null_values <= observed_value).sum()
        )
        low, high = np.percentile(null_values, [2.5, 97.5])
        summary_rows.append({
            **identity,
            "null_type": (
                "coin-flip judgments"
                if identity["method"] == "elo"
                else "permuted easy accuracy"
            ),
            "observed_value": observed_value,
            "null_median": np.median(null_values),
            "ci_low": low,
            "ci_high": high,
            "p_value": (extreme + 1) / (len(null_values) + 1),
            "n_randomizations": len(null_values),
        })
    return pd.DataFrame(summary_rows)
