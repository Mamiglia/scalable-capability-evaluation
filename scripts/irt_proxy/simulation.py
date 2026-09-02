"""Empirically subsample the collected easy-item comparison data.

This module intentionally cannot fit IRT: it receives a frozen item split.
Every replicate samples observed models, observed items, or observed judgment
rounds without replacement, then refits only Bradley-Terry and LOMO calibration.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from metrics import lomo_predictions, rank_metrics, value_metrics
from scores import analysis_frame, select_predictor_models

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
ELO_DIR = ROOT / "scripts" / "analyses" / "judge-elo"
sys.path.insert(0, str(ELO_DIR))
from elo import collect_pair_wins, components, fit_bt  # noqa: E402


def _consensus_components(
    rounds: pd.DataFrame,
    fit_name: str,
) -> list[tuple[int, set[str], dict[str, float]]]:
    """Fit consensus Elo separately inside every observed graph component."""
    if rounds.empty:
        return []

    consensus = rounds.copy()
    # Different judges on one source/item/pair are distinct empirical rounds.
    # The existing collector averages rows sharing item_id, so judge identity is
    # appended before pooling them into the single consensus group.
    consensus["item_id"] = consensus["vote_id"] + ":" + consensus["judge"]
    grouped = collect_pair_wins(consensus.to_dict("records"), ["fit"])
    pair_wins = grouped.get((fit_name,), {})
    if not pair_wins:
        return []

    graph_components = sorted(
        components(pair_wins),
        key=lambda members: (-len(members), tuple(sorted(members))),
    )
    fitted = []
    for component_id, members in enumerate(graph_components):
        component_wins = {
            pair: result
            for pair, result in pair_wins.items()
            if pair[0] in members and pair[1] in members
        }
        fitted.append((component_id, members, fit_bt(component_wins)))
    return fitted


def _model_budgets(n_models: int) -> list[int]:
    """Use every feasible small-model budget and sparse large-model steps."""
    if n_models < 4:
        return []
    if n_models <= 20:
        return list(range(4, n_models + 1))
    return sorted({4, 6, 8, 10, 15, 20, n_models, *range(30, n_models, 10)})


def _item_budgets(n_items: int) -> list[int]:
    """Clip the pre-specified item grid to the observed easy bank."""
    return sorted({n_items, *(value for value in (5, 10, 20, 40, 80) if value <= n_items)})


def _round_budgets(n_rounds: int) -> list[int]:
    """Convert fractions to feasible observed round counts, always including all."""
    return sorted({
        min(n_rounds, math.ceil(fraction * n_rounds))
        for fraction in (0.10, 0.25, 0.50, 0.75, 1.00)
    })


def _evaluate_sample(
    sample: pd.DataFrame,
    fit_name: str,
    accuracies: pd.DataFrame,
    full_component: int,
) -> list[dict]:
    """Refit a sampled graph and evaluate every component it produces."""
    fitted = _consensus_components(sample, fit_name)
    if not fitted:
        return [{
            "component": f"{full_component}.none",
            "full_component": full_component,
            "component_models": "",
            "matched_component_models": "",
            "n_models": 0,
            "tau": np.nan,
            "baseline_tau": np.nan,
            "mae": np.nan,
            "baseline_mae": np.nan,
            "valid": False,
            "reason": "sample contains no comparison edge",
            "rounds": len(sample),
            "calls": int(sample["n_calls"].sum()),
        }]

    results = []
    for child_id, members, strengths in fitted:
        component = f"{full_component}.{child_id}"
        elos = pd.DataFrame([
            {
                "irt_model": "fixed_3pl",
                "fit": fit_name,
                "bank": "easy",
                "judge": "consensus",
                "component": component,
                "model": model,
                "elo": strength,
            }
            for model, strength in strengths.items()
        ])
        frame = analysis_frame(elos, accuracies)
        n_models = frame["model"].nunique()
        matched_component_models = "|".join(sorted(frame["model"].unique()))

        tau = baseline_tau = mae = baseline_mae = np.nan
        if not frame.empty:
            ranks = rank_metrics(frame).set_index("method")
            tau = ranks.loc["elo", "tau_a"]
            baseline_tau = ranks.loc["easy_accuracy", "tau_a"]
            errors = value_metrics(lomo_predictions(frame)).set_index("method")
            mae = errors.loc["elo", "mae"]
            baseline_mae = errors.loc["easy_accuracy", "mae"]

        valid = n_models >= 4 and np.isfinite(tau) and np.isfinite(mae)
        component_rows = sample[
            sample["major"].isin(members) & sample["minor"].isin(members)
        ]
        results.append({
            "component": component,
            "full_component": full_component,
            "component_models": "|".join(sorted(members)),
            "matched_component_models": matched_component_models,
            "n_models": n_models,
            "tau": tau,
            "baseline_tau": baseline_tau,
            "mae": mae,
            "baseline_mae": baseline_mae,
            "valid": bool(valid),
            "reason": "" if valid else "fewer than 4 matched models or undefined metric",
            "rounds": len(component_rows),
            "calls": int(component_rows["n_calls"].sum()),
        })
    return results


def monte_carlo_budgets(
    rounds: pd.DataFrame,
    splits: pd.DataFrame,
    accuracies: pd.DataFrame,
    *,
    resamples: int = 1000,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate model/item/round requirements from observed consensus data.

    One axis changes at a time. A sampled graph is allowed to disconnect; every
    resulting component is saved, including components too small for a metric.
    Invalid draws are not retried because their frequency is part of the sample-
    requirement result.
    """
    random = np.random.default_rng(seed)
    fixed_splits = splits[
        (splits["irt_model"] == "fixed_3pl") & (splits["half"] == "easy")
    ]
    easy_items = fixed_splits[
        ["fit", "benchmark", "item"]
    ].rename(columns={"item": "item_id"})
    # This join is the firewall around the frozen IRT bank: later sampling code
    # sees judgment rounds and item labels, but no IRT fitter or response matrix.
    attached = rounds.merge(easy_items, on=["benchmark", "item_id"], how="inner")
    attached["item_key"] = attached["benchmark"] + ":" + attached["item_id"]

    raw_rows = []
    for fit_name in sorted(attached["fit"].unique()):
        fit_accuracy = accuracies[
            (accuracies["irt_model"] == "fixed_3pl")
            & (accuracies["fit"] == fit_name)
        ]
        unfiltered_rounds = attached[attached["fit"] == fit_name]
        graph_models = sorted(
            set(unfiltered_rounds["major"]) | set(unfiltered_rounds["minor"])
        )
        candidates = pd.DataFrame({
            "irt_model": "fixed_3pl",
            "fit": fit_name,
            "model": graph_models,
        })
        eligible, selection = select_predictor_models(
            fit_accuracy,
            candidates,
            "easy",
        )
        eligible_models = set(eligible["model"])
        selection_fields = selection.iloc[0].to_dict()
        fit_rounds = unfiltered_rounds[
            unfiltered_rounds["major"].isin(eligible_models)
            & unfiltered_rounds["minor"].isin(eligible_models)
        ].reset_index(drop=True)
        full_graph = _consensus_components(fit_rounds, fit_name)
        accuracy_models = set(
            fit_accuracy.pivot_table(
                index="model", columns="subset", values="accuracy",
            ).dropna(subset=["easy", "hard"]).index
        ) & eligible_models

        for full_component, members, _ in full_graph:
            component_rounds = fit_rounds[
                fit_rounds["major"].isin(members)
                & fit_rounds["minor"].isin(members)
            ].reset_index(drop=True)
            matched_models = sorted(members & accuracy_models)
            item_keys = sorted(component_rounds["item_key"].unique())

            # The full-data result is the stability target. It is computed once,
            # outside the replicate loop, and still does not touch IRT.
            reference_rows = _evaluate_sample(
                component_rounds,
                fit_name,
                fit_accuracy,
                full_component,
            )
            reference = reference_rows[0]

            budgets = {
                "models": _model_budgets(len(matched_models)),
                "items": _item_budgets(len(item_keys)),
                "rounds": _round_budgets(len(component_rounds)),
            }
            for axis, axis_budgets in budgets.items():
                for budget in axis_budgets:
                    for replicate in range(resamples):
                        # Change one resource at a time. Starting from the full
                        # component makes the three budget axes interpretable.
                        sample = component_rounds
                        sampled_models = set(matched_models)
                        if axis == "models":
                            chosen = set(random.choice(
                                matched_models,
                                size=budget,
                                replace=False,
                            ))
                            sampled_models = chosen
                            sample = sample[
                                sample["major"].isin(chosen)
                                & sample["minor"].isin(chosen)
                            ]
                        elif axis == "items":
                            chosen = set(random.choice(
                                item_keys,
                                size=budget,
                                replace=False,
                            ))
                            sample = sample[sample["item_key"].isin(chosen)]
                        else:
                            chosen_rows = random.choice(
                                len(sample),
                                size=budget,
                                replace=False,
                            )
                            sample = sample.iloc[chosen_rows]

                        for result in _evaluate_sample(
                            sample,
                            fit_name,
                            fit_accuracy,
                            full_component,
                        ):
                            covers_sampled_models = (
                                result["matched_component_models"]
                                == "|".join(sorted(sampled_models))
                            )
                            # These are pre-declared practical tolerances around
                            # the full collected-data estimate, not p-values.
                            stable = (
                                covers_sampled_models
                                and result["valid"]
                                and reference["valid"]
                                and abs(result["tau"] - reference["tau"]) <= 0.10
                                and abs(result["mae"] - reference["mae"]) <= 0.03
                            )
                            raw_rows.append({
                                "irt_model": "fixed_3pl",
                                "fit": fit_name,
                                "judge": "consensus",
                                "axis": axis,
                                "budget": budget,
                                "replicate": replicate,
                                "full_component_models": reference[
                                    "matched_component_models"
                                ],
                                "covers_sampled_models": covers_sampled_models,
                                "stable": bool(stable),
                                **selection_fields,
                                **result,
                            })

    raw = pd.DataFrame(raw_rows).sort_values(
        ["fit", "full_component", "axis", "budget", "replicate", "component"],
        kind="stable",
    ).reset_index(drop=True)
    summary = _summarize(raw, resamples)
    return raw, summary


def _summarize(raw: pd.DataFrame, resamples: int) -> pd.DataFrame:
    """Summarize empirical bands without calling them confidence intervals."""
    rows = []
    group_columns = [
        "irt_model",
        "fit",
        "judge",
        "full_component",
        "axis",
        "budget",
    ]
    for values, group in raw.groupby(group_columns, sort=True):
        identity = dict(zip(group_columns, values))
        # A disconnected child stays in `raw`, but it cannot stand in for the
        # sampled candidate set when estimating a resource requirement.
        complete = group[group["covers_sampled_models"]]
        valid = complete[complete["valid"]]

        def interval(column):
            if valid.empty:
                return (np.nan, np.nan, np.nan)
            return tuple(np.percentile(valid[column], [50, 5, 95]))

        tau_median, tau_p5, tau_p95 = interval("tau")
        mae_median, mae_p5, mae_p95 = interval("mae")
        rows.append({
            **identity,
            "component": str(identity["full_component"]),
            "component_models": group["full_component_models"].iloc[0],
            **{
                column: group[column].iloc[0]
                for column in (
                    "bank",
                    "predictor_accuracy_threshold",
                    "n_graph_candidates_before_threshold",
                    "n_threshold_candidates",
                    "threshold_target_models",
                    "threshold_target_met",
                )
                if column in group
            },
            "tau_median": tau_median,
            "tau_p5": tau_p5,
            "tau_p95": tau_p95,
            "mae_median": mae_median,
            "mae_p5": mae_p5,
            "mae_p95": mae_p95,
            "valid_fraction": valid["replicate"].nunique() / resamples,
            "stable_fraction": group.loc[group["stable"], "replicate"].nunique() / resamples,
            "elo_beats_baseline_probability": (
                (valid["tau"] > valid["baseline_tau"]).mean()
                if not valid.empty
                else np.nan
            ),
        })

    summary = pd.DataFrame(rows)
    recommendation_key = [
        "irt_model", "fit", "judge", "full_component", "axis"
    ]
    recommendations = {}
    for values, group in summary.groupby(recommendation_key, sort=True):
        # Invalid disconnected draws count against the 80% rule; retrying them
        # would make sparse budgets look artificially reliable.
        eligible = group[
            (group["valid_fraction"] >= 0.80)
            & (group["stable_fraction"] >= 0.80)
        ]
        recommendations[values] = (
            int(eligible["budget"].min()) if not eligible.empty else np.nan
        )
    summary["recommended_budget"] = [
        recommendations[tuple(row[column] for column in recommendation_key)]
        for _, row in summary.iterrows()
    ]
    return summary
