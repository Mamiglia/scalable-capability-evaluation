"""Fit the declared IRT item banks and turn difficulty into analysis splits.

This module only decides which response cell represents a model/item and which
items are easy or hard. It does not know which models were pair-judged; that
graph belongs to the prediction layer.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from irt.data import encode, pick_epoch
from irt.fitting import fit, summary

FIT_BENCHMARKS = {
    "gpqa": {"gpqa"},
    "aime": {"aime2025", "aime2026"},
    "joint": {"gpqa", "aime2025", "aime2026"},
}


def irt_cells(responses: pd.DataFrame) -> pd.DataFrame:
    """Choose one canonical scored cell, then reuse IRT's epoch fallback.

    Source selection must precede epoch selection. Otherwise epoch 1 from an
    imported log could beat epoch 2 from a locally rescored source, undoing the
    parser correction merely because it occurred in a later attempt.
    """
    answered = responses[
        responses["answer"].fillna("").astype(str).str.strip().ne("")
    ].copy()
    if answered.empty:
        return answered

    cell = ["model", "benchmark", "item"]
    best_priority = answered.groupby(cell)["source_priority"].transform("max")
    canonical_source = answered[answered["source_priority"] == best_priority]
    return pick_epoch(canonical_source, epoch=1)


def fit_parameters(
    responses: pd.DataFrame,
    *,
    floors: Mapping[str, float],
    warmup: int = 1000,
    samples: int = 1000,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict]]:
    """Run the three declared fits and retain item/model posterior draws.

    The floor is an item-level input derived from answer format. It is never a
    sampled parameter, so adding more response models cannot turn this into a
    free-guessing 3PL.
    """
    cells = irt_cells(responses)
    difficulty_rows, capability_rows, posteriors = [], [], {}
    for irt_model in ("fixed_3pl", "2pl"):
        for fit_name, benchmarks in FIT_BENCHMARKS.items():
            data = encode(cells[cells["benchmark"].isin(benchmarks)])

            # Index floors by encoded item order. Unknown benchmarks raise here
            # instead of quietly receiving the much smaller AIME floor.
            item_floors = np.array([
                floors[benchmark] for benchmark, _ in data.items
            ])
            model_arguments = (
                {"lower_asymptote": item_floors}
                if irt_model == "fixed_3pl"
                else {}
            )
            posterior = fit(
                data,
                model=irt_model,
                warmup=warmup,
                samples=samples,
                seed=seed,
                **model_arguments,
            )

            b = summary(data.items, posterior["b"])
            difficulty = pd.DataFrame(
                b.pop("name").tolist(),
                columns=["benchmark", "item"],
            )
            difficulty.insert(0, "fit", fit_name)
            difficulty.insert(0, "irt_model", irt_model)
            difficulty["lower_asymptote"] = item_floors if irt_model == "fixed_3pl" else 0.0
            difficulty["difficulty"] = b["mean"]
            difficulty["difficulty_p5"] = b["p5"]
            difficulty["difficulty_p95"] = b["p95"]
            difficulty["discrimination"] = np.asarray(posterior["a"]).mean(axis=0)
            difficulty_rows.append(difficulty)

            theta = summary(data.models, posterior["theta"])
            capability_rows.append(pd.DataFrame({
                "irt_model": irt_model, "fit": fit_name,
                "model": theta["name"], "theta": theta["mean"],
                "theta_p5": theta["p5"], "theta_p95": theta["p95"],
            }))
            posteriors[f"{irt_model}__{fit_name}"] = {
                "theta": np.asarray(posterior["theta"]),
                "a": np.asarray(posterior["a"]),
                "b": np.asarray(posterior["b"]),
                "floor": item_floors if irt_model == "fixed_3pl" else np.zeros_like(item_floors),
                "models": list(data.models),
                "items": list(data.items),
            }
    return (
        pd.concat(difficulty_rows, ignore_index=True),
        pd.concat(capability_rows, ignore_index=True),
        posteriors,
    )


def fit_difficulties(responses: pd.DataFrame, **kwargs) -> pd.DataFrame:
    return fit_parameters(responses, **kwargs)[0]


def assign_splits(difficulties: pd.DataFrame) -> pd.DataFrame:
    """Assign equal-count difficulty bins within each benchmark-year.

    A pooled numerical cut would let GPQA's difficulty scale decide which AIME
    items count as easy. Stratifying makes the joint fit benchmark-balanced by
    construction. Stable item-ID ordering resolves exact difficulty ties.
    """
    group_columns = ["irt_model", "fit", "benchmark"]
    splits = difficulties.sort_values(
        group_columns + ["difficulty", "item"],
        kind="stable",
    ).reset_index(drop=True)
    splits["half"] = ""
    for column in ("tertile", "quartile", "quintile"):
        splits[column] = ""

    for _, group in splits.groupby(group_columns, sort=False):
        indices = group.index.to_numpy()
        midpoint = len(indices) // 2

        # For an odd stratum, the pre-specified middle item belongs to hard.
        splits.loc[indices[:midpoint], "half"] = "easy"
        splits.loc[indices[midpoint:], "half"] = "hard"

        # array_split keeps bins within one item of equal size. This matters for
        # AIME, where a percentile interpolation rule would move whole items.
        for column, prefix, n_bins in (
            ("tertile", "T", 3),
            ("quartile", "Q", 4),
            ("quintile", "P", 5),
        ):
            for number, bin_indices in enumerate(np.array_split(indices, n_bins), 1):
                splits.loc[bin_indices, column] = f"{prefix}{number}"
    return splits
