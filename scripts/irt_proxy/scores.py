"""Build item-level outcomes and component-aware easy-item Elo scores.

Bradley-Terry strengths have an arbitrary origin, so an Elo from one connected
component is never joined to or ranked against an Elo from another component.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
ELO_DIR = ROOT / "scripts" / "analyses" / "judge-elo"
sys.path.insert(0, str(ELO_DIR))
from elo import collect_pair_wins, components, fit_bt  # noqa: E402

PREDICTOR_ACCURACY_THRESHOLD = 0.90
MIN_PREDICTOR_MODELS = 4
MIN_JUDGE_MODELS_BY_FIT = {"gpqa": 8}


def item_votes(votes: pd.DataFrame) -> pd.DataFrame:
    """Average presentation orders into one source/item/judge/pair round.

    A disagreement between the two counterbalanced presentations becomes 0.5,
    rather than two contradictory observations. The source prefix keeps a
    repeated pilot judgment distinct from the corresponding expansion round.
    """
    # Parse failures are useful loader diagnostics, but they are not ties.
    valid = votes[votes["parse_ok"]] if "parse_ok" in votes else votes
    round_identity = [
        "source",
        "benchmark",
        "item_id",
        "judge",
        "major",
        "minor",
    ]
    rounds = (
        valid.groupby(round_identity, as_index=False, sort=False)
        .agg(picked_major=("picked_major", "mean"), n_calls=("n_calls", "sum"))
    )
    rounds["vote_id"] = rounds["source"] + ":" + rounds["item_id"].astype(str)
    return rounds


def filter_judges_above_chance(
    rounds: pd.DataFrame,
    metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep judges agreeing with the higher-ECI model on over 50% of pairs.

    Only pairs with external ECI values for both candidates define judge
    inclusion. GPQA accuracy is the downstream outcome, so using it here would
    leak the answer into judge selection. Once a judge qualifies, all of that
    judge's item rounds remain available for Elo fitting.
    """
    eci_rows = metadata.dropna(subset=["eci"])[["model", "eci"]].drop_duplicates()
    conflicts = eci_rows.groupby("model")["eci"].nunique()
    conflicts = conflicts[conflicts > 1]
    if not conflicts.empty:
        raise ValueError(f"conflicting ECI values for {conflicts.index.tolist()}")
    eci = eci_rows.drop_duplicates("model").set_index("model")["eci"]

    judged = rounds.copy()
    left = judged[["major", "minor"]].min(axis=1)
    right = judged[["major", "minor"]].max(axis=1)
    judged["_pair"] = left + "\0" + right
    judged["_major_eci"] = judged["major"].map(eci)
    judged["_minor_eci"] = judged["minor"].map(eci)
    reference = judged[
        judged["_major_eci"].notna()
        & judged["_minor_eci"].notna()
        & judged["_major_eci"].ne(judged["_minor_eci"])
    ].copy()
    reference["agreed_with_higher_eci"] = reference["picked_major"].where(
        reference["_major_eci"] > reference["_minor_eci"],
        1 - reference["picked_major"],
    )

    # Average within a candidate pair first so a pair with more eligible items
    # cannot decide whether its judge clears the chance threshold.
    pair_rates = (
        reference.groupby(["benchmark", "judge", "_pair"], as_index=False)
        .agg(pair_agreement=("agreed_with_higher_eci", "mean"))
    )
    diagnostics = (
        pair_rates.groupby(["benchmark", "judge"], as_index=False)
        .agg(
            mean_pair_agreement_with_higher_eci=("pair_agreement", "mean"),
            n_eci_model_pairs=("_pair", "size"),
        )
    )
    reference_counts = (
        reference.groupby(["benchmark", "judge"], as_index=False)
        .agg(
            item_round_agreement_with_higher_eci=(
                "agreed_with_higher_eci", "mean",
            ),
            n_eci_item_rounds=("agreed_with_higher_eci", "size"),
        )
    )
    total_counts = (
        judged.groupby(["benchmark", "judge"], as_index=False)
        .size()
        .rename(columns={"size": "n_total_item_rounds"})
    )
    diagnostics = total_counts.merge(
        diagnostics,
        on=["benchmark", "judge"],
        how="left",
        validate="one_to_one",
    ).merge(
        reference_counts,
        on=["benchmark", "judge"],
        how="left",
        validate="one_to_one",
    )
    diagnostics[["n_eci_model_pairs", "n_eci_item_rounds"]] = diagnostics[[
        "n_eci_model_pairs", "n_eci_item_rounds",
    ]].fillna(0).astype(int)
    diagnostics["included"] = (
        diagnostics["mean_pair_agreement_with_higher_eci"] > 0.5
    )

    included = diagnostics.loc[
        diagnostics["included"], ["benchmark", "judge"]
    ]
    kept = rounds.merge(
        included,
        on=["benchmark", "judge"],
        how="inner",
        validate="many_to_one",
    )
    return kept, diagnostics


def model_accuracy(responses: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    """Measure single-epoch accuracy with equal item/year weights.

    The caller must supply one canonical response per model/item. Benchmark-year
    means are then averaged within a fit, preventing GPQA's much larger item bank
    from dominating joint accuracy.
    """
    answered = responses[
        responses["answer"].fillna("").astype(str).str.strip().ne("")
    ]
    cell = ["model", "benchmark", "item"]
    if answered.duplicated(cell).any():
        raise ValueError(
            "model_accuracy requires one canonical response per model/item"
        )
    selected_items = items[
        ["irt_model", "fit", "benchmark", "item", "subset"]
    ]
    joined = answered.merge(selected_items, on=["benchmark", "item"], how="inner")
    analysis_key = ["irt_model", "fit", "subset", "model"]

    # A single item or large benchmark never gets extra weight merely because
    # its stratum contains more questions.
    item_scores = (
        joined.groupby(analysis_key + ["benchmark", "item"], as_index=False)
        .agg(correct=("correct", "mean"))
    )
    stratum_scores = (
        item_scores.groupby(analysis_key + ["benchmark"], as_index=False)
        .agg(stratum_accuracy=("correct", "mean"), n_items=("item", "size"))
    )
    accuracy = (
        stratum_scores.groupby(analysis_key, as_index=False)
        .agg(accuracy=("stratum_accuracy", "mean"), n_items=("n_items", "sum"))
    )
    attempts = joined.groupby(analysis_key, as_index=False).size().rename(
        columns={"size": "n_attempts"},
    )
    return accuracy.merge(attempts, on=analysis_key, validate="one_to_one")


def select_predictor_models(
    accuracies: pd.DataFrame,
    candidates: pd.DataFrame,
    subset: str,
    *,
    threshold: float = PREDICTOR_ACCURACY_THRESHOLD,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep only models strictly above the requested predictor-bank accuracy."""
    keys = ["irt_model", "fit"]
    candidate_rows = candidates[keys + ["model"]].drop_duplicates()
    available = (
        accuracies[accuracies["subset"] == subset]
        .merge(candidate_rows, on=keys + ["model"], how="inner")
    )

    eligible_rows = []
    selection_rows = []
    for values, group in available.groupby(keys, sort=True):
        eligible = group[group["accuracy"] > threshold]
        eligible_rows.append(eligible[keys + ["model"]])
        selection_rows.append({
            **dict(zip(keys, values)),
            "bank": subset,
            "predictor_accuracy_threshold": threshold,
            "n_graph_candidates_before_threshold": group["model"].nunique(),
            "n_threshold_candidates": eligible["model"].nunique(),
            "threshold_target_models": MIN_PREDICTOR_MODELS,
            "threshold_target_met": eligible["model"].nunique()
            >= MIN_PREDICTOR_MODELS,
        })

    eligible = (
        pd.concat(eligible_rows, ignore_index=True)
        if eligible_rows
        else pd.DataFrame(columns=keys + ["model"])
    )
    return eligible, pd.DataFrame(selection_rows)


def fit_easy_elos(
    votes: pd.DataFrame,
    splits: pd.DataFrame,
    *,
    bank: str = "easy",
    eligible_models: pd.DataFrame | None = None,
    eligible_judges: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit every post-filter component for each judge and for consensus.

    Components are recomputed *after* filtering to the selected item bank. An
    edge that exists only on hard items cannot connect two easy-item Elo scales.
    """
    rounds = votes.copy() if "vote_id" in votes else item_votes(votes)
    if bank == "easy":
        selected_items = splits[splits["half"] == "easy"]
    else:
        bin_column = {
            "T": "tertile",
            "Q": "quartile",
            "P": "quintile",
        }.get(bank[:1])
        if bin_column is None:
            raise ValueError(f"unknown item bank: {bank}")
        selected_items = splits[splits[bin_column] == bank]
    selected_items = selected_items[
        ["irt_model", "fit", "benchmark", "item"]
    ].rename(columns={"item": "item_id"})
    selected = rounds.merge(
        selected_items,
        on=["benchmark", "item_id"],
        how="inner",
    )
    if eligible_models is not None:
        # Filter pair members before finding components. Otherwise an excluded
        # weak model could remain as a hidden bridge between retained models.
        keys = ["irt_model", "fit"]
        allowed = eligible_models[keys + ["model"]].drop_duplicates()
        for side in ("major", "minor"):
            selected = selected.merge(
                allowed.rename(columns={"model": side}),
                on=keys + [side],
                how="inner",
                validate="many_to_one",
            )
    if eligible_judges is not None:
        selected = selected.merge(
            eligible_judges[["irt_model", "fit", "judge"]].drop_duplicates(),
            on=["irt_model", "fit", "judge"],
            how="inner",
            validate="many_to_one",
        )

    # Keep per-judge fits for the uncertainty analysis. Consensus is a separate
    # pooled fit, not the numerical average of judge-specific Elo values.
    per_judge = selected.copy()
    per_judge["original_judge"] = per_judge["judge"]
    per_judge["item_id"] = per_judge["vote_id"]

    # Consensus pools judge/item rounds, not judge means. Adding the original
    # judge to item_id prevents collect_pair_wins from averaging different
    # judges who happened to see the same pair and item.
    consensus = per_judge.copy()
    consensus["item_id"] = consensus["vote_id"] + ":" + consensus["judge"]
    consensus["judge"] = "consensus"
    fit_rows = pd.concat([per_judge, consensus], ignore_index=True)

    group_columns = ["irt_model", "fit", "judge"]
    grouped_wins = collect_pair_wins(
        fit_rows.to_dict("records"),
        group_columns,
    )

    # Anchor every judge's child components to the pooled easy-item graph.
    # Without this, size-based IDs can swap the local and Epoch-imported scales
    # between judges even though the fitted strengths themselves are separate.
    parent_components = {}
    for (irt_model, fit_name, judge), pair_wins in grouped_wins.items():
        if judge != "consensus":
            continue
        parent_components[(irt_model, fit_name)] = sorted(
            components(pair_wins),
            key=lambda members: (-len(members), tuple(sorted(members))),
        )

    elo_rows = []
    coverage_rows = []
    for group_key, pair_wins in sorted(grouped_wins.items()):
        irt_model, fit_name, judge = group_key
        group_rows = fit_rows[
            (fit_rows["irt_model"] == irt_model)
            & (fit_rows["fit"] == fit_name)
            & (fit_rows["judge"] == judge)
        ]
        graph_models = {model for pair in pair_wins for model in pair}

        # A judge may see only fragments of a pooled component. Sort those
        # fragments within their parent, then give them IDs such as `0:0` and
        # `0:1`; no fragment is allowed to borrow another parent's Elo scale.
        graph_components = sorted(
            components(pair_wins),
            key=lambda member_set: (-len(member_set), tuple(sorted(member_set))),
        )
        anchored_components = []
        for members in graph_components:
            parents = [
                parent_id
                for parent_id, parent in enumerate(
                    parent_components[(irt_model, fit_name)],
                )
                if members <= parent
            ]
            if len(parents) != 1:
                raise ValueError(
                    f"component does not map to one pooled graph: {sorted(members)}"
                )
            anchored_components.append((parents[0], members))
        anchored_components.sort(
            key=lambda entry: (
                entry[0],
                -len(entry[1]),
                tuple(sorted(entry[1])),
            ),
        )

        child_count = {}
        for parent_id, members in anchored_components:
            child_id = child_count.get(parent_id, 0)
            child_count[parent_id] = child_id + 1
            component_id = f"{parent_id}:{child_id}"
            component_wins = {
                pair: result
                for pair, result in pair_wins.items()
                if pair[0] in members and pair[1] in members
            }
            strengths = fit_bt(component_wins)
            component_rows = group_rows[
                group_rows["major"].isin(members)
                & group_rows["minor"].isin(members)
            ]
            # `collect_pair_wins` already applies the repository's round
            # weighting. Reuse its totals so coverage describes the fit itself.
            rounds_by_model = {
                model: sum(
                    result[1]
                    for pair, result in component_wins.items()
                    if model in pair
                )
                for model in members
            }

            for model, strength in strengths.items():
                elo_rows.append({
                    "irt_model": irt_model,
                    "fit": fit_name,
                    "bank": bank,
                    "judge": judge,
                    "component": component_id,
                    "parent_component": parent_id,
                    "model": model,
                    "elo": strength,
                    "n_model_rounds": rounds_by_model[model],
                })

            # Small components remain in coverage even when they cannot support
            # tau or LOMO. Silence would look like deliberate model exclusion.
            coverage_rows.append({
                "irt_model": irt_model,
                "fit": fit_name,
                "bank": bank,
                "judge": judge,
                "component": component_id,
                "parent_component": parent_id,
                "n_graph_models": len(graph_models),
                "n_component_models": len(members),
                "component_models": "|".join(sorted(members)),
                "n_rounds": sum(result[1] for result in component_wins.values()),
                "n_calls": int(component_rows["n_calls"].sum()),
                "n_unique_items": component_rows["vote_id"].nunique(),
                "n_judges": component_rows["original_judge"].nunique(),
                "eligible_rank": len(members) >= 3,
                "eligible_lomo": len(members) >= 4,
                "reason": "" if len(members) >= 3 else "fewer than 3 models",
            })
    return pd.DataFrame(elo_rows), pd.DataFrame(coverage_rows)


def fit_saturated_elos(
    votes: pd.DataFrame,
    splits: pd.DataFrame,
    accuracies: pd.DataFrame,
    *,
    bank: str,
    threshold: float = PREDICTOR_ACCURACY_THRESHOLD,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Choose the bank-specific accuracy cutoff, then refit its induced graph."""
    unfiltered, _ = fit_easy_elos(votes, splits, bank=bank)
    candidates = unfiltered[unfiltered["judge"] == "consensus"][[
        "irt_model", "fit", "model",
    ]]
    eligible, selection = select_predictor_models(
        accuracies,
        candidates,
        bank,
        threshold=threshold,
    )
    provisional, _ = fit_easy_elos(
        votes,
        splits,
        bank=bank,
        eligible_models=eligible,
    )
    judge_counts = (
        provisional[provisional["judge"] != "consensus"]
        .groupby(["irt_model", "fit", "judge"], as_index=False)["model"]
        .nunique()
        .rename(columns={"model": "n_judge_models"})
    )
    judge_counts["minimum_judge_models"] = (
        judge_counts["fit"].map(MIN_JUDGE_MODELS_BY_FIT).fillna(0).astype(int)
    )
    judge_counts["included_by_coverage"] = (
        judge_counts["n_judge_models"] >= judge_counts["minimum_judge_models"]
    )
    eligible_judges = judge_counts[judge_counts["included_by_coverage"]]

    judge_selection_rows = []
    for (irt_model, fit_name), group in judge_counts.groupby(
        ["irt_model", "fit"], sort=True
    ):
        excluded = group.loc[~group["included_by_coverage"], "judge"]
        judge_selection_rows.append({
            "irt_model": irt_model,
            "fit": fit_name,
            "bank": bank,
            "minimum_judge_models": int(group["minimum_judge_models"].iloc[0]),
            "n_judges_before_coverage": len(group),
            "n_judges_after_coverage": int(group["included_by_coverage"].sum()),
            "coverage_excluded_judges": "|".join(sorted(excluded)),
        })
    judge_selection = pd.DataFrame(judge_selection_rows)

    elos, coverage = fit_easy_elos(
        votes,
        splits,
        bank=bank,
        eligible_models=eligible,
        eligible_judges=eligible_judges,
    )
    selection_key = ["irt_model", "fit", "bank"]
    selection = selection.merge(
        judge_selection,
        on=selection_key,
        how="left",
        validate="one_to_one",
    )
    return (
        elos.merge(selection, on=selection_key, validate="many_to_one"),
        coverage.merge(selection, on=selection_key, validate="many_to_one"),
    )


def analysis_frame(
    elos: pd.DataFrame,
    accuracies: pd.DataFrame,
    coverage: pd.DataFrame | None = None,
    *,
    predictor_subset: str = "easy",
    target_subset: str = "hard",
) -> pd.DataFrame:
    """Inner-join Elo, predictor accuracy, and target accuracy by component.

    The inner join is the matched baseline rule: Elo and easy accuracy are both
    evaluated on precisely the models for which all three quantities exist.
    """
    model_key = ["irt_model", "fit", "model"]
    predictor = accuracies[accuracies["subset"] == predictor_subset].rename(
        columns={"accuracy": "easy_accuracy"},
    )
    target = accuracies[accuracies["subset"] == target_subset].rename(
        columns={"accuracy": "hard_accuracy"},
    )
    frame = elos.merge(
        predictor[model_key + ["easy_accuracy"]],
        on=model_key,
        how="inner",
    ).merge(
        target[model_key + ["hard_accuracy"]],
        on=model_key,
        how="inner",
    )

    if coverage is not None and not coverage.empty:
        # Coverage is component metadata. A many-to-one validation catches an
        # accidental duplicate before it can multiply matched model rows.
        coverage_key = ["irt_model", "fit", "bank", "judge", "component"]
        extra_columns = [
            column
            for column in coverage
            if column not in coverage_key and column not in frame
        ]
        frame = frame.merge(
            coverage[coverage_key + extra_columns],
            on=coverage_key,
            how="left",
            validate="many_to_one",
        )

    comparison_key = ["irt_model", "fit", "bank", "judge", "component"]
    frame["n_matched_models"] = frame.groupby(comparison_key)["model"].transform(
        "nunique",
    )
    return frame
