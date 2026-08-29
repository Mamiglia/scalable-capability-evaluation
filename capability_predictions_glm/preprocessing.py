"""Fixed item quartiles and held-out-model row selection."""

import numpy as np
import pandas as pd


KEYS = ["bench", "model", "item"]


def pool_aime(frame: pd.DataFrame) -> pd.DataFrame:
    """Pool AIME years without colliding their numeric item identifiers."""
    out = frame.copy()
    years = out.bench.astype(str).str.fullmatch(r"aime\d{4}")
    out.loc[years, "item"] = out.loc[years, "bench"].astype(str) + "/" + out.loc[years, "item"].astype(str)
    out.loc[years, "bench"] = "aime"
    return out


def _model_item_rates(responses: pd.DataFrame) -> pd.DataFrame:
    required = {*KEYS, "correct"}
    if missing := required - set(responses):
        raise ValueError(f"response table missing columns: {sorted(missing)}")
    rows = pool_aime(responses)
    if "is_canonical" in rows:
        rows = rows[rows.is_canonical]
    return (rows.groupby(KEYS, as_index=False, sort=False).correct.mean()
            .rename(columns={"correct": "correct_rate"}))


def item_quartiles(responses: pd.DataFrame) -> pd.DataFrame:
    """Freeze Q1–Q4 from equal-model item solve rates in the canonical pool."""
    rates = _model_item_rates(responses)
    items = (rates.groupby(["bench", "item"], as_index=False).correct_rate.mean()
             .rename(columns={"correct_rate": "solve_rate"}))
    output = []
    for _, group in items.groupby("bench", sort=False):
        ordered = group.sort_values(["solve_rate", "item"], ascending=[False, True], kind="stable").copy()
        if len(ordered) < 4:
            raise ValueError(f"benchmark {ordered.bench.iloc[0]!r} has fewer than four items")
        for number, indices in enumerate(np.array_split(np.arange(len(ordered)), 4), 1):
            part = ordered.iloc[indices].copy()
            part["quartile"] = f"Q{number}"
            output.append(part)
    return pd.concat(output, ignore_index=True)[["bench", "item", "solve_rate", "quartile"]]


def quartile_accuracy(responses: pd.DataFrame, quartiles: pd.DataFrame) -> pd.DataFrame:
    """Equal-item model accuracy for each fixed item quartile."""
    rates = _model_item_rates(responses)
    joined = rates.merge(quartiles[["bench", "item", "quartile"]],
                         on=["bench", "item"], how="inner", validate="many_to_one")
    return (joined.groupby(["bench", "model", "quartile"], as_index=False).correct_rate.mean()
            .rename(columns={"correct_rate": "accuracy"}))


def _q1(table: pd.DataFrame, quartiles: pd.DataFrame) -> pd.DataFrame:
    required = {"bench", "item"}
    if missing := required - set(table):
        raise ValueError(f"table missing columns: {sorted(missing)}")
    q1 = quartiles.loc[quartiles.quartile == "Q1", ["bench", "item"]]
    return table.merge(q1, on=["bench", "item"], how="inner", validate="many_to_one")


def split_transcripts(
    transcripts: pd.DataFrame,
    quartiles: pd.DataFrame,
    held_out_models: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Q1 transcript rows split by whole held-out models."""
    source = _q1(transcripts, quartiles)
    held = source.model.isin(held_out_models)
    return source[~held].reset_index(drop=True), source[held].reset_index(drop=True)


def split_pairs(
    pairs: pd.DataFrame,
    quartiles: pd.DataFrame,
    held_out_models: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Q1 pairs: train on two seen models; test whenever either model is held out."""
    required = {"model_left", "model_right"}
    if missing := required - set(pairs):
        raise ValueError(f"pair table missing columns: {sorted(missing)}")
    source = _q1(pairs, quartiles)
    held = source.model_left.isin(held_out_models) | source.model_right.isin(held_out_models)
    return source[~held].reset_index(drop=True), source[held].reset_index(drop=True)


def attach_direct_target(
    transcripts: pd.DataFrame,
    accuracies: pd.DataFrame,
    target_quartile: str,
) -> pd.DataFrame:
    """Attach one model×benchmark target-quartile accuracy to transcript rows."""
    target = (accuracies[accuracies.quartile == target_quartile]
              [["bench", "model", "accuracy"]]
              .rename(columns={"accuracy": "target_accuracy"}))
    return transcripts.merge(target, on=["bench", "model"], how="inner", validate="many_to_one")


def _transcript_id(frame: pd.DataFrame, side: str) -> pd.Series:
    columns = [f"model_{side}"] + [
        f"{name}_{side}" for name in ("epoch", "log", "content_hash")
        if f"{name}_{side}" in frame
    ]
    return frame[columns].astype(str).agg("\x1f".join, axis=1)


def deduplicate_feature_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    """Keep one orientation of each underlying transcript pair, regardless of judge."""
    required = {"bench", "item", "model_left", "model_right"}
    if missing := required - set(pairs):
        raise ValueError(f"pair table missing columns: {sorted(missing)}")
    left, right = _transcript_id(pairs, "left"), _transcript_id(pairs, "right")
    keyed = pairs.assign(_pair_a=np.minimum(left, right), _pair_b=np.maximum(left, right))
    return (keyed.drop_duplicates(["bench", "item", "_pair_a", "_pair_b"], keep="first")
            .drop(columns=["_pair_a", "_pair_b"]).reset_index(drop=True))


def attach_pair_target(
    pairs: pd.DataFrame,
    accuracies: pd.DataFrame,
    target_quartile: str,
) -> pd.DataFrame:
    """Attach target-quartile accuracy to both sides and drop capability ties."""
    target = accuracies[accuracies.quartile == target_quartile][["bench", "model", "accuracy"]]
    output = pairs.drop(columns=["accuracy_left", "accuracy_right", "accuracy_difference",
                                 "left_stronger"], errors="ignore").copy()
    for side in ("left", "right"):
        side_target = target.rename(columns={"model": f"model_{side}", "accuracy": f"accuracy_{side}"})
        output = output.merge(side_target, on=["bench", f"model_{side}"],
                              how="inner", validate="many_to_one")
    output = output[output.accuracy_left != output.accuracy_right].copy()
    output["left_stronger"] = (output.accuracy_left > output.accuracy_right).astype(int)
    return output.reset_index(drop=True)
