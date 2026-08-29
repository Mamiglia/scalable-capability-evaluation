"""Compute univariate feature associations with verdicts and model capability."""
import numpy as np
import pandas as pd

def verdict_correlations(pairs: pd.DataFrame, dictionary: pd.DataFrame) -> pd.DataFrame:
    """Correlate each left-minus-right feature with whether the judge picked left.

    This is computed separately for every judge and benchmark. ``pick_left`` is
    1 when the left transcript was selected and 0 otherwise, so ordinary Pearson
    correlation with the continuous feature difference is the point-biserial
    coefficient. Positive values mean that more of the feature on the left is
    associated with choosing left; this is not a unique or causal effect.
    """
    required = {"bench", "judge", "pick_left"} | set(dictionary.column)
    if missing := required - set(pairs):
        raise ValueError(f"pair table missing columns: {sorted(missing)}")
    rows = []
    for (bench, judge), group in pairs.groupby(["bench", "judge"], sort=False):
        for feature in dictionary.itertuples():
            valid = group[[feature.column, "pick_left"]].dropna()
            value = (valid[feature.column].corr(valid.pick_left)
                     if valid[feature.column].nunique() > 1 and valid.pick_left.nunique() > 1
                     else np.nan)
            rows.append({"bench": bench, "judge": judge, "feature": feature.feature,
                         "block": feature.block, "correlation": value,
                         "n_complete": len(valid)})
    return pd.DataFrame(rows)


def capability_correlations(
    transcripts: pd.DataFrame,
    dictionary: pd.DataFrame,
    min_models: int = 5,
) -> pd.DataFrame:
    """Return pooled within-item Spearman correlations with model accuracy.

    Correct rollout repeats are first averaged within model and item. Each
    item's feature and accuracy values are then ranked and standardised before
    pooling, preventing easier items or repeated rollouts from dominating.
    """
    if min_models < 2:
        raise ValueError("min_models must be at least 2")
    if not (dictionary.feature == dictionary.column).all():
        raise ValueError("capability correlations require transcript-level columns")
    features = dictionary.feature.tolist()
    required = {"bench", "item", "model", "model_accuracy"} | set(features)
    if missing := required - set(transcripts):
        raise ValueError(f"transcript table missing columns: {sorted(missing)}")

    # A model can have multiple correct rollout transcripts for the same item.
    by_model = (transcripts.groupby(["bench", "item", "model"], as_index=False,
                                    sort=False)[["model_accuracy"] + features]
                .mean())
    blocks = dictionary.set_index("feature").block
    rows: list[dict[str, object]] = []
    for bench, benchmark in by_model.groupby("bench", sort=False):
        for feature in features:
            feature_ranks: list[np.ndarray] = []
            accuracy_ranks: list[np.ndarray] = []
            n_items = 0
            for _, item in benchmark.groupby("item", sort=False):
                valid = item[[feature, "model_accuracy"]].dropna()
                if (len(valid) < min_models or valid[feature].nunique() < 2
                        or valid.model_accuracy.nunique() < 2):
                    continue
                x = valid[feature].rank(method="average").to_numpy(float)
                y = valid.model_accuracy.rank(method="average").to_numpy(float)
                feature_ranks.append((x - x.mean()) / x.std())
                accuracy_ranks.append((y - y.mean()) / y.std())
                n_items += 1
            rho = (float(np.mean(np.concatenate(feature_ranks)
                                 * np.concatenate(accuracy_ranks)))
                   if feature_ranks else np.nan)
            rows.append({"bench": bench, "feature": feature,
                         "block": blocks[feature], "within_item_rank_rho": rho,
                         "n_items": n_items,
                         "n_observations": sum(len(x) for x in feature_ranks),
                         "min_models": min_models})
    return pd.DataFrame(rows)


def block_correlations(pairs: pd.DataFrame) -> pd.DataFrame:
    """Summarise absolute feature correlations for each unordered block pair.

    Diagonal cells average correlations between *different* features in the
    same block; they are within-block redundancy, not feature self-correlation.
    """
    required = {"bench", "block_a", "block_b", "abs_correlation"}
    if missing := required - set(pairs):
        raise ValueError(f"correlation table missing columns: {sorted(missing)}")
    frame = pairs.copy()
    ordered = np.sort(frame[["block_a", "block_b"]].astype(str).to_numpy(), axis=1)
    frame[["block_a", "block_b"]] = ordered
    return (frame.groupby(["bench", "block_a", "block_b"], sort=False, as_index=False)
            .agg(mean_abs_correlation=("abs_correlation", "mean"),
                 median_abs_correlation=("abs_correlation", "median"),
                 n_pairs=("abs_correlation", "size")))
