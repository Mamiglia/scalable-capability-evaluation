"""Within-item Spearman correlations between transcript features and capability."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _model_item_means(transcripts, dictionary, min_models):
    if min_models < 2:
        raise ValueError("min_models must be at least 2")
    features = dictionary.feature.tolist()
    required = {"bench", "item", "model", "accuracy"} | set(features)
    missing = required - set(transcripts)
    if missing:
        raise ValueError(f"transcript table missing columns: {sorted(missing)}")
    capability_counts = transcripts.groupby(
        ["bench", "item", "model"], sort=False
    ).accuracy.nunique(dropna=False)
    if (capability_counts > 1).any():
        raise ValueError("accuracy varies within a benchmark-item-model group")

    by_model = (transcripts.groupby(["bench", "item", "model"], as_index=False,
                                    sort=False)[["accuracy"] + features]
                .mean())
    groups = dictionary.set_index("feature").group
    return features, groups, by_model


def capability_correlations(transcripts, dictionary, min_models=5):
    """Return one feature correlation per benchmark item.

    Repeated correct rollouts are averaged within model and item before ranks
    are computed, so each model contributes once to each item.
    """
    features, groups, by_model = _model_item_means(
        transcripts, dictionary, min_models)
    rows = []
    for (bench, item), frame in by_model.groupby(["bench", "item"], sort=False):
        capability_rank = frame.accuracy.rank(method="average")
        feature_ranks = frame[features].rank(method="average")
        n_models = frame[features].notna().sum()
        variable = feature_ranks.nunique(dropna=True) > 1
        eligible = variable & (n_models >= min_models)
        rho = pd.Series(np.nan, index=features, dtype=float)
        if capability_rank.nunique(dropna=True) > 1:
            rho.loc[eligible] = feature_ranks.loc[:, eligible].corrwith(capability_rank)
        rows.append(pd.DataFrame({
            "bench": bench, "item": item, "feature": features,
            "group": [groups[feature] for feature in features],
            "n_models": n_models.to_numpy(), "spearman_rho": rho.to_numpy(),
            "min_models": min_models,
        }))
    return pd.concat(rows, ignore_index=True)


def benchmark_correlations(transcripts, dictionary, min_models=5):
    """Pooled within-item rank correlation for each benchmark and feature."""
    features, groups, by_model = _model_item_means(
        transcripts, dictionary, min_models)
    rows = []
    for bench, benchmark in by_model.groupby("bench", sort=False):
        for feature in features:
            x_parts, y_parts, n_items = [], [], 0
            for _, item in benchmark.groupby("item", sort=False):
                valid = item[[feature, "accuracy"]].dropna()
                if (len(valid) < min_models or valid[feature].nunique() < 2
                        or valid.accuracy.nunique() < 2):
                    continue
                x = valid[feature].rank(method="average").to_numpy(dtype=float)
                y = valid.accuracy.rank(method="average").to_numpy(dtype=float)
                x_parts.append((x - x.mean()) / x.std())
                y_parts.append((y - y.mean()) / y.std())
                n_items += 1
            rho = (float(np.mean(np.concatenate(x_parts) * np.concatenate(y_parts)))
                   if x_parts else np.nan)
            rows.append({"bench": bench, "feature": feature,
                         "group": groups[feature], "within_item_rank_rho": rho,
                         "n_items": n_items,
                         "n_observations": sum(len(x) for x in x_parts),
                         "min_models": min_models})
    return pd.DataFrame(rows)


def _summary(frame, keys):
    valid = frame.dropna(subset=["spearman_rho"]).assign(
        abs_rho=lambda x: x.spearman_rho.abs(),
        positive=lambda x: x.spearman_rho > 0,
        negative=lambda x: x.spearman_rho < 0,
    )
    return (valid.groupby(keys, as_index=False, sort=False)
            .agg(n_items=("spearman_rho", "size"),
                 median_item_rho=("spearman_rho", "median"),
                 mean_item_rho=("spearman_rho", "mean"),
                 mean_abs_item_rho=("abs_rho", "mean"),
                 fraction_positive=("positive", "mean"),
                 fraction_negative=("negative", "mean")))


def summarise(by_item):
    """Return overall and per-benchmark summaries of non-missing item rhos."""
    by_benchmark = _summary(by_item, ["bench", "feature", "group"])
    overall = _summary(by_item, ["feature", "group"])
    balanced = (by_benchmark.groupby(["feature", "group"], as_index=False)
                .agg(n_benchmarks=("bench", "size"),
                     benchmark_balanced_rho=("median_item_rho", "mean")))
    overall = overall.merge(balanced, on=["feature", "group"], validate="one_to_one")
    overall["abs_benchmark_balanced_rho"] = overall.benchmark_balanced_rho.abs()
    overall = overall.sort_values("abs_benchmark_balanced_rho", ascending=False)
    by_benchmark = by_benchmark.sort_values(
        ["bench", "median_item_rho"], ascending=[True, False])
    return overall.reset_index(drop=True), by_benchmark.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("transcript_features", type=Path)
    parser.add_argument("feature_dictionary", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--min-models", type=int, default=5)
    args = parser.parse_args()

    outputs = ["by-item.parquet", "by-benchmark.parquet", "summary.parquet",
               "summary.csv", "settings.parquet"]
    existing = [args.output_dir / name for name in outputs
                if (args.output_dir / name).exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite {existing[0]}")
    dictionary = pd.read_parquet(args.feature_dictionary)
    features = dictionary.feature.tolist()
    transcripts = pd.read_parquet(
        args.transcript_features,
        columns=["bench", "item", "model", "accuracy"] + features,
    )
    by_item = capability_correlations(transcripts, dictionary, args.min_models)
    overall, by_benchmark = summarise(by_item)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    by_item.to_parquet(args.output_dir / "by-item.parquet", index=False)
    by_benchmark.to_parquet(args.output_dir / "by-benchmark.parquet", index=False)
    overall.to_parquet(args.output_dir / "summary.parquet", index=False)
    overall.to_csv(args.output_dir / "summary.csv", index=False)
    pd.DataFrame([{
        "capability": "within-benchmark model accuracy",
        "unit": "benchmark-item-model",
        "repeat_rule": "mean feature across canonical-correct repeats",
        "correlation": "Spearman",
        "min_models": args.min_models,
        "interpretation": "conditional on the item having been answered correctly",
    }]).to_parquet(args.output_dir / "settings.parquet", index=False)
    print(f"items={by_item[['bench', 'item']].drop_duplicates().shape[0]} "
          f"features={by_item.feature.nunique()} correlations={len(by_item)}")


if __name__ == "__main__":
    main()
