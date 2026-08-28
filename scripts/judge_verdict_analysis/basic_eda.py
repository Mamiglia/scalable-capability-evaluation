"""Basic EDA for selected verdict-dissection feature blocks."""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CONFIG
from .data import load_features
from .preprocessing import CorrelationMethod, feature_correlations


MAX_CORRELATION_FEATURES = CONFIG.max_correlation_features


def feature_summary(table: pd.DataFrame, dictionary: pd.DataFrame) -> pd.DataFrame:
    """Summarise distribution, missingness and constancy by benchmark/feature."""
    rows: list[dict[str, object]] = []
    for bench, group in table.groupby("bench", sort=False):
        for feature in dictionary.itertuples():
            values = group[feature.column]
            present = values.dropna()
            rows.append({
                "bench": bench,
                "block": feature.block,
                "feature": feature.feature,
                "column": feature.column,
                "n_rows": len(values),
                "n_present": len(present),
                "missing_rate": values.isna().mean(),
                "n_unique": present.nunique(),
                "is_constant": present.nunique() <= 1,
                "zero_rate": (present == 0).mean() if len(present) else np.nan,
                "mean": present.mean() if len(present) else np.nan,
                "sd": present.std() if len(present) else np.nan,
                "min": present.min() if len(present) else np.nan,
                "median": present.median() if len(present) else np.nan,
                "max": present.max() if len(present) else np.nan,
            })
    return pd.DataFrame(rows)


def block_summary(features: pd.DataFrame) -> pd.DataFrame:
    """Aggregate feature-level quality metrics to benchmark-by-block rows."""
    return (features.groupby(["bench", "block"], sort=False, as_index=False)
            .agg(n_features=("feature", "size"),
                 n_constant=("is_constant", "sum"),
                 mean_missing_rate=("missing_rate", "mean"),
                 mean_zero_rate=("zero_rate", "mean")))


def _fingerprint(values: pd.Series) -> bytes:
    """Hash one column, including missing-value locations, for duplicate lookup."""
    hashes = pd.util.hash_pandas_object(values, index=False).to_numpy()
    return hashlib.sha256(hashes.tobytes()).digest()


def _opposites(a: pd.Series, b: pd.Series) -> bool:
    """Return whether two columns are exact sign flips with matching NaNs."""
    return bool(((a == -b) | (a.isna() & b.isna())).all())


def duplicate_features(table: pd.DataFrame, dictionary: pd.DataFrame) -> pd.DataFrame:
    """Find variable columns that are exactly equal or sign-flipped by benchmark.

    Constant columns are omitted here because ``feature_summary`` already reports
    them; including constants would create a combinatorial list of zero/zero pairs.
    """
    names = dictionary.set_index("column")[["feature", "block"]]
    rows: list[tuple[str, str, str, str]] = []
    for bench, group in table.groupby("bench", sort=False):
        seen: dict[bytes, list[str]] = {}
        for column in dictionary.column:
            values = group[column].reset_index(drop=True)
            if values.nunique(dropna=True) <= 1:
                continue  # constants are already reported in feature-summary.csv
            fingerprint = _fingerprint(values)
            opposite = _fingerprint(-values)
            for other in seen.get(fingerprint, []):
                if values.equals(group[other].reset_index(drop=True)):
                    rows.append((bench, other, column, "identical"))
            for other in seen.get(opposite, []):
                if _opposites(values, group[other].reset_index(drop=True)):
                    rows.append((bench, other, column, "sign_flipped"))
            seen.setdefault(fingerprint, []).append(column)
    return pd.DataFrame([
        {"bench": bench,
         "feature_a": names.loc[a, "feature"], "block_a": names.loc[a, "block"],
         "feature_b": names.loc[b, "feature"], "block_b": names.loc[b, "block"],
         "relation": relation}
        for bench, a, b, relation in rows
    ], columns=["bench", "feature_a", "block_a", "feature_b", "block_b", "relation"])


def correlation_pairs(
    table: pd.DataFrame,
    dictionary: pd.DataFrame,
    threshold: float = CONFIG.correlation_threshold,
    method: CorrelationMethod = CONFIG.correlation_method,
) -> pd.DataFrame:
    """Return within-benchmark feature pairs passing an absolute-correlation cutoff."""
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    if len(dictionary) > MAX_CORRELATION_FEATURES:
        raise ValueError(
            f"correlations are limited to {MAX_CORRELATION_FEATURES} features; select fewer blocks")
    return feature_correlations(table, dictionary, method).query(
        "abs_correlation >= @threshold").reset_index(drop=True)


def main() -> None:
    """Write reproducible summary, duplicate and optional correlation CSVs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path, nargs="?", default=CONFIG.basic_eda_dir)
    parser.add_argument("--level", choices=["transcript", "pair"], default="pair")
    parser.add_argument("--blocks", nargs="+", default=list(CONFIG.feature_blocks))
    parser.add_argument("--splits", nargs="+", default=list(CONFIG.splits))
    parser.add_argument("--correlations", nargs="*", choices=["pearson", "spearman"], default=[])
    parser.add_argument("--threshold", type=float, default=CONFIG.correlation_threshold)
    parser.add_argument("--config", type=Path, default=CONFIG.design_matrix_config)
    args = parser.parse_args()

    table, dictionary = load_features(
        args.level, args.blocks, splits=args.splits, config=args.config)
    features = feature_summary(table, dictionary)
    outputs: dict[str, pd.DataFrame] = {
        "feature-summary.csv": features,
        "block-summary.csv": block_summary(features),
        "duplicate-features.csv": duplicate_features(table, dictionary),
    }
    for method in args.correlations:
        outputs[f"correlation-pairs-{method}.csv"] = correlation_pairs(
            table, dictionary, args.threshold, method)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in outputs.items():
        frame.to_csv(args.output_dir / name, index=False)
    print(f"level={args.level} rows={len(table)} features={len(dictionary)} blocks={len(args.blocks)} -> {args.output_dir}")


if __name__ == "__main__":
    main()
