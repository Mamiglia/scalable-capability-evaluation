"""Export canonical verdict-dissection tables; normalised outputs are EDA-only."""
import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


TRANSCRIPT_KEYS = ["bench", "model", "item", "epoch", "log"]
KEYS = TRANSCRIPT_KEYS + ["sample_epoch", "content_hash"]
SURFACE_FEATURES = [
    "log_chars", "n_words", "personal_pronouns_count", "hedging_count", "backtracking_count",
    "uncertainty_count", "confidence_count", "questions_count", "verification_count",
    "connectives_count", "contrast_count", "numbered_steps_count", "bullets_count",
    "code_chars", "latex_chars",
]
OUTPUTS = (
    "feature-dictionary.parquet", "manifest.parquet",
    "raw/identity.parquet", "raw/model-benchmark-accuracy.parquet", "raw/surface-textual.parquet",
    "raw/context-predictability.parquet", "raw/transcript-features.parquet",
    "normalised/transcript-features-global.parquet",
    "normalised/transcript-features-within-benchmark.parquet",
    "normalised/normalisation-parameters.parquet", "paired/judging-rounds.parquet",
    "paired/raw-features.parquet", "paired/normalised-features.parquet",
)


def _require(frame, columns, name):
    missing = set(columns) - set(frame)
    if missing:
        raise ValueError(f"{name} missing columns: {sorted(missing)}")


def accuracy_table(identity):
    return (identity.groupby(["model", "bench"], as_index=False, sort=True)
            .agg(correct_n=("correct", "sum"), total_n=("correct", "size"))
            .assign(accuracy=lambda x: x.correct_n / x.total_n))


def normalise(frame, features, scope, within_benchmark=False):
    """Return z-scores and their moments; constants retain a unit denominator."""
    out, rows = frame.copy(), []
    groups = frame.groupby("bench", dropna=False, sort=False) if within_benchmark else [(None, frame)]
    for bench, group in groups:
        mean = group[features].mean()
        sd = group[features].std().replace(0, 1.0).fillna(1.0)
        out.loc[group.index, features] = (group[features] - mean) / sd
        rows.extend({"scope": scope, "benchmark": bench, "feature": feature,
                     "mean": mean[feature], "sd": sd[feature]} for feature in features)
    return out, pd.DataFrame(rows)


def paired_features(rounds, transcript_features, features):
    """Join each kept round to its two canonical-correct transcripts."""
    base = (transcript_features[KEYS + features].sort_values("sample_epoch")
            .drop_duplicates(TRANSCRIPT_KEYS))
    sides = {"model": "model_{side}", "epoch": "epoch_{side}", "log": "log_{side}",
             "sample_epoch": "sample_epoch_{side}", "content_hash": "content_hash_{side}"}
    left = base.rename(columns={key: value.format(side="left") for key, value in sides.items()}
                       | {f: f"{f}_left" for f in features})
    right = base.rename(columns={key: value.format(side="right") for key, value in sides.items()}
                        | {f: f"{f}_right" for f in features})
    pairs = rounds.merge(left, on=["bench", "item", "model_left", "epoch_left", "log_left"],
                         how="inner", validate="many_to_one")
    pairs = pairs.merge(right, on=["bench", "item", "model_right", "epoch_right", "log_right"],
                         how="inner", validate="many_to_one")
    numeric = [f for f in features if pd.api.types.is_numeric_dtype(base[f])
               and not pd.api.types.is_bool_dtype(base[f])]
    for feature in numeric:
        pairs[f"d_{feature}"] = pairs[f"{feature}_left"] - pairs[f"{feature}_right"]
    return pairs


def _write_all(root, tables):
    paths = {name: root / name for name in OUTPUTS}
    existing = [path for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite {existing[0]}")
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_parquet(paths[name], index=False)


def feature_dictionary(raw_features, surface, context):
    rows = []
    for group, source, features in (("surface-textual", "raw/surface-textual.parquet", surface),
                                    ("context-predictability", "raw/context-predictability.parquet", context)):
        rows.extend({"feature": feature, "group": group, "dtype": str(raw_features[feature].dtype),
                     "source_table": source, "description": f"Raw {group} feature."} for feature in features)
    return pd.DataFrame(rows)


def export_tables(store, verdicts, raw_features, output_root, content_hash=None, contract_hash=None):
    """Build and write all publication-friendly tables below ``output_root``."""
    _require(store, KEYS + ["correct", "is_canonical"], "transcript_store")
    _require(raw_features, KEYS, "raw_features_with_surprisal")
    _require(verdicts, ["bench", "item", "model_left", "model_right", "epoch_left", "epoch_right",
                        "log_left", "log_right"], "verdicts")
    identity = store.copy()
    canonical_identity = identity.loc[identity.is_canonical].copy()
    accuracy = accuracy_table(canonical_identity)
    raw_columns = [column for column in raw_features if column not in KEYS]
    surface = [column for column in SURFACE_FEATURES if column in raw_columns]
    context = [column for column in raw_columns if column.startswith("surprisal_")]
    canonical = canonical_identity.loc[canonical_identity.correct].merge(
        raw_features, on=KEYS, how="inner", validate="one_to_one")
    transcript = canonical.merge(accuracy, on=["model", "bench"], how="left", validate="many_to_one")
    numeric = [column for column in raw_columns if pd.api.types.is_numeric_dtype(transcript[column])
               and not pd.api.types.is_bool_dtype(transcript[column])]
    global_features, global_params = normalise(transcript, numeric, "global")
    within_features, within_params = normalise(transcript, numeric, "within_benchmark", True)
    raw_pairs = paired_features(verdicts, transcript, raw_columns)
    normalised_pairs = paired_features(verdicts, within_features, raw_columns)
    tables = {
        "feature-dictionary.parquet": feature_dictionary(raw_features, surface, context),
        "manifest.parquet": pd.DataFrame([{
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "transcript_rows": len(transcript), "judging_rounds": len(verdicts),
            "raw_pair_rows": len(raw_pairs), "content_hash": content_hash, "contract_hash": contract_hash,
        }]),
        "raw/identity.parquet": identity,
        "raw/model-benchmark-accuracy.parquet": accuracy,
        "raw/surface-textual.parquet": raw_features[KEYS + surface],
        "raw/context-predictability.parquet": raw_features[KEYS + context],
        "raw/transcript-features.parquet": transcript,
        "normalised/transcript-features-global.parquet": global_features,
        "normalised/transcript-features-within-benchmark.parquet": within_features,
        "normalised/normalisation-parameters.parquet": pd.concat([global_params, within_params], ignore_index=True),
        "paired/judging-rounds.parquet": verdicts,
        "paired/raw-features.parquet": raw_pairs,
        "paired/normalised-features.parquet": normalised_pairs.assign(normalisation_scope="within_benchmark_eda"),
    }
    _write_all(Path(output_root), tables)
    return tables


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("transcript_store", type=Path)
    parser.add_argument("verdicts", type=Path)
    parser.add_argument("raw_features_with_surprisal", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--content-hash")
    parser.add_argument("--contract-hash")
    args = parser.parse_args()
    tables = export_tables(pd.read_parquet(args.transcript_store), pd.read_parquet(args.verdicts),
                           pd.read_parquet(args.raw_features_with_surprisal), args.output_root,
                           content_hash=args.content_hash, contract_hash=args.contract_hash)
    print(f"transcripts={len(tables['raw/transcript-features.parquet'])} rounds={len(tables['paired/judging-rounds.parquet'])}")


if __name__ == "__main__":
    main()
