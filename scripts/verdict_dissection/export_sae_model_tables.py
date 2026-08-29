"""Export compact shared-SAE transcript and pair tables for downstream models."""
import argparse
from pathlib import Path

import pandas as pd

from export_feature_tables import KEYS, paired_features


SHARED_FEATURES = [f"sae_shared_{i}" for i in range(32)]
VERDICT_COLUMNS = [
    "bench", "judge", "item", "model_left", "model_right", "epoch_left", "epoch_right",
    "log_left", "log_right", "judge_log", "judge_sample", "judge_epoch", "judge_config_hash",
    "max_length", "max_length_recorded", "pick_left", "max_length_resolution",
]


def _bench_group(bench: pd.Series) -> pd.Series:
    return bench.mask(bench.str.startswith("aime"), "aime")


def _normalise_on_development(frame: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    out, parameters = frame.copy(), []
    out[features] = out[features].astype(float)
    development = frame["split_role"].isin({"representation_train", "representation_validation"})
    for bench, group in frame.groupby("bench_group", sort=False):
        reference = frame.loc[group.index[development.loc[group.index]], features]
        if reference.empty:
            raise ValueError(f"no development rows for {bench}")
        mean = reference.mean()
        sd = reference.std().replace(0, 1.0).fillna(1.0)
        out.loc[group.index, features] = (group[features] - mean) / sd
        parameters.extend({"scope": "development_within_benchmark", "benchmark": bench,
                           "feature": feature, "mean": mean[feature], "sd": sd[feature]}
                          for feature in features)
    out["normalisation_scope"] = "development_within_benchmark"
    return out, pd.DataFrame(parameters)


def _feature_dictionary(features: list[str]) -> pd.DataFrame:
    fired = [f"{feature}_fired" for feature in features]
    return pd.DataFrame({
        "feature": features + fired,
        "group": (["transcript-sae-shared-activation"] * len(features)
                  + ["transcript-sae-shared-firing"] * len(fired)),
        "description": (["Signed shared-SAE activation for this transcript."] * len(features)
                        + ["One when the shared SAE feature fires for this transcript, else zero."] * len(fired)),
    })


def build_tables(transcripts: pd.DataFrame, verdicts: pd.DataFrame) -> dict[str, pd.DataFrame]:
    features = sorted((column for column in transcripts if column.startswith("sae_shared_")),
                      key=lambda column: int(column.rsplit("_", 1)[1]))
    if not features:
        raise ValueError("no shared SAE features")
    required = set(KEYS + ["split_role", *features])
    if missing := required - set(transcripts):
        raise ValueError(f"transcript table missing columns: {sorted(missing)}")
    valid_roles = {"representation_train", "representation_validation", "confirmation"}
    if unknown := set(transcripts["split_role"]) - valid_roles:
        raise ValueError(f"unknown split_role values: {sorted(unknown)}")

    raw = transcripts[KEYS + ["split_role", *features]].copy()
    raw.insert(len(KEYS) + 1, "bench_group", _bench_group(raw["bench"]))
    fired = [f"{feature}_fired" for feature in features]
    raw[fired] = (raw[features] != 0).astype("int8").set_axis(fired, axis=1)
    normalised, parameters = _normalise_on_development(raw, features)

    roles = raw[["bench", "item", "split_role"]].drop_duplicates()
    if roles.duplicated(["bench", "item"]).any():
        raise ValueError("split_role is inconsistent within an item")
    rounds = verdicts.merge(roles, on=["bench", "item"], how="left", validate="many_to_one")
    if rounds["split_role"].isna().any():
        raise ValueError("verdict items are missing split roles")
    rounds.insert(1, "bench_group", _bench_group(rounds["bench"]))
    pair_features = features + fired
    raw_pairs = paired_features(rounds, raw, pair_features)
    normalised_pairs = paired_features(rounds, normalised, pair_features)
    normalised_pairs["normalisation_scope"] = "development_within_benchmark"
    return {
        "transcript/raw.parquet": raw,
        "transcript/normalised.parquet": normalised,
        "normalisation-parameters.parquet": parameters,
        "pair/raw.parquet": raw_pairs,
        "pair/normalised.parquet": normalised_pairs,
        "feature-dictionary.parquet": _feature_dictionary(features),
    }


def write_tables(tables: dict[str, pd.DataFrame], output_dir: Path) -> None:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    for name, frame in tables.items():
        path = output_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("transcript_features", type=Path)
    parser.add_argument("verdicts", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    transcripts = pd.read_parquet(
        args.transcript_features, columns=KEYS + ["split_role", *SHARED_FEATURES])
    verdicts = pd.read_parquet(args.verdicts, columns=VERDICT_COLUMNS)
    tables = build_tables(transcripts, verdicts)
    write_tables(tables, args.output_dir)
    print(f"transcripts={len(tables['transcript/raw.parquet'])} "
          f"pairs={len(tables['pair/raw.parquet'])} features={len(SHARED_FEATURES)}")


if __name__ == "__main__":
    main()
