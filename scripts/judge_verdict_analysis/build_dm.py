"""Build the readable-feature composite design matrix."""
import argparse
from collections.abc import Sequence
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
from tempfile import mkdtemp

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

from .config import AnalysisConfig, CONFIG
from .preprocessing import (
    apply_composites,
    fit_composites,
)


SOURCE_ARTIFACTS = (
    "transcript.parquet", "pair.parquet", "feature-dictionary.parquet",
    "scaling-transcript.parquet", "scaling-pair.parquet",
)
PROCESSED_ARTIFACTS = SOURCE_ARTIFACTS + (
    "composite-provenance.json", "manifest.json", "design-matrix.yaml",
)
ARTIFACTS = SOURCE_ARTIFACTS + (
    "representative-map.csv", "manifest.json", "design-matrix.yaml",
)  # Historical expanded-matrix contract.


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_root(config: Path) -> Path:
    values = yaml.safe_load(config.read_text())
    if not isinstance(values, dict) or "output_root" not in values:
        raise ValueError(f"design-matrix config has no output_root: {config}")
    return Path(values["output_root"])


def _deduplicate_judges(pair: pd.DataFrame) -> pd.DataFrame:
    """Keep one row per underlying transcript pair while selecting correlations."""
    identity = [
        column for column in (
            "bench", "item", "pair_id", "model_left", "model_right", "epoch_left", "epoch_right",
            "sample_epoch_left", "sample_epoch_right", "log_left", "log_right",
            "content_hash_left", "content_hash_right",
        ) if column in pair
    ]
    if not identity:
        ignored = {"pick_left", "left_stronger", "accuracy_difference"}
        identity = [column for column in pair if "judge" not in column and column not in ignored]
    return pair.drop_duplicates(identity, keep="first")


def _selected_dictionary(dictionary: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
    required = {"feature", "block"}
    if missing := required - set(dictionary):
        raise ValueError(f"feature dictionary missing columns: {sorted(missing)}")
    selected = dictionary[dictionary.block.isin(config.feature_blocks)].copy()
    if config.regressors:
        selected = selected[selected.feature.isin(config.regressors)].copy()
    if selected.empty:
        raise ValueError("no configured readable features found in source dictionary")
    if selected.feature.duplicated().any():
        raise ValueError("feature dictionary contains duplicate feature names")
    selected["column"] = selected.feature
    return selected


def _scale_composites(
    frame: pd.DataFrame,
    features: list[str],
    splits: tuple[str, ...],
    *,
    centre: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Scale final composites per benchmark on the designated training rows."""
    if "split" not in frame:
        raise ValueError("composite matrix has no split column for scaling")
    values = frame[features].to_numpy(dtype=float, copy=True)
    training = frame.split.isin(splits).to_numpy()
    rows = []
    for benchmark in pd.unique(frame.bench):
        mask = (frame.bench == benchmark).to_numpy()
        reference = values[mask & training]
        if not len(reference):
            raise ValueError(f"no composite scaling rows for benchmark {benchmark!r}")
        count = np.isfinite(reference).sum(axis=0)
        observed_mean = np.divide(
            np.nansum(reference, axis=0), count,
            out=np.zeros(len(features)), where=count > 0,
        )
        mean = observed_mean if centre else np.zeros(len(features))
        squared_deviation = np.nansum((reference - observed_mean) ** 2, axis=0)
        sd = np.sqrt(np.divide(
            squared_deviation, count - 1,
            out=np.ones(len(features)), where=count > 1,
        ))
        sd = np.where(np.isfinite(sd) & (sd > 0), sd, 1.0)
        values[mask] = (values[mask] - mean) / sd
        rows.extend({
            "benchmark": benchmark,
            "feature": feature,
            "mean": float(feature_mean),
            "sd": float(feature_sd),
        } for feature, feature_mean, feature_sd in zip(features, mean, sd))
    output = frame.copy()
    output.loc[:, features] = np.nan_to_num(values)
    return output, pd.DataFrame(rows)


def _manifest(
    config: AnalysisConfig,
    source_config: Path,
    source_root: Path,
    tables: dict[str, object],
    original_features: int,
) -> dict[str, object]:
    source_files = [source_config] + [source_root / name for name in SOURCE_ARTIFACTS]
    provenance = tables["composite-provenance.json"]
    provenance_hash = hashlib.sha256(json.dumps(
        provenance, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()
    output_dictionary = tables["feature-dictionary.parquet"]
    assert isinstance(output_dictionary, pd.DataFrame)
    settings = {
        "feature_blocks": list(config.feature_blocks), "regressors": list(config.regressors),
        "benchmarks": list(config.benchmarks), "splits": list(config.splits),
        "correlation_method": config.correlation_method,
        "correlation_threshold": config.correlation_threshold,
        "correlation_threshold_inclusive": True,
        "correlation_grouping": "greedy-maximum-correlation-hierarchical-mean",
        "support_splits": list(config.support_splits),
    }
    return {
        "source_hf_repo": config.source_hf_repo,
        "source_hf_revision": config.source_hf_revision,
        "source_status": "resolved" if config.source_hf_revision else "unresolved/test-only",
        "source_paths": {path.name: str(path) for path in source_files},
        "source_sha256": {path.name: _sha256(path) for path in source_files if path.exists()},
        "composite_provenance_sha256": provenance_hash,
        "settings": settings,
        "rows": {"transcript": len(tables["transcript.parquet"]), "pair": len(tables["pair.parquet"])},
        "features": {
            "selected_original": original_features,
            "retained": len(output_dictionary),
            "inventory": output_dictionary.feature.astype(str).tolist(),
        },
    }


def build_design_matrix(config: AnalysisConfig = CONFIG) -> dict[str, pd.DataFrame | dict[str, object] | str]:
    """Materialize a source-compatible matrix containing hierarchical composites."""
    config = config.validate()
    source_config = Path(config.design_matrix_config)
    source_root = _source_root(source_config)
    dictionary = pd.read_parquet(source_root / "feature-dictionary.parquet")
    transcript = pd.read_parquet(source_root / "transcript.parquet")
    pair = pd.read_parquet(source_root / "pair.parquet")
    transcript = transcript[transcript.bench.isin(config.benchmarks)].copy()
    pair = pair[pair.bench.isin(config.benchmarks)].copy()
    selected = _selected_dictionary(dictionary, config)
    names = selected.feature.tolist()
    if missing := set(names) - set(transcript):
        raise ValueError(f"transcript matrix missing selected features: {sorted(missing)}")
    pair_names = [f"d_{name}" for name in names]
    if missing := set(pair_names) - set(pair):
        raise ValueError(f"pair matrix missing selected features: {sorted(missing)}")

    pair_dictionary = selected.assign(column=selected.feature.map(lambda name: f"d_{name}"))
    correlation_input = pair[pair.bench.isin(config.benchmarks)].copy()
    if "split" in correlation_input:
        correlation_input = correlation_input[
            correlation_input.split.isin(config.support_splits)
        ]
    _, output_dictionary, provenance = fit_composites(
        _deduplicate_judges(correlation_input), pair_dictionary,
        config.correlation_threshold, config.correlation_method,
    )
    final_features = output_dictionary.feature.astype(str).tolist()
    source_features = set(dictionary.feature)
    transcript_metadata = transcript[[column for column in transcript if column not in source_features]]
    transcript_values = apply_composites(
        transcript, selected, provenance, neutralize=False,
    )
    transcript_output = pd.concat([transcript_metadata, transcript_values], axis=1)
    pair_source_features = {f"d_{feature}" for feature in source_features}
    pair_metadata = pair[[column for column in pair if column not in pair_source_features]]
    pair_values = apply_composites(
        pair, pair_dictionary, provenance, neutralize=False,
    ).rename(columns=lambda feature: f"d_{feature}")
    pair_output = pd.concat([pair_metadata, pair_values], axis=1)
    transcript_output, transcript_scaling = _scale_composites(
        transcript_output, final_features, config.support_splits, centre=True,
    )
    pair_features = [f"d_{feature}" for feature in final_features]
    pair_output, pair_scaling = _scale_composites(
        pair_output, pair_features, config.support_splits, centre=False,
    )
    output_dictionary = output_dictionary.drop(columns="column", errors="ignore")
    tables: dict[str, pd.DataFrame | dict[str, object] | str] = {
        "transcript.parquet": transcript_output,
        "pair.parquet": pair_output,
        "feature-dictionary.parquet": output_dictionary,
        "scaling-transcript.parquet": transcript_scaling,
        "scaling-pair.parquet": pair_scaling,
        "composite-provenance.json": provenance,
    }
    tables["manifest.json"] = _manifest(config, source_config, source_root, tables, len(selected))
    tables["design-matrix.yaml"] = yaml.safe_dump({
        "output_root": str(config.processed_matrix_dir),
        "source_hf_repo": config.source_hf_repo,
        "source_hf_revision": config.source_hf_revision,
        "source_status": "resolved" if config.source_hf_revision else "unresolved/test-only",
    }, sort_keys=False)
    return tables


def _expanded_table(
    baseline_root: Path,
    source_root: Path,
    level: str,
    baseline_features: list[str],
    additions: list[str],
) -> pd.DataFrame:
    prefix = "d_" if level == "pair" else ""
    baseline = pd.read_parquet(baseline_root / f"{level}.parquet")
    baseline_columns = {f"{prefix}{feature}" for feature in baseline_features}
    metadata = [column for column in baseline if column not in baseline_columns]
    source_path = source_root / f"{level}.parquet"
    source_schema = pq.read_schema(source_path).names
    added_columns = [f"{prefix}{feature}" for feature in additions]
    if missing := [column for column in added_columns if column not in source_schema]:
        raise ValueError(f"source {level} matrix missing additions: {missing}")
    if missing := [column for column in metadata if column not in source_schema]:
        raise ValueError(f"source {level} matrix missing baseline metadata: {missing}")
    source = pd.read_parquet(source_path, columns=metadata + added_columns)
    if not source[metadata].equals(baseline[metadata]):
        raise ValueError(f"source {level} rows/metadata do not match baseline")
    return pd.concat([baseline, source[added_columns]], axis=1)


def _expanded_scaling(
    baseline_root: Path,
    source_root: Path,
    filename: str,
    additions: list[str],
    prefix: str,
) -> pd.DataFrame:
    baseline = pd.read_parquet(baseline_root / filename)
    source = pd.read_parquet(source_root / filename)
    if missing := sorted(set(baseline) - set(source)):
        raise ValueError(f"source {filename} missing baseline columns: {missing}")
    names = [f"{prefix}{feature}" for feature in additions]
    added = source[source.feature.isin(names)].copy()
    if missing := [feature for feature in names if feature not in set(added.feature)]:
        raise ValueError(f"source {filename} missing additions: {missing}")
    if added.duplicated(["benchmark", "feature"]).any():
        raise ValueError(f"source {filename} has duplicate benchmark/feature rows")
    benchmark_order = baseline.benchmark.drop_duplicates().tolist()
    benchmarks = set(benchmark_order)
    incomplete = [
        feature for feature in names
        if set(added.loc[added.feature == feature, "benchmark"]) != benchmarks
    ]
    if incomplete:
        raise ValueError(f"source {filename} additions lack baseline benchmark coverage: {incomplete}")
    added = (added.assign(
        _benchmark_order=added.benchmark.map({name: i for i, name in enumerate(benchmark_order)}),
        _feature_order=added.feature.map({name: i for i, name in enumerate(names)}),
    ).sort_values(["_benchmark_order", "_feature_order"])
        .drop(columns=["_benchmark_order", "_feature_order"]))
    return pd.concat([baseline, added], ignore_index=True, sort=False)


def build_expanded_design_matrix(
    baseline_config: str | Path,
    source_config: str | Path,
    additions: Sequence[str],
    output_root: str | Path,
) -> dict[str, pd.DataFrame | dict[str, object] | str]:
    """Append exact canonical features to a processed baseline without regrouping them."""
    baseline_config, source_config = Path(baseline_config), Path(source_config)
    baseline_root, source_root = _source_root(baseline_config), _source_root(source_config)
    output_root = Path(output_root)
    additions = list(additions)
    if len(additions) != len(set(additions)):
        raise ValueError("additions must be unique")

    baseline_dictionary = pd.read_parquet(baseline_root / "feature-dictionary.parquet")
    source_dictionary = pd.read_parquet(source_root / "feature-dictionary.parquet")
    synthesized_dictionary_fields = {
        "correlation_group", "representative", "retained", "original_block",
        "group_size", "processed_name",
    }
    required_source_fields = set(baseline_dictionary) - synthesized_dictionary_fields
    if missing := sorted(required_source_fields - set(source_dictionary)):
        raise ValueError(f"source feature dictionary missing baseline columns: {missing}")
    baseline_features = baseline_dictionary.feature.tolist()
    if overlap := [feature for feature in additions if feature in set(baseline_features)]:
        raise ValueError(f"additions already present in baseline: {overlap}")
    source_index = source_dictionary.set_index("feature")
    if missing := [feature for feature in additions if feature not in source_index.index]:
        raise ValueError(f"source feature dictionary missing additions: {missing}")
    added_dictionary = source_index.loc[additions].reset_index()
    added_dictionary = added_dictionary.assign(
        correlation_group=pd.NA,
        representative=additions,
        retained=True,
        original_block=added_dictionary.block,
        group_size=1,
        processed_name=additions,
    )
    output_dictionary = pd.concat(
        [baseline_dictionary, added_dictionary], ignore_index=True, sort=False)

    transcript = _expanded_table(
        baseline_root, source_root, "transcript", baseline_features, additions)
    pair = _expanded_table(
        baseline_root, source_root, "pair", baseline_features, additions)
    transcript_scaling = _expanded_scaling(
        baseline_root, source_root, "scaling-transcript.parquet", additions, "")
    pair_scaling = _expanded_scaling(
        baseline_root, source_root, "scaling-pair.parquet", additions, "d_")

    representative_map = pd.read_csv(baseline_root / "representative-map.csv")
    addition_map = pd.DataFrame({
        "feature": additions,
        "status": "retained",
        "correlation_group": pd.NA,
        "representative": additions,
        "retained": True,
        "original_block": added_dictionary.block.tolist(),
        "group_size": 1,
        "processed_name": additions,
    })
    representative_map = pd.concat(
        [representative_map, addition_map], ignore_index=True, sort=False)

    source_files = {
        "baseline_config": baseline_config,
        "canonical_config": source_config,
        "baseline_manifest": baseline_root / "manifest.json",
        "baseline_representative_map": baseline_root / "representative-map.csv",
        **{
            f"baseline_{name.removesuffix('.parquet').replace('-', '_')}": baseline_root / name
            for name in ARTIFACTS[:5]
        },
        **{
            f"canonical_{name.removesuffix('.parquet').replace('-', '_')}": source_root / name
            for name in ARTIFACTS[:5]
        },
    }
    baseline_manifest = json.loads((baseline_root / "manifest.json").read_text())
    inventory = baseline_features + additions
    manifest = {
        **{
            key: baseline_manifest.get(key)
            for key in ("source_hf_repo", "source_hf_revision", "source_status")
        },
        "source_paths": {key: str(path) for key, path in source_files.items()},
        "source_sha256": {key: _sha256(path) for key, path in source_files.items()},
        "settings": {
            "construction": "processed-baseline-plus-explicit-additions",
            "additions": additions,
            "correlation_grouping": False,
        },
        "rows": {"transcript": len(transcript), "pair": len(pair)},
        "features": {
            "baseline": len(baseline_features), "added": len(additions),
            "retained": len(inventory), "inventory": inventory,
            "additions": additions,
        },
    }
    tables: dict[str, pd.DataFrame | dict[str, object] | str] = {
        "transcript.parquet": transcript,
        "pair.parquet": pair,
        "feature-dictionary.parquet": output_dictionary,
        "scaling-transcript.parquet": transcript_scaling,
        "scaling-pair.parquet": pair_scaling,
        "representative-map.csv": representative_map,
        "manifest.json": manifest,
        "design-matrix.yaml": yaml.safe_dump({
            "output_root": str(output_root),
            "source_hf_repo": manifest["source_hf_repo"],
            "source_hf_revision": manifest["source_hf_revision"],
            "source_status": manifest["source_status"],
        }, sort_keys=False),
    }
    return tables


def write_design_matrix(tables: dict[str, pd.DataFrame | dict[str, object] | str], output_dir: Path) -> None:
    """Atomically write one complete matrix variant, refusing to replace an existing one."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        for name, value in tables.items():
            path = stage / name
            if name.endswith(".parquet"):
                assert isinstance(value, pd.DataFrame)
                value.to_parquet(path, index=False)
            elif name.endswith(".csv"):
                assert isinstance(value, pd.DataFrame)
                value.to_csv(path, index=False)
            elif name.endswith(".json"):
                path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
            else:
                path.write_text(str(value))
        os.replace(stage, output_dir)
    except BaseException:
        for path in stage.iterdir() if stage.exists() else ():
            path.unlink()
        if stage.exists():
            stage.rmdir()
        raise


def main(argv: list[str] | None = None) -> None:
    """Build the configured processed design matrix."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-repo", default=CONFIG.source_hf_repo)
    parser.add_argument("--source-revision", default=CONFIG.source_hf_revision)
    args = parser.parse_args(argv)
    config = replace(
        CONFIG, design_matrix_config=args.source_config,
        processed_matrix_root=args.output_root,
        source_hf_repo=args.source_repo, source_hf_revision=args.source_revision,
    ).validate()
    tables = build_design_matrix(config)
    write_design_matrix(tables, config.processed_matrix_dir)
    print(f"transcripts={len(tables['transcript.parquet'])} pairs={len(tables['pair.parquet'])} "
          f"features={len(tables['feature-dictionary.parquet'])} -> {config.processed_matrix_dir}")


if __name__ == "__main__":
    main()
