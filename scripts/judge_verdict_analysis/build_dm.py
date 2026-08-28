"""Build the representative readable-feature design matrix."""
import hashlib
import json
import os
from pathlib import Path
from tempfile import mkdtemp

import pandas as pd
import yaml

from .config import AnalysisConfig, CONFIG
from .preprocessing import (
    common_support,
    correlation_groups,
    feature_correlations,
    select_representatives,
)


ARTIFACTS = (
    "transcript.parquet", "pair.parquet", "feature-dictionary.parquet",
    "scaling-transcript.parquet", "scaling-pair.parquet", "representative-map.csv",
    "manifest.json", "design-matrix.yaml",
)


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


def _scaling_rows(scaling: pd.DataFrame, names: list[str], label: str) -> pd.DataFrame:
    if "feature" not in scaling:
        raise ValueError(f"{label} scaling metadata has no feature column")
    result = scaling[scaling.feature.isin(names)].copy()
    if missing := set(names) - set(result.feature):
        raise ValueError(f"{label} scaling metadata missing retained features: {sorted(missing)}")
    return result


def _manifest(
    config: AnalysisConfig,
    source_config: Path,
    source_root: Path,
    tables: dict[str, pd.DataFrame],
    original_features: int,
) -> dict[str, object]:
    source_files = [source_config] + [source_root / name for name in ARTIFACTS[:5]]
    settings = {
        "feature_blocks": list(config.feature_blocks), "regressors": list(config.regressors),
        "benchmarks": list(config.benchmarks), "splits": list(config.splits),
        "correlation_method": config.correlation_method,
        "correlation_threshold": config.correlation_threshold,
        "correlation_transitive": config.correlation_transitive,
        "common_support": config.common_support, "support_splits": list(config.support_splits),
        "representative_rule": config.representative_rule,
        "representative_overrides": [list(pair) for pair in config.representative_overrides],
        "processed_aliases": [list(pair) for pair in config.processed_aliases],
    }
    return {
        "source_hf_repo": config.source_hf_repo,
        "source_hf_revision": config.source_hf_revision,
        "source_status": "resolved" if config.source_hf_revision else "unresolved/test-only",
        "source_paths": {path.name: str(path) for path in source_files},
        "source_sha256": {path.name: _sha256(path) for path in source_files if path.exists()},
        "settings": settings,
        "rows": {"transcript": len(tables["transcript.parquet"]), "pair": len(tables["pair.parquet"])},
        "features": {"selected_original": original_features,
                     "retained": len(tables["feature-dictionary.parquet"])},
    }


def build_design_matrix(config: AnalysisConfig = CONFIG) -> dict[str, pd.DataFrame | dict[str, object] | str]:
    """Materialize a source-compatible matrix containing one feature per correlation group."""
    config = config.validate()
    source_config = Path(config.design_matrix_config)
    source_root = _source_root(source_config)
    dictionary = pd.read_parquet(source_root / "feature-dictionary.parquet")
    transcript = pd.read_parquet(source_root / "transcript.parquet")
    pair = pd.read_parquet(source_root / "pair.parquet")
    transcript_scaling = pd.read_parquet(source_root / "scaling-transcript.parquet")
    pair_scaling = pd.read_parquet(source_root / "scaling-pair.parquet")
    selected = _selected_dictionary(dictionary, config)
    names = selected.feature.tolist()
    if missing := set(names) - set(transcript):
        raise ValueError(f"transcript matrix missing selected features: {sorted(missing)}")
    pair_names = [f"d_{name}" for name in names]
    if missing := set(pair_names) - set(pair):
        raise ValueError(f"pair matrix missing selected features: {sorted(missing)}")

    supported = (common_support(transcript, selected, config.benchmarks, config.support_splits)
                 if config.common_support else selected.copy())
    support_names = supported.feature.tolist()
    pair_dictionary = supported.assign(column=supported.feature.map(lambda name: f"d_{name}"))
    correlation_input = pair[pair.bench.isin(config.benchmarks)].copy()
    if "split" in correlation_input:
        correlation_input = correlation_input[correlation_input.split.isin(config.splits)]
    correlations = feature_correlations(
        _deduplicate_judges(correlation_input), pair_dictionary, config.correlation_method)
    groups = correlation_groups(correlations, supported, config.correlation_threshold, config.correlation_transitive)
    representatives = select_representatives(
        groups, config.representative_rule, dict(config.representative_overrides), dict(config.processed_aliases))
    retained = representatives[representatives.retained].feature.tolist()

    representative_map = selected[["feature", "block"]].rename(columns={"block": "original_block"})
    representative_map = representative_map.merge(
        representatives, on=["feature", "original_block"], how="left", validate="one_to_one")
    retained_mask = representative_map.retained.fillna(False).astype(bool)
    representative_map["status"] = "outside_common_support"
    representative_map.loc[representative_map.feature.isin(support_names) & ~retained_mask, "status"] = "grouped_out"
    representative_map.loc[retained_mask, "status"] = "retained"
    representative_map["retained"] = retained_mask
    representative_map = representative_map[[
        "feature", "status", "correlation_group", "representative", "retained", "original_block",
        "group_size", "processed_name",
    ]]
    if representative_map.feature.duplicated().any() or len(representative_map) != len(selected):
        raise ValueError("representative map must cover each selected feature exactly once")
    if not representatives.groupby("correlation_group").retained.sum().eq(1).all():
        raise ValueError("every correlation group must have exactly one representative")

    source_features = set(dictionary.feature)
    transcript_output = transcript[[column for column in transcript if column not in source_features or column in retained]].copy()
    pair_source_features = {f"d_{feature}" for feature in source_features}
    pair_output = pair[[column for column in pair if column not in pair_source_features or column in {f"d_{name}" for name in retained}]].copy()
    output_dictionary = dictionary[dictionary.feature.isin(retained)].copy()
    output_dictionary = output_dictionary.merge(
        representative_map[representative_map.retained].drop(columns="status"), on="feature", how="left",
        validate="one_to_one")
    tables: dict[str, pd.DataFrame | dict[str, object] | str] = {
        "transcript.parquet": transcript_output,
        "pair.parquet": pair_output,
        "feature-dictionary.parquet": output_dictionary,
        "scaling-transcript.parquet": _scaling_rows(transcript_scaling, retained, "transcript"),
        "scaling-pair.parquet": _scaling_rows(pair_scaling, [f"d_{name}" for name in retained], "pair"),
        "representative-map.csv": representative_map,
    }
    tables["manifest.json"] = _manifest(config, source_config, source_root, tables, len(selected))
    tables["design-matrix.yaml"] = yaml.safe_dump({
        "output_root": str(config.processed_matrix_dir),
        "source_hf_repo": config.source_hf_repo,
        "source_hf_revision": config.source_hf_revision,
        "source_status": "resolved" if config.source_hf_revision else "unresolved/test-only",
    }, sort_keys=False)
    return tables


def write_design_matrix(tables: dict[str, pd.DataFrame | dict[str, object] | str], output_dir: Path) -> None:
    """Atomically write one complete matrix variant, refusing to replace an existing one."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        for name in ARTIFACTS:
            value = tables[name]
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


def main() -> None:
    """Build the configured processed design matrix."""
    tables = build_design_matrix(CONFIG)
    write_design_matrix(tables, CONFIG.processed_matrix_dir)
    print(f"transcripts={len(tables['transcript.parquet'])} pairs={len(tables['pair.parquet'])} "
          f"features={len(tables['feature-dictionary.parquet'])} -> {CONFIG.processed_matrix_dir}")


if __name__ == "__main__":
    main()
