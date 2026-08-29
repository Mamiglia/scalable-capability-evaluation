"""Build and fit exact expanded judge-verdict matrix variants."""
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import socket
from tempfile import TemporaryDirectory

import pandas as pd
import yaml

from .build_dm import (
    ARTIFACTS,
    _sha256,
    build_expanded_design_matrix,
    write_design_matrix,
)
from .config import CONFIG
from .fit import run_remote_fits


_SOURCE_KEYS = {
    "baseline_config", "canonical_config", "baseline_manifest",
    "baseline_representative_map",
    *{
        f"{prefix}_{name.removesuffix('.parquet').replace('-', '_')}"
        for prefix in ("baseline", "canonical") for name in ARTIFACTS[:5]
    },
}


def _resume_variant(
    output: Path,
    additions: tuple[str, ...],
    baseline_config: Path | None = None,
    source_config: Path | None = None,
) -> bool:
    if not output.exists():
        return False
    if missing := [name for name in ARTIFACTS if not (output / name).is_file()]:
        raise RuntimeError(f"cannot resume partial variant {output}; missing {missing}")
    try:
        transcript = pd.read_parquet(output / "transcript.parquet")
        pair = pd.read_parquet(output / "pair.parquet")
        dictionary = pd.read_parquet(output / "feature-dictionary.parquet")
        transcript_scaling = pd.read_parquet(output / "scaling-transcript.parquet")
        pair_scaling = pd.read_parquet(output / "scaling-pair.parquet")
        representative_map = pd.read_csv(output / "representative-map.csv")
        manifest = json.loads((output / "manifest.json").read_text())
        matrix_config = yaml.safe_load((output / "design-matrix.yaml").read_text())
        settings, features = manifest["settings"], manifest["features"]
        inventory = features["inventory"]
        source_paths = manifest["source_paths"]
        source_sha256 = manifest["source_sha256"]
        inventory_set = set(inventory)
        pair_inventory = [f"d_{feature}" for feature in inventory]
        dictionary_fields = {
            "feature", "block", "correlation_group", "representative", "retained",
            "original_block", "group_size", "processed_name",
        }
        representative_fields = (dictionary_fields - {"block"}) | {"status"}
        scaling_fields = {"benchmark", "feature", "mean", "sd"}
        transcript_fields = {"bench", "model", "item", "split", "model_accuracy"}
        pair_fields = {
            "bench", "judge", "item", "model_left", "model_right", "split", "pick_left",
        }
        expected_transcript_scaling = {
            (benchmark, feature)
            for benchmark in set(transcript.bench) for feature in inventory
        }
        expected_pair_scaling = {
            (benchmark, feature)
            for benchmark in set(pair.bench) for feature in pair_inventory
        }
        transcript_scaling_keys = set(zip(
            transcript_scaling.benchmark, transcript_scaling.feature, strict=True,
        ))
        pair_scaling_keys = set(zip(
            pair_scaling.benchmark, pair_scaling.feature, strict=True,
        ))
        valid = (
            settings == {
                "construction": "processed-baseline-plus-explicit-additions",
                "additions": list(additions),
                "correlation_grouping": False,
            }
            and features["baseline"] == 117
            and features["added"] == len(additions)
            and features["additions"] == list(additions)
            and features["retained"] == features["baseline"] + len(additions)
            and len(inventory) == features["retained"]
            and len(inventory_set) == len(inventory)
            and features["inventory"][-len(additions):] == list(additions)
            and manifest["rows"] == {"transcript": len(transcript), "pair": len(pair)}
            and dictionary_fields <= set(dictionary)
            and dictionary.feature.tolist() == inventory
            and not dictionary.columns.duplicated().any()
            and transcript_fields <= set(transcript)
            and [column for column in transcript if column in inventory_set] == inventory
            and not transcript.columns.duplicated().any()
            and pair_fields <= set(pair)
            and [column for column in pair if column in set(pair_inventory)] == pair_inventory
            and not pair.columns.duplicated().any()
            and scaling_fields <= set(transcript_scaling)
            and transcript_scaling_keys == expected_transcript_scaling
            and len(transcript_scaling) == len(expected_transcript_scaling)
            and scaling_fields <= set(pair_scaling)
            and pair_scaling_keys == expected_pair_scaling
            and len(pair_scaling) == len(expected_pair_scaling)
            and representative_fields <= set(representative_map)
            and representative_map.feature.tolist() == inventory
            and not representative_map.feature.duplicated().any()
            and isinstance(matrix_config, dict)
            and Path(matrix_config["output_root"]).resolve() == output.resolve()
            and all(
                matrix_config[key] == manifest[key]
                for key in ("source_hf_repo", "source_hf_revision", "source_status")
            )
            and set(source_paths) == _SOURCE_KEYS
            and set(source_sha256) == _SOURCE_KEYS
            and all(
                Path(path).is_absolute()
                and Path(path).is_file()
                and _sha256(Path(path)) == source_sha256[key]
                for key, path in source_paths.items()
            )
        )
        if baseline_config is not None and source_config is not None:
            valid = valid and (
                source_paths["baseline_config"] == str(baseline_config.resolve())
                and source_paths["canonical_config"] == str(source_config.resolve())
            )
    except Exception as error:
        raise RuntimeError(f"cannot resume invalid variant {output}") from error
    if not valid:
        raise RuntimeError(f"cannot resume invalid variant {output}")
    return True


def build_variants(
    baseline_config: Path,
    source_config: Path,
    output_matrix_root: Path,
) -> None:
    """Materialize each configured exact-addition matrix."""
    _guard_swc()
    baseline_config = Path(baseline_config).expanduser().resolve()
    source_config = Path(source_config).expanduser().resolve()
    output_matrix_root = Path(output_matrix_root).expanduser().resolve()
    _validate_build_inputs(baseline_config, source_config)
    with TemporaryDirectory(
        prefix="judge-expanded-", dir=os.environ.get("SLURM_TMPDIR")
    ) as temporary:
        temporary = Path(temporary)
        normalized_baseline = temporary / "baseline.yaml"
        normalized_source = temporary / "canonical.yaml"
        normalized_baseline.write_text(yaml.safe_dump(_config_values(baseline_config)))
        normalized_source.write_text(yaml.safe_dump(_config_values(source_config)))
        for name, additions in CONFIG.expanded_feature_sets:
            output = output_matrix_root / name
            if _resume_variant(output, additions, baseline_config, source_config):
                continue
            tables = build_expanded_design_matrix(
                normalized_baseline, normalized_source, additions, output
            )
            manifest = tables["manifest.json"]
            assert isinstance(manifest, dict)
            for key, config in (
                ("baseline_config", baseline_config), ("canonical_config", source_config),
            ):
                manifest["source_paths"][key] = str(config)
                manifest["source_sha256"][key] = _sha256(config)
            write_design_matrix(tables, output)


def _guard_swc() -> None:
    hostname = socket.gethostname().lower()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("expanded work requires a Slurm allocation")
    if hostname.startswith(("gateway", "sgw")):
        raise RuntimeError("refusing expanded work on the SWC gateway")


def _config_values(config: Path) -> dict[str, object]:
    try:
        values = yaml.safe_load(config.read_text())
        root = Path(values["output_root"]).expanduser()
    except Exception as error:
        raise RuntimeError(f"invalid design-matrix config: {config}") from error
    if root.is_absolute():
        resolved = root.resolve()
    else:
        candidates = {
            (Path.cwd() / root).resolve(),
            (config.parent / root).resolve(),
        }
        existing = [candidate for candidate in candidates if candidate.exists()]
        if not existing:
            raise RuntimeError(
                f"relative output_root does not resolve from cwd or config parent: {root}"
            )
        if len(existing) > 1:
            raise RuntimeError(f"ambiguous relative output_root {root}: {sorted(map(str, existing))}")
        resolved = existing[0]
    values["output_root"] = str(resolved)
    return values


def _config_root(config: Path) -> Path:
    return Path(_config_values(config)["output_root"])


def _validate_build_inputs(baseline_config: Path, source_config: Path) -> None:
    baseline_root, source_root = _config_root(baseline_config), _config_root(source_config)
    baseline_manifest = baseline_root / "manifest.json"
    if _sha256(baseline_manifest) != CONFIG.processed_matrix_manifest_sha256:
        raise RuntimeError("baseline manifest SHA does not match the authorized config pin")
    try:
        baseline = pd.read_parquet(baseline_root / "feature-dictionary.parquet")
        source = pd.read_parquet(source_root / "feature-dictionary.parquet")
        baseline_features = baseline.feature.tolist()
        source_counts = source.feature.value_counts()
    except Exception as error:
        raise RuntimeError("invalid baseline or canonical feature dictionary") from error
    if len(baseline_features) != 117 or len(set(baseline_features)) != 117:
        raise RuntimeError("authorized baseline must contain exactly 117 features")
    additions = tuple(dict.fromkeys(
        feature for _, features in CONFIG.expanded_feature_sets for feature in features
    ))
    if any(source_counts.get(feature, 0) != 1 for feature in additions):
        raise RuntimeError("canonical dictionary additions must each appear exactly once")
    if set(additions) & set(baseline_features):
        raise RuntimeError("canonical dictionary additions overlap the readable baseline")


def _expanded_blocks(additions: tuple[str, ...]) -> tuple[str, ...]:
    blocks = list(CONFIG.feature_blocks)
    for feature, block in (
        ("surprisal_", "surprisal-llama"),
        ("position_adjusted", "surprisal-position-adjusted"),
        ("sae_shared_", "embedding-sae-shared"),
    ):
        if any(feature in addition for addition in additions) and block not in blocks:
            blocks.append(block)
    return tuple(blocks)


def fit_variant(
    variant: str,
    expanded_matrix_root: Path,
    results_root: Path,
    model_fold_index: int,
) -> None:
    """Run one model fold and its four item folds for one expanded variant."""
    _guard_swc()
    expanded_matrix_root = Path(expanded_matrix_root).expanduser().resolve()
    results_root = Path(results_root).expanduser().resolve()
    variants = dict(CONFIG.expanded_feature_sets)
    if variant not in variants:
        raise ValueError(f"unknown expanded variant: {variant}")
    if not 0 <= model_fold_index < len(CONFIG.model_folds):
        raise ValueError("model_fold_index must be between 0 and 3")
    matrix = expanded_matrix_root / variant
    _resume_variant(matrix, variants[variant])
    manifest_sha = hashlib.sha256((matrix / "manifest.json").read_bytes()).hexdigest()
    run_remote_fits(replace(
        CONFIG,
        processed_matrix_root=matrix,
        processed_matrix_manifest_sha256=manifest_sha,
        results_dir=results_root / variant,
        feature_blocks=_expanded_blocks(variants[variant]),
        fit_backends=("frequentist",),
        execution_mode="execute",
        execution_target="swc_hpc",
        resume_completed_folds=True,
        model_folds=(CONFIG.model_folds[model_fold_index],),
    ))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="build every expanded matrix variant")
    build.add_argument("baseline_config", type=Path)
    build.add_argument("source_config", type=Path)
    build.add_argument("output_matrix_root", type=Path)
    fit = commands.add_parser("fit", help="fit one variant and model fold")
    fit.add_argument("variant", choices=[name for name, _ in CONFIG.expanded_feature_sets])
    fit.add_argument("expanded_matrix_root", type=Path)
    fit.add_argument("results_root", type=Path)
    fit.add_argument("model_fold_index", type=int, choices=range(len(CONFIG.model_folds)))
    args = parser.parse_args(argv)
    if args.command == "build":
        build_variants(
            args.baseline_config.expanduser().resolve(),
            args.source_config.expanduser().resolve(),
            args.output_matrix_root.expanduser().resolve(),
        )
    else:
        fit_variant(
            args.variant,
            args.expanded_matrix_root.expanduser().resolve(),
            args.results_root.expanduser().resolve(),
            args.model_fold_index,
        )


if __name__ == "__main__":
    main()
