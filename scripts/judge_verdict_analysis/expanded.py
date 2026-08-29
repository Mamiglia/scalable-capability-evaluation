"""Build and fit exact expanded judge-verdict matrix variants."""
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import socket

import yaml

from .build_dm import (
    ARTIFACTS,
    _sha256,
    build_expanded_design_matrix,
    write_design_matrix,
)
from .config import CONFIG
from .fit import run_remote_fits


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
        manifest = json.loads((output / "manifest.json").read_text())
        settings, features = manifest["settings"], manifest["features"]
        valid = (
            settings == {
                "construction": "processed-baseline-plus-explicit-additions",
                "additions": list(additions),
                "correlation_grouping": False,
            }
            and features["added"] == len(additions)
            and features["additions"] == list(additions)
            and features["retained"] == features["baseline"] + len(additions)
            and features["inventory"][-len(additions):] == list(additions)
            and Path(yaml.safe_load(
                (output / "design-matrix.yaml").read_text()
            )["output_root"]) == output
        )
        if baseline_config is not None and source_config is not None:
            source_paths = manifest["source_paths"]
            source_sha256 = manifest["source_sha256"]
            valid = valid and (
                source_paths["baseline_config"] == str(baseline_config)
                and source_paths["canonical_config"] == str(source_config)
                and source_paths.keys() == source_sha256.keys()
                and all(
                    Path(path).is_file() and _sha256(Path(path)) == source_sha256[key]
                    for key, path in source_paths.items()
                )
            )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError, yaml.YAMLError) as error:
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
    for name, additions in CONFIG.expanded_feature_sets:
        output = Path(output_matrix_root) / name
        if _resume_variant(output, additions, Path(baseline_config), Path(source_config)):
            continue
        tables = build_expanded_design_matrix(
            baseline_config, source_config, additions, output
        )
        write_design_matrix(tables, output)


def _guard_swc() -> None:
    hostname = socket.gethostname().lower()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("expanded fits require a Slurm allocation")
    if hostname.startswith(("gateway", "sgw")):
        raise RuntimeError("refusing expanded fit on the SWC gateway")


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
    variants = dict(CONFIG.expanded_feature_sets)
    if variant not in variants:
        raise ValueError(f"unknown expanded variant: {variant}")
    if not 0 <= model_fold_index < len(CONFIG.model_folds):
        raise ValueError("model_fold_index must be between 0 and 3")
    matrix = Path(expanded_matrix_root) / variant
    _resume_variant(matrix, variants[variant])
    manifest_sha = hashlib.sha256((matrix / "manifest.json").read_bytes()).hexdigest()
    run_remote_fits(replace(
        CONFIG,
        processed_matrix_root=matrix,
        processed_matrix_manifest_sha256=manifest_sha,
        results_dir=Path(results_root) / variant,
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
        build_variants(args.baseline_config, args.source_config, args.output_matrix_root)
    else:
        fit_variant(
            args.variant, args.expanded_matrix_root, args.results_root, args.model_fold_index
        )


if __name__ == "__main__":
    main()
