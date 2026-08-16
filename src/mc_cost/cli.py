"""Reproducible command-line workflow for the Monte Carlo cost simulator."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit

from mc_cost.catalogue import build_capability_draws, build_catalogue, pool_enabled, validate_catalogue
from mc_cost.data import (
    RELEASE_EPOCH,
    candidate_token_rows,
    judge_token_rows,
    prepare_judgments,
    standardize_judgments,
    write_prepared_tables,
)
from mc_cost.design import AutomaticConfig, automatic_comparisons, manual_comparisons, prediction_eligible
from mc_cost.gpqa import prepare_for_mc as prepare_mc_gpqa
from mc_cost.model import (
    FEATURE_COLUMNS,
    JUDGMENT_SPECS,
    PosteriorArtifact,
    encode_judgments,
    fit_judgment_model,
    left_logits,
    load_artifact,
    save_artifact,
)
from mc_cost.simulation import (
    SimulationConfig,
    dashboard_payload,
    dashboard_ranking_recovery,
    simulate,
)
from mc_cost.tokens import draw_token_usage, draw_transcript_lengths, fit_token_model
from mc_cost.validation import (
    choose_judge_gate,
    choose_judgment_spec,
    cross_validate_judgments,
    decision_payload,
    posterior_predictive_replay,
    ranking_recovery,
    require_decision,
    summarize_token_predictions,
    support_distance,
    validate_tokens,
    write_decision,
)


_COMMANDS = ("catalogue", "prepare", "fit", "validate", "simulate", "export-dashboard")
_MANUAL_COLUMNS = (
    "judge", "m1", "m2", "judge_reasoning_class", "m1_reasoning_class", "m2_reasoning_class",
)
_CATALOGUE_FILES = ("catalogue.csv", "capability_draws.npz")
_PREPARE_FILES = (
    "judgments.parquet", "token_candidate.parquet", "token_judge.parquet",
    "judgment_standardization.json", "scales.json", "prepare_accounting.json",
)
_MODEL_FILES = tuple(
    f"model/{name}.{suffix}"
    for name in (*JUDGMENT_SPECS, "candidate_token", "judge_token")
    for suffix in ("npz", "json")
)
_DASHBOARD_SHARD_FILES = (
    "automatic.json", "manual.json", "ranking_recovery.json", "warnings.json", "metadata.json",
)


class CliError(ValueError):
    pass


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in _COMMANDS:
        command = subparsers.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--seed", type=int)
        if name == "fit":
            command.add_argument("--warmup", type=_positive)
            command.add_argument("--samples", type=_positive)
            command.add_argument("--chains", type=_positive)
        elif name == "validate":
            command.add_argument("--shard-index", type=int)
            command.add_argument("--shard-count", type=_positive)
            command.add_argument("--reduce-shards", action="store_true")
        elif name == "simulate":
            command.add_argument("--manual", type=Path)
            command.add_argument("--judges", type=_positive)
            command.add_argument("--anchors", type=_positive)
            command.add_argument("--comparators", type=_positive)
            command.add_argument("--items", type=_positive)
            command.add_argument("--presentations", type=_positive)
            command.add_argument("--trials", type=_positive)
            responses = command.add_mutually_exclusive_group()
            responses.add_argument("--reuse-responses", dest="reuse_responses", action="store_true")
            responses.add_argument("--generate-responses", dest="reuse_responses", action="store_false")
            command.set_defaults(reuse_responses=None)
            command.add_argument("--accuracy-target", choices=("higher_theta",))
        elif name == "export-dashboard":
            command.add_argument("--out", type=Path)
            command.add_argument("--trials", type=_positive)
            command.add_argument("--provisional", action="store_true")
            command.add_argument("--shard-index", type=int)
            command.add_argument("--shard-count", type=_positive)
            command.add_argument("--reduce-shards", action="store_true")
    return parser


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
    temporary.replace(path)


def write_run_config(path: str | Path, config: dict, source_hashes: dict[str, str]) -> None:
    _write_json(Path(path), {"config": config, "source_hashes": source_hashes})


def _load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise CliError(f"configuration file not found: {path}")
    try:
        config = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise CliError(f"cannot read configuration {path}: {error}") from None
    if not isinstance(config, dict) or not isinstance(config.get("paths"), dict):
        raise CliError("configuration must contain a paths object")
    return config


def _manual_header(path: Path) -> None:
    try:
        with path.open(newline="") as handle:
            header = next(csv.reader(handle), [])
    except OSError as error:
        raise CliError(f"cannot read manual comparisons: {error}") from None
    if header[:3] != list(_MANUAL_COLUMNS[:3]) or len(header) != len(set(header)) or any(
        column not in _MANUAL_COLUMNS for column in header
    ):
        raise CliError(
            "manual comparison columns must be judge,m1,m2 with optional *_reasoning_class overrides"
        )


def _output(config: dict[str, Any]) -> Path:
    value = config["paths"].get("output")
    if not isinstance(value, str) or not value:
        raise CliError("configuration paths.output must be a nonempty path")
    return Path(value)


def _configured_path(config: dict[str, Any], name: str) -> Path:
    value = config["paths"].get(name)
    if not isinstance(value, str) or not value:
        raise CliError(f"configuration paths.{name} must be a nonempty path")
    return Path(value)


def _require(path: Path, command: str) -> None:
    if not path.is_file():
        raise CliError(f"missing prerequisite {path}; run `mc-cost {command} --config ...` first")


def _require_source(config: dict[str, Any], name: str) -> Path:
    path = _configured_path(config, name)
    if not path.exists():
        guidance = (
            "; run `uv run python scripts/irt_proxy/run.py` first"
            if name in {"irt_capabilities", "irt_items", "irt_posterior"}
            else ""
        )
        raise CliError(f"configured source does not exist: {path}{guidance}")
    return path


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _publish(stage: Path, output: Path, names: list[str]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for name in names:
        source, target = stage / name, output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        source.replace(target)


def _write_manifest(output: Path, manifest: str, names) -> None:
    _write_json(output / manifest, {
        "schema_version": 1,
        "files": {name: _hash_file(output / name) for name in names},
    })


def _verify_generation(
    output: Path, manifest: str, required, command: str, label: str,
) -> None:
    path = output / manifest
    _require(path, command)
    try:
        payload = json.loads(path.read_text())
        files = payload["files"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        raise CliError(f"invalid {label} generation manifest; rerun `mc-cost {command}`") from None
    if not isinstance(files, dict) or not set(required) <= set(files):
        raise CliError(f"incomplete {label} generation manifest; rerun `mc-cost {command}`")
    for name, expected in files.items():
        relative = Path(name)
        if (
            not isinstance(name, str) or not isinstance(expected, str)
            or relative.is_absolute() or ".." in relative.parts
        ):
            raise CliError(f"invalid {label} generation manifest; rerun `mc-cost {command}`")
        artifact = output / relative
        if not artifact.is_file() or _hash_file(artifact) != expected:
            raise CliError(f"{label} generation hash mismatch for {name}; rerun `mc-cost {command}`")


def _temporary_directory(parent: Path):
    parent.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(dir=parent, prefix=".mc-cost-")


def _catalogue(args: argparse.Namespace, config: dict[str, Any]) -> None:
    sources = {
        name: _require_source(config, name)
        for name in ("canonical_models", "model_pool", "irt_capabilities", "irt_posterior")
    }
    if "model_overrides" in config["paths"]:
        sources["model_overrides"] = _require_source(config, "model_overrides")
    catalogue = build_catalogue(
        sources["canonical_models"], sources["model_pool"], sources["irt_capabilities"],
        sources.get("model_overrides"),
    )
    validate_catalogue(catalogue)
    with np.load(sources["irt_posterior"], allow_pickle=False) as stored:
        irt = {name: stored[name].copy() for name in stored.files}
    seed = args.seed if args.seed is not None else int(config.get("fit", {}).get("seed", 0))
    models, draws = build_capability_draws(catalogue, irt, np.random.default_rng(seed))
    hashes = {name: _hash_file(path) for name, path in sources.items()}
    output = _output(config)
    with _temporary_directory(output.parent) as directory:
        stage = Path(directory)
        catalogue.to_csv(stage / "catalogue.csv", index=False)
        np.savez_compressed(
            stage / "capability_draws.npz", models=np.asarray(models, dtype=str), draws=draws,
            seed=np.asarray(seed), source_names=np.asarray(list(hashes), dtype=str),
            source_hashes=np.asarray(list(hashes.values()), dtype=str),
        )
        _publish(stage, output, list(_CATALOGUE_FILES))
        _write_manifest(output, "catalogue.manifest.json", _CATALOGUE_FILES)


def _prepare(args: argparse.Namespace, config: dict[str, Any]) -> None:
    output = _output(config)
    _require(output / "catalogue.csv", "catalogue")
    _verify_generation(
        output, "catalogue.manifest.json", _CATALOGUE_FILES, "catalogue", "catalogue",
    )
    items_path = _require_source(config, "irt_items")
    logs = _require_source(config, "judge_logs")
    roots = config["paths"].get("candidate_log_roots", [])
    if not isinstance(roots, list) or any(not isinstance(root, str) for root in roots):
        raise CliError("configuration paths.candidate_log_roots must be a list of paths")
    root_paths = [Path(root) for root in roots]
    for root in root_paths:
        if not root.is_dir():
            raise CliError(f"configured candidate log root does not exist: {root}")
    catalogue = pd.read_csv(output / "catalogue.csv")
    items = pd.read_csv(items_path)
    attempts, judgment_counts, judgment_eligibility = prepare_mc_gpqa(logs, catalogue)
    judgment_exclusions = list(attempts.attrs.get("exclusions", []))
    judgments = prepare_judgments(attempts, catalogue, items)
    max_chars = pd.to_numeric(judgments["presentation_max_chars"], errors="coerce")
    if max_chars.isna().any() or max_chars.nunique() != 1:
        raise CliError("judgment logs must declare one presentation_max_chars value")
    candidates = candidate_token_rows(
        root_paths, catalogue, items, skip_incomplete=True,
        presentation_max_chars=int(max_chars.iloc[0]),
    )
    candidate_counts = dict(candidates.attrs.get("accounting", {}))
    candidate_exclusions = list(candidates.attrs.get("exclusions", []))
    judges = judge_token_rows(judgments)
    judge_token_counts = dict(judges.attrs.get("accounting", {}))
    judge_token_exclusions = list(judges.attrs.get("exclusions", []))
    accounting = {
        "schema_version": 1,
        "judgment_counts": json.loads(
            judgment_counts.rename_axis("judge").reset_index().to_json(orient="records")
        ),
        "judgment_eligibility": json.loads(
            judgment_eligibility.rename_axis("judge").reset_index().to_json(orient="records")
        ),
        "candidate_counts": candidate_counts,
        "judge_token_counts": judge_token_counts,
        "exclusions": judgment_exclusions + candidate_exclusions + judge_token_exclusions,
    }
    with _temporary_directory(output.parent) as directory:
        stage = Path(directory)
        paths = write_prepared_tables(
            stage, judgments, candidates, judges, training_rows=np.ones(len(judgments), dtype=bool),
        )
        shutil.copyfile(paths["judgment_standardization"], stage / "scales.json")
        _write_json(stage / "prepare_accounting.json", accounting)
        _publish(stage, output, list(_PREPARE_FILES))
        _write_manifest(output, "prepare.manifest.json", _PREPARE_FILES)


def _scales_path(output: Path) -> Path:
    preferred = output / "scales.json"
    return preferred if preferred.is_file() else output / "judgment_standardization.json"


def _load_scales(path: Path) -> dict[str, dict[str, float]]:
    payload = json.loads(path.read_text())
    if {"mean", "scale"} <= payload.keys():
        return pd.DataFrame(payload).to_dict(orient="index")
    return payload


def _fit_sources(output: Path) -> dict[str, Path]:
    sources = {
        "catalogue": output / "catalogue.csv",
        "capability_draws": output / "capability_draws.npz",
        "judgments": output / "judgments.parquet",
        "candidate_tokens": output / "token_candidate.parquet",
        "judge_tokens": output / "token_judge.parquet",
        "scales": _scales_path(output),
    }
    for path in sources.values():
        _require(path, "prepare" if path.name not in {"catalogue.csv", "capability_draws.npz"} else "catalogue")
    _verify_generation(
        output, "catalogue.manifest.json", _CATALOGUE_FILES, "catalogue", "catalogue",
    )
    _verify_generation(output, "prepare.manifest.json", _PREPARE_FILES, "prepare", "prepared-data")
    return sources


def _tag_judgment_artifact(
    artifact: PosteriorArtifact, spec, source_hashes: dict[str, str], conditional_judges: bool,
) -> PosteriorArtifact:
    return PosteriorArtifact(
        artifact.posterior,
        {
            **artifact.metadata,
            "feature_order": list(spec.global_features),
            "include_length": spec.include_length,
            "source_hashes": source_hashes,
            "conditional_judges": conditional_judges,
        },
        artifact.diagnostics,
    )


def _fit(args: argparse.Namespace, config: dict[str, Any]) -> None:
    output = _output(config)
    sources = _fit_sources(output)
    hashes = {name: _hash_file(path) for name, path in sources.items()}
    catalogue = pd.read_csv(sources["catalogue"])
    judgments = pd.read_parquet(sources["judgments"])
    candidate_tokens = pd.read_parquet(sources["candidate_tokens"])
    judge_tokens = pd.read_parquet(sources["judge_tokens"])
    scales = _load_scales(sources["scales"])
    configured = config.get("fit", {})
    settings = {
        name: getattr(args, name) if getattr(args, name) is not None else int(configured.get(name, default))
        for name, default in (("warmup", 1000), ("samples", 1000), ("chains", 1), ("seed", 0))
    }
    with _temporary_directory(output.parent) as directory:
        stage = Path(directory) / "model"
        for name, spec in JUDGMENT_SPECS.items():
            data = encode_judgments(
                judgments, catalogue, spec, training_scales=scales, source_hashes=hashes,
            )
            artifact = _tag_judgment_artifact(
                fit_judgment_model(data, **settings), spec, hashes, True,
            )
            save_artifact(artifact, stage / name)
        for role, frame in (("candidate", candidate_tokens), ("judge", judge_tokens)):
            artifact = fit_token_model(
                frame, role, reasoning_classes=catalogue, source_hashes=hashes, **settings,
            )
            save_artifact(artifact, stage / f"{role}_token")
        _publish(Path(directory), output, list(_MODEL_FILES))
        _write_manifest(output, "model/manifest.json", _MODEL_FILES)


def _fit_settings(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, int]:
    configured = config.get("fit", {})
    return {
        name: (
            getattr(args, name, None)
            if getattr(args, name, None) is not None
            else int(configured.get(name, default))
        )
        for name, default in (("warmup", 1000), ("samples", 1000), ("chains", 1), ("seed", 0))
    }


def _held_out_judgment_probabilities(
    artifact: PosteriorArtifact, training, held_out, catalogue: pd.DataFrame,
    rng: np.random.Generator,
) -> np.ndarray:
    posterior, train_maps = artifact.posterior, training.category_maps
    test_maps = held_out.category_maps
    draws = np.asarray(posterior["judge_effects"]).shape[0]
    metadata = catalogue.copy()
    metadata["model_id"] = metadata["model_id"].astype(str)
    metadata["release_day"] = (
        pd.to_datetime(metadata["release"], errors="coerce") - RELEASE_EPOCH
    ).dt.days
    metadata = metadata.set_index("model_id")
    scales = artifact.metadata["training_scales"]

    judge_effects = np.empty((draws, held_out.num_judges, 2))
    length_deviation = np.empty((draws, held_out.num_judges)) if held_out.include_length else None
    for judge, target in test_maps["judges"].items():
        if judge in train_maps["judges"]:
            source = train_maps["judges"][judge]
            judge_effects[:, target] = posterior["judge_effects"][:, source]
            if length_deviation is not None:
                length_deviation[:, target] = posterior["length_deviation"][:, source]
            continue
        population = np.asarray(posterior["judge_population"])
        design = [1.0]
        if population.shape[2] == 3:
            for name, value in (
                ("judge_theta", float(metadata.at[judge, "theta"])),
                ("judge_release_days", float(metadata.at[judge, "release_day"])),
            ):
                design.append((value - float(scales[name]["mean"])) / float(scales[name]["scale"]))
        raw = rng.normal(size=(draws, 2))
        residual = np.einsum("di,dji->dj", raw, np.asarray(posterior["judge_corr"]))
        judge_effects[:, target] = (
            np.einsum("dij,j->di", population, np.asarray(design))
            + residual * np.asarray(posterior["judge_scales"])
        )
        if length_deviation is not None:
            length_deviation[:, target] = rng.normal(size=draws) * np.asarray(posterior["length_scale"])

    candidate_effects = np.empty((draws, held_out.num_models))
    for model, target in test_maps["models"].items():
        if model in train_maps["models"]:
            candidate_effects[:, target] = posterior["candidate_effects"][:, train_maps["models"][model]]
        else:
            candidate_effects[:, target] = rng.normal(size=draws) * np.asarray(posterior["candidate_scale"])

    item_effects = np.empty((draws, held_out.num_items))
    for item, target in test_maps["items"].items():
        if item in train_maps["items"]:
            item_effects[:, target] = posterior["item_effects"][:, train_maps["items"][item]]
        else:
            item_effects[:, target] = rng.normal(size=draws) * np.asarray(posterior["item_scale"])

    local = {
        "judge_effects": judge_effects,
        "candidate_effects": candidate_effects,
        "item_effects": item_effects,
    }
    for name in ("global_beta", "length_mean"):
        if name in posterior:
            local[name] = posterior[name]
    if length_deviation is not None:
        local["length_deviation"] = length_deviation
    return expit(left_logits(local, held_out))


def _judgment_validator(
    catalogue: pd.DataFrame, hashes: dict[str, str], settings: dict[str, int],
    rng: np.random.Generator,
):
    def fit_predict(train, test, spec, conditional_judges):
        combined = pd.concat([train, test], ignore_index=True)
        training_rows = np.arange(len(combined)) < len(train)
        standardized, scales = standardize_judgments(combined, training_rows)
        scale_map = scales.to_dict(orient="index")
        training = encode_judgments(
            standardized.iloc[:len(train)], catalogue, spec,
            training_scales=scale_map, source_hashes=hashes,
        )
        child_settings = {**settings, "seed": int(rng.integers(0, np.iinfo(np.int32).max))}
        artifact = fit_judgment_model(
            training, conditional_judges=conditional_judges, **child_settings,
        )
        held_out_rows = standardized.iloc[len(train):].copy()
        held_out_rows["chosen_left"] = 0  # Predictor encoding never consumes this dummy outcome.
        held_out = encode_judgments(
            held_out_rows, catalogue, spec,
            training_scales=scale_map, source_hashes=hashes,
        )
        return _held_out_judgment_probabilities(artifact, training, held_out, catalogue, rng)

    return fit_predict


def _token_validator(
    catalogue: pd.DataFrame, hashes: dict[str, str], settings: dict[str, int],
    rng: np.random.Generator,
):
    def fit_predict(train, test, role):
        child_settings = {**settings, "seed": int(rng.integers(0, np.iinfo(np.int32).max))}
        artifact = fit_token_model(
            train, role, reasoning_classes=catalogue, source_hashes=hashes, **child_settings,
        )
        count = min(
            np.asarray(value).shape[0]
            for value in artifact.posterior.values() if np.asarray(value).ndim
        )
        outcomes = {name: [] for name in ("input_tokens", "output_tokens", "reasoning_tokens")}
        if role == "candidate":
            outcomes["response_chars"] = []
        rows = test.copy()
        if "role" not in rows:
            rows["role"] = role
        for draw in range(count):
            usage = draw_token_usage(artifact, rows, rng, draw)
            for outcome in ("input_tokens", "output_tokens", "reasoning_tokens"):
                outcomes[outcome].append(np.log1p(usage[outcome].to_numpy(dtype=float)))
            if role == "candidate":
                lengths = draw_transcript_lengths(artifact, rows, rng, draw)
                outcomes["response_chars"].append(np.log1p(lengths))
        result = {}
        for outcome, values in outcomes.items():
            draws = np.asarray(values)
            result[f"{outcome}_mean"] = draws.mean(axis=0)
            result[f"{outcome}_p5"] = np.quantile(draws, 0.05, axis=0)
            result[f"{outcome}_p95"] = np.quantile(draws, 0.95, axis=0)
        return pd.DataFrame(result, index=test.index)

    return fit_predict


def _fitted_artifacts(output: Path, hashes: dict[str, str]) -> dict[str, PosteriorArtifact]:
    _verify_generation(output, "model/manifest.json", _MODEL_FILES, "fit", "model")
    artifacts = {
        name: load_artifact(output / f"model/{name}")
        for name in (*JUDGMENT_SPECS, "candidate_token", "judge_token")
    }
    for name, artifact in artifacts.items():
        if artifact.metadata.get("source_hashes") != hashes:
            raise CliError(f"{name} artifact source hash mismatch; rerun `mc-cost fit`")
    for name, role in (("candidate_token", "candidate"), ("judge_token", "judge")):
        if artifacts[name].metadata.get("role") != role:
            raise CliError(f"{name} artifact role mismatch; rerun `mc-cost fit`")
    return artifacts


def _selected_artifact_matches(
    artifact: PosteriorArtifact, selected: str, conditional_judges: bool,
) -> bool:
    spec = JUDGMENT_SPECS[selected]
    return (
        artifact.metadata.get("feature_order") == list(spec.global_features)
        and artifact.metadata.get("include_length") is spec.include_length
        and artifact.metadata.get("conditional_judges") is conditional_judges
        and artifact.metadata.get("presentation_include_reasoning") is False
        and isinstance(artifact.metadata.get("presentation_max_chars"), int)
        and not isinstance(artifact.metadata.get("presentation_max_chars"), bool)
        and artifact.metadata["presentation_max_chars"] > 0
        and artifact.metadata.get("presentation_truncate_frac") is None
        and artifact.metadata.get("presentation_truncate_keep") == "first"
    )


def _judgment_support(
    judgments: pd.DataFrame, spec, conditional_judges: bool,
    scales: dict[str, dict[str, float]],
) -> tuple[np.ndarray, list[str], float]:
    predictor_names = ["theta_gap", *(
        FEATURE_COLUMNS.get(feature, feature) for feature in spec.global_features
    )]
    if spec.include_length:
        predictor_names.append("length_difference")
    support_frame = judgments.copy()
    if conditional_judges:
        missing = sorted({"judge_theta", "judge_release_day"} - set(judgments))
        if missing:
            raise CliError(
                "judgments lack conditional support predictors: " + ", ".join(missing)
            )
        support_frame = judgments.assign(judge_release_days=judgments["judge_release_day"])
        predictor_names.extend(["judge_theta", "judge_release_days"])
    predictor_names = list(dict.fromkeys(predictor_names))
    for name in predictor_names:
        if name in scales:
            support_frame[name] = (
                pd.to_numeric(support_frame[name], errors="raise") * scales[name]["scale"]
                + scales[name]["mean"]
            )
    predictors = support_frame[predictor_names].apply(pd.to_numeric, errors="raise").to_numpy(float)
    support = support_distance(predictors, predictors[:1])
    return predictors, predictor_names, float(support["threshold"].iloc[0])


_VALIDATION_SHARD_FILES = (
    "judgment_predictions.csv", "judgment_folds.csv",
    "token_candidate_predictions.csv", "token_judge_predictions.csv", "metadata.json",
)


def _validation_task_counts(
    judgments: pd.DataFrame, candidates: pd.DataFrame, judges: pd.DataFrame,
) -> tuple[int, int, int]:
    models = set(judgments["left_model"].astype(str)) | set(judgments["right_model"].astype(str))
    judgment_count = len(JUDGMENT_SPECS) * (
        2 * judgments["judge"].astype(str).nunique() + len(models)
    )

    def token_count(frame: pd.DataFrame, role: str) -> int:
        rows = frame.assign(role=role) if "role" not in frame else frame
        return len(rows[["model", "role", "reasoning_class"]].drop_duplicates())

    return judgment_count, token_count(candidates, "candidate"), token_count(judges, "judge")


def _validation_shard_metadata(
    hashes: dict[str, str], settings: dict[str, int], index: int, count: int, total: int,
) -> tuple[str, dict[str, Any]]:
    identity = {
        "data_hashes": hashes, "validation_config": settings,
        "shard_count": count, "task_total": total,
    }
    generation = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    return generation, {
        "schema_version": 1, **identity, "shard_index": index,
        "task_ordinals": list(range(index, total, count)),
    }


def _validation_shard_names(generation: str, index: int, count: int) -> tuple[str, list[str]]:
    relative = f"validation/shards/{generation}/{index:04d}-of-{count:04d}"
    return relative, [f"{relative}/{name}" for name in _VALIDATION_SHARD_FILES]


def _completed_validation_shard(
    output: Path, relative: str, files: list[str], expected: dict[str, Any],
) -> bool:
    manifest = f"{relative}/manifest.json"
    if not (output / manifest).is_file():
        return False
    _verify_generation(output, manifest, files, "validate", "validation shard")
    try:
        actual = json.loads((output / relative / "metadata.json").read_text())
    except (OSError, json.JSONDecodeError):
        raise CliError("invalid validation shard metadata") from None
    if actual != expected:
        raise CliError("stale validation shard metadata")
    return True


def _write_validation_shard(
    output: Path, relative: str, files: list[str], metadata: dict[str, Any],
    judgment_predictions: pd.DataFrame, judgment_folds: pd.DataFrame,
    candidate_predictions: pd.DataFrame, judge_predictions: pd.DataFrame,
) -> None:
    frames = {
        "judgment_predictions.csv": judgment_predictions,
        "judgment_folds.csv": judgment_folds,
        "token_candidate_predictions.csv": candidate_predictions,
        "token_judge_predictions.csv": judge_predictions,
    }
    with _temporary_directory(output.parent) as directory:
        stage = Path(directory) / relative
        stage.mkdir(parents=True)
        for name, frame in frames.items():
            frame.to_csv(stage / name, index=False)
        _write_json(stage / "metadata.json", metadata)
        _write_manifest(Path(directory), f"{relative}/manifest.json", files)
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise CliError("validation shard target already exists without a valid manifest")
        stage.rename(target)


def _read_shard_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _load_validation_shards(
    output: Path, hashes: dict[str, str], settings: dict[str, int], count: int,
    task_counts: tuple[int, int, int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    total = sum(task_counts)
    generation, _ = _validation_shard_metadata(hashes, settings, 0, count, total)
    frames = {name: [] for name in _VALIDATION_SHARD_FILES[:-1]}
    ordinals = []
    for index in range(count):
        relative, files = _validation_shard_names(generation, index, count)
        manifest = output / relative / "manifest.json"
        if not manifest.is_file():
            raise CliError(f"missing validation shard {index} of {count}")
        _verify_generation(
            output, str(manifest.relative_to(output)), files, "validate", "validation shard",
        )
        metadata = json.loads((output / relative / "metadata.json").read_text())
        if not isinstance(metadata, dict):
            raise CliError("invalid validation shard metadata")
        _, expected = _validation_shard_metadata(hashes, settings, index, count, total)
        expected.pop("task_ordinals")
        actual_ordinals = metadata.pop("task_ordinals", None)
        if not isinstance(actual_ordinals, list) or any(
            type(ordinal) is not int for ordinal in actual_ordinals
        ):
            raise CliError("invalid validation shard metadata")
        if metadata != expected:
            raise CliError("stale validation shard metadata")
        ordinals.extend(actual_ordinals)
        for name in frames:
            frames[name].append(_read_shard_csv(output / relative / name))
    if len(ordinals) != len(set(ordinals)):
        raise CliError("duplicate validation shard tasks")
    if sorted(ordinals) != list(range(total)):
        raise CliError("validation shards do not cover every task")

    def joined(name: str) -> pd.DataFrame:
        nonempty = [frame for frame in frames[name] if not frame.empty]
        return pd.concat(nonempty, ignore_index=True) if nonempty else pd.DataFrame()

    return tuple(joined(name) for name in frames)


def _validate(args: argparse.Namespace, config: dict[str, Any]) -> None:
    output = _output(config)
    sources = _fit_sources(output)
    hashes = {name: _hash_file(path) for name, path in sources.items()}
    fitted_artifacts = _fitted_artifacts(output, hashes)
    catalogue = pd.read_csv(sources["catalogue"])
    judgments = pd.read_parquet(sources["judgments"])
    candidates = pd.read_parquet(sources["candidate_tokens"])
    judges = pd.read_parquet(sources["judge_tokens"])
    settings = _fit_settings(args, config)
    if args.seed is not None:
        settings["seed"] = args.seed
    rng = np.random.default_rng(settings["seed"])
    shard_index = getattr(args, "shard_index", None)
    shard_count = getattr(args, "shard_count", None)
    reduce_shards = bool(getattr(args, "reduce_shards", False))
    if reduce_shards:
        if shard_count is None or shard_index is not None:
            raise CliError("--reduce-shards requires --shard-count and no --shard-index")
    elif (shard_index is None) != (shard_count is None):
        raise CliError("--shard-index and --shard-count must be supplied together")
    elif shard_index is not None and not 0 <= shard_index < shard_count:
        raise CliError("--shard-index must satisfy 0 <= index < shard-count")

    task_counts = _validation_task_counts(judgments, candidates, judges)
    if reduce_shards:
        (
            judgment_predictions, judgment_folds,
            candidate_predictions, judge_predictions,
        ) = _load_validation_shards(output, hashes, settings, shard_count, task_counts)
        candidate_summary, candidate_strata = summarize_token_predictions(
            candidate_predictions, "candidate",
        )
        judge_summary, judge_strata = summarize_token_predictions(judge_predictions, "judge")
    elif shard_index is not None:
        total = sum(task_counts)
        generation, metadata = _validation_shard_metadata(
            hashes, settings, shard_index, shard_count, total,
        )
        relative, files = _validation_shard_names(generation, shard_index, shard_count)
        if _completed_validation_shard(output, relative, files, metadata):
            return
        judgment_predictions, judgment_folds = cross_validate_judgments(
            judgments, _judgment_validator(catalogue, hashes, settings, rng),
            shard=(shard_index, shard_count, 0),
        )
        candidate_predictions, _, _ = validate_tokens(
            candidates, "candidate", _token_validator(catalogue, hashes, settings, rng),
            shard=(shard_index, shard_count, task_counts[0]),
        )
        judge_predictions, _, _ = validate_tokens(
            judges, "judge", _token_validator(catalogue, hashes, settings, rng),
            shard=(shard_index, shard_count, task_counts[0] + task_counts[1]),
        )
        _write_validation_shard(
            output, relative, files, metadata, judgment_predictions, judgment_folds,
            candidate_predictions, judge_predictions,
        )
        return
    else:
        judgment_predictions, judgment_folds = cross_validate_judgments(
            judgments, _judgment_validator(catalogue, hashes, settings, rng),
        )
        candidate_predictions, candidate_summary, candidate_strata = validate_tokens(
            candidates, "candidate", _token_validator(catalogue, hashes, settings, rng),
        )
        judge_predictions, judge_summary, judge_strata = validate_tokens(
            judges, "judge", _token_validator(catalogue, hashes, settings, rng),
        )
    selected = choose_judgment_spec(judgment_folds)
    selected_folds = judgment_folds.loc[judgment_folds["spec"].eq(selected["selected"])]
    pooled = selected_folds.loc[selected_folds["population"].eq("pooled")]
    conditional = selected_folds.loc[
        selected_folds["population"].eq("conditional") & selected_folds["fold"].isin(pooled["fold"])
    ]
    judge_gate = choose_judge_gate(conditional, pooled)
    spec = JUDGMENT_SPECS[selected["selected"]]
    scales = _load_scales(sources["scales"])
    predictors, predictor_names, support_threshold = _judgment_support(
        judgments, spec, bool(judge_gate["enabled"]), scales,
    )
    token_coverage = pd.concat([candidate_summary, judge_summary], ignore_index=True)
    payload = decision_payload(
        selected_spec=selected["selected"], judge_gate=judge_gate,
        token_coverage=token_coverage, predictors=predictors,
        predictor_names=predictor_names, support_threshold=support_threshold,
        data_hashes=hashes, validation_timestamp=datetime.now(timezone.utc).isoformat(),
    )
    payload["validation_config"] = {
        **settings, **({"shard_count": shard_count} if reduce_shards else {}),
    }
    expected_conditional = bool(judge_gate["enabled"])
    selected_artifact = fitted_artifacts[selected["selected"]]
    refitted_artifact = None
    if not _selected_artifact_matches(
        selected_artifact, selected["selected"], expected_conditional,
    ):
        full_data = encode_judgments(
            judgments, catalogue, spec,
            training_scales=scales, source_hashes=hashes,
        )
        refitted_artifact = _tag_judgment_artifact(
            fit_judgment_model(
                full_data, conditional_judges=expected_conditional, **settings,
            ),
            spec, hashes, expected_conditional,
        )
        selected_artifact = refitted_artifact
    if not _selected_artifact_matches(selected_artifact, selected["selected"], expected_conditional):
        raise CliError("selected full-data judgment artifact does not match the validated specification")

    replay_predictions = judgment_predictions.loc[
        judgment_predictions["spec"].eq(selected["selected"])
        & judgment_predictions["population"].eq("conditional" if judge_gate["enabled"] else "pooled")
        & judgment_predictions["validation"].eq("judge")
    ].copy()
    observed = judgments.reset_index(names="source_row")
    replay_predictions = replay_predictions.merge(observed, on="source_row", how="left", validate="many_to_one")
    replay_rows = []
    for position, row in enumerate(replay_predictions.to_dict("records")):
        replay_rows.append({
            "row_id": row["source_row"], "draw": int(row.get("draw", position)),
            "observed_choice": row["chosen_left"],
            "predicted_choice": int(rng.binomial(1, row["probability"])),
            **{name: row[name] for name in (
                "judge", "theta_gap", "oversight", "difficulty",
                "release_gap_days", "length_difference", "correct", "visible_reasoning",
            ) if name in row},
            **({"m1_is_left": row["major_is_left"]} if "major_is_left" in row else {}),
            "probability": row["probability"],
        })
    replay = posterior_predictive_replay(pd.DataFrame(replay_rows))
    votes = judgments.rename(columns={
        "left_model": "major", "right_model": "minor", "item": "item_id",
        "chosen_left": "picked_major",
    }).assign(fit="gpqa").loc[lambda frame: frame["major"].ne(frame["minor"])]
    recovery, recovery_warnings = ranking_recovery(
        votes, catalogue.rename(columns={"model_id": "model", "theta": "score"}),
    )

    names = {
        "judgment_predictions.csv": judgment_predictions,
        "judgment_folds.csv": judgment_folds,
        "token_candidate_predictions.csv": candidate_predictions,
        "token_candidate_summary.csv": candidate_summary,
        "token_candidate_length_strata.csv": candidate_strata,
        "token_judge_predictions.csv": judge_predictions,
        "token_judge_summary.csv": judge_summary,
        "token_judge_length_strata.csv": judge_strata,
        "posterior_predictive_replay.csv": replay,
        "ranking_recovery.csv": recovery,
        "ranking_warnings.csv": recovery_warnings,
    }
    with _temporary_directory(output.parent) as directory:
        stage = Path(directory) / "validation"
        stage.mkdir()
        for name, frame in names.items():
            frame.to_csv(stage / name, index=False)
        write_decision(stage / "decision.json", payload)
        extra = []
        if refitted_artifact is not None:
            save_artifact(refitted_artifact, Path(directory) / f"model/{selected['selected']}")
            extra = [
                f"model/{selected['selected']}.npz", f"model/{selected['selected']}.json",
            ]
        _publish(Path(directory), output, [
            *extra, *(f"validation/{name}" for name in names), "validation/decision.json",
        ])
        if extra:
            _write_manifest(output, "model/manifest.json", _MODEL_FILES)
        _write_manifest(
            output, "validation/manifest.json",
            [*(f"validation/{name}" for name in names), "validation/decision.json"],
        )


def _semantic_decision_hash(decision: dict[str, Any]) -> str:
    semantic = {key: value for key, value in decision.items() if key != "validation_timestamp"}
    encoded = json.dumps(semantic, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _capability_inputs(
    config: dict[str, Any], path: Path, catalogue: pd.DataFrame, irt: dict[str, np.ndarray],
) -> tuple[list[str], np.ndarray]:
    with np.load(path, allow_pickle=False) as stored:
        required = {"models", "draws", "source_names", "source_hashes"}
        if missing := required - set(stored.files):
            raise CliError(f"capability draws lack provenance arrays: {', '.join(sorted(missing))}")
        models = stored["models"].astype(str).tolist()
        draws = np.asarray(stored["draws"], dtype=float)
        source_names = stored["source_names"].astype(str).tolist()
        source_values = stored["source_hashes"].astype(str).tolist()
    expected_names = {"canonical_models", "model_pool", "irt_capabilities", "irt_posterior"}
    if "model_overrides" in config["paths"]:
        expected_names.add("model_overrides")
    if (
        len(source_names) != len(source_values) or len(source_names) != len(set(source_names))
        or set(source_names) != expected_names
    ):
        raise CliError("capability draws contain invalid source_names/source_hashes provenance")
    embedded = dict(zip(source_names, source_values))
    for name in sorted(expected_names):
        current = _hash_file(_require_source(config, name))
        if embedded[name] != current:
            label = "IRT posterior" if name == "irt_posterior" else name
            raise CliError(f"capability draws {label} hash mismatch; rerun `mc-cost catalogue`")

    catalogue_ids = set(catalogue["model_id"].astype(str))
    required_models = set()
    for column in ("judge_pool", "candidate_pool"):
        if column in catalogue:
            required_models.update(catalogue.loc[catalogue[column].map(pool_enabled), "model_id"].astype(str))
    if (
        len(models) != len(set(models)) or draws.ndim != 2 or draws.shape[1] != len(models)
        or not np.isfinite(draws).all() or not len(draws)
        or not set(models) <= catalogue_ids or not required_models <= set(models)
    ):
        raise CliError("capability model alignment does not match the current catalogue")
    a_key = "fixed_3pl__gpqa__a" if "fixed_3pl__gpqa__a" in irt else "a"
    if a_key not in irt or len(np.asarray(irt[a_key])) != len(draws):
        raise CliError("capability and IRT posterior draw counts are not aligned")
    return models, draws


def _provisional_decision(
    output: Path, sources: dict[str, Path], hashes: dict[str, str],
    fitted: dict[str, PosteriorArtifact],
) -> tuple[Path, dict[str, Any]]:
    selected = "core_release"
    decision_path = output / "provisional/decision.json"
    manifest = output / "provisional/manifest.json"
    if manifest.is_file():
        _verify_generation(
            output, "provisional/manifest.json", ["provisional/decision.json"],
            "export-dashboard --provisional", "provisional decision",
        )
        decision = require_decision(decision_path, hashes)
        if (
            decision.get("validation_status") != "provisional"
            or decision.get("selected_judgment_spec") != selected
            or decision.get("judge_covariate_gate", {}).get("enabled") is not True
        ):
            raise CliError("existing provisional decision is incompatible with the preview policy")
        return decision_path, decision
    if decision_path.exists() or manifest.exists():
        raise CliError("incomplete provisional decision generation")
    if not _selected_artifact_matches(fitted[selected], selected, True):
        raise CliError(
            "provisional export requires the conditional core_release artifact; rerun `mc-cost fit`"
        )
    predictors, predictor_names, threshold = _judgment_support(
        pd.read_parquet(sources["judgments"]), JUDGMENT_SPECS[selected], True,
        _load_scales(sources["scales"]),
    )
    decision = decision_payload(
        selected_spec=selected,
        judge_gate={"enabled": True, "reason": "fixed provisional preview policy"},
        token_coverage=pd.DataFrame(columns=("role", "reasoning_class", "outcome", "coverage_90")),
        predictors=predictors, predictor_names=predictor_names, support_threshold=threshold,
        data_hashes=hashes, validation_timestamp=datetime.now(timezone.utc).isoformat(),
    )
    decision["validation_status"] = "provisional"
    with _temporary_directory(output.parent) as directory:
        stage = Path(directory) / "provisional"
        stage.mkdir()
        write_decision(stage / "decision.json", decision)
        _write_manifest(
            Path(directory), "provisional/manifest.json", ["provisional/decision.json"],
        )
        target = output / "provisional"
        target.parent.mkdir(parents=True, exist_ok=True)
        stage.rename(target)
    return decision_path, decision


def _simulation_inputs(config: dict[str, Any], *, provisional: bool = False) -> dict[str, Any]:
    output = _output(config)
    sources = _fit_sources(output)
    hashes = {name: _hash_file(path) for name, path in sources.items()}
    fitted = _fitted_artifacts(output, hashes)
    if provisional:
        decision_path, decision = _provisional_decision(output, sources, hashes, fitted)
    else:
        decision_path = output / "validation/decision.json"
        _verify_generation(
            output, "validation/manifest.json", ["validation/decision.json"],
            "validate", "validation",
        )
        decision = require_decision(decision_path, hashes)
    selected = decision.get("selected_judgment_spec")
    if selected not in JUDGMENT_SPECS:
        raise CliError("decision.json does not select a sanctioned judgment specification")
    selected_spec = JUDGMENT_SPECS[selected]
    expected_spec = {
        "global_features": list(selected_spec.global_features),
        "include_length": selected_spec.include_length,
    }
    if decision.get("judgment_spec") != expected_spec:
        raise CliError("decision judgment specification does not match the sanctioned selection")
    artifacts = {
        "judgment": fitted[selected],
        "candidate_token": fitted["candidate_token"],
        "judge_token": fitted["judge_token"],
    }
    gate = decision.get("judge_covariate_gate", {}).get("enabled")
    if type(gate) is not bool:
        raise CliError("decision.json lacks a boolean judge-covariate gate")
    if not _selected_artifact_matches(artifacts["judgment"], selected, gate):
        raise CliError(
            "selected judgment specification or judge-covariate gate disagrees with its artifact; "
            f"rerun `mc-cost {'fit' if provisional else 'validate'}`"
        )

    irt_path = _require_source(config, "irt_posterior")
    with np.load(irt_path, allow_pickle=False) as stored:
        irt = {name: stored[name].copy() for name in stored.files}
    catalogue = pd.read_csv(sources["catalogue"])
    capability_models, capability_draws = _capability_inputs(
        config, sources["capability_draws"], catalogue, irt,
    )
    item_key = "fixed_3pl__gpqa__items" if "fixed_3pl__gpqa__items" in irt else "items"
    if item_key not in irt:
        raise CliError("IRT posterior lacks fixed_3pl GPQA item labels")
    source_hashes = {
        **hashes,
        "decision": _semantic_decision_hash(decision),
        "irt_posterior": _hash_file(irt_path),
    }
    artifact_paths = {
        "judgment": output / f"model/{selected}",
        "candidate_token": output / "model/candidate_token",
        "judge_token": output / "model/judge_token",
    }
    for name, path in artifact_paths.items():
        source_hashes[f"{name}_posterior"] = _hash_file(path.with_suffix(".npz"))
        source_hashes[f"{name}_metadata"] = _hash_file(path.with_suffix(".json"))
    return {
        "output": output, "catalogue": catalogue,
        "observed_cells": pd.read_parquet(sources["candidate_tokens"]),
        "capability_models": capability_models, "capability_draws": capability_draws,
        "irt": irt, "item_ids": irt[item_key].astype(str).tolist(),
        "judgment_artifact": artifacts["judgment"],
        "candidate_token_artifact": artifacts["candidate_token"],
        "judge_token_artifact": artifacts["judge_token"],
        "decision_path": decision_path, "decision": decision,
        "data_hashes": hashes, "source_hashes": source_hashes,
    }


def _simulation_defaults(config: dict[str, Any]) -> dict[str, Any]:
    values = config.get("simulation")
    if not isinstance(values, dict):
        raise CliError("configuration must contain a simulation object")
    return values


def _resolved_simulation(args: argparse.Namespace, config: dict[str, Any], mode: str) -> dict[str, Any]:
    defaults = _simulation_defaults(config)

    def value(argument: str, configured: str, fallback=None):
        supplied = getattr(args, argument, None)
        return supplied if supplied is not None else defaults.get(configured, fallback)

    resolved = {
        "mode": mode,
        "trials": value("trials", "trials", 1000),
        "items": value("items", "attempted_items"),
        "presentations": value("presentations", "presentations", 2),
        "reuse_responses": (
            args.reuse_responses
            if getattr(args, "reuse_responses", None) is not None
            else bool(defaults.get("reuse_responses", True))
        ),
        "accuracy_target": value("accuracy_target", "accuracy_target", "higher_theta"),
        "seed": value("seed", "seed", 0),
    }
    if mode == "automatic":
        resolved.update({
            "judges": value("judges", "judges", defaults.get("j")),
            "anchors": value("anchors", "anchors", defaults.get("m")),
            "comparators": value("comparators", "comparators", defaults.get("n")),
            "selected_judges": list(defaults.get("selected_judges", ())),
            "require_stronger": bool(defaults.get("require_stronger", True)),
        })
        missing = [name for name in ("judges", "anchors", "comparators") if resolved[name] is None]
        if missing:
            raise CliError(f"automatic simulation requires {', '.join('--' + name for name in missing)}")
    return resolved


def _scenario_name(resolved: dict[str, Any]) -> str:
    encoded = json.dumps(resolved, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()[:12]


def _comparison_source(
    resolved: dict[str, Any], defaults: dict[str, Any], catalogue: pd.DataFrame,
    manual: Path | None = None, *, capability_models,
):
    if manual is not None:
        return manual_comparisons(
            catalogue, pd.read_csv(manual), capability_models=capability_models,
        )
    automatic = AutomaticConfig(
        j=int(resolved["judges"]), m=int(resolved["anchors"]), n=int(resolved["comparators"]),
        attempted_items=int(resolved["items"]), presentations=int(resolved["presentations"]),
        selected_judges=tuple(defaults.get("selected_judges", ())),
        require_stronger=bool(resolved.get("require_stronger", True)),
    )
    return lambda rng: automatic_comparisons(
        catalogue, automatic, rng, capability_models=capability_models,
    )


def _call_simulation(comparisons, resolved: dict[str, Any], inputs: dict[str, Any]):
    simulation_config = SimulationConfig(
        trials=int(resolved["trials"]), attempted_items=int(resolved["items"]),
        presentations=int(resolved["presentations"]),
        reuse_responses=bool(resolved["reuse_responses"]), mode=resolved["mode"],
        accuracy_target=resolved["accuracy_target"], configuration=_scenario_name(resolved),
    )
    return simulate(
        comparisons, simulation_config,
        item_ids=inputs["item_ids"], observed_cells=inputs["observed_cells"],
        capability_models=inputs["capability_models"], capability_draws=inputs["capability_draws"],
        irt=inputs["irt"], judgment_artifact=inputs["judgment_artifact"],
        candidate_token_artifact=inputs["candidate_token_artifact"],
        judge_token_artifact=inputs["judge_token_artifact"], catalogue=inputs["catalogue"],
        decision_path=inputs["decision_path"], data_hashes=inputs["data_hashes"],
        seed=int(resolved["seed"]),
    )


def _simulate_command(args: argparse.Namespace, config: dict[str, Any]) -> None:
    inputs = _simulation_inputs(config)
    resolved = _resolved_simulation(args, config, "manual" if args.manual is not None else "automatic")
    if args.manual is not None:
        resolved["manual_comparisons"] = _records(pd.read_csv(args.manual))
    comparisons = _comparison_source(
        resolved, _simulation_defaults(config), inputs["catalogue"], args.manual,
        capability_models=inputs["capability_models"],
    )
    result = _call_simulation(comparisons, resolved, inputs)
    identity = {"config": resolved, "source_hashes": inputs["source_hashes"]}
    run_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()[:16]
    runs = inputs["output"] / "runs"
    with _temporary_directory(runs) as directory:
        stage = Path(directory) / run_id
        stage.mkdir()
        result.trials.to_parquet(stage / "trials.parquet", index=False)
        result.summary.to_csv(stage / "summary.csv", index=False)
        result.warnings.to_csv(stage / "warnings.csv", index=False)
        write_run_config(stage / "config.json", resolved, inputs["source_hashes"])
        target = runs / run_id
        run_files = ("trials.parquet", "summary.csv", "warnings.csv", "config.json")
        _publish(stage, target, list(run_files))
        _write_manifest(target, "manifest.json", run_files)


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records"))


def _dashboard_scenarios(
    dashboard: dict[str, Any], defaults: dict[str, Any],
) -> list[tuple[int, int, int, int, int, bool]]:
    keys = (
        "judge_counts", "anchor_counts", "comparator_counts",
        "attempted_item_counts", "presentation_counts",
    )
    axes = []
    for key in keys:
        values = dashboard.get(key)
        if (
            not isinstance(values, list) or not values
            or any(not isinstance(value, int) or isinstance(value, bool) or value < 1 for value in values)
        ):
            raise CliError(f"dashboard.{key} must be a nonempty list of positive integers")
        axes.append(sorted(set(values)))
    strengths = dashboard.get("strength_filters", [bool(defaults.get("require_stronger", True))])
    if (
        not isinstance(strengths, list) or not strengths
        or any(type(value) is not bool for value in strengths)
    ):
        raise CliError("dashboard.strength_filters must be a nonempty list of booleans")
    strengths = list(dict.fromkeys(strengths))
    mode = dashboard.get("grid_mode", "one_at_a_time")
    if mode == "cartesian":
        scenarios = list(product(*axes, strengths))
    elif mode != "one_at_a_time":
        raise CliError("dashboard.grid_mode must be one_at_a_time or cartesian")
    else:
        baseline = tuple([*(axis[0] for axis in axes), strengths[0]])
        scenarios = [baseline]
        for index, values in enumerate([*axes, strengths]):
            for value in values[1:]:
                scenario = list(baseline)
                scenario[index] = value
                scenarios.append(tuple(scenario))
    joint_keys = (
        "judges", "anchors", "comparators", "attempted_items",
        "presentations", "require_stronger",
    )
    joint = dashboard.get("joint_scenarios", [])
    if not isinstance(joint, list):
        raise CliError("dashboard.joint_scenarios must be a list of exact scenario objects")
    for index, row in enumerate(joint):
        if not isinstance(row, dict) or set(row) != set(joint_keys):
            raise CliError(
                f"dashboard.joint_scenarios[{index}] must contain exactly: "
                + ", ".join(joint_keys)
            )
        if any(
            not isinstance(row[key], int) or isinstance(row[key], bool) or row[key] < 1
            for key in joint_keys[:-1]
        ) or type(row["require_stronger"]) is not bool:
            raise CliError(
                f"dashboard.joint_scenarios[{index}] requires positive integer counts "
                "and boolean require_stronger"
            )
        scenarios.append(tuple(row[key] for key in joint_keys))
    return list(dict.fromkeys(scenarios))


def _dashboard_manual_bank(
    catalogue: pd.DataFrame, configured: Any, *, seed: int, limit: Any, capability_models,
) -> tuple[list[dict[str, str]], dict[str, Any] | None]:
    if not isinstance(configured, list):
        raise CliError("dashboard.manual_triples must be a list")
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
        raise CliError("dashboard.manual_triple_limit must be a non-negative integer")

    explicit = []
    if configured:
        comparisons, _ = manual_comparisons(
            catalogue, pd.DataFrame(configured), capability_models=capability_models,
        )
        for record in _records(comparisons):
            identity = {key: record.get(key) for key in _MANUAL_COLUMNS}
            if any(not isinstance(identity[key], str) or not identity[key] for key in _MANUAL_COLUMNS):
                raise CliError(
                    "dashboard.manual_triples require reasoning-class overrides when catalogue defaults are missing"
                )
            explicit.append({**identity, "manual_source": "configured"})

    eligible = catalogue.loc[prediction_eligible(catalogue, capability_models)]
    classes = eligible.set_index("model_id")["reasoning_class"].to_dict()
    default_models = sorted(
        str(model) for model, reasoning_class in classes.items()
        if isinstance(reasoning_class, str) and reasoning_class in {"reasoning", "non_reasoning"}
    )
    key = lambda row: tuple(row[name] for name in _MANUAL_COLUMNS)
    explicit_by_key = {key(row): row for row in explicit}
    model_count = len(default_models)
    default_count = model_count * model_count * max(model_count - 1, 0)

    def default_row(index: int) -> dict[str, str]:
        judge_index, remainder = divmod(index, model_count * (model_count - 1))
        m1_index, compressed_m2 = divmod(remainder, model_count - 1)
        m2_index = compressed_m2 + int(compressed_m2 >= m1_index)
        judge, m1, m2 = (
            default_models[judge_index], default_models[m1_index], default_models[m2_index],
        )
        return {
            "judge": judge, "m1": m1, "m2": m2,
            "judge_reasoning_class": classes[judge],
            "m1_reasoning_class": classes[m1], "m2_reasoning_class": classes[m2],
            "manual_source": "deterministic_default_sample",
        }

    def is_default(identity: tuple[str, ...]) -> bool:
        judge, m1, m2, judge_class, m1_class, m2_class = identity
        return (
            judge in classes and m1 in classes and m2 in classes and m1 != m2
            and (judge_class, m1_class, m2_class) == (classes[judge], classes[m1], classes[m2])
        )

    overlap = sum(is_default(identity) for identity in explicit_by_key)
    candidate_count = 0 if limit == 0 else min(default_count, limit + overlap)
    indexes = (
        np.random.default_rng(seed).choice(default_count, candidate_count, replace=False)
        if candidate_count and candidate_count < default_count else range(candidate_count)
    )
    sampled = []
    for index in indexes:
        row = default_row(int(index))
        if key(row) not in explicit_by_key:
            sampled.append(row)
            if len(sampled) == limit:
                break
    exported_by_key = {**explicit_by_key, **{key(row): row for row in sampled}}
    total_count = default_count + sum(not is_default(identity) for identity in explicit_by_key)
    omitted_count = total_count - len(exported_by_key)
    warning = None
    if omitted_count:
        omitted_hash = hashlib.sha256(json.dumps({
            "models": [[model, classes[model]] for model in default_models],
            "excluded": [list(identity) for identity in sorted(exported_by_key) if is_default(identity)],
            "rule": "judge,m1,m2; m1 != m2",
        }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        warning = {
            "code": "manual_bank_truncated",
            "message": (
                f"exported {len(exported_by_key)} of {total_count} exact manual identities; "
                "add an omitted identity to dashboard.manual_triples and rerun mc-cost export-dashboard"
            ),
            "total_count": total_count, "exported_count": len(exported_by_key),
            "omitted_count": omitted_count, "omitted_population_hash": omitted_hash,
        }
    return [exported_by_key[name] for name in sorted(exported_by_key)], warning


def _dashboard_context(
    resolved: dict[str, Any], dashboard: dict[str, Any], *, require_stronger: bool | None,
    manual_identity: dict[str, str] | None = None, manual_source: str | None = None,
) -> dict[str, Any]:
    judge_pool = dashboard.get("judge_pool_column", "judge_pool")
    candidate_pool = dashboard.get("candidate_pool_column", "candidate_pool")
    j = int(resolved["judges"]) if "judges" in resolved else None
    m = int(resolved["anchors"]) if "anchors" in resolved else None
    n = int(resolved["comparators"]) if "comparators" in resolved else None
    identity = dict(manual_identity) if manual_identity is not None else None
    normalized = {
        "trials": int(resolved["trials"]), "seed": int(resolved["seed"]),
        "response_policy": "reuse_responses" if resolved["reuse_responses"] else "generate_responses",
        "accuracy_target": str(resolved["accuracy_target"]),
        "selected_judges": list(resolved.get("selected_judges", ())),
        "strength_filter": require_stronger,
        "grid": {"J": j, "M": m, "N": n, "attempted_items": int(resolved["items"])},
        "presentations": int(resolved["presentations"]),
        "manual": {
            "limit": int(dashboard.get("manual_triple_limit", 0)),
            "source": manual_source, "identity": identity,
        },
        "pools": {
            "judge_column": judge_pool, "candidate_column": candidate_pool,
            "comparator": "all_released",
        },
    }
    context = {
        "configuration": _scenario_name(resolved), "mode": resolved["mode"],
        "trials": normalized["trials"], "seed": normalized["seed"],
        "response_policy": normalized["response_policy"],
        "accuracy_target": normalized["accuracy_target"],
        "selected_judges": normalized["selected_judges"],
        "J": j, "M": m, "N": n, "attempted_items": int(resolved["items"]),
        "presentations": normalized["presentations"], "strength_filter": require_stronger,
        "judge_pool": judge_pool, "anchor_pool": candidate_pool,
        "comparator_pool": "all_released", "manual_triple_limit": normalized["manual"]["limit"],
        "manual_source": manual_source, "resolved_configuration": normalized,
    }
    if identity is not None:
        context.update(identity)
    return context


def _dashboard_selection_catalogue(
    catalogue: pd.DataFrame, dashboard: dict[str, Any],
) -> pd.DataFrame:
    result = catalogue.copy()
    for target, setting in (
        ("judge_pool", "judge_pool_column"), ("candidate_pool", "candidate_pool_column"),
    ):
        source = dashboard.get(setting, target)
        if not isinstance(source, str) or not source or source not in result:
            raise CliError(f"dashboard.{setting} must name a catalogue column")
        result[target] = result[source]
    return result


def _dashboard_records(
    result, context: dict[str, Any], catalogue: pd.DataFrame,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    models = catalogue.set_index("model_id", drop=False)
    theta = pd.to_numeric(models.get("theta"), errors="coerce").to_dict()

    def is_observed(model: str, role: str) -> bool:
        column = "observed_candidate" if role == "candidate" else "observed_judge"
        return model in models.index and column in models and pool_enabled(models.at[model, column])

    def identity_observed(identity: dict[str, Any]) -> bool:
        return all(
            identity.get(role) in models.index
            and is_observed(identity[role], "judge" if role == "judge" else "candidate")
            and identity.get(f"{role}_reasoning_class") == models.at[identity[role], "reasoning_class"]
            for role in ("judge", "m1", "m2")
        )

    manual_identity = (
        {key: context.get(key) for key in _MANUAL_COLUMNS}
        if all(isinstance(context.get(key), str) and context[key] for key in _MANUAL_COLUMNS)
        else None
    )
    scenario_identities = []
    if manual_identity is not None:
        scenario_identities.append(manual_identity)
    else:
        seen = set()
        for source in _records(result.summary):
            identities = tuple(source.get(role) for role in ("judge", "m1", "m2"))
            if any(identity is None for identity in identities) or identities in seen:
                continue
            seen.add(identities)
            scenario_identities.append({
                "judge": identities[0], "m1": identities[1], "m2": identities[2],
                **{
                    f"{role}_reasoning_class": (
                        models.at[model, "reasoning_class"] if model in models.index else None
                    )
                    for role, model in zip(("judge", "m1", "m2"), identities)
                },
            })
    scenario_observed = bool(scenario_identities) and all(map(identity_observed, scenario_identities))
    scenario_status = "observed" if scenario_observed else "predicted"
    support_warnings = [
        row for row in _records(result.warnings)
        if row.get("code") in {
            "missing_prediction_metadata", "out_of_support",
            "outside_training_range", "outside_joint_support",
        }
    ]
    scenario_extrapolation = not scenario_observed or bool(support_warnings)
    support_evidence = [
        f"{identity['judge']}/{identity['judge_reasoning_class']} judge; "
        f"{identity['m1']}/{identity['m1_reasoning_class']} and "
        f"{identity['m2']}/{identity['m2_reasoning_class']} candidates: "
        f"{'observed' if identity_observed(identity) else 'predicted'}"
        for identity in scenario_identities
    ] or ["no comparison-level support evidence was exported"]
    support_evidence.extend(
        f"{warning['code']}: {warning.get('message', '')}" for warning in support_warnings
    )

    rows = []
    for row in _records(result.summary):
        original = dict(row)
        record = {**row, **context}
        if record.get("m1") in theta and record.get("m2") in theta:
            left, right = theta[record["m1"]], theta[record["m2"]]
            record["capability_gap"] = float(left - right) if np.isfinite([left, right]).all() else None
        original_identities = [original.get(role) for role in ("judge", "m1", "m2")]
        if all(identity in models.index for identity in original_identities):
            identity = manual_identity or {
                "judge": original_identities[0], "m1": original_identities[1],
                "m2": original_identities[2],
                **{
                    f"{role}_reasoning_class": models.at[model, "reasoning_class"]
                    for role, model in zip(("judge", "m1", "m2"), original_identities)
                },
            }
            observed = identity_observed(identity)
        elif original.get("model") in models.index and original.get("role") in {"candidate", "judge"}:
            observed = (
                is_observed(original["model"], original["role"])
                and original.get("reasoning_class") == models.at[original["model"], "reasoning_class"]
            )
        else:
            observed = scenario_observed
        record["row_prediction_status"] = "observed" if observed else "predicted"
        record["prediction_status"] = record["row_prediction_status"]
        record["scenario_prediction_status"] = scenario_status
        record["scenario_extrapolation"] = scenario_extrapolation
        record["extrapolation"] = scenario_extrapolation
        record["support_evidence"] = support_evidence
        rows.append(record)
    warning_rows = [{**row, **context} for row in _records(result.warnings)]
    return rows, warning_rows


def _dashboard_plan_hash(
    dashboard: dict[str, Any], defaults: dict[str, Any], trials: int, seed: int,
    provisional: bool, tasks: list[dict[str, Any]],
) -> str:
    return hashlib.sha256(json.dumps({
        "dashboard": dashboard, "simulation": defaults, "trials": trials, "seed": seed,
        "provisional": provisional, "tasks": tasks,
    }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _dashboard_shard_metadata(
    source_hashes: dict[str, str], plan_hash: str, status: str,
    index: int, count: int, total: int,
) -> tuple[str, dict[str, Any]]:
    identity = {
        "source_hashes": source_hashes, "plan_hash": plan_hash,
        "validation_status": status, "shard_count": count, "task_total": total,
    }
    generation = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    return generation, {
        "schema_version": 1, **identity, "shard_index": index,
        "task_ordinals": list(range(index, total, count)),
    }


def _dashboard_shard_names(generation: str, index: int, count: int) -> tuple[str, list[str]]:
    relative = f"dashboard/shards/{generation}/{index:04d}-of-{count:04d}"
    return relative, [f"{relative}/{name}" for name in _DASHBOARD_SHARD_FILES]


def _completed_dashboard_shard(
    output: Path, relative: str, files: list[str], expected: dict[str, Any],
) -> bool:
    manifest = f"{relative}/manifest.json"
    if not (output / manifest).is_file():
        return False
    _verify_generation(output, manifest, files, "export-dashboard", "dashboard shard")
    try:
        actual = json.loads((output / relative / "metadata.json").read_text())
    except (OSError, json.JSONDecodeError):
        raise CliError("invalid dashboard shard metadata") from None
    if actual != expected:
        raise CliError("stale dashboard shard metadata")
    return True


def _write_dashboard_shard(
    output: Path, relative: str, files: list[str], metadata: dict[str, Any],
    automatic: list[dict[str, Any]], manual: list[dict[str, Any]],
    ranking_recovery: list[dict[str, Any]], warnings: list[dict[str, Any]],
) -> None:
    arrays = {
        "automatic.json": automatic, "manual.json": manual,
        "ranking_recovery.json": ranking_recovery, "warnings.json": warnings,
    }
    with _temporary_directory(output.parent) as directory:
        stage = Path(directory) / relative
        stage.mkdir(parents=True)
        for name, rows in arrays.items():
            _write_json(stage / name, rows)
        _write_json(stage / "metadata.json", metadata)
        _write_manifest(Path(directory), f"{relative}/manifest.json", files)
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise CliError("dashboard shard target already exists without a valid manifest")
        stage.rename(target)


def _read_dashboard_shard_rows(path: Path) -> list[dict[str, Any]]:
    try:
        rows = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        raise CliError("invalid dashboard shard data") from None
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise CliError("invalid dashboard shard data")
    return rows


def _load_dashboard_shards(
    output: Path, source_hashes: dict[str, str], plan_hash: str,
    status: str, count: int, total: int,
) -> tuple[list[dict[str, Any]], ...]:
    generation, _ = _dashboard_shard_metadata(
        source_hashes, plan_hash, status, 0, count, total,
    )
    arrays = {name: [] for name in _DASHBOARD_SHARD_FILES[:-1]}
    ordinals = []
    ordinal_mismatch = False
    for index in range(count):
        relative, files = _dashboard_shard_names(generation, index, count)
        manifest = output / relative / "manifest.json"
        if not manifest.is_file():
            raise CliError(f"missing dashboard shard {index} of {count}")
        _verify_generation(
            output, str(manifest.relative_to(output)), files,
            "export-dashboard", "dashboard shard",
        )
        try:
            metadata = json.loads((output / relative / "metadata.json").read_text())
        except (OSError, json.JSONDecodeError):
            raise CliError("invalid dashboard shard metadata") from None
        if not isinstance(metadata, dict):
            raise CliError("invalid dashboard shard metadata")
        _, expected = _dashboard_shard_metadata(
            source_hashes, plan_hash, status, index, count, total,
        )
        actual_ordinals = metadata.get("task_ordinals")
        if not isinstance(actual_ordinals, list) or any(
            type(ordinal) is not int for ordinal in actual_ordinals
        ):
            raise CliError("invalid dashboard shard metadata")
        actual = {key: value for key, value in metadata.items() if key != "task_ordinals"}
        identity = {key: value for key, value in expected.items() if key != "task_ordinals"}
        if actual != identity:
            raise CliError("stale dashboard shard metadata")
        ordinal_mismatch |= actual_ordinals != expected["task_ordinals"]
        ordinals.extend(actual_ordinals)
        for name in arrays:
            arrays[name].extend(_read_dashboard_shard_rows(output / relative / name))
    if len(ordinals) != len(set(ordinals)):
        raise CliError("duplicate dashboard shard tasks")
    if sorted(ordinals) != list(range(total)):
        raise CliError("dashboard shards do not cover every task")
    if ordinal_mismatch:
        raise CliError("stale dashboard shard metadata")
    return tuple(arrays[name] for name in arrays)


def _export_dashboard(args: argparse.Namespace, config: dict[str, Any]) -> None:
    inputs = _simulation_inputs(config, provisional=args.provisional)
    dashboard = config.get("dashboard")
    if not isinstance(dashboard, dict):
        raise CliError("configuration must contain a dashboard object")
    defaults = _simulation_defaults(config)
    selection_catalogue = _dashboard_selection_catalogue(inputs["catalogue"], dashboard)
    trials = args.trials if args.trials is not None else int(defaults.get("trials", 1000))
    seed = args.seed if args.seed is not None else int(defaults.get("seed", 0))
    scenarios = _dashboard_scenarios(dashboard, defaults)
    manual_identities, truncation_warning = _dashboard_manual_bank(
        selection_catalogue, dashboard.get("manual_triples", []), seed=seed,
        limit=dashboard.get("manual_triple_limit", 0),
        capability_models=inputs["capability_models"],
    )
    cap = dashboard.get("max_judgment_calls", 2_000_000)
    if not isinstance(cap, int) or isinstance(cap, bool) or cap < 1:
        raise CliError("dashboard.max_judgment_calls must be a positive integer")
    planned = trials * sum(
        judges * anchors * comparators * items * presentations
        for judges, anchors, comparators, items, presentations, _ in scenarios
    )
    planned += (
        trials * len(manual_identities) * int(defaults.get("attempted_items"))
        * int(defaults.get("presentations", 2))
    )
    if planned > cap:
        raise CliError(
            f"dashboard planned {planned} presented judgments, above "
            f"dashboard.max_judgment_calls={cap}; reduce trials/grid/manual triples or raise the cap"
        )

    shard_index = getattr(args, "shard_index", None)
    shard_count = getattr(args, "shard_count", None)
    reduce_shards = bool(getattr(args, "reduce_shards", False))
    if reduce_shards:
        if shard_count is None or shard_index is not None:
            raise CliError("--reduce-shards requires --shard-count and no --shard-index")
    elif (shard_index is None) != (shard_count is None):
        raise CliError("--shard-index and --shard-count must be supplied together")
    elif shard_index is not None and not 0 <= shard_index < shard_count:
        raise CliError("--shard-index must satisfy 0 <= index < shard-count")

    tasks = [
        {"mode": "automatic", "scenario": list(scenario)} for scenario in scenarios
    ] + [
        {"mode": "manual", "identity": identity} for identity in manual_identities
    ]
    status = "provisional" if args.provisional else "validated"
    plan_hash = _dashboard_plan_hash(
        dashboard, defaults, trials, seed, bool(args.provisional), tasks,
    )
    total = len(tasks)
    if reduce_shards:
        automatic, manual, ranking_rows, warnings = _load_dashboard_shards(
            inputs["output"], inputs["source_hashes"], plan_hash,
            status, shard_count, total,
        )
        selected_scenarios, selected_manual = [], []
    else:
        selected_scenarios = [
            scenario for ordinal, scenario in enumerate(scenarios)
            if shard_index is None or ordinal % shard_count == shard_index
        ]
        selected_manual = [
            identity for ordinal, identity in enumerate(manual_identities, start=len(scenarios))
            if shard_index is None or ordinal % shard_count == shard_index
        ]
        automatic, manual, ranking_rows, warnings = [], [], [], []
    if shard_index is not None:
        generation, metadata = _dashboard_shard_metadata(
            inputs["source_hashes"], plan_hash, status,
            shard_index, shard_count, total,
        )
        relative, files = _dashboard_shard_names(generation, shard_index, shard_count)
        if _completed_dashboard_shard(inputs["output"], relative, files, metadata):
            return

    for judges, anchors, comparators, items, presentations, require_stronger in selected_scenarios:
        resolved = {
            "mode": "automatic", "trials": trials, "items": items,
            "presentations": presentations, "reuse_responses": bool(defaults.get("reuse_responses", True)),
            "accuracy_target": defaults.get("accuracy_target", "higher_theta"), "seed": seed,
            "judges": judges, "anchors": anchors, "comparators": comparators,
            "selected_judges": list(defaults.get("selected_judges", ())),
            "require_stronger": require_stronger,
        }
        context = _dashboard_context(
            resolved, dashboard, require_stronger=require_stronger,
        )
        try:
            result = _call_simulation(
                _comparison_source(
                    resolved, defaults, selection_catalogue,
                    capability_models=inputs["capability_models"],
                ),
                resolved, inputs,
            )
        except ValueError as error:
            message = str(error)
            if "requested " not in message and "eligible" not in message:
                raise
            warnings.append({**context, "code": "infeasible_grid", "message": message})
            continue
        scenario_rows, warning_rows = _dashboard_records(result, context, inputs["catalogue"])
        recovery_rows, recovery_warnings = dashboard_ranking_recovery(
            result, inputs["catalogue"], context,
        )
        automatic.extend(scenario_rows)
        ranking_rows.extend(recovery_rows)
        warnings.extend(warning_rows)
        warnings.extend(recovery_warnings)

    if truncation_warning and shard_index is None:
        warnings.append(truncation_warning)
    for exported_identity in selected_manual:
        manual_source = exported_identity["manual_source"]
        identity = {key: exported_identity[key] for key in _MANUAL_COLUMNS}
        resolved = {
            "mode": "manual", "trials": trials,
            "items": int(defaults.get("attempted_items")),
            "presentations": int(defaults.get("presentations", 2)),
            "reuse_responses": bool(defaults.get("reuse_responses", True)),
            "accuracy_target": defaults.get("accuracy_target", "higher_theta"), "seed": seed,
            "manual_comparisons": [identity],
        }
        context = _dashboard_context(
            resolved, dashboard, require_stronger=None,
            manual_identity=identity, manual_source=manual_source,
        )
        comparisons = manual_comparisons(
            selection_catalogue, pd.DataFrame([identity]),
            capability_models=inputs["capability_models"],
        )
        result = _call_simulation(comparisons, resolved, inputs)
        scenario_rows, warning_rows = _dashboard_records(result, context, inputs["catalogue"])
        recovery_rows, recovery_warnings = dashboard_ranking_recovery(
            result, inputs["catalogue"], context,
        )
        manual.extend(scenario_rows)
        ranking_rows.extend(recovery_rows)
        warnings.extend(warning_rows)
        warnings.extend(recovery_warnings)

    if shard_index is not None:
        _write_dashboard_shard(
            inputs["output"], relative, files, metadata,
            automatic, manual, ranking_rows, warnings,
        )
        return

    target = args.out or Path(dashboard.get("out", "interactive_viewer/data/mc_cost.json"))
    _write_json(target, dashboard_payload(
        generated_at=inputs["decision"].get("validation_timestamp"),
        source_hashes=inputs["source_hashes"], catalogue=_records(inputs["catalogue"]),
        validation={
            "status": status,
            "selected_spec": inputs["decision"]["selected_judgment_spec"],
            "judge_covariates": inputs["decision"]["judge_covariate_gate"]["enabled"],
            "message": (
                "Provisional — held-out validation pending."
                if args.provisional else "Held-out validation complete."
            ),
        },
        automatic=automatic, manual=manual, ranking_recovery=ranking_rows, warnings=warnings,
    ))


def _dispatch(args: argparse.Namespace, config: dict[str, Any]) -> None:
    if args.command == "catalogue":
        _catalogue(args, config)
    elif args.command == "prepare":
        _prepare(args, config)
    elif args.command == "fit":
        _fit(args, config)
    elif args.command == "validate":
        _validate(args, config)
    elif args.command == "simulate":
        _simulate_command(args, config)
    elif args.command == "export-dashboard":
        _export_dashboard(args, config)
    else:
        raise CliError(f"{args.command} orchestration is not implemented")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "simulate" and args.manual is not None:
            _manual_header(args.manual)
        config = _load_config(args.config)
        _dispatch(args, config)
    except (CliError, FileNotFoundError, ValueError) as error:
        print(f"mc-cost: error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
