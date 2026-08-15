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
from mc_cost.design import AutomaticConfig, automatic_comparisons, manual_comparisons
from mc_cost.gpqa import prepare as prepare_gpqa
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
    "judgment_standardization.json", "scales.json",
)
_MODEL_FILES = tuple(
    f"model/{name}.{suffix}"
    for name in (*JUDGMENT_SPECS, "candidate_token", "judge_token")
    for suffix in ("npz", "json")
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
    catalogue = build_catalogue(
        sources["canonical_models"], sources["model_pool"], sources["irt_capabilities"],
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
    models = _require_source(config, "judge_models")
    responses = _require_source(config, "epoch_responses")
    roots = config["paths"].get("candidate_log_roots", [])
    if not isinstance(roots, list) or any(not isinstance(root, str) for root in roots):
        raise CliError("configuration paths.candidate_log_roots must be a list of paths")
    root_paths = [Path(root) for root in roots]
    for root in root_paths:
        if not root.is_dir():
            raise CliError(f"configured candidate log root does not exist: {root}")
    catalogue = pd.read_csv(output / "catalogue.csv")
    items = pd.read_csv(items_path)
    attempts, _, _ = prepare_gpqa(logs, models, responses)
    judgments = prepare_judgments(attempts, catalogue, items)
    candidates = candidate_token_rows(root_paths, catalogue, items)
    judges = judge_token_rows(judgments)
    with _temporary_directory(output.parent) as directory:
        stage = Path(directory)
        paths = write_prepared_tables(
            stage, judgments, candidates, judges, training_rows=np.ones(len(judgments), dtype=bool),
        )
        shutil.copyfile(paths["judgment_standardization"], stage / "scales.json")
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
    )


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
    predictor_names = ["theta_gap", *(
        FEATURE_COLUMNS.get(feature, feature) for feature in spec.global_features
    )]
    if spec.include_length:
        predictor_names.append("length_difference")
    predictor_names = list(dict.fromkeys(predictor_names))
    predictors = judgments[predictor_names].apply(pd.to_numeric, errors="raise").to_numpy(float)
    support = support_distance(predictors, predictors[:1])
    token_coverage = pd.concat([candidate_summary, judge_summary], ignore_index=True)
    payload = decision_payload(
        selected_spec=selected["selected"], judge_gate=judge_gate,
        token_coverage=token_coverage, predictors=predictors,
        predictor_names=predictor_names, support_threshold=float(support["threshold"].iloc[0]),
        data_hashes=hashes, validation_timestamp=datetime.now(timezone.utc).isoformat(),
    )
    payload["validation_config"] = settings
    expected_conditional = bool(judge_gate["enabled"])
    selected_artifact = fitted_artifacts[selected["selected"]]
    refitted_artifact = None
    if not _selected_artifact_matches(
        selected_artifact, selected["selected"], expected_conditional,
    ):
        full_data = encode_judgments(
            judgments, catalogue, spec,
            training_scales=_load_scales(sources["scales"]), source_hashes=hashes,
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
    }).assign(fit="gpqa")
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


def _simulation_inputs(config: dict[str, Any]) -> dict[str, Any]:
    output = _output(config)
    sources = _fit_sources(output)
    hashes = {name: _hash_file(path) for name, path in sources.items()}
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
    fitted = _fitted_artifacts(output, hashes)
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
            "rerun `mc-cost validate`"
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
    manual: Path | None = None,
):
    if manual is not None:
        return manual_comparisons(catalogue, pd.read_csv(manual))
    automatic = AutomaticConfig(
        j=int(resolved["judges"]), m=int(resolved["anchors"]), n=int(resolved["comparators"]),
        attempted_items=int(resolved["items"]), presentations=int(resolved["presentations"]),
        selected_judges=tuple(defaults.get("selected_judges", ())),
        require_stronger=bool(resolved.get("require_stronger", True)),
    )
    return lambda rng: automatic_comparisons(catalogue, automatic, rng)


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


def _dashboard_manual_bank(
    catalogue: pd.DataFrame, configured: Any, *, seed: int, limit: Any,
) -> tuple[list[dict[str, str]], dict[str, Any] | None]:
    if not isinstance(configured, list):
        raise CliError("dashboard.manual_triples must be a list")
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
        raise CliError("dashboard.manual_triple_limit must be a non-negative integer")

    explicit = []
    if configured:
        comparisons, _ = manual_comparisons(catalogue, pd.DataFrame(configured))
        for record in _records(comparisons):
            identity = {key: record.get(key) for key in _MANUAL_COLUMNS}
            if any(not isinstance(identity[key], str) or not identity[key] for key in _MANUAL_COLUMNS):
                raise CliError(
                    "dashboard.manual_triples require reasoning-class overrides when catalogue defaults are missing"
                )
            explicit.append(identity)

    classes = catalogue.set_index("model_id")["reasoning_class"].to_dict()
    default_models = sorted(
        str(model) for model, reasoning_class in classes.items()
        if isinstance(reasoning_class, str) and reasoning_class in {"reasoning", "non_reasoning"}
    )
    defaults = [
        {
            "judge": judge, "m1": m1, "m2": m2,
            "judge_reasoning_class": classes[judge],
            "m1_reasoning_class": classes[m1],
            "m2_reasoning_class": classes[m2],
        }
        for judge, m1, m2 in product(default_models, repeat=3)
    ]
    key = lambda row: tuple(row[name] for name in _MANUAL_COLUMNS)
    explicit_by_key = {key(row): row for row in explicit}
    defaults_by_key = {key(row): row for row in defaults}
    candidates = [defaults_by_key[name] for name in sorted(defaults_by_key.keys() - explicit_by_key.keys())]
    if len(candidates) > limit:
        indexes = np.random.default_rng(seed).choice(len(candidates), limit, replace=False)
        sampled = [candidates[int(index)] for index in indexes]
    else:
        sampled = candidates
    exported_by_key = {**explicit_by_key, **{key(row): row for row in sampled}}
    all_by_key = {**defaults_by_key, **explicit_by_key}
    omitted = [all_by_key[name] for name in sorted(all_by_key.keys() - exported_by_key.keys())]
    warning = None
    if omitted:
        warning = {
            "code": "manual_bank_truncated",
            "message": (
                f"exported {len(exported_by_key)} of {len(all_by_key)} exact manual identities; "
                "add an omitted identity to dashboard.manual_triples and rerun mc-cost export-dashboard"
            ),
            "total_count": len(all_by_key), "exported_count": len(exported_by_key),
            "omitted_count": len(omitted), "omitted_identities": omitted,
        }
    return [exported_by_key[name] for name in sorted(exported_by_key)], warning


def _dashboard_context(
    resolved: dict[str, Any], dashboard: dict[str, Any], *, require_stronger: bool,
) -> dict[str, Any]:
    return {
        "configuration": _scenario_name(resolved), "mode": "automatic",
        "J": int(resolved["judges"]), "M": int(resolved["anchors"]),
        "N": int(resolved["comparators"]), "attempted_items": int(resolved["items"]),
        "presentations": int(resolved["presentations"]), "strength_filter": require_stronger,
        "judge_pool": str(dashboard.get("judge_pool_name", "Catalogue judge pool")),
        "anchor_pool": str(dashboard.get("anchor_pool_name", "Catalogue candidate pool")),
        "comparator_pool": str(dashboard.get("comparator_pool_name", "All released catalogue models")),
    }


def _dashboard_records(
    result, context: dict[str, Any], catalogue: pd.DataFrame,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    models = catalogue.set_index("model_id", drop=False)
    theta = pd.to_numeric(models.get("theta"), errors="coerce").to_dict()

    def is_observed(model: str, role: str) -> bool:
        column = "observed_candidate" if role == "candidate" else "observed_judge"
        return column in models and pool_enabled(models.at[model, column])

    rows = []
    for row in _records(result.summary):
        record = {**context, **row}
        if record.get("m1") in theta and record.get("m2") in theta:
            left, right = theta[record["m1"]], theta[record["m2"]]
            record["capability_gap"] = float(left - right) if np.isfinite([left, right]).all() else None
        identities = [record.get(role) for role in ("judge", "m1", "m2")]
        if all(identity in models.index for identity in identities):
            observed = (
                is_observed(identities[0], "judge")
                and all(is_observed(identity, "candidate") for identity in identities[1:])
                and all(
                    record.get(f"{role}_reasoning_class") == models.at[identity, "reasoning_class"]
                    for role, identity in zip(("judge", "m1", "m2"), identities)
                )
            )
        elif record.get("model") in models.index and record.get("role") in {"candidate", "judge"}:
            observed = (
                is_observed(record["model"], record["role"])
                and record.get("reasoning_class") == models.at[record["model"], "reasoning_class"]
            )
        else:
            observed = False
        record["prediction_status"] = "observed" if observed else "predicted"
        record["extrapolation"] = not observed
        rows.append(record)
    warning_rows = [{**context, **row} for row in _records(result.warnings)]
    return rows, warning_rows


def _export_dashboard(args: argparse.Namespace, config: dict[str, Any]) -> None:
    inputs = _simulation_inputs(config)
    dashboard = config.get("dashboard")
    if not isinstance(dashboard, dict):
        raise CliError("configuration must contain a dashboard object")
    defaults = _simulation_defaults(config)
    trials = args.trials if args.trials is not None else int(defaults.get("trials", 1000))
    seed = args.seed if args.seed is not None else int(defaults.get("seed", 0))
    keys = (
        "judge_counts", "anchor_counts", "comparator_counts",
        "attempted_item_counts", "presentation_counts",
    )
    grid = []
    for key in keys:
        values = dashboard.get(key)
        if (
            not isinstance(values, list) or not values
            or any(not isinstance(value, int) or isinstance(value, bool) or value < 1 for value in values)
        ):
            raise CliError(f"dashboard.{key} must be a nonempty list of positive integers")
        grid.append(sorted(set(values)))
    strength_filters = dashboard.get("strength_filters", [bool(defaults.get("require_stronger", True))])
    if (
        not isinstance(strength_filters, list) or not strength_filters
        or any(type(value) is not bool for value in strength_filters)
    ):
        raise CliError("dashboard.strength_filters must be a nonempty list of booleans")

    automatic, manual, ranking_rows, warnings = [], [], [], []
    for judges, anchors, comparators, items, presentations, require_stronger in product(
        *grid, sorted(set(strength_filters)),
    ):
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
                _comparison_source(resolved, defaults, inputs["catalogue"]), resolved, inputs,
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

    manual_identities, truncation_warning = _dashboard_manual_bank(
        inputs["catalogue"], dashboard.get("manual_triples", []), seed=seed,
        limit=dashboard.get("manual_triple_limit", 0),
    )
    if truncation_warning:
        warnings.append(truncation_warning)
    for identity in manual_identities:
        resolved = {
            "mode": "manual", "trials": trials,
            "items": int(defaults.get("attempted_items")),
            "presentations": int(defaults.get("presentations", 2)),
            "reuse_responses": bool(defaults.get("reuse_responses", True)),
            "accuracy_target": defaults.get("accuracy_target", "higher_theta"), "seed": seed,
            "manual_comparisons": [identity],
        }
        comparisons = manual_comparisons(inputs["catalogue"], pd.DataFrame([identity]))
        result = _call_simulation(comparisons, resolved, inputs)
        scenario_rows, warning_rows = _dashboard_records(result, identity, inputs["catalogue"])
        recovery_rows, recovery_warnings = dashboard_ranking_recovery(
            result, inputs["catalogue"], identity,
        )
        manual.extend(scenario_rows)
        ranking_rows.extend(recovery_rows)
        warnings.extend(warning_rows)
        warnings.extend(recovery_warnings)

    target = args.out or Path(dashboard.get("out", "interactive_viewer/data/mc_cost.json"))
    _write_json(target, dashboard_payload(
        generated_at=inputs["decision"].get("validation_timestamp"),
        source_hashes=inputs["source_hashes"], catalogue=_records(inputs["catalogue"]),
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
