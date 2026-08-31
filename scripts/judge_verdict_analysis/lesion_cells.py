"""Summarize saved lesion predictions by benchmark and judge×benchmark."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


INDEX_COLUMNS = {
    "fold", "specification", "kind", "variable", "reference", "artifact", "predictions",
}
PREDICTION_COLUMNS = {"fold", "judge", "bench", "item", "observed", "probability"}
SUMMARY_COLUMNS = (
    "specification", "kind", "variable", "reference", "artifact",
    "scope", "judge", "benchmark", "lesion_metric", "estimate", "lower", "upper",
    "delta_log_loss", "delta_log_loss_lower", "delta_log_loss_upper",
    "delta_accuracy", "delta_accuracy_lower", "delta_accuracy_upper",
    "fold_count", "row_count",
)
ITEM_DELTA_COLUMNS = (
    "specification", "kind", "variable", "reference", "artifact",
    "fold", "judge", "bench", "item", "delta_log_loss", "delta_accuracy", "row_count",
)
REFERENCE_METRIC_COLUMNS = (
    "reference", "judge", "benchmark", "log_loss", "accuracy",
    "fold_count", "row_count",
)


def _validate_production_inventory(
    kinds: pd.Series, aliases: dict[str, str], folds: list[str],
) -> None:
    expected_folds = {
        f"model-fold-{model}__item-{item}" for model in range(4) for item in range(4)
    }
    if set(folds) != expected_folds:
        raise ValueError("production fold inventory is not the exact 4×4 outer-fold plan")
    expected_kinds = {
        "lesion-feature": 117, "singular-feature": 117,
        "lesion-block": 9, "singular-block": 9,
        "lesion-family": 5, "singular-family": 5,
    }
    if kinds.value_counts().to_dict() != expected_kinds:
        raise ValueError("production specification kind inventory is not 117/117/9/9/5/5")
    expected_aliases = {
        "lesion-family-planning": "lesion-block-clause-plan",
        "singular-family-planning": "singular-block-clause-plan",
    }
    if aliases != expected_aliases:
        raise ValueError("production plan must contain exactly the two planning aliases")


def validate_artifact_index(
    path: str | Path, *, expected_folds: int = 16,
    expected_labelled: int = 262, expected_unique: int = 260,
) -> tuple[pd.DataFrame, dict[str, str]]:
    path = Path(path).resolve()
    table = pd.read_csv(path, keep_default_na=False)
    if missing := sorted(INDEX_COLUMNS - set(table)):
        raise ValueError(f"artifact index missing columns: {missing}")
    if table.duplicated(["fold", "specification"]).any():
        raise ValueError("artifact index has duplicate fold/specification rows")
    folds = table.fold.astype(str).drop_duplicates().tolist()
    if len(folds) != expected_folds:
        raise ValueError(f"expected {expected_folds} folds, found {len(folds)}")
    references = table[table.kind == "reference"]
    labelled = table[table.kind != "reference"]
    expected_references = {"full", "intercept-only"}
    for fold in folds:
        names = set(references.loc[references.fold.astype(str) == fold, "specification"])
        if names != expected_references:
            raise ValueError(f"fold {fold} must contain exactly full and intercept-only references")
    reference_plan = set(references[
        ["specification", "variable", "reference", "artifact"]
    ].itertuples(index=False, name=None))
    if reference_plan != {
        ("full", "full", "", "full"),
        ("intercept-only", "intercept-only", "", "intercept-only"),
    }:
        raise ValueError("reference rows do not identify the exact full/intercept-only plan")
    plan_columns = ["specification", "kind", "variable", "reference", "artifact"]
    first = None
    for fold in folds:
        current = (labelled[labelled.fold.astype(str) == fold][plan_columns]
                   .sort_values("specification").reset_index(drop=True))
        if first is None:
            first = current
        elif not current.equals(first):
            raise ValueError(f"labelled specification plan differs in fold {fold}")
    assert first is not None
    if len(first) != expected_labelled:
        raise ValueError(f"expected {expected_labelled} labelled specifications, found {len(first)}")
    if first.artifact.nunique() != expected_unique:
        raise ValueError(
            f"expected {expected_unique} unique artifacts, found {first.artifact.nunique()}")
    allowed_kinds = {
        "lesion-feature", "lesion-block", "lesion-family",
        "singular-feature", "singular-block", "singular-family",
    }
    if not set(first.kind).issubset(allowed_kinds):
        raise ValueError("artifact index contains an unknown labelled specification kind")
    wrong_lesion = first[first.kind.str.startswith("lesion") & first.reference.ne("full")]
    if not wrong_lesion.empty:
        raise ValueError(f"{wrong_lesion.iloc[0].specification} lesions must use full reference")
    wrong_singular = first[
        first.kind.str.startswith("singular") & first.reference.ne("intercept-only")]
    if not wrong_singular.empty:
        raise ValueError(
            f"{wrong_singular.iloc[0].specification} singulars must use intercept-only reference")
    canonical = set(first.specification)
    aliases = dict(first.loc[first.specification.ne(first.artifact), ["specification", "artifact"]]
                   .itertuples(index=False, name=None))
    if any(target not in canonical for target in aliases.values()):
        raise ValueError("alias target is not a labelled canonical specification")
    by_name = first.set_index("specification")
    if any(by_name.loc[target, "artifact"] != target for target in aliases.values()):
        raise ValueError("alias target must be a canonical artifact")
    if (expected_folds, expected_labelled, expected_unique) == (16, 262, 260):
        _validate_production_inventory(first.kind, aliases, folds)
    for fold in folds:
        fold_rows = table[table.fold.astype(str) == fold]
        by_name = fold_rows.set_index("specification")
        for alias, target in aliases.items():
            if by_name.loc[alias, "predictions"] != by_name.loc[target, "predictions"]:
                raise ValueError(f"alias prediction path differs for {alias} in fold {fold}")
        artifact_paths = fold_rows[fold_rows.kind != "reference"].drop_duplicates("artifact")
        if artifact_paths.predictions.duplicated().any():
            raise ValueError(f"distinct artifacts share a prediction path in fold {fold}")
    base = path.parent
    table = table.copy()
    table["fold"] = table.fold.astype(str)
    table["predictions"] = table.predictions.map(
        lambda value: (base / str(value)).resolve() if not Path(str(value)).is_absolute()
        else Path(str(value)).resolve())
    for prediction in table.predictions.drop_duplicates():
        if not prediction.is_file():
            raise FileNotFoundError(f"prediction artifact does not exist: {prediction}")
    return table, aliases


def _validate_predictions(frame: pd.DataFrame, expected_fold: str | None = None) -> pd.DataFrame:
    if frame.columns.duplicated().any():
        raise ValueError("prediction table has duplicate columns")
    if missing := sorted(PREDICTION_COLUMNS - set(frame)):
        raise ValueError(f"prediction table missing columns: {missing}")
    if frame.empty:
        raise ValueError("prediction table is empty")
    if any(frame[column].isna().any() for column in ("fold", "judge", "bench", "item")):
        raise ValueError("prediction row identifiers cannot be missing")
    if any(frame[column].astype(str).eq("").any() for column in ("fold", "judge", "bench", "item")):
        raise ValueError("prediction row identifiers cannot be empty")
    if expected_fold is not None and not frame.fold.astype(str).eq(expected_fold).all():
        raise ValueError(f"prediction table fold does not match {expected_fold}")
    observed = pd.to_numeric(frame.observed, errors="raise")
    if observed.isna().any() or not observed.isin([0, 1]).all():
        raise ValueError("observed values must be binary 0/1")
    probability = pd.to_numeric(frame.probability, errors="raise").to_numpy(float)
    if not np.isfinite(probability).all():
        raise ValueError("probabilities must be finite")
    if np.any((probability < 0) | (probability > 1)):
        raise ValueError("probabilities must lie in [0, 1]")
    output = frame.reset_index(drop=True).copy()
    output["observed"] = observed.astype(int).to_numpy()
    output["probability"] = probability
    return output


def read_predictions(path: str | Path, expected_fold: str) -> pd.DataFrame:
    return _validate_predictions(pd.read_parquet(path), expected_fold)


def _log_loss(observed: np.ndarray, probability: np.ndarray) -> np.ndarray:
    probability = np.clip(probability, 1e-12, 1 - 1e-12)
    return -(observed * np.log(probability) + (1 - observed) * np.log1p(-probability))


def pair_predictions(
    candidate: pd.DataFrame, reference: pd.DataFrame, reference_name: str,
) -> pd.DataFrame:
    candidate = _validate_predictions(candidate)
    reference = _validate_predictions(reference)
    compared = reference.columns.drop("probability")
    if (candidate.columns.tolist() != reference.columns.tolist()
            or len(candidate) != len(reference)
            or any(not candidate[column].equals(reference[column]) for column in compared)):
        raise ValueError("candidate predictions are not row-aligned with their reference")
    observed = reference.observed.to_numpy(dtype=int)
    candidate_probability = candidate.probability.to_numpy(dtype=float)
    reference_probability = reference.probability.to_numpy(dtype=float)
    candidate_loss = _log_loss(observed, candidate_probability)
    reference_loss = _log_loss(observed, reference_probability)
    candidate_accuracy = ((candidate_probability >= 0.5) == observed).astype(float)
    reference_accuracy = ((reference_probability >= 0.5) == observed).astype(float)
    output = reference.drop(columns="probability").copy()
    if reference_name == "full":
        output["delta_log_loss"] = candidate_loss - reference_loss
        output["delta_accuracy"] = reference_accuracy - candidate_accuracy
    elif reference_name == "intercept-only":
        output["delta_log_loss"] = reference_loss - candidate_loss
        output["delta_accuracy"] = candidate_accuracy - reference_accuracy
    else:
        raise ValueError(f"unknown reference: {reference_name}")
    return output


def reference_cell_metrics(plan: pd.DataFrame) -> pd.DataFrame:
    """Return absolute held-out metrics for full and intercept-only references."""
    rows = []
    for row in plan[plan.kind == "reference"].itertuples(index=False):
        frame = read_predictions(row.predictions, str(row.fold))
        observed = frame.observed.to_numpy(int)
        probability = frame.probability.to_numpy(float)
        values = frame[["fold", "judge", "bench"]].copy()
        values["reference"] = row.specification
        values["log_loss"] = _log_loss(observed, probability)
        values["accuracy"] = ((probability >= 0.5) == observed).astype(float)
        rows.append(values)
    combined = pd.concat(rows, ignore_index=True)
    output = combined.groupby(
        ["reference", "judge", "bench"], sort=True, as_index=False,
    ).agg(
        log_loss=("log_loss", "mean"), accuracy=("accuracy", "mean"),
        fold_count=("fold", "nunique"), row_count=("accuracy", "size"),
    ).rename(columns={"bench": "benchmark"})
    return output[list(REFERENCE_METRIC_COLUMNS)]


def summarize_cells(
    rows: pd.DataFrame, bootstrap_iterations: int = 1_000, seed: int = 0,
) -> pd.DataFrame:
    if bootstrap_iterations <= 0:
        raise ValueError("bootstrap_iterations must be positive")
    metadata = [
        name for name in ("specification", "kind", "variable", "reference", "artifact")
        if name in rows
    ]
    scoped = pd.concat([
        rows.assign(scope="benchmark", group_judge="", benchmark=rows.bench),
        rows.assign(
            scope="judge×benchmark", group_judge=rows.judge.astype(str),
            benchmark=rows.bench,
        ),
    ], ignore_index=True)
    output = []
    keys = [*metadata, "scope", "group_judge", "benchmark"]
    for key_values, group in scoped.groupby(keys, sort=True):
        if not isinstance(key_values, tuple):
            key_values = (key_values,)
        values_by_key = dict(zip(keys, key_values))
        scope = values_by_key.pop("scope")
        judge = values_by_key.pop("group_judge")
        benchmark = values_by_key.pop("benchmark")
        weight = (group.row_count.to_numpy(dtype=int)
                  if "row_count" in group else np.ones(len(group), dtype=int))
        clusters = pd.DataFrame({
            "item": group.item.to_numpy(),
            "row_count": weight,
            "delta_log_loss": group.delta_log_loss.to_numpy(float) * weight,
            "delta_accuracy": group.delta_accuracy.to_numpy(float) * weight,
        }).groupby("item", sort=False, as_index=False).sum()
        group_identity = {
            "artifact": values_by_key.get("artifact", values_by_key.get("specification", "rows")),
            "reference": values_by_key.get("reference", ""),
            "scope": scope, "judge": judge, "benchmark": benchmark, "seed": seed,
        }
        group_seed = int.from_bytes(hashlib.sha256(json.dumps(
            group_identity, sort_keys=True, separators=(",", ":")).encode()).digest()[:8])
        draws = np.random.default_rng(group_seed).integers(
            len(clusters), size=(bootstrap_iterations, len(clusters)))
        sampled_counts = clusters.row_count.to_numpy()[draws].sum(axis=1)
        values = {}
        for metric in ("delta_log_loss", "delta_accuracy"):
            estimate = float(np.average(group[metric], weights=weight))
            sampled = clusters[metric].to_numpy()[draws].sum(axis=1) / sampled_counts
            lower, upper = np.percentile(sampled, [2.5, 97.5])
            values[metric] = estimate
            values[f"{metric}_lower"] = float(lower)
            values[f"{metric}_upper"] = float(upper)
        output.append({
            **values_by_key, "scope": scope, "judge": judge, "benchmark": benchmark,
            **values, "lesion_metric": "log_loss",
            "estimate": values["delta_log_loss"],
            "lower": values["delta_log_loss_lower"],
            "upper": values["delta_log_loss_upper"],
            "fold_count": group.fold.nunique(), "row_count": int(weight.sum()),
        })
    return pd.DataFrame(output)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _signature(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _input_signatures(plan: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for path in sorted(set(plan.predictions), key=str):
        stat = path.stat()
        rows.append({
            "path": str(path), "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns, "sha256": _sha256(path),
        })
    return pd.DataFrame(rows)


def _write_json_once(path: Path, value: dict[str, object]) -> None:
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise RuntimeError(f"refusing to overwrite mismatched artifact: {path}")
        return
    partial = path.with_name(f".{path.name}.partial")
    partial.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    if json.loads(partial.read_text()) != value:
        raise RuntimeError(f"failed to validate partial artifact: {partial}")
    os.replace(partial, path)


def _write_frame_once(path: Path, frame: pd.DataFrame) -> None:
    def read(target: Path) -> pd.DataFrame:
        return (pd.read_parquet(target) if path.suffix == ".parquet"
                else pd.read_csv(target, keep_default_na=False))

    def same(target: Path) -> bool:
        try:
            pd.testing.assert_frame_equal(read(target), frame, check_dtype=False)
            return True
        except (AssertionError, OSError, ValueError):
            return False

    if path.exists():
        if not same(path):
            raise RuntimeError(f"refusing to overwrite mismatched artifact: {path}")
        return
    partial = path.with_name(f".{path.name}.partial")
    if not partial.exists():
        if path.suffix == ".parquet":
            frame.to_parquet(partial, index=False)
        else:
            frame.to_csv(partial, index=False)
    if not same(partial):
        raise RuntimeError(f"invalid partial artifact preserved at {partial}")
    os.replace(partial, path)


def _validate_summary(frame: pd.DataFrame, expected_labelled: int) -> None:
    if frame.columns.tolist() != list(SUMMARY_COLUMNS):
        raise RuntimeError("cell summary schema is not recoverable")
    if (frame.empty or frame.specification.nunique() != expected_labelled
            or frame.duplicated(["specification", "scope", "judge", "benchmark"]).any()
            or not frame.groupby("specification").scope.agg(set).eq(
                {"benchmark", "judge×benchmark"}).all()
            or not frame.lesion_metric.eq("log_loss").all()):
        raise RuntimeError("cell summary inventory is not recoverable")
    numeric = frame[[
        "estimate", "lower", "upper", "delta_log_loss", "delta_log_loss_lower",
        "delta_log_loss_upper", "delta_accuracy", "delta_accuracy_lower",
        "delta_accuracy_upper", "fold_count", "row_count",
    ]].to_numpy(dtype=float)
    if (not np.isfinite(numeric).all()
            or not np.array_equal(frame.estimate, frame.delta_log_loss)
            or not np.array_equal(frame.lower, frame.delta_log_loss_lower)
            or not np.array_equal(frame.upper, frame.delta_log_loss_upper)
            or (frame.fold_count.astype(int) <= 0).any()
            or (frame.row_count.astype(int) <= 0).any()):
        raise RuntimeError("cell summary values are not recoverable")


def _validate_item_deltas(frame: pd.DataFrame, expected_labelled: int) -> None:
    if frame.columns.tolist() != list(ITEM_DELTA_COLUMNS):
        raise RuntimeError("item-cell delta schema is not recoverable")
    keys = ["specification", "fold", "judge", "bench", "item"]
    if (frame.empty or frame.specification.nunique() != expected_labelled
            or frame.duplicated(keys).any()
            or (frame.row_count.astype(int) <= 0).any()
            or not np.isfinite(frame[
                ["delta_log_loss", "delta_accuracy"]].to_numpy(dtype=float)).all()):
        raise RuntimeError("item-cell delta values are not recoverable")


def _item_cell_deltas(plan: pd.DataFrame) -> pd.DataFrame:
    output = []
    plan_columns = ["specification", "kind", "variable", "reference", "artifact"]
    for fold, fold_rows in plan.groupby("fold", sort=False):
        by_specification = fold_rows.set_index("specification")
        references = {
            name: read_predictions(by_specification.loc[name, "predictions"], fold)
            for name in ("full", "intercept-only")
        }
        pair_predictions(references["intercept-only"], references["full"], "full")
        labelled = fold_rows[fold_rows.kind != "reference"]
        for _, artifact_rows in labelled.groupby("artifact", sort=False):
            artifact_row = next(artifact_rows.itertuples(index=False))
            paired = pair_predictions(
                read_predictions(artifact_row.predictions, fold),
                references[artifact_row.reference], artifact_row.reference)
            base_cells = paired.groupby(
                ["fold", "judge", "bench", "item"], sort=False, as_index=False,
            ).agg(
                delta_log_loss=("delta_log_loss", "mean"),
                delta_accuracy=("delta_accuracy", "mean"),
                row_count=("delta_log_loss", "size"),
            )
            for row in artifact_rows.itertuples(index=False):
                cells = base_cells.copy()
                for name in reversed(plan_columns):
                    cells.insert(0, name, getattr(row, name))
                output.append(cells)
    return pd.concat(output, ignore_index=True)


def _settings(
    index: Path, plan: pd.DataFrame, aliases: dict[str, str],
    bootstrap_iterations: int, seed: int, write_item_deltas: bool,
    expected_inventory: tuple[int, int, int],
) -> dict[str, object]:
    first_fold = plan.fold.iloc[0]
    columns = ["specification", "kind", "variable", "reference", "artifact"]
    specification_plan = (plan[(plan.fold == first_fold) & (plan.kind != "reference")][columns]
                          .sort_values("specification").to_dict(orient="records"))
    return {
        "schema_version": 2,
        "source_artifact_index": str(index),
        "source_artifact_index_sha256": _sha256(index),
        "bootstrap_iterations": bootstrap_iterations,
        "seed": seed,
        "primary_metric": "log_loss",
        "write_item_deltas": write_item_deltas,
        "expected_inventory": list(expected_inventory),
        "folds": plan.fold.drop_duplicates().tolist(),
        "aliases": dict(sorted(aliases.items())),
        "plan_signature": _signature(specification_plan),
    }


def _validate_outputs(directory: Path, settings: dict[str, object]) -> dict[str, object]:
    try:
        saved_settings = json.loads((directory / "settings.json").read_text())
        manifest = json.loads((directory / "manifest.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise RuntimeError(f"output is not a completed lesion-cell summary: {directory}") from error
    expected_settings = {"signature": _signature(settings), "settings": settings}
    if saved_settings != expected_settings:
        raise RuntimeError(f"summary settings mismatch at {directory}")
    inventory = settings["expected_inventory"]
    expected_manifest = {
        "state": "completed", "schema_version": 2,
        "settings_signature": expected_settings["signature"],
        "plan_signature": settings["plan_signature"],
        "source_artifact_index_sha256": settings["source_artifact_index_sha256"],
        "fold_count": inventory[0], "labelled_specifications": inventory[1],
        "unique_artifacts": inventory[2],
        "bootstrap_iterations": settings["bootstrap_iterations"], "seed": settings["seed"],
        "primary_metric": "log_loss",
    }
    if any(manifest.get(name) != value for name, value in expected_manifest.items()):
        raise RuntimeError(f"invalid completion manifest at {directory}")
    expected_artifacts = {
        "settings.json", "input-signatures.csv", "cell-summary.csv",
        "reference-cell-metrics.csv",
        *(["item-cell-deltas.parquet"] if settings["write_item_deltas"] else []),
    }
    hashes = manifest.get("artifact_sha256")
    if not isinstance(hashes, dict) or set(hashes) != expected_artifacts:
        raise RuntimeError(f"completion manifest artifact inventory mismatch at {directory}")
    for name, expected_hash in hashes.items():
        path = directory / name
        if not path.is_file() or _sha256(path) != expected_hash:
            raise RuntimeError(f"completed output artifact mismatch: {path}")
    expected_labelled = int(settings["expected_inventory"][1])
    summary = pd.read_csv(directory / "cell-summary.csv", keep_default_na=False)
    _validate_summary(summary, expected_labelled)
    reference_metrics = pd.read_csv(
        directory / "reference-cell-metrics.csv", keep_default_na=False)
    if (reference_metrics.columns.tolist() != list(REFERENCE_METRIC_COLUMNS)
            or set(reference_metrics.reference) != {"full", "intercept-only"}
            or not np.isfinite(reference_metrics[
                ["log_loss", "accuracy", "fold_count", "row_count"]].to_numpy(float)).all()):
        raise RuntimeError("reference cell metrics are not recoverable")
    if settings["write_item_deltas"]:
        _validate_item_deltas(
            pd.read_parquet(directory / "item-cell-deltas.parquet"), expected_labelled)
    if (manifest.get("summary_rows") != len(summary)
            or (settings["write_item_deltas"] and manifest.get("item_delta_rows") != len(
                pd.read_parquet(directory / "item-cell-deltas.parquet")))):
        raise RuntimeError(f"completion manifest row counts mismatch at {directory}")
    return manifest


def build(
    artifact_index: str | Path, output_dir: str | Path, *,
    bootstrap_iterations: int = 1_000, seed: int = 0,
    write_item_deltas: bool = False,
    expected_inventory: tuple[int, int, int] = (16, 262, 260),
) -> Path:
    if bootstrap_iterations <= 0:
        raise ValueError("bootstrap_iterations must be positive")
    artifact_index = Path(artifact_index).resolve()
    output_dir = Path(output_dir).resolve()
    plan, aliases = validate_artifact_index(
        artifact_index, expected_folds=expected_inventory[0],
        expected_labelled=expected_inventory[1], expected_unique=expected_inventory[2])
    settings = _settings(
        artifact_index, plan, aliases, bootstrap_iterations, seed,
        write_item_deltas, expected_inventory)
    signatures = _input_signatures(plan)
    if output_dir.exists():
        manifest = _validate_outputs(output_dir, settings)
        saved_signatures = pd.read_csv(
            output_dir / "input-signatures.csv", keep_default_na=False)
        if not saved_signatures.equals(signatures):
            raise RuntimeError("input provenance mismatch for completed lesion-cell summary")
        if manifest.get("input_signatures_sha256") != _sha256(
            output_dir / "input-signatures.csv"):
            raise RuntimeError("input signature artifact mismatch")
        return output_dir

    stage = output_dir.with_name(f".{output_dir.name}.partial")
    if stage.exists() and not (stage / "settings.json").is_file():
        raise RuntimeError(f"partial output has no resumable settings: {stage}")
    stage.mkdir(parents=True, exist_ok=True)
    settings_payload = {"signature": _signature(settings), "settings": settings}
    _write_json_once(stage / "settings.json", settings_payload)
    _write_frame_once(stage / "input-signatures.csv", signatures)

    summary_path = stage / "cell-summary.csv"
    reference_path = stage / "reference-cell-metrics.csv"
    item_path = stage / "item-cell-deltas.parquet"
    if summary_path.exists():
        _validate_summary(
            pd.read_csv(summary_path, keep_default_na=False), expected_inventory[1])
    if item_path.exists():
        if not write_item_deltas:
            raise RuntimeError(f"unexpected staged item deltas: {item_path}")
        _validate_item_deltas(pd.read_parquet(item_path), expected_inventory[1])
    required_outputs = [
        summary_path, reference_path, *([item_path] if write_item_deltas else []),
    ]
    if not all(path.is_file() for path in required_outputs):
        item_deltas = _item_cell_deltas(plan)
        summary = summarize_cells(item_deltas, bootstrap_iterations, seed)
        summary = summary[list(SUMMARY_COLUMNS)].sort_values(
            ["specification", "scope", "judge", "benchmark"]).reset_index(drop=True)
        _write_frame_once(summary_path, summary)
        _write_frame_once(reference_path, reference_cell_metrics(plan))
        if write_item_deltas:
            _write_frame_once(item_path, item_deltas)
    summary_rows = len(pd.read_csv(summary_path))
    item_rows = len(pd.read_parquet(item_path)) if write_item_deltas else 0
    artifact_names = [
        "settings.json", "input-signatures.csv", "cell-summary.csv",
        "reference-cell-metrics.csv",
        *(["item-cell-deltas.parquet"] if write_item_deltas else []),
    ]
    manifest = {
        "state": "completed", "schema_version": 2,
        "settings_signature": settings_payload["signature"],
        "plan_signature": settings["plan_signature"],
        "source_artifact_index_sha256": settings["source_artifact_index_sha256"],
        "input_signatures_sha256": _sha256(stage / "input-signatures.csv"),
        "input_file_count": len(signatures),
        "fold_count": expected_inventory[0],
        "labelled_specifications": expected_inventory[1],
        "unique_artifacts": expected_inventory[2],
        "bootstrap_iterations": bootstrap_iterations, "seed": seed,
        "primary_metric": "log_loss", "summary_rows": summary_rows,
        "item_delta_rows": item_rows,
        "artifact_sha256": {name: _sha256(stage / name) for name in artifact_names},
    }
    _write_json_once(stage / "manifest.json", manifest)
    _validate_outputs(stage, settings)
    if output_dir.exists():
        raise RuntimeError(f"refusing to overwrite output directory: {output_dir}")
    os.replace(stage, output_dir)
    return output_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--write-item-deltas", action="store_true")
    parser.add_argument("--expected-folds", type=int, default=16)
    parser.add_argument("--expected-labelled", type=int, default=262)
    parser.add_argument("--expected-unique", type=int, default=260)
    args = parser.parse_args(argv)
    build(
        args.artifact_index, args.output_dir,
        bootstrap_iterations=args.bootstrap_iterations, seed=args.seed,
        write_item_deltas=args.write_item_deltas,
        expected_inventory=(
            args.expected_folds, args.expected_labelled, args.expected_unique),
    )


if __name__ == "__main__":
    main()
