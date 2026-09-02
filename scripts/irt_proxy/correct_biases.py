"""Restartable SWC commands for the GPQA bias-correction analysis."""

from __future__ import annotations

import argparse
from io import StringIO
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
CONFIG_PATH = HERE / "config.yaml"
IRT_MODELS = ("fixed_3pl",)
ARMS = ("corrected", "uncorrected")
CORRECTION_SOURCES = {
    "gpqa-pilot-rd2", "data-expansion-gpqa", "hf-judging-2026-08-17",
}
EXPECTED_TABLES = {
    "predictions.csv", "coefficients.csv", "coverage.csv", "metrics.csv",
    "gap_pairs.csv",
}
DIAGNOSTIC_COLUMNS = [
    "irt_model", "fold", "arm", "phase", "inference", "parameter",
    "divergences", "r_hat_max", "n_eff_min",
]
IDENTITY_COLUMNS = ["irt_model", "fold", "arm"]
FOLD_TABLES = {
    "predictions", "coverage", "calibration", "coefficients", "diagnostics",
    "identity", "target_exclusions",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the GPQA bias-correction pipeline on SWC CPU nodes.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("validate-model", "Run synthetic recovery (SWC compute node)."),
        ("prepare", "Prepare validated inputs (SWC compute node)."),
        ("fit-reference", "Fit frozen IRT banks (SWC compute node)."),
        ("reduce", "Reduce twenty completed fold artifacts (SWC compute node)."),
    ):
        command = commands.add_parser(name, help=help_text, description=help_text)
        command.add_argument("--config", type=Path, default=CONFIG_PATH)
    fold = commands.add_parser(
        "fit-fold", help="Fit one cross-validation arm (SWC compute node).",
        description="Fit one cross-validation arm (SWC compute node).",
    )
    fold.add_argument("--config", type=Path, default=CONFIG_PATH)
    fold.add_argument("--irt-model", choices=IRT_MODELS, required=True)
    fold.add_argument("--fold", type=int, choices=range(5), required=True)
    fold.add_argument("--model", choices=ARMS, required=True)
    return parser.parse_args(argv)


def _write_csv_once(table: pd.DataFrame, path: Path) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    table.to_csv(temporary, index=False)
    temporary.replace(path)


def _write_json_once(value: dict, path: Path) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _write_parquet_once(table: pd.DataFrame, path: Path) -> None:
    if path.exists():
        return
    temporary = path.with_name(f".{path.name}.tmp")
    table.to_parquet(temporary, index=False)
    temporary.replace(path)


def write_tables(tables: dict[str, pd.DataFrame], output_directory: Path) -> None:
    if set(tables) != EXPECTED_TABLES:
        missing = sorted(EXPECTED_TABLES - set(tables))
        extra = sorted(set(tables) - EXPECTED_TABLES)
        raise ValueError(f"output table mismatch; missing={missing}, extra={extra}")
    existing = sorted(name for name in tables if (output_directory / name).exists())
    if existing:
        raise FileExistsError(output_directory / existing[0])
    for name, table in tables.items():
        _write_csv_once(table, output_directory / name)


def write_fold_artifact(path: Path, tables: dict[str, pd.DataFrame]) -> None:
    """Write DataFrames as JSON-table strings in a pickle-free NPZ."""
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp.npz")
    np.savez_compressed(temporary, **{
        name: np.asarray(table.to_json(orient="table", index=False))
        for name, table in sorted(tables.items())
    })
    temporary.replace(path)


def read_fold_artifact(path: Path) -> dict[str, pd.DataFrame]:
    with np.load(path, allow_pickle=False) as stored:
        return {
            name: pd.read_json(StringIO(str(stored[name].item())), orient="table")
            for name in stored.files
        }


def _fold_artifact_paths(output: Path) -> list[Path]:
    return [
        output / "folds" / irt_model / f"fold-{fold}" / f"{arm}.npz"
        for irt_model in IRT_MODELS
        for fold in range(5)
        for arm in ARMS
    ]


def require_fold_artifacts(output: Path) -> list[Path]:
    paths = _fold_artifact_paths(output)
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"missing {len(missing)} of {len(paths)} fold artifacts: "
            + ", ".join(str(path) for path in missing)
        )
    return paths


def validate_fold_artifact(
    path: Path, irt_model: str, fold: int, arm: str,
) -> dict[str, pd.DataFrame]:
    from bias_crossfit import (
        CALIBRATION_COLUMNS, COEFFICIENT_COLUMNS, COVERAGE_COLUMNS,
        PREDICTION_COLUMNS, TARGET_EXCLUSION_COLUMNS,
    )

    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"{path} must be a nonempty readable NPZ")
    try:
        tables = read_fold_artifact(path)
    except Exception as exc:
        raise ValueError(f"{path} must be a nonempty readable NPZ") from exc
    if set(tables) != FOLD_TABLES:
        missing = sorted(FOLD_TABLES - set(tables))
        extra = sorted(set(tables) - FOLD_TABLES)
        raise ValueError(f"{path} fold tables mismatch; missing={missing}, extra={extra}")
    schemas = {
        "predictions": PREDICTION_COLUMNS,
        "coverage": COVERAGE_COLUMNS,
        "calibration": CALIBRATION_COLUMNS,
        "coefficients": COEFFICIENT_COLUMNS,
        "target_exclusions": TARGET_EXCLUSION_COLUMNS,
        "diagnostics": DIAGNOSTIC_COLUMNS,
        "identity": IDENTITY_COLUMNS,
    }
    for name, columns in schemas.items():
        if list(tables[name].columns) != list(columns):
            raise ValueError(f"{path} {name} schema mismatch")
    identity = tables["identity"]
    expected = [irt_model, fold, arm]
    if len(identity) != 1 or identity.iloc[0].tolist() != expected:
        raise ValueError(
            f"{path} identity mismatch; expected {irt_model}/fold-{fold}/{arm}"
        )
    diagnostics = tables["diagnostics"]
    if not diagnostics.empty and not (
        diagnostics["irt_model"].eq(irt_model).all()
        and diagnostics["fold"].eq(fold).all()
        and diagnostics["arm"].eq(arm).all()
    ):
        raise ValueError(f"{path} diagnostics identity mismatch")
    return tables


def load_fold_artifacts(output: Path) -> list[dict[str, pd.DataFrame]]:
    require_fold_artifacts(output)
    return [
        validate_fold_artifact(
            output / "folds" / irt_model / f"fold-{fold}" / f"{arm}.npz",
            irt_model, fold, arm,
        )
        for irt_model in IRT_MODELS
        for fold in range(5)
        for arm in ARMS
    ]


def _output(config: dict) -> Path:
    return ROOT / config["correction"].get("output", "analysis/bias_correction")


def _load_config(path: Path) -> dict:
    from inputs import load_config

    return load_config(path)


def _promote_directory(temporary: Path, final: Path) -> None:
    if final.exists():
        raise FileExistsError(final)
    temporary.replace(final)


def _validate_prepared(votes: pd.DataFrame, responses: pd.DataFrame) -> None:
    vote_columns = {
        "source", "benchmark", "judge", "item_id", "left_model",
        "right_model", "chosen_left",
    }
    response_columns = {
        "benchmark", "model", "item", "answer", "correct", "source_priority",
    }
    for name, frame, columns in (
        ("votes", votes, vote_columns), ("responses", responses, response_columns),
    ):
        missing = sorted(columns - set(frame))
        if missing:
            raise ValueError(f"{name} missing columns: {missing}")
        if set(frame["benchmark"].dropna()) != {"gpqa"}:
            raise ValueError(f"{name} must contain GPQA only")
    if votes["chosen_left"].isna().any() or not votes["chosen_left"].isin(
        [0.0, 1.0],
    ).all():
        raise ValueError("votes contain missing or non-binary verdicts")
    if set(votes["source"].dropna()) != CORRECTION_SOURCES:
        raise ValueError(
            "votes must contain exactly the three declared sources: "
            f"{sorted(CORRECTION_SOURCES)}"
        )
    feature_columns = [column for column in votes if column.startswith("delta_")]
    if not feature_columns or not np.isfinite(votes[feature_columns]).all().all():
        raise ValueError("votes need finite pairwise bias features")


def primary_responses(responses: pd.DataFrame) -> pd.DataFrame:
    """Keep the sole declared correction benchmark."""
    return responses[responses["benchmark"].eq("gpqa")].copy()


def candidate_closure(
    responses: pd.DataFrame,
    candidates: set[str],
    *,
    splits: pd.DataFrame | None = None,
    anchors: pd.DataFrame | None = None,
    irt_model: str | None = None,
    minimum_items: int = 2,
) -> pd.DataFrame:
    """Validate exact-candidate scored cells and optional frozen anchors."""
    from item_banks import irt_cells

    candidates = set(candidates)
    cells = irt_cells(primary_responses(responses))
    if cells.duplicated(["model", "benchmark", "item"]).any():
        raise ValueError("candidate closure requires unique canonical GPQA cells")
    counts = cells[cells["model"].isin(candidates)].groupby("model")["item"].nunique()
    missing = sorted(model for model in candidates if counts.get(model, 0) < minimum_items)
    if missing:
        raise ValueError(f"candidates lack canonical GPQA cells: {missing}")

    coverage = pd.DataFrame({"model": sorted(candidates)})
    coverage["n_cells"] = coverage["model"].map(counts).fillna(0).astype(int)
    if splits is not None:
        selected = splits[
            splits["benchmark"].eq("gpqa")
            & (splits["irt_model"].eq(irt_model) if "irt_model" in splits else True)
            & (splits["fit"].eq("gpqa") if "fit" in splits else True)
        ][["benchmark", "item", "half"]]
        scoped = cells[cells["model"].isin(candidates)].merge(
            selected, on=["benchmark", "item"], validate="many_to_one",
        )
        by_half = scoped.groupby(["model", "half"])["item"].nunique().unstack(fill_value=0)
        for half in ("easy", "hard"):
            coverage[f"n_{half}"] = coverage["model"].map(
                by_half[half] if half in by_half else {},
            ).fillna(0).astype(int)
        insufficient = coverage[
            coverage["n_easy"].lt(minimum_items) | coverage["n_hard"].lt(minimum_items)
        ]
        if not insufficient.empty:
            details = ", ".join(
                f"{row.model}: easy={row.n_easy}, hard={row.n_hard}"
                for row in insufficient.itertuples()
            )
            raise ValueError(f"candidate easy/hard closure failure: {details}")
    if anchors is not None:
        selected = anchors
        if irt_model and "irt_model" in selected:
            selected = selected[selected["irt_model"].eq(irt_model)]
        anchor_counts = selected[selected["model"].isin(candidates)]["model"].value_counts()
        coverage["n_anchors"] = coverage["model"].map(anchor_counts).fillna(0).astype(int)
        invalid = coverage[coverage["n_anchors"].ne(1)]
        if not invalid.empty:
            details = ", ".join(
                f"{row.model}={row.n_anchors}" for row in invalid.itertuples()
            )
            raise ValueError(f"every candidate needs exactly one anchor: {details}")
    return coverage


def validate_input_paths(root: Path, config: dict) -> None:
    viewer_root = root / config["paths"]["viewer_data"]
    missing = [
        viewer_root / source
        for source in config["correction"]["viewer_datasets"]
        if not (viewer_root / source).is_dir()
    ]
    if missing:
        raise FileNotFoundError(
            "missing correction viewer datasets: "
            + ", ".join(str(path) for path in missing)
        )


def prepare(config: dict) -> None:
    from bias_crossfit import assign_exact_model_folds
    from inputs import load_correction_data

    output = _output(config)
    final = output / "input"
    if (final / "manifest.json").is_file():
        print(f"skip existing final: {final}")
        return
    if final.exists():
        raise FileExistsError(f"incomplete final input directory: {final}")

    validate_input_paths(ROOT, config)
    votes, scored, exclusions = load_correction_data(
        ROOT, config, return_exclusions=True,
    )
    scored = primary_responses(scored)
    _validate_prepared(votes, scored)
    candidates = sorted(set(votes["left_model"]) | set(votes["right_model"]))
    if len(candidates) < 5:
        raise ValueError("GPQA correction needs at least five candidate models")
    candidate_closure(scored, set(candidates))
    folds = assign_exact_model_folds(
        candidates, n_splits=5, seed=config["irt"]["seed"],
    )
    counts = {
        "sources": int(votes["source"].nunique()),
        "judges": int(votes["judge"].nunique()),
        "models": len(candidates),
        "items": int(votes["item_id"].nunique()),
        "excluded_verdict_rows": int(exclusions["count"].sum()),
    }
    feature_columns = [column for column in votes if column.startswith("delta_")]
    feature_identity = [
        column for column in (
            "source", "artifact", "round", "judge", "item_id", "order",
            "left_model", "right_model",
        ) if column in votes
    ]
    features = votes[feature_identity + feature_columns].copy()

    temporary = output / ".input.tmp"
    temporary.mkdir(parents=True, exist_ok=True)
    for name, table in (
        ("votes.parquet", votes), ("responses.parquet", scored),
        ("features.parquet", features), ("folds.parquet", folds),
        ("exclusions.parquet", exclusions),
    ):
        _write_parquet_once(table, temporary / name)
    if not (temporary / "manifest.json").exists():
        _write_json_once(counts, temporary / "manifest.json")
    _promote_directory(temporary, final)
    print(json.dumps(counts, sort_keys=True))


def _write_reference_posterior(path: Path, fits: dict[str, dict]) -> None:
    if path.exists():
        return
    arrays = {}
    for irt_model, posterior in fits.items():
        for name in ("a", "b", "floor"):
            arrays[f"{irt_model}__{name}"] = np.asarray(posterior[name])
        arrays[f"{irt_model}__items"] = np.asarray(posterior["items"], dtype=str)
    temporary = path.with_name(".posterior.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def _diagnostic_rows(
    artifacts: list[dict], *, irt_model: str, fold: int, arm: str,
) -> pd.DataFrame:
    rows = []
    for artifact in artifacts:
        for parameter, values in artifact.get("diagnostics", {}).items():
            r_hat = np.asarray(values.get("r_hat", np.nan), dtype=float)
            n_eff = np.asarray(values.get("n_eff", np.nan), dtype=float)
            r_hat = r_hat[np.isfinite(r_hat)]
            n_eff = n_eff[np.isfinite(n_eff)]
            rows.append({
                "irt_model": irt_model, "fold": fold, "arm": arm,
                "phase": artifact.get("phase", "calibration"),
                "inference": artifact.get("inference", arm),
                "parameter": parameter,
                "divergences": artifact.get("divergences", np.nan),
                "r_hat_max": float(r_hat.max()) if len(r_hat) else np.nan,
                "n_eff_min": float(n_eff.min()) if len(n_eff) else np.nan,
            })
    return pd.DataFrame(rows, columns=DIAGNOSTIC_COLUMNS)


def write_reference_artifacts(
    directory: Path, splits: pd.DataFrame, anchors: pd.DataFrame,
    posteriors: dict[str, dict],
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    if not (directory / "items.csv").exists():
        _write_csv_once(splits, directory / "items.csv")
    if not (directory / "anchors.csv").exists():
        _write_csv_once(anchors, directory / "anchors.csv")
    if not (directory / "diagnostics.csv").exists():
        diagnostics = pd.concat([
            _diagnostic_rows(
                [{**posterior, "phase": "reference", "inference": irt_model}],
                irt_model=irt_model, fold=-1, arm="reference",
            )
            for irt_model, posterior in posteriors.items()
        ], ignore_index=True)
        _write_csv_once(diagnostics, directory / "diagnostics.csv")
    _write_reference_posterior(directory / "posterior.npz", posteriors)


def fit_reference(config: dict) -> None:
    from item_banks import fit_reference_bank, score_fixed_item_capabilities

    output = _output(config)
    final = output / "reference"
    if (final / "posterior.npz").is_file():
        print(f"skip existing final: {final}")
        return
    if final.exists():
        raise FileExistsError(f"incomplete final reference directory: {final}")
    responses = pd.read_parquet(output / "input" / "responses.parquet")
    folds = pd.read_parquet(output / "input" / "folds.parquet")
    candidates = set(folds["model"])
    settings = config["irt"]
    splits, anchors, posteriors = [], [], {}
    candidate_rows = responses[responses["model"].isin(candidates)]
    for irt_model in IRT_MODELS:
        split, posterior = fit_reference_bank(
            responses, candidates, floor=settings["floors"]["gpqa"],
            irt_model=irt_model, warmup=settings["warmup"],
            samples=settings["samples"], seed=settings["seed"], chains=4,
            collect_diagnostics=True,
        )
        candidate_closure(
            candidate_rows, candidates, splits=split, irt_model=irt_model,
        )
        scored = score_fixed_item_capabilities(candidate_rows, posterior)
        scored.insert(0, "fit", "gpqa")
        scored.insert(0, "irt_model", irt_model)
        candidate_closure(
            candidate_rows, candidates, splits=split, anchors=scored,
            irt_model=irt_model,
        )
        splits.append(split)
        anchors.append(scored)
        posteriors[irt_model] = posterior

    temporary = output / ".reference.tmp"
    write_reference_artifacts(
        temporary, pd.concat(splits, ignore_index=True),
        pd.concat(anchors, ignore_index=True), posteriors,
    )
    _promote_directory(temporary, final)


def _hard_accuracy(responses: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    from item_banks import irt_cells

    cells = irt_cells(responses)
    rows = []
    for irt_model in IRT_MODELS:
        items = splits[
            splits["irt_model"].eq(irt_model)
            & splits["fit"].eq("gpqa")
            & splits["benchmark"].eq("gpqa")
        ][["benchmark", "item", "half"]]
        merged = cells.merge(items, on=["benchmark", "item"], validate="many_to_one")
        hard = merged[merged["half"].eq("hard")].groupby("model")["correct"].mean()
        rows.extend({
            "irt_model": irt_model, "model": model,
            "hard_accuracy": value,
        } for model, value in hard.items())
    return pd.DataFrame(rows)


def fit_fold(config: dict, irt_model: str, fold: int, arm: str) -> None:
    from bias_crossfit import (
        COEFFICIENT_COLUMNS, COVERAGE_COLUMNS, build_fold_frames, run_fold,
    )
    from bias_model import fit_calibration, infer_targets
    from inputs import BIAS_FEATURES

    output = _output(config)
    final = output / "folds" / irt_model / f"fold-{fold}" / f"{arm}.npz"
    if final.is_file():
        print(f"skip existing final: {final}")
        return
    votes = pd.read_parquet(output / "input" / "votes.parquet")
    responses = pd.read_parquet(output / "input" / "responses.parquet")
    folds = pd.read_parquet(output / "input" / "folds.parquet")
    splits = pd.read_csv(output / "reference" / "items.csv")
    anchors = pd.read_csv(output / "reference" / "anchors.csv")
    held_out = set(folds.loc[folds["fold"].eq(fold), "model"])
    fold_anchors = anchors[~anchors["model"].isin(held_out)].copy()
    bin_spec = config["correction"].get("predictor_bin") or {
        "column": "half", "value": "easy",
    }
    calibration, target = build_fold_frames(
        votes, fold_anchors, splits, folds, fold, irt_model,
        bin_column=bin_spec["column"], bin_value=bin_spec["value"],
    )
    hard = _hard_accuracy(responses, splits)
    captured = []

    def capture_fit(*args, **kwargs):
        artifact = fit_calibration(*args, **kwargs)
        captured.append({
            **artifact, "phase": "calibration", "inference": arm,
        })
        return artifact

    def capture_infer(*args, inference_name, **kwargs):
        result = infer_targets(*args, inference_name=inference_name, **kwargs)
        captured.append({
            **result.attrs, "phase": "target", "inference": inference_name,
        })
        return result

    bin_scope = pd.Series(True, index=splits.index)
    if "irt_model" in splits:
        bin_scope &= splits["irt_model"].eq(irt_model)
    bin_items = set(splits.loc[
        bin_scope & splits[bin_spec["column"]].eq(bin_spec["value"]), "item",
    ])
    settings = config["irt"]
    tables = run_fold(
        fold, calibration, target, fold_anchors, hard,
        responses=responses, easy_items=bin_items, irt_model=irt_model,
        bin_column=bin_spec["column"], bin_value=bin_spec["value"],
        feature_columns=[f"delta_{name}" for name in BIAS_FEATURES],
        feature_ablations={
            group: [f"delta_{name}" for name in names]
            for group, names in (
                config["correction"].get("feature_ablations") or {}
            ).items()
        },
        warmup=settings["warmup"], samples=settings["samples"],
        seed=settings["seed"], arm=arm,
        target_accept=settings.get("target_accept", 0.8),
        fit_calibration_fn=capture_fit,
        inference_options={
            "chains": 4, "collect_diagnostics": True,
            "target_models": held_out,
        },
        infer_targets_fn=capture_infer,
    )
    if arm == "uncorrected":
        tables["coverage"] = pd.DataFrame(columns=COVERAGE_COLUMNS)
        tables["coefficients"] = pd.DataFrame(columns=COEFFICIENT_COLUMNS)
    tables["diagnostics"] = _diagnostic_rows(
        captured, irt_model=irt_model, fold=fold, arm=arm,
    )
    tables["identity"] = pd.DataFrame(
        [[irt_model, fold, arm]], columns=IDENTITY_COLUMNS,
    )
    write_fold_artifact(final, tables)
    if not tables["diagnostics"].empty:
        print(tables["diagnostics"].to_string(index=False))


def validate_model(config: dict) -> None:
    from bias_model import fit_calibration, simulate_calibration

    output = _output(config) / "synthetic_recovery.csv"
    if output.is_file():
        print(f"skip existing final: {output}")
        return
    frame, truth = simulate_calibration(seed=config["irt"]["seed"])
    fit = fit_calibration(
        frame, ["delta_log_chars", "delta_personal_pronouns"],
        warmup=config["irt"]["warmup"], samples=config["irt"]["samples"],
        seed=config["irt"]["seed"],
    )
    rows = []
    for index, judge in enumerate(fit["judges"]):
        planted = truth.set_index("judge").loc[judge]
        values = {
            "alpha": fit["alpha"][:, index],
            "discrimination": fit["discrimination"][:, index],
            **{
                feature: fit["beta"][:, index, feature_index]
                for feature_index, feature in enumerate(fit["features"])
            },
        }
        for parameter, draws in values.items():
            expected = float(planted[parameter])
            p5, p95 = np.percentile(draws, [5, 95])
            rows.append({
                "judge": judge, "parameter": parameter, "truth": expected,
                "mean": np.mean(draws), "p5": p5, "p95": p95,
                "sign_correct": (
                    expected == 0
                    or np.sign(np.mean(draws)) == np.sign(expected)
                ),
                "interval_covers_truth": p5 <= expected <= p95,
            })
    _write_csv_once(pd.DataFrame(rows), output)


def reduce(config: dict) -> None:
    from bias_crossfit import reduce_folds
    from correction_figures import render_correction_figures
    from metrics import capability_gap_rows, correction_metrics

    output = _output(config)
    manifest = output / "reduce-manifest.json"
    if manifest.is_file():
        print(f"skip existing final: {manifest}")
        return
    artifacts = load_fold_artifacts(output)
    folded = reduce_folds(artifacts)
    predictions = folded["predictions"]
    gap_pairs = capability_gap_rows(predictions)
    tables = {
        "predictions.csv": predictions,
        "coefficients.csv": folded["coefficients"],
        "coverage.csv": folded["coverage"],
        "metrics.csv": correction_metrics(
            predictions,
            n_resamples=config["correction"].get("bootstrap_resamples", 5000),
            seed=config["irt"]["seed"],
        ),
        "gap_pairs.csv": gap_pairs,
    }
    table_directory = output / "tables"
    if not table_directory.exists():
        temporary = output / ".tables.tmp"
        if temporary.exists():
            existing = {path.name for path in temporary.glob("*.csv")}
            for name in existing:
                tables.pop(name)
            for name, table in tables.items():
                _write_csv_once(table, temporary / name)
            if {path.name for path in temporary.glob("*.csv")} != EXPECTED_TABLES:
                raise ValueError("incomplete temporary reduction tables")
        else:
            write_tables(tables, temporary)
        _promote_directory(temporary, table_directory)
    figure_directory = output / "figures"
    if not figure_directory.exists():
        temporary = output / ".figures.tmp"
        render_correction_figures(predictions, gap_pairs, temporary)
        _promote_directory(temporary, figure_directory)
    _write_json_once({"fold_artifacts": len(artifacts)}, manifest)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = _load_config(args.config)
    if args.command == "validate-model":
        validate_model(config)
    elif args.command == "prepare":
        prepare(config)
    elif args.command == "fit-reference":
        fit_reference(config)
    elif args.command == "fit-fold":
        fit_fold(config, args.irt_model, args.fold, args.model)
    else:
        reduce(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
