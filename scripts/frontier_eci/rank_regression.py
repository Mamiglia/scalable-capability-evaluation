"""Descriptive in-sample rank-to-accuracy fits for the frozen frontier study."""

import argparse
import hashlib
import json
import multiprocessing
import shutil
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from scipy.special import expit


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from ranking import rank  # noqa: E402
from ranking.scoring import resample_items  # noqa: E402
from scripts.frontier_eci.analyze import plot_frontier_style, save  # noqa: E402
from scripts.frontier_eci.canonical_analysis import (  # noqa: E402
    _atomic_write_json, _checkpoint_tables, cap_pair_items, filter_protocol_verdicts,
    fit_binomial_logit, merge_rollout_sources, _resample_hard_items,
)
from scripts.frontier_eci.canonical_coverage import (  # noqa: E402
    _load_config, _path, historical_rollouts, release_date,
)
from scripts.judge.verdict_tables import read_verdicts  # noqa: E402


FIT_LABEL = "Rank–accuracy fit (in-sample)"
TIME_TREND_LABEL = "Pre-release time trend"
QUARTILES = (2, 3, 4)
HARD_SLICE = "Q2–Q4"
TIME_ORIGIN = pd.Timestamp("2024-01-01")
REQUIRED_OUTPUTS = (
    "abilities.csv", "ranking_blocks.csv", "fits.csv", "training_fitted.csv",
    "frontier_fitted.csv", "checkpoint-main.json", "summary.json",
)
MAIN_OUTPUT_MANIFEST = {
    "status": "authoritative",
    "main_analysis": "analysis/rank_regression",
    "main_glob": "analysis/rank_regression/**",
    "label": FIT_LABEL,
    "supplemental_controls": [{
        "analysis": TIME_TREND_LABEL,
        "root": "analysis/rank_regression/time_trend_control",
        "scientific_scope": "same quarterly frontier points as judge figures; no ranking values or judge verdict outcomes",
    }],
    "result_document": "../../../docs/results/2026-08-30-canonical-anchor-frontier-redo.md",
    "lifecycle_manifest": "provenance/final-lifecycle.json",
    "discarded_provenance": [
        {"root": ".", "glob": "analysis/**",
         "except": ["analysis/rank_regression/**"]},
        {"root": ".", "glob": "capability_baseline/**"},
        {"root": "../eci-semiannual", "glob": "analysis*/**"},
        {"root": "../eci-semiannual", "glob": "calibration-audit/**"},
    ],
    "scientific_scope": "in-sample; no temporal cutoff, anchors, target exclusion, or ECI",
    "target": "combined Q2–Q4 accuracy",
    "uncertainty": "95% two-source item bootstrap",
}


def select_method(linear_fitted: np.ndarray, linear_rmse: float,
                  sigmoid_rmse: float) -> str:
    """Apply the declared equal-parameter deterministic selection rule."""
    if not np.isfinite(sigmoid_rmse):
        return "linear"
    leaves_bounds = bool(np.any((linear_fitted < 0) | (linear_fitted > 1)))
    return "sigmoid" if leaves_bounds or sigmoid_rmse < linear_rmse else "linear"


def _observed_accuracy(outcomes: pd.DataFrame) -> pd.DataFrame:
    return (outcomes.groupby(["bench", "target_quartile", "model"], sort=True)
            .agg(actual=("correct", "mean"), n_items=("correct", "size"))
            .reset_index())


def fit_accuracy_candidates(ability: pd.Series,
                            outcomes: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Fit free-intercept linear and sigmoid curves to every joined model."""
    observed = (outcomes.groupby("model", sort=True)
                .agg(actual=("correct", "mean"), n_items=("correct", "size"))
                .reset_index())
    observed["ability"] = observed.model.map(ability)
    points = observed.dropna(subset=["actual", "ability"]).sort_values("model").reset_index(drop=True)
    if len(points) < 2 or points.ability.nunique() < 2:
        raise ValueError("fewer than two joined models with distinct BT abilities")

    x = points.ability.to_numpy(dtype=float)
    y = points.actual.to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(points)), x])
    linear_intercept, linear_slope = np.linalg.lstsq(design, y, rcond=None)[0]
    linear_fitted = design @ np.array([linear_intercept, linear_slope])
    linear_residual = linear_fitted - y
    linear_sse = float(np.square(linear_residual).sum())
    linear_rmse = float(np.sqrt(linear_sse / len(points)))

    sigmoid_status, sigmoid_failure = "ok", ""
    try:
        joined_ids = points.model.astype(str).tolist()
        fit = fit_binomial_logit(
            ability.reindex(joined_ids), outcomes[outcomes.model.isin(joined_ids)])
        sigmoid_intercept, sigmoid_slope = fit.intercept, fit.slope
        sigmoid_fitted = expit(sigmoid_intercept + sigmoid_slope * x)
        sigmoid_residual = sigmoid_fitted - y
        sigmoid_sse = float(np.square(sigmoid_residual).sum())
        sigmoid_rmse = float(np.sqrt(sigmoid_sse / len(points)))
    except (RuntimeError, ValueError) as error:
        sigmoid_intercept = sigmoid_slope = sigmoid_sse = sigmoid_rmse = np.nan
        sigmoid_fitted = np.full(len(points), np.nan)
        sigmoid_residual = np.full(len(points), np.nan)
        sigmoid_status, sigmoid_failure = "failed", str(error)

    selected = select_method(linear_fitted, linear_rmse, sigmoid_rmse)
    selected_fitted = sigmoid_fitted if selected == "sigmoid" else linear_fitted
    selected_residual = selected_fitted - y
    ids = "|".join(points.model.astype(str))
    summary = {
        "linear_intercept": float(linear_intercept),
        "linear_slope": float(linear_slope),
        "linear_sse": linear_sse, "linear_rmse": linear_rmse,
        "linear_leaves_unit_interval": bool(np.any((linear_fitted < 0) | (linear_fitted > 1))),
        "sigmoid_intercept": float(sigmoid_intercept),
        "sigmoid_slope": float(sigmoid_slope),
        "sigmoid_sse": float(sigmoid_sse), "sigmoid_rmse": float(sigmoid_rmse),
        "sigmoid_fit_status": sigmoid_status, "sigmoid_failure_reason": sigmoid_failure,
        "selected_method": selected, "n_training_models": len(points),
        "training_model_ids": ids, "fit_label": FIT_LABEL,
        "linear_equation": f"accuracy = {linear_intercept:.12g} + {linear_slope:.12g} * ability",
        "sigmoid_equation": (
            f"accuracy = logistic({sigmoid_intercept:.12g} + {sigmoid_slope:.12g} * ability)"
            if sigmoid_status == "ok" else ""),
    }
    points = points.assign(
        linear_fitted=linear_fitted, linear_residual=linear_residual,
        sigmoid_fitted=sigmoid_fitted, sigmoid_residual=sigmoid_residual,
        selected_method=selected, selected_fitted=selected_fitted,
        selected_residual=selected_residual, fit_label=FIT_LABEL,
    )
    return summary, points


def rank_all_blocks(protocol: pd.DataFrame, pair_limit: int | None = 8,
                    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    table = cap_pair_items(protocol, pair_limit) if pair_limit is not None else protocol.copy()
    abilities, blocks = [], []
    for (bench, judge), block in table.groupby(["bench", "judge"], sort=True):
        fitted = rank(block).assign(bench=bench, judge=judge)
        abilities.append(fitted)
        pair = np.sort(block[["model_left", "model_right"]].astype(str).to_numpy(), axis=1)
        counted = block.assign(pair_a=pair[:, 0], pair_b=pair[:, 1], item=block.item.astype(str))
        maximum = int(counted.groupby(["pair_a", "pair_b"]).item.nunique().max())
        blocks.append({"bench": bench, "judge": judge, "n_protocol_verdicts": len(block),
                       "n_ranked_models": len(fitted), "n_pair_items_max": maximum,
                       "pair_item_cap": pair_limit})
    return (pd.concat(abilities, ignore_index=True) if abilities else pd.DataFrame(),
            pd.DataFrame(blocks))


def fit_rank_regressions(abilities: pd.DataFrame, hard_outcomes: pd.DataFrame,
                         quartiles: tuple[int, ...] = QUARTILES,
                         ) -> tuple[pd.DataFrame, pd.DataFrame]:
    fits, fitted_points = [], []
    for (bench, judge), ranked in abilities.groupby(["bench", "judge"], sort=True):
        ability = ranked.set_index("model").ability
        for quartile in quartiles:
            outcomes = hard_outcomes[
                hard_outcomes.bench.eq(bench) & hard_outcomes.target_quartile.eq(quartile)]
            try:
                fit, points = fit_accuracy_candidates(ability, outcomes)
                fit.update(bench=bench, judge=judge, target_quartile=quartile,
                           fit_status="ok", failure_reason="")
                points = points.assign(bench=bench, judge=judge,
                                       target_quartile=quartile)
                fitted_points.append(points)
            except (RuntimeError, ValueError) as error:
                fit = {"bench": bench, "judge": judge, "target_quartile": quartile,
                       "fit_status": "failed", "failure_reason": str(error),
                       "fit_label": FIT_LABEL, "n_training_models": 0,
                       "training_model_ids": ""}
            fits.append(fit)
    return (pd.DataFrame(fits),
            pd.concat(fitted_points, ignore_index=True) if fitted_points else pd.DataFrame())


def _combined_hard(hard_outcomes: pd.DataFrame) -> pd.DataFrame:
    """Return one Q2–Q4 item table; Q1 never enters the hard target."""
    table = hard_outcomes[hard_outcomes.target_quartile.isin(QUARTILES)].copy()
    return table.drop(columns="target_quartile")


def fit_combined_rank_regressions(abilities: pd.DataFrame,
                                  hard_outcomes: pd.DataFrame
                                  ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit one combined-Q2–Q4 curve per judge and benchmark."""
    hard = _combined_hard(hard_outcomes)
    fits, fitted_points = [], []
    for (bench, judge), ranked in abilities.groupby(["bench", "judge"], sort=True):
        ability = ranked.set_index("model").ability
        outcomes = hard[hard.bench.eq(bench)]
        try:
            fit, points = fit_accuracy_candidates(ability, outcomes)
            fit.update(bench=bench, judge=judge, target_slice=HARD_SLICE,
                       fit_status="ok", failure_reason="")
            fitted_points.append(points.assign(
                bench=bench, judge=judge, target_slice=HARD_SLICE))
        except (RuntimeError, ValueError) as error:
            fit = {"bench": bench, "judge": judge, "target_slice": HARD_SLICE,
                   "fit_status": "failed", "failure_reason": str(error),
                   "fit_label": FIT_LABEL, "n_training_models": 0,
                   "training_model_ids": ""}
        fits.append(fit)
    return (pd.DataFrame(fits),
            pd.concat(fitted_points, ignore_index=True) if fitted_points else pd.DataFrame())


def _bootstrap_seed(seed: int, bench: str, judge: str) -> int:
    digest = hashlib.sha256(f"{seed}|{bench}|{judge}|{HARD_SLICE}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def _fit_combined_block(task) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    bench, judge, protocol, hard, resamples, seed, pair_limit = task
    capped = cap_pair_items(protocol, pair_limit) if pair_limit is not None else protocol.copy()
    ability_frame = rank(capped).assign(bench=bench, judge=judge)
    ability = ability_frame.set_index("model").ability
    fit, points = fit_accuracy_candidates(ability, hard)
    fit.update(bench=bench, judge=judge, target_slice=HARD_SLICE,
               fit_status="ok", failure_reason="", n_bootstrap=resamples,
               bootstrap_unit="Q1 and combined Q2–Q4 item IDs")
    rng = np.random.default_rng(_bootstrap_seed(seed, bench, judge))
    draws = {model: [] for model in points.model.astype(str)}
    for _ in range(resamples):
        try:
            draw_ability = rank(resample_items(capped, rng)).set_index("model").ability
            draw_hard = _resample_hard_items(hard, rng)
            _, draw_points = fit_accuracy_candidates(draw_ability, draw_hard)
            values = draw_points.set_index("model").selected_fitted
            for model in draws:
                if model in values and pd.notna(values[model]):
                    draws[model].append(float(values[model]))
        except (RuntimeError, ValueError):
            continue
    lows, highs, counts = [], [], []
    for row in points.itertuples():
        values = draws[str(row.model)]
        if not values:
            lows.append(np.nan)
            highs.append(np.nan)
            counts.append(0)
            continue
        low, high = np.quantile(values, [0.025, 0.975])
        lows.append(min(float(low), float(row.selected_fitted)))
        highs.append(max(float(high), float(row.selected_fitted)))
        counts.append(len(values))
    points = points.assign(
        bench=bench, judge=judge, target_slice=HARD_SLICE,
        fitted_low=lows, fitted_high=highs, n_bootstrap=counts,
        bootstrap_unstable=np.asarray(counts) < 0.8 * resamples,
    )
    fit["n_bootstrap_min"] = int(min(counts)) if counts else 0
    return ability_frame, pd.DataFrame([fit]), points


def fit_combined_with_bootstrap(protocol: pd.DataFrame, hard_outcomes: pd.DataFrame,
                                resamples: int = 1000, seed: int = 0, workers: int = 1,
                                pair_limit: int | None = 8
                                ) -> tuple[pd.DataFrame, pd.DataFrame,
                                           pd.DataFrame, pd.DataFrame]:
    hard = _combined_hard(hard_outcomes)
    tasks = [
        (bench, judge, block.copy(), hard[hard.bench.eq(bench)].copy(),
         resamples, seed, pair_limit)
        for (bench, judge), block in protocol.groupby(["bench", "judge"], sort=True)
    ]
    if workers <= 1:
        results = [_fit_combined_block(task) for task in tasks]
    else:
        results = [None] * len(tasks)
        with ProcessPoolExecutor(
                max_workers=min(workers, len(tasks)),
                mp_context=multiprocessing.get_context("spawn")) as executor:
            futures = {executor.submit(_fit_combined_block, task): index
                       for index, task in enumerate(tasks)}
            for future in as_completed(futures):
                results[futures[future]] = future.result()
    abilities = pd.concat([result[0] for result in results], ignore_index=True)
    fits = pd.concat([result[1] for result in results], ignore_index=True)
    points = pd.concat([result[2] for result in results], ignore_index=True)
    _, ranking_blocks = rank_all_blocks(protocol, pair_limit=pair_limit)
    return abilities, ranking_blocks, fits, points


def fit_time_trend_controls(actuals: pd.DataFrame, targets: list[str],
                            judges: list[str], releases: dict[str, str],
                            resamples: int = 1000, seed: int = 0
                            ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit frontier hard accuracy on release time using targets older than each judge."""
    actuals = (actuals[actuals.model.isin(targets)]
               .drop_duplicates(["bench", "model"]).copy())
    actuals["target_release"] = actuals.model.map(
        lambda model: release_date(releases, model))
    fits, predictions = [], []
    for bench in sorted(actuals.bench.unique()):
        bench_actuals = actuals[actuals.bench.eq(bench) & actuals.target_release.notna()]
        for judge in judges:
            judge_release = release_date(releases, judge)
            training = bench_actuals[bench_actuals.target_release < judge_release].copy()
            shown = bench_actuals.copy()
            if len(training) < 2 or training.target_release.nunique() < 2:
                reason = f"only {len(training)} strictly pre-release frontier points"
                fits.append({
                    "bench": bench, "judge": judge, "fit_label": TIME_TREND_LABEL,
                    "intercept_2024": np.nan, "slope_per_year": np.nan, "rmse": np.nan,
                    "n_training_models": len(training),
                    "training_model_ids": "|".join(sorted(training.model.astype(str))),
                    "judge_release": judge_release, "target_slice": HARD_SLICE,
                    "n_bootstrap": 0, "bootstrap_unit": "hard-item outcomes within model",
                    "fit_status": "failed", "failure_reason": reason,
                })
                predictions.append(shown.assign(
                    judge=judge, judge_release=judge_release,
                    predicted=np.nan, predicted_low=np.nan, predicted_high=np.nan,
                    is_fit_point=shown.model.isin(training.model), is_frontier=True,
                    post_judge_release=shown.target_release > judge_release,
                    residual=np.nan, fit_label=TIME_TREND_LABEL,
                    target_slice=HARD_SLICE, n_bootstrap=0,
                    fit_status="failed", failure_reason=reason,
                ))
                continue
            x_train = ((training.target_release - TIME_ORIGIN).dt.days / 365.25).to_numpy()
            x_shown = ((shown.target_release - TIME_ORIGIN).dt.days / 365.25).to_numpy()
            design = np.column_stack([np.ones(len(training)), x_train])
            coefficients = np.linalg.lstsq(design, training.actual.to_numpy(), rcond=None)[0]
            predicted = np.column_stack([np.ones(len(shown)), x_shown]) @ coefficients
            rng = np.random.default_rng(_bootstrap_seed(seed + 1, bench, judge))
            draws = []
            n_items = training.n_items.astype(int).to_numpy()
            for _ in range(resamples):
                draw_actual = rng.binomial(n_items, training.actual.to_numpy()) / n_items
                draw_coefficients = np.linalg.lstsq(design, draw_actual, rcond=None)[0]
                draws.append(np.column_stack([np.ones(len(shown)), x_shown]) @ draw_coefficients)
            low, high = np.quantile(np.asarray(draws), [0.025, 0.975], axis=0)
            low, high = np.minimum(low, predicted), np.maximum(high, predicted)
            residuals = design @ coefficients - training.actual.to_numpy()
            training_ids = "|".join(sorted(training.model.astype(str)))
            fits.append({
                "bench": bench, "judge": judge, "fit_label": TIME_TREND_LABEL,
                "intercept_2024": coefficients[0], "slope_per_year": coefficients[1],
                "rmse": float(np.sqrt(np.mean(residuals ** 2))),
                "n_training_models": len(training), "training_model_ids": training_ids,
                "judge_release": judge_release, "target_slice": HARD_SLICE,
                "n_bootstrap": resamples,
                "bootstrap_unit": "hard-item outcomes within model (aggregate binomial)",
                "fit_status": "ok", "failure_reason": "",
            })
            predictions.append(shown.assign(
                judge=judge, judge_release=judge_release,
                predicted=predicted, predicted_low=low, predicted_high=high,
                is_fit_point=shown.model.isin(training.model),
                is_frontier=True,
                post_judge_release=shown.target_release > judge_release,
                residual=predicted - shown.actual.to_numpy(),
                fit_label=TIME_TREND_LABEL, target_slice=HARD_SLICE,
                n_bootstrap=resamples, fit_status="ok", failure_reason="",
            ))
    return pd.DataFrame(fits), pd.concat(predictions, ignore_index=True)


def merge_prediction_comparison(judge: pd.DataFrame,
                                trend: pd.DataFrame) -> pd.DataFrame:
    """Join judge and time predictions only when their frontier cells match exactly."""
    keys = ["bench", "judge", "model"]
    judge_columns = keys + [
        "actual", "n_items", "target_release", "judge_release",
        "fitted", "fitted_low", "fitted_high",
    ]
    trend_columns = keys + [
        "actual", "target_release", "judge_release",
        "predicted", "predicted_low", "predicted_high",
    ]
    comparison = judge[judge_columns].merge(
        trend[trend_columns], on=keys, how="outer", validate="one_to_one",
        suffixes=("_judge", "_trend"), indicator=True,
    )
    if not comparison._merge.eq("both").all():
        raise ValueError("judge and time control must contain exact frontier cells")
    if not np.allclose(comparison.actual_judge, comparison.actual_trend, equal_nan=True):
        raise ValueError("judge and time control must contain exact frontier actuals")
    for column in ("target_release", "judge_release"):
        if not pd.to_datetime(comparison[f"{column}_judge"]).equals(
                pd.to_datetime(comparison[f"{column}_trend"])):
            raise ValueError(f"judge and time control must contain exact frontier {column}s")
    comparison = comparison.rename(columns={
        "actual_judge": "actual", "target_release_judge": "target_release",
        "judge_release_judge": "judge_release", "fitted": "judge_predicted",
        "fitted_low": "judge_low", "fitted_high": "judge_high",
        "predicted": "trend_predicted", "predicted_low": "trend_low",
        "predicted_high": "trend_high",
    })
    comparison["post_judge_release"] = (
        pd.to_datetime(comparison.target_release) > pd.to_datetime(comparison.judge_release))
    return (comparison.drop(columns=[
        "actual_trend", "target_release_trend", "judge_release_trend", "_merge",
    ]).sort_values(["bench", "judge", "target_release", "model"]).reset_index(drop=True))


def combined_frontier_grid(points: pd.DataFrame, actual: pd.DataFrame,
                           targets: list[str], judges: list[str],
                           releases: dict[str, str], benches: list[str]) -> pd.DataFrame:
    grid = pd.MultiIndex.from_product(
        [benches, judges, targets], names=["bench", "judge", "model"]
    ).to_frame(index=False)
    grid = grid.merge(actual, on=["bench", "model"], how="left")
    columns = ["bench", "judge", "model", "ability", "selected_fitted",
               "fitted_low", "fitted_high", "n_bootstrap", "bootstrap_unstable",
               "selected_method", "fit_label"]
    grid = grid.merge(points[columns], on=["bench", "judge", "model"], how="left")
    grid = grid.rename(columns={"selected_fitted": "fitted"})
    grid["residual"] = grid.fitted - grid.actual
    grid["target_slice"] = HARD_SLICE
    grid["fit_status"] = np.where(grid.fitted.notna(), "ok", "missing_ability")
    grid["failure_reason"] = np.where(
        grid.fit_status.eq("ok"), "", "frontier model absent from judge/benchmark BT ranking")
    grid["fit_label"] = grid.fit_label.fillna(FIT_LABEL)
    grid["target_release"] = grid.model.map(lambda model: release_date(releases, model))
    grid["judge_release"] = grid.judge.map(lambda judge: release_date(releases, judge))
    grid["post_judge_release"] = grid.target_release > grid.judge_release
    return grid


def frontier_grid(points: pd.DataFrame, actual: pd.DataFrame, targets: list[str],
                  judges: list[str], releases: dict[str, str],
                  benches: list[str], quartiles: tuple[int, ...] = QUARTILES) -> pd.DataFrame:
    grid = pd.MultiIndex.from_product(
        [benches, judges, quartiles, targets],
        names=["bench", "judge", "target_quartile", "model"],
    ).to_frame(index=False)
    grid = grid.merge(actual, on=["bench", "target_quartile", "model"], how="left")
    columns = ["bench", "judge", "target_quartile", "model", "ability",
               "selected_fitted", "selected_method", "fit_label"]
    available = points[[column for column in columns if column in points]].copy()
    grid = grid.merge(available, on=["bench", "judge", "target_quartile", "model"], how="left")
    grid = grid.rename(columns={"selected_fitted": "fitted"})
    grid["residual"] = grid.fitted - grid.actual
    grid["fitted_low"] = grid.fitted
    grid["fitted_high"] = grid.fitted
    grid["fit_status"] = np.where(
        grid.fitted.notna(), "ok",
        np.where(grid.actual.notna(), "missing_ability", "missing_actual_and_ability"))
    grid["failure_reason"] = np.where(
        grid.fit_status.eq("missing_ability"), "frontier model absent from judge/benchmark BT ranking",
        np.where(grid.fit_status.eq("missing_actual_and_ability"),
                 "frontier hard accuracy and BT ability unavailable", ""))
    if "fit_label" not in grid:
        grid["fit_label"] = FIT_LABEL
    else:
        grid["fit_label"] = grid.fit_label.fillna(FIT_LABEL)
    grid["target_release"] = grid.model.map(lambda model: release_date(releases, model))
    grid["judge_release"] = grid.judge.map(lambda judge: release_date(releases, judge))
    grid["post_judge_release"] = grid.target_release > grid.judge_release
    return grid


def _write_figures(frontier: pd.DataFrame, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for (bench, judge, quartile), table in frontier.groupby(
            ["bench", "judge", "target_quartile"], sort=True):
        for style in ("line", "points"):
            plot_frontier_style(
                table, judge, bench, quartile, out, style,
                prediction_label=FIT_LABEL, filename_prefix="rank-regression_",
                value_column="fitted", low_column="fitted_low", high_column="fitted_high",
                clip_prediction_display=True,
            )


def _write_combined_figures(frontier: pd.DataFrame, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for (bench, judge), table in frontier.groupby(["bench", "judge"], sort=True):
        bench_label = "GPQA" if bench == "gpqa" else "SWE-mini"
        for style in ("band", "points"):
            plot_frontier_style(
                table, judge, bench, "2-4", out, style,
                prediction_label=FIT_LABEL, filename_prefix="rank-regression_",
                value_column="fitted", low_column="fitted_low", high_column="fitted_high",
                clip_prediction_display=True, panel_label=f"{bench_label} · Q2–Q4",
                y_label="Combined hard accuracy", file_quartile="2-q4",
            )


def write_time_trend_figures(points: pd.DataFrame, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for (bench, judge), table in points.groupby(["bench", "judge"], sort=True):
        bench_label = "GPQA" if bench == "gpqa" else "SWE-mini"
        for style in ("band", "points"):
            plot_frontier_style(
                table, judge, bench, "2-4", out, style,
                prediction_label=TIME_TREND_LABEL, filename_prefix="time-trend_",
                value_column="predicted", low_column="predicted_low",
                high_column="predicted_high", clip_prediction_display=True,
                panel_label=f"{bench_label} · Q2–Q4", y_label="Combined hard accuracy",
                file_quartile="2-q4",
            )


def write_prediction_comparison(comparison: pd.DataFrame, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for (bench, judge), table in comparison.groupby(["bench", "judge"], sort=True):
        table = table.sort_values("target_release").copy()
        table["target_release"] = pd.to_datetime(table.target_release)
        columns = ["judge_predicted", "judge_low", "judge_high",
                   "trend_predicted", "trend_low", "trend_high"]
        table[columns] = table[columns].clip(0, 1)
        judge_date = pd.Timestamp(table.judge_release.iloc[0])
        fig, ax = plt.subplots(figsize=(1.75, 1.55))
        judge_band = ax.fill_between(
            table.target_release, table.judge_low, table.judge_high,
            color="#2a78d6", alpha=0.16, lw=0)
        judge_line, = ax.plot(
            table.target_release, table.judge_predicted, color="#2a78d6", lw=1)
        trend_band = ax.fill_between(
            table.target_release, table.trend_low, table.trend_high,
            color="#eb6834", alpha=0.16, lw=0)
        trend_line, = ax.plot(
            table.target_release, table.trend_predicted, color="#eb6834", lw=1)
        actual, = ax.plot(
            table.target_release, table.actual, "o", ms=3.5,
            color="#222222", clip_on=False, zorder=4)
        release = ax.axvline(judge_date, color="#777777", lw=0.8, ls="--")
        first = min(table.target_release.min(), judge_date).replace(month=1, day=1)
        last = (max(table.target_release.max(), judge_date).replace(month=1, day=1)
                + pd.DateOffset(years=1))
        ax.set_xticks(pd.date_range(first, last, freq="YS"))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.set_xlim(first, last)
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.5, 1])
        ax.set_ylabel("Combined hard accuracy")
        ax.set_xlabel("Model release")
        bench_label = "GPQA" if bench == "gpqa" else "SWE-mini"
        ax.text(0, 1.02, f"{bench_label} · Q2–Q4", color="#777777", fontsize=7,
                ha="left", va="bottom", transform=ax.transAxes)
        ax.spines["left"].set_bounds(0, 1)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_position(("outward", 5))
        ax.legend(
            [actual, (judge_band, judge_line), (trend_band, trend_line), release],
            ["Actual", "Judge prediction", "Time trend", "Judge release"],
            handler_map={tuple: HandlerTuple(ndivide=1)}, loc="lower left",
            bbox_to_anchor=(0, 1.08), frameon=False, handletextpad=0.5,
            borderaxespad=0, ncol=1)
        save(fig, out / f"comparison_{judge}_{bench}_q2-q4_frontier")


def write_time_trend_output(fits: pd.DataFrame, points: pd.DataFrame,
                            judge_predictions: pd.DataFrame, output: Path, resamples: int,
                            figures: bool = True) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    fits.to_csv(output / "fits.csv", index=False)
    points.to_csv(output / "predictions.csv", index=False)
    if figures:
        write_time_trend_figures(points, output / "figures")
    comparison = merge_prediction_comparison(judge_predictions, points)
    comparison_root = output / "comparison"
    comparison_root.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(comparison_root / "comparison.csv", index=False)
    if figures:
        write_prediction_comparison(comparison, comparison_root / "figures")
    _atomic_write_json({
        "analysis": "Judge prediction vs pre-release time trend",
        "target": f"combined {HARD_SLICE} accuracy",
        "rows": len(comparison),
        "judges": int(comparison.judge.nunique()),
        "benchmarks": int(comparison.bench.nunique()),
        "missing_time_trend_rows": int(comparison.trend_predicted.isna().sum()),
    }, comparison_root / "summary.json")
    summary = {
        "analysis": TIME_TREND_LABEL,
        "target": f"combined {HARD_SLICE} accuracy",
        "fit": "free-intercept linear OLS on frontier models released strictly before each judge",
        "uncertainty": "95% aggregate binomial item bootstrap",
        "resamples": resamples,
        "fit_blocks": len(fits),
        "fit_failures": int(fits.fit_status.ne("ok").sum()),
        "prediction_rows": len(points),
        "frontier_prediction_rows": int(points.is_frontier.sum()),
        "comparison_rows": len(comparison),
    }
    _atomic_write_json(summary, output / "summary.json")
    return summary


def _validate_staged_output(staging: Path, require_figures: bool) -> None:
    missing = [name for name in REQUIRED_OUTPUTS if not (staging / name).is_file()]
    if missing:
        raise ValueError(f"incomplete staged rank-regression output: missing {','.join(missing)}")
    checkpoint = json.loads((staging / "checkpoint-main.json").read_text())
    summary = json.loads((staging / "summary.json").read_text())
    if checkpoint.get("phase") != "main" or summary.get("analysis") != FIT_LABEL:
        raise ValueError("incomplete staged rank-regression output: invalid completion metadata")
    for name, expected_rows in checkpoint.get("rows", {}).items():
        if len(pd.read_csv(staging / name)) != expected_rows:
            raise ValueError(f"incomplete staged rank-regression output: row count mismatch for {name}")
    fits = pd.read_csv(staging / "fits.csv")
    frontier = pd.read_csv(staging / "frontier_fitted.csv")
    ranking_blocks = pd.read_csv(staging / "ranking_blocks.csv")
    training = pd.read_csv(staging / "training_fitted.csv")
    valid = (
        len(ranking_blocks) == summary.get("ranking_blocks") == 10
        and ranking_blocks.n_pair_items_max.max() <= 8
        and len(fits) == summary.get("fit_blocks") == 10
        and fits.fit_status.eq("ok").all()
        and fits.target_slice.eq(HARD_SLICE).all()
        and len(frontier) == summary.get("frontier_rows") == 80
        and not frontier[["actual", "fitted"]].isna().any().any()
        and frontier.target_slice.eq(HARD_SLICE).all()
        and all(
            set(training[(training.bench == row.bench)
                         & (training.judge == row.judge)
                         & (training.target_slice == row.target_slice)].model)
            == set(row.training_model_ids.split("|"))
            for row in fits.itertuples()
        )
    )
    if not valid:
        raise ValueError("invalid main-analysis invariants in staged rank-regression output")
    if require_figures:
        figures = staging / "figures"
        counts = {suffix: len(list(figures.glob(f"*.{suffix}")))
                  for suffix in ("png", "pdf", "svg")}
        if counts != {"png": 20, "pdf": 20, "svg": 20}:
            raise ValueError(f"incomplete staged rank-regression output: figure counts {counts}")


def publish_staged_output(staging: Path, output: Path, require_figures: bool = True) -> None:
    """Validate and atomically-ish swap a same-parent staging directory into place."""
    _validate_staged_output(staging, require_figures)
    _atomic_write_json({
        "status": "complete", "analysis": FIT_LABEL,
        "required_files": list(REQUIRED_OUTPUTS), "figures_required": require_figures,
    }, staging / "COMPLETE.json")
    backup = output.with_name(f".{output.name}.previous")
    if backup.exists():
        if output.exists():
            shutil.rmtree(backup)
        else:
            backup.replace(output)
    if output.exists():
        output.replace(backup)
    try:
        staging.replace(output)
    except Exception:
        if backup.exists() and not output.exists():
            backup.replace(output)
        raise
    else:
        if backup.exists():
            shutil.rmtree(backup)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path,
                        default=ROOT / "configs/frontier_eci_canonical_anchors/experiment.yaml")
    parser.add_argument("--logs-root", type=Path,
                        default=ROOT / "logs/frontier-eci-canonical-anchors")
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    config = _load_config(args.config)
    result_root = _path(config["results"])
    output = result_root / "analysis/rank_regression"
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))

    try:
        quartiles = pd.read_csv(_path(config["quartiles"]), dtype={"item": str})
        canonical = pd.read_parquet(
            _path(config["canonical_identity"]),
            columns=["bench", "model", "item", "epoch", "correct", "is_canonical"])
        frontier = pd.read_parquet(_path(config["frontier_rollouts"])).copy()
        historical = historical_rollouts(config.get("historical_backfill", {}))
        historical["epoch"] = 0
        supplements = historical_rollouts(config.get("actual_supplements", {}))
        supplements["epoch"] = 0
        rollouts = merge_rollout_sources(canonical, frontier, historical, supplements)
        rollouts["item"] = rollouts.item.astype(str)
        hard = rollouts.merge(quartiles[["bench", "item", "item_quartile"]],
                              on=["bench", "item"], how="inner")
        hard = hard.rename(columns={"item_quartile": "target_quartile"})

        judges = [judge.split("/")[-1] for group in config["judges"].values() for judge in group]
        old = [pd.read_parquet(_path(path)) for path in config["verdicts"]]
        paid, candidate_logs = read_verdicts([args.logs_root])
        columns = ["bench", "judge", "item", "model_left", "model_right",
                   "epoch_left", "epoch_right", "judge_epoch", "pick_left"]
        verdicts = pd.concat([table[columns] for table in old] + [paid[columns]], ignore_index=True)
        verdicts = (verdicts[verdicts.judge.isin(judges)]
                    .drop_duplicates(columns[:-1], keep="last").reset_index(drop=True))
        protocol = filter_protocol_verdicts(verdicts, rollouts, quartiles)

        targets = list(config["candidates"])
        abilities, ranking_blocks, fits, training = fit_combined_with_bootstrap(
            protocol, hard, resamples=args.resamples,
            seed=args.seed, workers=args.workers, pair_limit=8)
        combined = _combined_hard(hard)
        actual = (combined.groupby(["bench", "model"], sort=True)
                  .agg(actual=("correct", "mean"), n_items=("correct", "size"))
                  .reset_index())
        benches = sorted(hard.bench.unique())
        frontier_fitted = combined_frontier_grid(
            training, actual, targets, judges, config["releases"], benches)
        control_fits, control_points = fit_time_trend_controls(
            actual, targets, judges, config["releases"],
            resamples=args.resamples, seed=args.seed)

        _checkpoint_tables(staging, "main", {
            "abilities.csv": abilities,
            "ranking_blocks.csv": ranking_blocks,
            "fits.csv": fits,
            "training_fitted.csv": training,
            "frontier_fitted.csv": frontier_fitted,
        })
        if not args.no_figures:
            _write_combined_figures(frontier_fitted, staging / "figures")
        write_time_trend_output(
            control_fits, control_points, frontier_fitted,
            staging / "time_trend_control",
            args.resamples, figures=not args.no_figures)
        missing = frontier_fitted[frontier_fitted.fit_status.ne("ok")]
        summary = {
            "analysis": FIT_LABEL,
            "target": "combined Q2–Q4 accuracy",
            "uncertainty": "95% two-source item bootstrap",
            "resamples": args.resamples,
            "pair_item_cap": 8,
            "q1_protocol_rows": len(protocol),
            "ranking_blocks": len(ranking_blocks),
            "fit_blocks": len(fits),
            "fit_failures": int(fits.fit_status.ne("ok").sum()),
            "frontier_rows": len(frontier_fitted),
            "frontier_actual_missing": int(frontier_fitted.actual.isna().sum()),
            "frontier_ability_gaps": int(missing.actual.notna().sum()),
            "frontier_gap_ids": sorted(missing.loc[missing.actual.notna(), "model"].unique()),
            "paid_success_rows": len(paid),
            "paid_candidate_logs": len(candidate_logs),
        }
        _atomic_write_json(summary, staging / "summary.json")
        publish_staged_output(staging, output, require_figures=not args.no_figures)
        _atomic_write_json(MAIN_OUTPUT_MANIFEST, result_root / "MAIN_OUTPUT.json")
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
