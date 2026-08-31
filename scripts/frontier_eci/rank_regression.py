"""Descriptive in-sample rank-to-accuracy fits for the frozen frontier study."""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from ranking import rank  # noqa: E402
from scripts.frontier_eci.analyze import plot_frontier_style  # noqa: E402
from scripts.frontier_eci.canonical_analysis import (  # noqa: E402
    _atomic_write_json, _checkpoint_tables, cap_pair_items, filter_protocol_verdicts,
    fit_binomial_logit, merge_rollout_sources,
)
from scripts.frontier_eci.canonical_coverage import (  # noqa: E402
    _load_config, _path, historical_rollouts, release_date,
)
from scripts.judge.verdict_tables import read_verdicts  # noqa: E402


FIT_LABEL = "Rank–accuracy fit (in-sample)"
QUARTILES = (2, 3, 4)


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path,
                        default=ROOT / "configs/frontier_eci_canonical_anchors/experiment.yaml")
    parser.add_argument("--logs-root", type=Path,
                        default=ROOT / "logs/frontier-eci-canonical-anchors")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    config = _load_config(args.config)
    result_root = _path(config["results"])
    output = result_root / "analysis/rank_regression"
    if output.exists():
        shutil.rmtree(output)

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

    abilities, ranking_blocks = rank_all_blocks(protocol, pair_limit=8)
    fits, training = fit_rank_regressions(abilities, hard)
    actual = _observed_accuracy(hard)
    targets = list(config["candidates"])
    benches = sorted(hard.bench.unique())
    frontier_fitted = frontier_grid(
        training, actual, targets, judges, config["releases"], benches=benches)

    _checkpoint_tables(output, "main", {
        "abilities.csv": abilities,
        "ranking_blocks.csv": ranking_blocks,
        "fits.csv": fits,
        "training_fitted.csv": training,
        "frontier_fitted.csv": frontier_fitted,
    })
    if not args.no_figures:
        _write_figures(frontier_fitted, output / "figures")
    missing = frontier_fitted[frontier_fitted.fit_status.ne("ok")]
    summary = {
        "analysis": FIT_LABEL,
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
    _atomic_write_json(summary, output / "summary.json")
    _atomic_write_json({
        "main_analysis": "analysis/rank_regression",
        "label": FIT_LABEL,
        "discarded_non_main": ["analysis/figures/frontier_style", "capability_baseline/figures"],
        "scientific_scope": "in-sample; no temporal cutoff, anchors, target exclusion, or ECI",
    }, result_root / "MAIN_OUTPUT.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
