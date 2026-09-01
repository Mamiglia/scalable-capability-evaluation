"""Balanced pair ranking and hard-outcome calibration for canonical anchors."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
import sys
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict, deque

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from ranking import rank  # noqa: E402
from ranking.scoring import resample_items  # noqa: E402
from scripts.frontier_eci.canonical_coverage import (  # noqa: E402
    _load_config, _path, canonical_anchor_models, historical_rollouts,
    prediction_cutoff, release_date, stable_items,
)
from scripts.frontier_eci.analyze import plot_frontier_style  # noqa: E402
from scripts.judge.verdict_tables import read_verdicts  # noqa: E402


MODEL_ALIASES = {"gemma3-12b-it": "gemma-3-12b-it"}


@dataclass
class Fit:
    intercept: float
    slope: float
    converged: bool
    gradient_intercept: float
    gradient_slope: float
    anchor_table: pd.DataFrame


@dataclass
class Prediction:
    predicted: float
    intercept: float
    slope: float
    converged: bool
    gradient_intercept: float
    gradient_slope: float
    anchor_table: pd.DataFrame
    actual: float
    residual: float


def cap_pair_items(verdicts: pd.DataFrame, limit: int = 8) -> pd.DataFrame:
    """Keep every row for the same stable item IDs within each unordered pair."""
    columns = list(verdicts.columns)
    table = verdicts.copy()
    for column in ("model_left", "model_right"):
        table[column] = table[column].replace(MODEL_ALIASES)
    pair = np.sort(table[["model_left", "model_right"]].astype(str).to_numpy(), axis=1)
    table["_pair_a"], table["_pair_b"] = pair[:, 0], pair[:, 1]
    table["_item"] = table.item.astype(str)
    groups = [column for column in ("bench", "judge") if column in table] + ["_pair_a", "_pair_b"]
    keep = table.groupby(groups, sort=False)["_item"].transform(
        lambda values: values.isin(stable_items(values.tolist(), limit)))
    sort = groups + ["_item", "model_left", "model_right"]
    sort += [column for column in columns if column not in sort and column != "item"]
    return (table[keep].sort_values(sort, key=lambda values: values.astype(str), kind="stable")
            [columns].reset_index(drop=True))


def filter_protocol_verdicts(verdicts: pd.DataFrame, rollouts: pd.DataFrame,
                             quartiles: pd.DataFrame) -> pd.DataFrame:
    """Keep frozen-Q1 verdicts whose exact rollout epochs are both correct."""
    columns = list(verdicts.columns)
    table = verdicts.copy()
    for column in ("model_left", "model_right"):
        table[column] = table[column].replace(MODEL_ALIASES)
    table["item"] = table.item.astype(str)
    q1 = quartiles[quartiles.item_quartile.eq(1)][["bench", "item"]].copy()
    q1["item"] = q1.item.astype(str)
    table = table.merge(q1, on=["bench", "item"], how="inner")

    outcomes = rollouts.copy()
    outcomes["item"] = outcomes.item.astype(str)
    if "epoch" not in outcomes:
        outcomes["epoch"] = 0
    correct = (outcomes.groupby(["bench", "model", "item", "epoch"], as_index=False)
               .correct.all())
    left = correct.rename(columns={"model": "model_left", "epoch": "epoch_left",
                                   "correct": "_left_correct"})
    right = correct.rename(columns={"model": "model_right", "epoch": "epoch_right",
                                    "correct": "_right_correct"})
    table = table.merge(left, on=["bench", "model_left", "item", "epoch_left"], how="left")
    table = table.merge(right, on=["bench", "model_right", "item", "epoch_right"], how="left")
    keep = table._left_correct.fillna(False) & table._right_correct.fillna(False)
    key = [column for column in columns if column != "pick_left"]
    return table.loc[keep, columns].drop_duplicates(key, keep="last").reset_index(drop=True)


def _connected_models(verdicts: pd.DataFrame, target: str) -> set[str]:
    graph: dict[str, set[str]] = defaultdict(set)
    for left, right in verdicts[["model_left", "model_right"]].drop_duplicates().itertuples(index=False):
        graph[str(left)].add(str(right))
        graph[str(right)].add(str(left))
    connected, queue = {target}, deque([target])
    while queue:
        model = queue.popleft()
        for neighbor in graph.get(model, set()) - connected:
            connected.add(neighbor)
            queue.append(neighbor)
    return connected


def select_anchor_ids(verdicts: pd.DataFrame, target: str,
                      canonical_models: list[str], frontier_targets: set[str],
                      approved_backfill: list[str], minimum_anchors: int = 4,
                      ) -> tuple[list[str], list[str]]:
    """Return connected canonical anchors and only-needed approved backfill."""
    connected = _connected_models(verdicts, target)
    canonical = sorted((set(canonical_models) - frontier_targets) & connected)
    backfill = (sorted(set(approved_backfill) & connected)
                if len(canonical) < minimum_anchors else [])
    return canonical, backfill


def summarize_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """MAE, bias, and rank correlation overall and by anchor-range position."""
    scored = predictions.dropna(subset=["actual", "predicted"]).copy()
    scored["error"] = scored.predicted - scored.actual
    groups = [scored.assign(anchor_range="overall")]
    if "inside_anchor_range" in scored:
        groups.extend([
            scored[scored.inside_anchor_range.eq(True)].assign(anchor_range="inside"),
            scored[scored.inside_anchor_range.eq(False)].assign(anchor_range="outside"),
        ])
    table = pd.concat(groups, ignore_index=True)
    keys = ["bench", "judge", "target_quartile", "anchor_range"]
    rows = []
    for key, block in table.groupby(keys, sort=False):
        rho = (float(spearmanr(block.predicted, block.actual).statistic)
               if len(block) >= 2 and block.predicted.nunique() > 1
               and block.actual.nunique() > 1 else np.nan)
        rows.append(dict(zip(keys, key), mae=float(block.error.abs().mean()),
                         bias=float(block.error.mean()), spearman=rho,
                         n_models=len(block)))
    return pd.DataFrame(rows)


def merge_rollout_sources(canonical: pd.DataFrame, frontier: pd.DataFrame,
                          historical: pd.DataFrame,
                          supplements: pd.DataFrame) -> pd.DataFrame:
    """Prefer frozen canonical/frontier rows; supplements only fill absent models."""
    canonical = canonical[canonical.is_canonical].copy()
    canonical["is_historical_backfill"] = False
    frontier = frontier.copy()
    frontier["is_canonical"], frontier["is_historical_backfill"] = False, False
    historical = historical.copy()
    historical["is_canonical"], historical["is_historical_backfill"] = False, True
    supplements = supplements.copy()
    supplements["is_canonical"], supplements["is_historical_backfill"] = False, False
    kept = canonical
    for table in (frontier, historical, supplements):
        present = pd.MultiIndex.from_frame(kept[["bench", "model"]]).unique()
        keys = pd.MultiIndex.from_frame(table[["bench", "model"]])
        kept = pd.concat([kept, table[~keys.isin(present)]], ignore_index=True)
    return kept


def _failed_prediction(bench: str, judge: str, target: str, quartile: int,
                       actual: float, reason: str, anchor_ids: str = "") -> dict:
    return {"bench": bench, "judge": judge, "model": target,
            "target_quartile": quartile, "actual": actual,
            "predicted": np.nan, "residual": np.nan,
            "predicted_low": np.nan, "predicted_median": np.nan,
            "predicted_high": np.nan, "anchor_ids": anchor_ids,
            "n_anchors": len(anchor_ids.split("|")) if anchor_ids else 0,
            "n_bootstrap": 0, "bootstrap_unstable": False,
            "intercept": np.nan, "slope": np.nan, "converged": False,
            "gradient_intercept": np.nan, "gradient_slope": np.nan,
            "target_ability": np.nan, "anchor_ability_min": np.nan,
            "anchor_ability_max": np.nan, "inside_anchor_range": np.nan,
            "fit_status": "failed", "failure_reason": reason}


def _block_seed(seed: int, block: str, quartile: int) -> int:
    digest = hashlib.sha256(f"{seed}|{block}|{quartile}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def analyze_blocks(coverage: pd.DataFrame, verdicts: pd.DataFrame,
                   hard_outcomes: pd.DataFrame,
                   canonical_models: dict[str, list[str]],
                   frontier_targets: set[str], releases: dict[str, str],
                   resamples: int, seed: int, pair_limit: int | None = 8,
                   target_quartiles: tuple[int, ...] = (2, 3, 4),
                   anchor_overrides: dict[str, list[str]] | None = None,
                   ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit every frozen coverage block while retaining explicit failures."""
    predictions, fits, residuals, exported = [], [], [], []
    for block in coverage.itertuples(index=False):
        block_id = f"{block.bench}|{block.judge}|{block.target}"
        judged = verdicts[(verdicts.bench.eq(block.bench))
                           & (verdicts.judge.eq(block.judge))].copy()
        cutoff = prediction_cutoff(block.target, block.judge, releases)
        left_dates = judged.model_left.map(lambda model: release_date(releases, model))
        right_dates = judged.model_right.map(lambda model: release_date(releases, model))
        prefix = judged[(left_dates <= cutoff) & (right_dates <= cutoff)]
        component = _connected_models(prefix, block.target)
        prefix = prefix[prefix.model_left.isin(component) & prefix.model_right.isin(component)]
        eligible = canonical_anchor_models(
            canonical_models.get(block.bench, []), block.target, block.judge, releases)
        approved = json.loads(block.historical_backfill_anchors)
        approved = [model for model in approved
                    if pd.notna(release_date(releases, model))
                    and release_date(releases, model) < cutoff]
        canonical, backfill = select_anchor_ids(
            prefix, block.target, eligible, frontier_targets, approved)
        anchors = canonical + backfill
        headline_anchors = anchors
        override_failure = ""
        if anchor_overrides is not None and block_id in anchor_overrides:
            anchors = anchor_overrides[block_id]
            invalid = sorted(set(anchors) - set(headline_anchors))
            if invalid:
                override_failure = f"paired anchors absent from headline:{'|'.join(invalid)}"
        anchor_ids = "|".join(anchors)
        frozen_failure = block.status != "identifiable"
        failure = (f"coverage:{block.status}" if frozen_failure else
                   (override_failure or
                    (f"only_{len(anchors)}_connected_anchors" if len(anchors) < 4 else "")))
        exported.append({**block._asdict(), "block": block_id,
                         "canonical_anchor_ids": "|".join(canonical),
                         "backfill_anchor_ids": "|".join(backfill),
                         "anchor_ids": anchor_ids, "analysis_status": "failed" if failure else "ok",
                         "analysis_failure_reason": failure,
                         "n_protocol_verdicts": len(prefix)})
        for quartile in target_quartiles:
            outcomes = hard_outcomes[(hard_outcomes.bench.eq(block.bench))
                                     & (hard_outcomes.target_quartile.eq(quartile))]
            if failure:
                actual = _target_actual(outcomes, block.target)
                failed = _failed_prediction(
                    block.bench, block.judge, block.target, quartile, actual, failure, anchor_ids)
                failed["block"] = block_id
                predictions.append(failed)
                fits.append({key: failed.get(key) for key in (
                    "block", "bench", "judge", "model", "target_quartile", "anchor_ids",
                    "n_anchors", "intercept", "slope", "converged",
                    "gradient_intercept", "gradient_slope", "fit_status", "failure_reason")})
                continue
            try:
                summary, anchor_table = predict_block(
                    prefix, outcomes, block.target, anchors, resamples,
                    _block_seed(seed, block_id, quartile), pair_limit=pair_limit)
                summary.update(bench=block.bench, judge=block.judge,
                               target_quartile=quartile, fit_status="ok",
                               failure_reason="", block=block_id)
                predictions.append(summary)
                fits.append({key: summary[key] for key in (
                    "block", "bench", "judge", "model", "target_quartile", "anchor_ids",
                    "n_anchors", "intercept", "slope", "converged",
                    "gradient_intercept", "gradient_slope", "fit_status", "failure_reason")})
                residuals.append(anchor_table.assign(
                    block=block_id, bench=block.bench, judge=block.judge,
                    target=block.target, target_quartile=quartile))
            except (RuntimeError, ValueError) as error:
                reason = str(error)
                actual = _target_actual(outcomes, block.target)
                failed = _failed_prediction(block.bench, block.judge, block.target,
                                             quartile, actual, reason, anchor_ids)
                failed["block"] = block_id
                predictions.append(failed)
                fits.append({key: failed.get(key) for key in (
                    "block", "bench", "judge", "model", "target_quartile", "anchor_ids",
                    "n_anchors", "intercept", "slope", "converged",
                    "gradient_intercept", "gradient_slope", "fit_status", "failure_reason")})
    residual_table = pd.concat(residuals, ignore_index=True) if residuals else pd.DataFrame()
    return (pd.DataFrame(predictions), pd.DataFrame(fits), residual_table,
            pd.DataFrame(exported))


def _analysis_tasks(coverage: pd.DataFrame,
                    target_quartiles: tuple[int, ...]) -> list[tuple[pd.DataFrame, int]]:
    return [(coverage.iloc[[index]].copy(), quartile)
            for index in range(len(coverage)) for quartile in target_quartiles]


def _combine_analysis_results(results) -> tuple[pd.DataFrame, pd.DataFrame,
                                                    pd.DataFrame, pd.DataFrame]:
    tables = []
    for index in range(4):
        parts = [result[index] for result in results if not result[index].empty]
        table = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        if index == 3 and not table.empty:
            table = table.drop_duplicates("block", keep="first").reset_index(drop=True)
        tables.append(table)
    return tuple(tables)


def analyze_blocks_parallel(coverage: pd.DataFrame, verdicts: pd.DataFrame,
                            hard_outcomes: pd.DataFrame,
                            canonical_models: dict[str, list[str]],
                            frontier_targets: set[str], releases: dict[str, str],
                            resamples: int, seed: int, pair_limit: int | None = 8,
                            target_quartiles: tuple[int, ...] = (2, 3, 4),
                            workers: int = 1,
                            anchor_overrides: dict[str, list[str]] | None = None,
                            ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tasks = _analysis_tasks(coverage, target_quartiles)
    if not tasks:
        return tuple(pd.DataFrame() for _ in range(4))
    if workers <= 1:
        return _combine_analysis_results([
            analyze_blocks(task, verdicts, hard_outcomes, canonical_models,
                           frontier_targets, releases, resamples, seed,
                           pair_limit, (quartile,), anchor_overrides)
            for task, quartile in tasks
        ])
    count = min(workers, 18, len(tasks))
    with ProcessPoolExecutor(max_workers=count,
                             mp_context=multiprocessing.get_context("spawn")) as executor:
        futures = {executor.submit(
            analyze_blocks, task, verdicts, hard_outcomes, canonical_models,
            frontier_targets, releases, resamples, seed, pair_limit, (quartile,),
            anchor_overrides): index for index, (task, quartile) in enumerate(tasks)}
        results = [None] * len(tasks)
        for completed, future in enumerate(as_completed(futures), 1):
            results[futures[future]] = future.result()
            print(f"analysis task {completed}/{len(tasks)} complete", flush=True)
    return _combine_analysis_results(results)


def _atomic_write_csv(table: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    table.to_csv(temporary, index=False)
    temporary.replace(path)


def _atomic_write_json(values: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(values, indent=2) + "\n")
    temporary.replace(path)


def _checkpoint_tables(out: Path, phase: str,
                       tables: dict[str, pd.DataFrame]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        _atomic_write_csv(table, out / name)
    _atomic_write_json({"phase": phase, "rows": {name: len(table)
                                                   for name, table in tables.items()}},
                       out / f"checkpoint-{phase}.json")


def _clear_checkpoints(out: Path) -> None:
    for path in out.glob("checkpoint-*.json"):
        path.unlink()


def _aggregate_outcomes(outcomes: pd.DataFrame) -> pd.DataFrame:
    if {"successes", "total"}.issubset(outcomes.columns):
        return outcomes.groupby("model", as_index=False)[["successes", "total"]].sum()
    if "correct" not in outcomes:
        raise ValueError("hard outcomes need successes/total or item-level correct rows")
    return (outcomes.groupby("model", as_index=False).correct
            .agg(successes="sum", total="size"))


def _target_actual(outcomes: pd.DataFrame, target: str) -> float:
    target_outcome = _aggregate_outcomes(outcomes[outcomes.model.eq(target)])
    return (float(target_outcome.iloc[0].successes / target_outcome.iloc[0].total)
            if len(target_outcome) and target_outcome.iloc[0].total > 0 else np.nan)


def _selected_anchor_inputs(ability: pd.Series, outcomes: pd.DataFrame,
                            target: str, anchors: list[str],
                            ) -> tuple[list[str], pd.Series, pd.DataFrame]:
    if target not in ability or pd.isna(ability.get(target)):
        raise ValueError(f"target {target!r} has no fitted ability")
    anchor_ids = [anchor for anchor in anchors if anchor != target]
    missing_ability = [anchor for anchor in anchor_ids
                       if anchor not in ability or pd.isna(ability.get(anchor))]
    present_outcomes = set(_aggregate_outcomes(outcomes).model.astype(str))
    missing_outcomes = [anchor for anchor in anchor_ids if anchor not in present_outcomes]
    missing = []
    if missing_ability:
        missing.append(f"ability={'|'.join(missing_ability)}")
    if missing_outcomes:
        missing.append(f"outcomes={'|'.join(missing_outcomes)}")
    if missing:
        raise ValueError(f"selected anchors unavailable: {'; '.join(missing)}")
    return (anchor_ids, ability.reindex(anchor_ids),
            outcomes[outcomes.model.isin(anchor_ids)].copy())


def fit_binomial_logit(ability: pd.Series, outcomes: pd.DataFrame) -> Fit:
    """Fit an unregularized intercept and slope to grouped binomial outcomes."""
    table = _aggregate_outcomes(outcomes)
    table["ability"] = table.model.map(ability)
    table = table.dropna(subset=["ability", "successes", "total"]).copy()
    if len(table) < 4:
        raise ValueError(f"only {len(table)} calibration anchors")
    if (table.total <= 0).any() or (table.successes < 0).any() or (table.successes > table.total).any():
        raise ValueError("invalid binomial outcomes")

    x = table.ability.to_numpy(float)
    successes = table.successes.to_numpy(float)
    total = table.total.to_numpy(float)
    if np.ptp(x) == 0:
        raise RuntimeError("binomial logit is unidentifiable: anchor abilities are constant")
    positive = x[successes > 0]
    negative = x[successes < total]
    if (not len(positive) or not len(negative)
            or negative.max() <= positive.min() or positive.max() <= negative.min()):
        raise RuntimeError("binomial logit outcomes are separated")

    center, scale = x.mean(), x.std()
    z = (x - center) / scale
    design = np.column_stack([np.ones(len(x)), z])

    def loss(beta):
        eta = beta[0] + beta[1] * z
        return np.sum(total * np.logaddexp(0, eta) - successes * eta)

    def gradient(beta):
        residual = total * expit(beta[0] + beta[1] * z) - successes
        return design.T @ residual

    def hessian(beta):
        fitted = expit(beta[0] + beta[1] * z)
        weights = total * fitted * (1 - fitted)
        return design.T @ (weights[:, None] * design)

    beta = np.zeros(2)
    converged = False
    for _ in range(1_000):
        step = np.linalg.solve(hessian(beta), gradient(beta))
        previous = loss(beta)
        fraction = 1.0
        while fraction > 2 ** -20 and loss(beta - fraction * step) > previous:
            fraction /= 2
        beta -= fraction * step
        slope = beta[1] / scale
        intercept = beta[0] - slope * center
        fitted = expit(intercept + slope * x)
        residual = total * fitted - successes
        grad = np.array([residual.sum(), residual @ x])
        if np.abs(grad).max() < 1e-6:
            converged = True
            break
    slope = beta[1] / scale
    intercept = beta[0] - slope * center
    fitted = expit(intercept + slope * x)
    residual = total * fitted - successes
    grad = np.array([residual.sum(), residual @ x])
    if (not converged or not np.isfinite(beta).all()
            or np.abs(beta).max() > 50 or np.abs(grad).max() >= 1e-6):
        raise RuntimeError("binomial logit did not converge: weighted gradient tolerance exceeded")

    table["observed"] = table.successes / table.total
    table["fitted"] = fitted
    table["residual"] = table.observed - table.fitted
    table["weighted_residual"] = table.successes - table.total * table.fitted
    nonzero = table.residual[~np.isclose(table.residual, 0, atol=1e-12)]
    if len(nonzero) and ((nonzero > 0).all() or (nonzero < 0).all()):
        raise RuntimeError("nonzero anchor residuals all have one sign")
    return Fit(float(intercept), float(slope), True,
               float(grad[0]), float(grad[1]), table.reset_index(drop=True))


def predict_from_ability(ability: pd.Series, hard_outcomes: pd.DataFrame,
                         target: str, anchors: list[str]) -> Prediction:
    """Fit only anchor outcomes, then predict the target from its judged ability."""
    anchor_ids, anchor_ability, anchor_outcomes = _selected_anchor_inputs(
        ability, hard_outcomes, target, anchors)
    fit = fit_binomial_logit(anchor_ability, anchor_outcomes)
    fit.anchor_table = (fit.anchor_table.set_index("model").reindex(anchor_ids)
                        .rename_axis("model").reset_index())
    predicted = float(expit(fit.intercept + fit.slope * ability[target]))
    actual = _target_actual(hard_outcomes, target)
    return Prediction(predicted, fit.intercept, fit.slope, fit.converged,
                      fit.gradient_intercept, fit.gradient_slope, fit.anchor_table,
                      actual, predicted - actual)


def _resample_hard_items(outcomes: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    if "item" not in outcomes:
        raise ValueError("item-level hard outcomes are required for bootstrap")
    keys = [column for column in ("bench", "item") if column in outcomes]
    items = outcomes[keys].drop_duplicates()
    draws = items.iloc[rng.integers(len(items), size=len(items))].copy()
    draws["_draw"] = np.arange(len(draws)).astype(str)
    return (outcomes.merge(draws, on=keys).drop(columns="item")
            .rename(columns={"_draw": "item"}))


def predict_block(verdicts: pd.DataFrame, hard_outcomes: pd.DataFrame, target: str,
                  anchors: list[str], resamples: int, seed: int,
                  pair_limit: int | None = 8) -> tuple[dict, pd.DataFrame]:
    """Fit one capped block and independently bootstrap judged and hard item IDs."""
    capped = cap_pair_items(verdicts, pair_limit) if pair_limit is not None else verdicts.copy()
    ability = rank(capped).set_index("model").ability
    anchor_ids, _, anchor_outcomes = _selected_anchor_inputs(
        ability, hard_outcomes, target, anchors)
    point = predict_from_ability(ability, anchor_outcomes, target, anchor_ids)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(resamples):
        try:
            draw_ability = rank(resample_items(capped, rng)).set_index("model").ability
            draw_outcomes = _resample_hard_items(anchor_outcomes, rng)
            draws.append(predict_from_ability(
                draw_ability, draw_outcomes, target, anchor_ids).predicted)
        except (RuntimeError, ValueError):
            continue
    if not draws:
        raise RuntimeError("every two-source item-bootstrap calibration failed")

    low, median, high = np.quantile(draws, [0.025, 0.5, 0.975])
    actual = _target_actual(hard_outcomes, target)
    anchor_ability = ability.reindex(anchor_ids)
    ability_min, ability_max = float(anchor_ability.min()), float(anchor_ability.max())
    target_ability = float(ability[target])
    pairs = np.sort(capped[["model_left", "model_right"]].astype(str).to_numpy(), axis=1)
    counts = (capped.assign(_pair_a=pairs[:, 0], _pair_b=pairs[:, 1],
                            _item=capped.item.astype(str))
              .groupby(["_pair_a", "_pair_b"])._item.nunique())
    summary = {
        "model": target,
        "predicted": point.predicted,
        "actual": actual,
        "residual": point.predicted - actual,
        "predicted_low": float(low),
        "predicted_median": float(median),
        "predicted_high": float(high),
        "n_anchors": len(anchor_ids),
        "n_bootstrap": len(draws),
        "bootstrap_unstable": len(draws) < 0.8 * resamples,
        "n_verdicts": len(capped),
        "n_pair_items_max": int(counts.max()),
        "anchor_ids": "|".join(anchor_ids),
        "target_ability": target_ability,
        "anchor_ability_min": ability_min,
        "anchor_ability_max": ability_max,
        "inside_anchor_range": ability_min <= target_ability <= ability_max,
        "intercept": point.intercept,
        "slope": point.slope,
        "converged": point.converged,
        "gradient_intercept": point.gradient_intercept,
        "gradient_slope": point.gradient_slope,
    }
    return summary, point.anchor_table


def _decorate_predictions(table: pd.DataFrame, releases: dict[str, str]) -> pd.DataFrame:
    table = table.copy()
    table["target_release"] = table.model.map(lambda model: release_date(releases, model))
    table["judge_release"] = table.judge.map(lambda judge: release_date(releases, judge))
    table["post_judge_release"] = table.target_release > table.judge_release
    return table


def _write_figures(predictions: pd.DataFrame, out: Path, prefix: str = "",
                   label: str = "Judge prediction") -> None:
    out.mkdir(parents=True, exist_ok=True)
    for (bench, judge, quartile), table in predictions.groupby(
            ["bench", "judge", "target_quartile"], sort=False):
        for style in ("band", "points"):
            plot_frontier_style(table, judge, bench, quartile, out, style,
                                prediction_label=label, filename_prefix=prefix)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path,
                        default=ROOT / "configs/frontier_eci_canonical_anchors/experiment.yaml")
    parser.add_argument("--logs-root", type=Path,
                        default=ROOT / "logs/frontier-eci-canonical-anchors")
    parser.add_argument("--eci", type=Path,
                        default=ROOT / "external_data/epoch-eci/eci_scores.csv")
    parser.add_argument("--resamples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    config = _load_config(args.config)
    result_root = _path(config["results"])
    analysis_out = result_root / "analysis"
    baseline_out = result_root / "capability_baseline"
    _clear_checkpoints(analysis_out)
    _clear_checkpoints(baseline_out)
    coverage = pd.read_csv(result_root / "coverage/blocks.csv")
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
    canonical_models = (canonical.groupby("bench").model.unique()
                        .map(lambda values: sorted(map(str, values))).to_dict())
    targets = set(config["candidates"])
    releases = config["releases"]

    print("fitting headline capped judge forecasts", flush=True)
    predictions, fits, residuals, exported_coverage = analyze_blocks_parallel(
        coverage, protocol, hard, canonical_models, targets, releases,
        args.resamples, args.seed, workers=args.workers)
    predictions = _decorate_predictions(predictions, releases)
    _checkpoint_tables(analysis_out, "headline", {
        "predictions.csv": predictions,
        "anchor_fits.csv": fits,
        "anchor_residuals.csv": residuals,
        "metrics.csv": summarize_metrics(predictions),
        "coverage.csv": exported_coverage,
    })
    print("fitting uncapped judge sensitivity", flush=True)
    uncapped, uncapped_fits, _, _ = analyze_blocks_parallel(
        coverage, protocol, hard, canonical_models, targets, releases,
        args.resamples, args.seed, pair_limit=None, workers=args.workers)
    uncapped = _decorate_predictions(uncapped, releases)
    model_residuals = pd.concat([
        predictions.assign(analysis="capped"),
        uncapped.assign(analysis="uncapped"),
    ], ignore_index=True)[[
        "analysis", "bench", "judge", "target_quartile", "model", "actual",
        "predicted", "residual", "inside_anchor_range", "fit_status", "failure_reason",
    ]]
    _checkpoint_tables(analysis_out, "uncapped", {
        "predictions_uncapped.csv": uncapped,
        "anchor_fits_uncapped.csv": uncapped_fits,
        "metrics_uncapped.csv": summarize_metrics(uncapped),
        "model_residuals.csv": model_residuals,
    })

    from scripts.frontier_eci.canonical_capability_baseline import (
        eci_anchor_subsets, eci_mapping, fit_eci_blocks_parallel,
    )
    eci_models = sorted(
        targets
        | {model for models in canonical_models.values() for model in models}
        | {model for models in config.get("historical_backfill", {}).values() for model in models}
    )
    mapping = eci_mapping(args.eci, eci_models)
    eci = mapping.set_index("model").eci
    subsets = eci_anchor_subsets(fits, eci)
    subset_blocks = subsets.drop_duplicates("block")
    anchor_overrides = {
        row.block: [model for model in row.eci_anchor_ids.split("|") if model]
        for row in subset_blocks.itertuples(index=False)
    }
    print("fitting mapped-anchor paired judge sensitivity", flush=True)
    paired_predictions, paired_fits, paired_residuals, _ = analyze_blocks_parallel(
        coverage, protocol, hard, canonical_models, targets, releases,
        args.resamples, args.seed, workers=args.workers, anchor_overrides=anchor_overrides)
    paired_predictions = _decorate_predictions(paired_predictions, releases)
    subset_metadata = subset_blocks[[
        "block", "headline_anchor_ids", "eci_anchor_ids", "dropped_missing_eci_ids"]]
    paired_predictions = paired_predictions.merge(subset_metadata, on="block", how="left")
    paired_fits = paired_fits.merge(subset_metadata, on="block", how="left")
    _checkpoint_tables(baseline_out, "paired-judge", {
        "eci_model_mapping.csv": mapping,
        "paired_judge_predictions.csv": paired_predictions,
        "paired_judge_anchor_fits.csv": paired_fits,
        "paired_judge_anchor_residuals.csv": paired_residuals,
        "paired_judge_metrics.csv": summarize_metrics(paired_predictions),
    })
    target_by_block = coverage.assign(
        block=lambda frame: frame.bench + "|" + frame.judge + "|" + frame.target
    ).set_index("block").target.to_dict()
    print("fitting matched mapped-anchor ECI baseline", flush=True)
    eci_table = fit_eci_blocks_parallel(
        paired_fits, eci, hard, target_by_block, resamples=args.resamples,
        seed=args.seed, workers=args.workers)
    eci_residuals = []
    for row in eci_table.itertuples(index=False):
        if isinstance(row.anchor_table, pd.DataFrame):
            eci_residuals.append(row.anchor_table.assign(
                block=row.block, bench=row.bench, judge=row.judge,
                target=row.model, target_quartile=row.target_quartile))
    eci_predictions = _decorate_predictions(eci_table.drop(columns="anchor_table"), releases)
    eci_fits = eci_predictions[[
        "block", "bench", "judge", "model", "target_quartile", "headline_anchor_ids",
        "eci_anchor_ids", "dropped_missing_eci_ids", "anchor_ids", "n_anchors",
        "intercept", "slope", "converged", "gradient_intercept", "gradient_slope",
        "fit_status", "failure_reason",
    ]]
    eci_coverage = exported_coverage.copy()
    eci_coverage["eci_anchor_ids"] = eci_coverage.anchor_ids.map(
        lambda value: "|".join(model for model in str(value).split("|")
                               if model and pd.notna(eci.get(model))))
    eci_coverage["dropped_missing_eci_ids"] = eci_coverage.anchor_ids.map(
        lambda value: "|".join(model for model in str(value).split("|")
                               if model and pd.isna(eci.get(model))))
    mapped_counts = eci_coverage.eci_anchor_ids.map(
        lambda value: len([model for model in value.split("|") if model]))
    eci_coverage["analysis_status"] = np.where(
        mapped_counts.lt(4), "failed", eci_coverage.analysis_status)
    eci_coverage["analysis_failure_reason"] = np.where(
        mapped_counts.lt(4), "only " + mapped_counts.astype(str) + " mapped ECI anchors",
        eci_coverage.analysis_failure_reason)
    eci_residual_table = (pd.concat(eci_residuals, ignore_index=True)
                          if eci_residuals else pd.DataFrame())
    eci_model_residuals = eci_predictions[[
        "bench", "judge", "target_quartile", "model", "actual", "predicted",
        "residual", "inside_anchor_range", "fit_status", "failure_reason",
    ]]
    _checkpoint_tables(baseline_out, "eci", {
        "predictions.csv": eci_predictions,
        "anchor_fits.csv": eci_fits,
        "anchor_residuals.csv": eci_residual_table,
        "metrics.csv": summarize_metrics(eci_predictions),
        "coverage.csv": eci_coverage,
        "model_residuals.csv": eci_model_residuals,
    })

    if not args.no_figures:
        print("rendering release-time figures", flush=True)
        _write_figures(predictions, analysis_out / "figures/frontier_style")
        _write_figures(eci_predictions, baseline_out / "figures",
                       prefix="eci-baseline_", label="ECI prediction")
    summary = {
        "resamples": args.resamples,
        "verdict_rows_old": int(sum(len(table) for table in old)),
        "paid_success_rows": len(paid),
        "paid_candidate_logs": len(candidate_logs),
        "deduplicated_verdict_rows": len(verdicts),
        "q1_both_correct_rows": len(protocol),
        "coverage_blocks": len(exported_coverage),
        "coverage_identifiable": int(coverage.status.eq("identifiable").sum()),
        "coverage_missing": int(coverage.status.ne("identifiable").sum()),
        "judge_fit_failures": int(fits.fit_status.ne("ok").sum()),
        "eci_fit_failures": int(eci_fits.fit_status.ne("ok").sum()),
        "paired_judge_fit_failures": int(paired_fits.fit_status.ne("ok").sum()),
        "unmapped_eci_anchor_ids": sorted(
            model for model in set(
                anchor for value in exported_coverage.anchor_ids
                for anchor in str(value).split("|") if anchor)
            if pd.isna(eci.get(model))),
    }
    _atomic_write_json(summary, analysis_out / "summary.json")
    _atomic_write_json(summary, baseline_out / "summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
