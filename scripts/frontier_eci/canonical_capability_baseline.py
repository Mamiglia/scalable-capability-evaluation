"""Matched-anchor Epoch ECI baseline for canonical frontier forecasts."""

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.frontier_eci.canonical_analysis import (  # noqa: E402
    _resample_hard_items, _target_actual, predict_from_ability,
)


ECI_NAMES = {
    "gpt-4-1106": "GPT-4 Turbo (Nov 2023)",
    "claude-3-opus": "Claude 3 Opus",
    "claude-3.5-sonnet-0620": "Claude 3.5 Sonnet",
    "claude-3.5-sonnet-1022": "Claude 3.5 Sonnet (October 2024)",
    "gpt-4o-0806": "GPT-4o (Aug 2024)",
    "gpt-4o-1120": "GPT-4o (Nov 2024)",
    "claude-3.7-sonnet": "Claude 3.7 Sonnet",
    "o3-mini-0131-medium": "o3-mini",
    "gemini-2.0-flash-001": "Gemini 2.0 Flash (Feb 2025)",
    "claude-opus-4.7": "Claude Opus 4.7",
    "claude-sonnet-4.5": "Claude Sonnet 4.5",
    "deepseek-r1-0528": "DeepSeek-R1 (May 2025)",
    "deepseek-v3.2": "DeepSeek-V3.2",
    "gemini-3.5-flash": "Gemini 3.5 Flash",
    "gemma-3-12b-it": None,
    "gemma-4-e2b-it": "Gemma 4 31B IT",
    "glm-4.6": "GLM-4.6",
    "glm-5.2": "GLM-5.2",
    "glm-5.3": None,
    "gpt-4.1": "GPT-4.1",
    "gpt-5": "GPT-5",
    "granite-4.1-8b": None,
    "kimi-k2.6": "Kimi K2.6",
    "kimi-k3": "Kimi K3",
    "ling-2.6-flash": None,
    "mistral-medium-3": "Mistral Medium 3",
    "muse-glimmer-30b": None,
    "nvidia-nemotron-3-ultra-550b-a55b-bf16": "Nemotron 3 Ultra",
    "nvidia-nemotron-3.5-lightning-30b-a3b-bf16": None,
    "o3": "o3",
    "qwen3-32b": None,
    "qwen3.5-4b": None,
    "qwen3.5-9b": None,
    "qwen3.8-2.4t-a95b": None,
    "qwen3.8-27b": None,
    "gemini-3-pro-preview": "Gemini 3 Pro",
    "claude-opus-4.6": "Claude Opus 4.6",
}


def eci_mapping(path: Path, models: list[str]) -> pd.DataFrame:
    """Validate and report the reviewed internal-ID to Epoch-name mapping."""
    table = pd.read_csv(path)
    rows = []
    for model in models:
        if model not in ECI_NAMES:
            raise ValueError(f"unreviewed ECI model identity: {model}")
        name = ECI_NAMES[model]
        if name is None:
            rows.append({"model": model, "epoch_model": "", "eci": np.nan,
                         "mapping_status": "not_in_epoch_eci"})
            continue
        matches = table[table.Model.eq(name)]
        if len(matches) != 1:
            raise ValueError(f"expected exactly one Epoch ECI row for {name!r}, found {len(matches)}")
        rows.append({"model": model, "epoch_model": name,
                     "eci": float(matches.iloc[0].eci), "mapping_status": "mapped"})
    mapping = pd.DataFrame(rows)
    mapped = mapping[mapping.mapping_status.eq("mapped")]
    if mapped.epoch_model.duplicated().any():
        raise ValueError("reviewed ECI mapping is not one-to-one")
    return mapping


def load_eci(path: Path, models: list[str]) -> pd.Series:
    mapping = eci_mapping(path, models)
    return mapping.set_index("model").eci


def eci_anchor_subsets(judge_fits: pd.DataFrame, eci: pd.Series) -> pd.DataFrame:
    rows = []
    for index, fit in judge_fits.iterrows():
        headline = str(fit.get("headline_anchor_ids", fit["anchor_ids"]))
        anchors = [model for model in headline.split("|") if model]
        mapped = [model for model in anchors if pd.notna(eci.get(model))]
        missing = [model for model in anchors if pd.isna(eci.get(model))]
        rows.append({"_index": index, "block": fit["block"],
                     "headline_anchor_ids": headline,
                     "eci_anchor_ids": "|".join(mapped),
                     "dropped_missing_eci_ids": "|".join(missing)})
    return pd.DataFrame(rows).set_index("_index")


def fit_eci_blocks(judge_fits: pd.DataFrame, eci: pd.Series,
                   hard_outcomes: pd.DataFrame,
                   targets: dict[str, str], resamples: int = 0,
                   seed: int = 0) -> pd.DataFrame:
    """Refit every judge block with the identical anchor IDs and ECI as ability."""
    rows = []
    subsets = eci_anchor_subsets(judge_fits, eci)
    for index, judge_fit in judge_fits.iterrows():
        row = judge_fit.to_dict()
        subset = subsets.loc[index]
        row.update(headline_anchor_ids=subset.headline_anchor_ids,
                   eci_anchor_ids=subset.eci_anchor_ids,
                   dropped_missing_eci_ids=subset.dropped_missing_eci_ids,
                   anchor_ids=subset.eci_anchor_ids)
        anchors = [model for model in subset.eci_anchor_ids.split("|") if model]
        row["n_anchors"] = len(anchors)
        target = targets[row["block"]]
        outcomes = hard_outcomes
        for column in ("bench", "target_quartile"):
            if column in outcomes and column in row and pd.notna(row[column]):
                outcomes = outcomes[outcomes[column].eq(row[column])]
        anchor_outcomes = outcomes[outcomes.model.isin(anchors)].copy()
        if len(anchors) < 4:
            actual = _target_actual(outcomes, target)
            row.update(predicted=np.nan, actual=actual, residual=np.nan,
                       intercept=np.nan, slope=np.nan, converged=False,
                       gradient_intercept=np.nan, gradient_slope=np.nan,
                       fit_status="failed", failure_reason=f"only {len(anchors)} mapped ECI anchors",
                       predicted_low=np.nan, predicted_median=np.nan,
                       predicted_high=np.nan, n_bootstrap=0,
                       bootstrap_unstable=False, anchor_table=None)
        elif pd.isna(eci.get(target)):
            actual = _target_actual(outcomes, target)
            row.update(predicted=np.nan, actual=actual, residual=np.nan,
                       intercept=np.nan, slope=np.nan, converged=False,
                       gradient_intercept=np.nan, gradient_slope=np.nan,
                       fit_status="failed", failure_reason=f"missing target ECI: {target}",
                       predicted_low=np.nan, predicted_median=np.nan,
                       predicted_high=np.nan, n_bootstrap=0,
                       bootstrap_unstable=False, anchor_table=None)
        else:
            try:
                fit = predict_from_ability(eci, anchor_outcomes, target, anchors)
                draws = []
                rng = np.random.default_rng(seed + int(index))
                for _ in range(resamples):
                    try:
                        draws.append(predict_from_ability(
                            eci, _resample_hard_items(anchor_outcomes, rng),
                            target, anchors).predicted)
                    except (RuntimeError, ValueError):
                        continue
                if resamples and not draws:
                    raise RuntimeError("every hard-item bootstrap calibration failed")
                low, median, high = ((np.quantile(draws, [0.025, 0.5, 0.975])
                                      if draws else [np.nan, np.nan, np.nan]))
                anchor_values = eci.reindex(anchors)
                actual = _target_actual(outcomes, target)
                row.update(predicted=fit.predicted, actual=actual,
                           residual=fit.predicted - actual, intercept=fit.intercept,
                           slope=fit.slope, converged=fit.converged,
                           gradient_intercept=fit.gradient_intercept,
                           gradient_slope=fit.gradient_slope,
                           predicted_low=float(low), predicted_median=float(median),
                           predicted_high=float(high), n_bootstrap=len(draws),
                           bootstrap_unstable=len(draws) < 0.8 * resamples,
                           target_ability=float(eci[target]),
                           anchor_ability_min=float(anchor_values.min()),
                           anchor_ability_max=float(anchor_values.max()),
                           inside_anchor_range=(float(anchor_values.min()) <= eci[target]
                                                <= float(anchor_values.max())),
                           fit_status="ok", failure_reason="",
                           anchor_table=fit.anchor_table)
            except (RuntimeError, ValueError) as error:
                actual = _target_actual(outcomes, target)
                row.update(predicted=np.nan, actual=actual, residual=np.nan,
                           intercept=np.nan, slope=np.nan, converged=False,
                           gradient_intercept=np.nan, gradient_slope=np.nan,
                           predicted_low=np.nan, predicted_median=np.nan,
                           predicted_high=np.nan, n_bootstrap=0,
                           bootstrap_unstable=False, fit_status="failed",
                           failure_reason=str(error), anchor_table=None)
        rows.append(row)
    return pd.DataFrame(rows)


def _eci_tasks(judge_fits: pd.DataFrame) -> list[pd.DataFrame]:
    return [judge_fits.iloc[[index]].copy() for index in range(len(judge_fits))]


def fit_eci_blocks_parallel(judge_fits: pd.DataFrame, eci: pd.Series,
                            hard_outcomes: pd.DataFrame, targets: dict[str, str],
                            resamples: int = 0, seed: int = 0,
                            workers: int = 1) -> pd.DataFrame:
    tasks = _eci_tasks(judge_fits)
    if not tasks:
        return pd.DataFrame()
    if workers <= 1:
        return pd.concat([
            fit_eci_blocks(task, eci, hard_outcomes, targets, resamples, seed)
            for task in tasks
        ], ignore_index=True) if tasks else pd.DataFrame()
    count = min(workers, 18, len(tasks))
    with ProcessPoolExecutor(max_workers=count,
                             mp_context=multiprocessing.get_context("spawn")) as executor:
        futures = {executor.submit(
            fit_eci_blocks, task, eci, hard_outcomes, targets, resamples, seed): index
                   for index, task in enumerate(tasks)}
        results = [None] * len(tasks)
        for completed, future in enumerate(as_completed(futures), 1):
            results[futures[future]] = future.result()
            print(f"ECI task {completed}/{len(tasks)} complete", flush=True)
    return pd.concat(results, ignore_index=True)
