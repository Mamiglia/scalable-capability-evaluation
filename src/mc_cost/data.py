"""Prepare fitted-IRT judgment features and empirical token rows."""

from __future__ import annotations

from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from inspect_ai.log import list_eval_logs, read_eval_log

from judge.util import extract_transcript


RELEASE_EPOCH = pd.Timestamp("2020-01-01")

JUDGMENT_COLUMNS = [
    "chosen_left", "judge", "item", "left_model", "right_model",
    "theta_gap", "oversight", "release_gap_days", "relative_release_days",
    "difficulty", "theta_gap_x_oversight", "theta_gap_x_difficulty",
    "length_difference", "same_family_x_theta_gap", "self_comparison",
    "reasoning_difference", "left_chars", "right_chars",
]

CONTINUOUS_JUDGMENT_COLUMNS = [
    "theta_gap", "oversight", "release_gap_days", "relative_release_days",
    "difficulty", "theta_gap_x_oversight", "theta_gap_x_difficulty",
    "length_difference", "same_family_x_theta_gap", "reasoning_difference",
]


def _short_model(slug: str) -> str:
    return slug.removeprefix("openrouter/").split("/")[-1]


def _sample_tokens(sample) -> dict[str, int]:
    """Match the Inspect accounting used by ``scripts/cost/eval_cost.py``."""
    total = defaultdict(int)
    for usage in (sample.model_usage or {}).values():
        total["input"] += usage.input_tokens or 0
        total["output"] += usage.output_tokens or 0
        total["reasoning"] += getattr(usage, "reasoning_tokens", 0) or 0
        total["total"] += usage.total_tokens or 0
    return {key: total[key] for key in ("input", "output", "reasoning", "total")}


def _sample_correct(sample) -> int:
    """Match the existing Inspect scorer convention for correct answers."""
    for score in (sample.scores or {}).values():
        value = score.value
        if value in ("C", 1, True) or (isinstance(value, float) and value >= 0.5):
            return 1
    return 0


def _gpqa_items(items: pd.DataFrame) -> pd.DataFrame:
    required = {"irt_model", "fit", "benchmark", "item", "difficulty"}
    if not required <= set(items):
        return pd.DataFrame(columns=["benchmark", "item", "difficulty"])
    return items.loc[
        (items["irt_model"] == "fixed_3pl")
        & (items["fit"] == "gpqa")
        & (items["benchmark"] == "gpqa"),
        ["benchmark", "item", "difficulty"],
    ].copy()


def _release_days(values: pd.Series) -> pd.Series:
    return (pd.to_datetime(values, errors="coerce") - RELEASE_EPOCH).dt.days


def _catalogue_metadata(catalogue: pd.DataFrame) -> pd.DataFrame:
    required = {"model_id", "theta", "release"}
    missing = required - set(catalogue)
    if missing:
        raise ValueError(f"catalogue lacks required columns: {', '.join(sorted(missing))}")
    frame = catalogue.copy()
    if frame["model_id"].duplicated().any():
        raise ValueError("duplicate model_id in catalogue")
    frame["theta"] = pd.to_numeric(frame["theta"], errors="coerce")
    frame["release_day"] = _release_days(frame["release"])
    return frame.set_index("model_id", drop=False)


def _missing_judgment_metadata(frame: pd.DataFrame) -> list[str]:
    missing = []
    for role in ("judge", "left", "right"):
        identity = frame[f"{role}_model"] if role != "judge" else frame["judge"]
        for column, label in ((f"{role}_theta", "theta"), (f"{role}_release_day", "release")):
            for value in identity[frame[column].isna()].astype(str).unique():
                missing.append(f"{role} {value} missing {label}")
    for value in frame.loc[frame["difficulty"].isna(), "item"].astype(str).unique():
        missing.append(f"item {value} missing difficulty")
    return missing


def prepare_judgments(attempts: pd.DataFrame, catalogue: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    """Join observed attempts to fixed-3PL metadata and derive model features."""
    frame = attempts.copy()
    metadata = _catalogue_metadata(catalogue)
    for role, identity_column in (("judge", "judge"), ("left", "left_model"), ("right", "right_model")):
        joined = metadata.reindex(frame[identity_column])
        frame[f"{role}_theta"] = joined["theta"].to_numpy()
        frame[f"{role}_release_day"] = joined["release_day"].to_numpy()

    difficulty = _gpqa_items(items).drop_duplicates(["benchmark", "item"])
    frame = frame.merge(difficulty.drop(columns="benchmark"), on="item", how="left", validate="many_to_one")
    missing = _missing_judgment_metadata(frame)
    if missing:
        raise ValueError("; ".join(missing))

    frame["theta_gap"] = frame["left_theta"] - frame["right_theta"]
    frame["oversight"] = (frame["left_theta"] + frame["right_theta"]) / 2 - frame["judge_theta"]
    frame["release_gap_days"] = frame["left_release_day"] - frame["right_release_day"]
    frame["relative_release_days"] = (
        (frame["left_release_day"] + frame["right_release_day"]) / 2 - frame["judge_release_day"]
    )
    frame["theta_gap_x_oversight"] = frame["theta_gap"] * frame["oversight"]
    frame["theta_gap_x_difficulty"] = frame["theta_gap"] * frame["difficulty"]
    frame["same_family_x_theta_gap"] = frame["same_model_family"] * frame["theta_gap"]
    return frame.reindex(columns=JUDGMENT_COLUMNS + [column for column in frame if column not in JUDGMENT_COLUMNS])


def _candidate_item_id(sample) -> str:
    return str(sample.id).rsplit("_", 1)[0]


def candidate_token_rows(log_roots, catalogue: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    """Extract one valid candidate-generation usage row per model/item/epoch."""
    metadata = _catalogue_metadata(catalogue)
    difficulty = _gpqa_items(items).set_index(["benchmark", "item"])["difficulty"]
    rows, seen_logs, seen_samples = [], set(), set()
    excluded_zero_or_error = 0
    for root in log_roots:
        for info in list_eval_logs(str(root)):
            if info.name in seen_logs:
                continue
            seen_logs.add(info.name)
            log = read_eval_log(info)
            if not log.samples:
                continue
            benchmark = log.eval.task.split("/")[-1]
            model = _short_model(log.eval.model)
            for sample in log.samples:
                tokens = _sample_tokens(sample)
                if getattr(sample, "error", None) or tokens["total"] == 0:
                    excluded_zero_or_error += 1
                    continue
                item, epoch = _candidate_item_id(sample), getattr(sample, "epoch", None)
                key = benchmark, model, item, epoch
                if key in seen_samples:
                    continue
                seen_samples.add(key)
                model_data = metadata.loc[model] if model in metadata.index else None
                item_difficulty = difficulty.get((benchmark, item), np.nan)
                missing = []
                if model_data is None or pd.isna(model_data["theta"]):
                    missing.append("theta")
                if model_data is None or pd.isna(model_data["release_day"]):
                    missing.append("release")
                if pd.isna(item_difficulty):
                    missing.append("difficulty")
                if missing:
                    details = []
                    model_fields = [field for field in missing if field != "difficulty"]
                    if model_fields:
                        details.append(f"model {model} missing {', '.join(model_fields)}")
                    if "difficulty" in missing:
                        details.append(f"item {item} missing difficulty")
                    raise ValueError("; ".join(details))
                transcript = extract_transcript(sample, include_reasoning=True)
                reasoning_tokens = tokens["reasoning"]
                rows.append({
                    "role": "candidate", "benchmark": benchmark, "model": model, "item": item, "epoch": epoch,
                    "response_chars": len(transcript), "correct": _sample_correct(sample),
                    "theta": model_data["theta"], "release_day": model_data["release_day"],
                    "difficulty": item_difficulty,
                    "visible_reasoning": int("<THOUGHT_START>" in transcript or "<think>" in transcript),
                    "reasoning_class": model_data.get("reasoning_class", pd.NA),
                    "input_tokens": tokens["input"], "output_tokens": tokens["output"],
                    "reasoning_tokens": reasoning_tokens, "ordinary_output_tokens": max(tokens["output"] - reasoning_tokens, 0),
                    "total_tokens": tokens["total"],
                })
    result = pd.DataFrame(rows)
    result.attrs["accounting"] = {"zero_usage_or_error": excluded_zero_or_error}
    return result


def judge_token_rows(judgments: pd.DataFrame) -> pd.DataFrame:
    """Convert observed judge attempts into token-model rows."""
    frame = judgments.copy()
    output = pd.to_numeric(frame["judge_output_tokens"], errors="coerce").fillna(0)
    reasoning = pd.to_numeric(frame["judge_reasoning_tokens"], errors="coerce").fillna(0)
    columns = {
        "role": "judge", "model": frame["judge"], "judge": frame["judge"], "item": frame["item"],
        "chosen_left": frame["chosen_left"], "transcript_chars": frame["left_chars"] + frame["right_chars"],
        "input_tokens": frame["judge_input_tokens"], "output_tokens": output,
        "reasoning_tokens": reasoning, "ordinary_output_tokens": (output - reasoning).clip(lower=0),
        "total_tokens": frame["judge_total_tokens"],
    }
    if "difficulty" in frame:
        columns["difficulty"] = frame["difficulty"]
    return pd.DataFrame(columns)


def standardize_judgments(
    judgments: pd.DataFrame, training_rows,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Standardize continuous predictors using the supplied training rows."""
    frame = judgments.copy()
    training_rows = np.asarray(training_rows, dtype=bool)
    if training_rows.shape != (len(frame),) or not training_rows.any():
        raise ValueError("training_rows must select at least one judgment row")
    stats = []
    for column in CONTINUOUS_JUDGMENT_COLUMNS:
        values = pd.to_numeric(frame[column], errors="raise")
        training_values = values[training_rows]
        mean = float(training_values.mean())
        scale = float(training_values.std(ddof=0)) or 1.0
        frame[column] = (values - mean) / scale
        stats.append({"feature": column, "mean": mean, "scale": scale})
    return frame, pd.DataFrame(stats).set_index("feature")


def write_prepared_tables(
    output_directory: str | Path,
    judgments: pd.DataFrame,
    candidate_tokens: pd.DataFrame,
    judge_tokens: pd.DataFrame,
    training_rows,
) -> dict[str, Path]:
    """Write standardized training judgments, token rows, and their scales."""
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    standardized, scales = standardize_judgments(judgments, training_rows)
    paths = {
        "judgments": output_directory / "judgments.parquet",
        "candidate_tokens": output_directory / "token_candidate.parquet",
        "judge_tokens": output_directory / "token_judge.parquet",
        "judgment_standardization": output_directory / "judgment_standardization.json",
    }
    standardized.to_parquet(paths["judgments"], index=False)
    candidate_tokens.to_parquet(paths["candidate_tokens"], index=False)
    judge_tokens.to_parquet(paths["judge_tokens"], index=False)
    scales.to_json(paths["judgment_standardization"], indent=2)
    return paths
