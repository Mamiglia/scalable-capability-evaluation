"""Packaged extraction of attempt-level GPQA judge data."""

from __future__ import annotations

import json
import re
import zipfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import zipfile_zstd  # noqa: F401 - registers ZIP_ZSTANDARD with zipfile

from mc_cost.catalogue import catalogue_aliases, normalized_name


EXTERNAL_CAPABILITY_MODELS = {
    "claude-3-5-sonnet-0620": "claude-3-5-sonnet-20240620",
    "gemini-3.5-flash": "gemini-3.5-flash_high",
    "gpt-4.1": "gpt-4.1-2025-04-14",
    "gpt-4o": "gpt-4o-2024-08-06",
    "gpt-5.2-medium": "gpt-5.2-2025-12-11_medium",
    "gpt-5.5-low": "gpt-5.5_low",
    "llama-3.1-70b": "Llama-3.1-70B-Instruct",
    "mistral-medium": "mistral-medium-2505",
    "qwen3.6-flash": "qwen3.6-flash",
    "qwen3.6-plus": "qwen3.6-plus",
}
TRANSCRIPTS = re.compile(
    r"Transcript (?:A|1):\s*\n---\n(.*?)\n---\n.*?"
    r"Transcript (?:B|2):\s*\n---\n(.*?)\n---\n\s*Which of",
    re.DOTALL,
)
LEGACY_PRESENTATION_MAX_CHARS = 65536


def model_family(name: str) -> str:
    match = re.match(r"[a-z]+", name.lower())
    return match.group() if match else name.lower()


def standardise(data: pd.DataFrame, columns: Sequence[str]) -> None:
    for column in columns:
        scale = data[column].std()
        if not scale:
            raise ValueError(f"Cannot standardise constant feature: {column}")
        data[f"{column}_sd"] = (data[column] - data[column].mean()) / scale


def load_capabilities(models_path: Path, responses_path: Path) -> dict[str, float]:
    capabilities = pd.read_csv(models_path).set_index("model")["accuracy"].to_dict()
    responses = pd.read_parquet(responses_path, columns=["model", "correct"])
    epoch_accuracy = responses.groupby("model")["correct"].mean()
    for short_name, epoch_name in EXTERNAL_CAPABILITY_MODELS.items():
        if epoch_name not in epoch_accuracy:
            raise ValueError(f"Missing Epoch capability estimate for {epoch_name}")
        capabilities[short_name] = float(epoch_accuracy[epoch_name])
    return capabilities


def log_identity(path: Path, root: Path) -> tuple[str, str, str]:
    parts = path.relative_to(root).parts
    if parts[0].startswith("trunc-"):
        condition, judge, pair = parts[:3]
        condition = f"truncated_{int(condition.removeprefix('trunc-')):02d}"
    else:
        judge, pair = parts[:2]
        condition = "raw"
    if "-vs-" not in pair:
        raise ValueError(f"Cannot identify model pair from {path}")
    return judge, pair, condition


def transcript_text(sample: dict) -> tuple[str, str]:
    prompt = sample.get("input", "")
    if not isinstance(prompt, str):
        prompt = next(
            (
                message.get("content", "")
                for message in sample.get("messages", [])
                if message.get("role") == "user"
            ),
            "",
        )
    match = TRANSCRIPTS.search(prompt)
    if not match:
        raise ValueError(f"Could not parse transcripts for sample {sample.get('id')}")
    return match.group(1), match.group(2)


def judge_answer(sample: dict) -> tuple[str | None, float | None]:
    score = next(iter((sample.get("scores") or {}).values()), {})
    answer = score.get("answer")
    confidence = (score.get("metadata") or {}).get("confidence")
    answer = {"1": "A", "2": "B"}.get(str(answer), answer)
    return (answer if answer in {"A", "B"} else None), confidence


def major_is_left(sample: dict) -> bool:
    return str(sample.get("target")) in {"A", "1"}


def token_usage(sample: dict) -> dict[str, int]:
    usage = list((sample.get("model_usage") or {}).values())
    return {
        key: int(sum(model.get(key, 0) or 0 for model in usage))
        for key in ("input_tokens", "output_tokens", "reasoning_tokens", "total_tokens")
    }


def _exclude(exclusions: Counter, *, count: int = 1, **details) -> None:
    exclusions[json.dumps(details, sort_keys=True, separators=(",", ":"))] += count


def _exclusion_rows(exclusions: Counter) -> list[dict]:
    return [{**json.loads(key), "count": count} for key, count in sorted(exclusions.items())]


def extract_attempts(
    log_root: Path, capabilities: Mapping[str, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    seen_samples = set()
    accounting = defaultdict(Counter)
    exclusions = Counter()
    for path in sorted(log_root.rglob("*.eval")):
        judge, pair, condition = log_identity(path, log_root)
        accounting[judge]["eval_files_found"] += 1
        major, minor = pair.split("-vs-", 1)
        if capabilities is not None and (major not in capabilities or minor not in capabilities):
            raise ValueError(f"Missing capability estimate for pair {pair}")
        try:
            with zipfile.ZipFile(path) as archive:
                if "header.json" not in archive.namelist():
                    accounting[judge]["files_skipped_no_header"] += 1
                    _exclude(
                        exclusions, source="judgment", source_log=str(path), judge=judge,
                        model="|".join(sorted((major, minor))), item=None,
                        reason="missing_header", missing_fields=[],
                    )
                    continue
                header = json.loads(archive.read("header.json"))
                if header.get("status") != "success":
                    accounting[judge]["files_skipped_not_success"] += 1
                    _exclude(
                        exclusions, source="judgment", source_log=str(path), judge=judge,
                        model="|".join(sorted((major, minor))), item=None,
                        reason="non_success", missing_fields=[],
                    )
                    continue
                eval_id = header["eval"].get("eval_id")
                eval_metadata = header["eval"].get("metadata") or {}
                include_reasoning = bool(eval_metadata.get("include_reasoning"))
                try:
                    presentation_max_chars = int(
                        eval_metadata.get("max_length", LEGACY_PRESENTATION_MAX_CHARS)
                    )
                except (TypeError, ValueError):
                    raise ValueError("invalid judgment presentation max_length") from None
                if presentation_max_chars <= 0:
                    raise ValueError("judgment presentation max_length must be positive")
                truncate_frac = eval_metadata.get("truncate_frac")
                if truncate_frac is None and condition.startswith("truncated_"):
                    truncate_frac = int(condition.removeprefix("truncated_")) / 100
                truncate_keep = eval_metadata.get("truncate_keep", "first")
                phase_match = re.search(r"(?:phase-|_p)([12])", str(path))
                for name in archive.namelist():
                    if not name.startswith("samples/") or not name.endswith(".json"):
                        continue
                    sample = json.loads(archive.read(name))
                    item = str(sample.get("id", "")).rsplit("_", 1)[0]
                    sample_key = sample.get("uuid") or (eval_id, sample.get("id"), sample.get("epoch"))
                    if sample_key in seen_samples:
                        accounting[judge]["duplicate_attempts_skipped"] += 1
                        _exclude(
                            exclusions, source="judgment", source_log=str(path), judge=judge,
                            model="|".join(sorted((major, minor))), item=item,
                            reason="duplicate_attempt", missing_fields=[],
                        )
                        continue
                    answer, confidence = judge_answer(sample)
                    if answer is None:
                        accounting[judge]["attempts_skipped_invalid_answer"] += 1
                        _exclude(
                            exclusions, source="judgment", source_log=str(path), judge=judge,
                            model="|".join(sorted((major, minor))), item=item,
                            reason="invalid_answer", missing_fields=[],
                        )
                        seen_samples.add(sample_key)
                        continue
                    try:
                        left_text, right_text = transcript_text(sample)
                    except ValueError:
                        accounting[judge]["attempts_skipped_unparseable_transcript"] += 1
                        _exclude(
                            exclusions, source="judgment", source_log=str(path), judge=judge,
                            model="|".join(sorted((major, minor))), item=item,
                            reason="unparseable_transcript", missing_fields=[],
                        )
                        seen_samples.add(sample_key)
                        continue
                    major_left = major_is_left(sample)
                    left_model, right_model = (major, minor) if major_left else (minor, major)
                    usage = token_usage(sample)
                    row = {
                        "chosen_left": int(answer == "A"), "judge": judge,
                        "item": item,
                        "model_pair": "|".join(sorted((major, minor))),
                        "left_model": left_model, "right_model": right_model,
                        "transcript_condition": condition,
                        "length_difference": np.log1p(len(left_text)) - np.log1p(len(right_text)),
                        "reasoning_difference": int("<THOUGHT_START>" in left_text)
                        - int("<THOUGHT_START>" in right_text),
                        "contains_reasoning_and_the_other_does_not": int(
                            ("<THOUGHT_START>" in left_text) != ("<THOUGHT_START>" in right_text)
                        ),
                        "same_model_family": int(
                            left_model != right_model
                            and model_family(left_model) == model_family(right_model)
                        ),
                        "self_comparison": int(left_model == right_model),
                        "include_reasoning": include_reasoning,
                        "presentation_max_chars": presentation_max_chars,
                        "presentation_truncate_frac": truncate_frac,
                        "presentation_truncate_keep": truncate_keep,
                        "major_is_left": major_left, "answer": answer,
                        "confidence": confidence, "left_chars": len(left_text),
                        "right_chars": len(right_text),
                        "judge_input_tokens": usage["input_tokens"],
                        "judge_output_tokens": usage["output_tokens"],
                        "judge_reasoning_tokens": usage["reasoning_tokens"],
                        "judge_total_tokens": usage["total_tokens"],
                        "judge_model": header["eval"].get("model"),
                        "task_name": header["eval"].get("task"),
                        "experiment_name": eval_metadata.get("experiment_name"),
                        "phase": int(phase_match.group(1)) if phase_match else None,
                        "eval_id": eval_id, "sample_uuid": sample.get("uuid"),
                        "log_file": str(path),
                    }
                    if capabilities is not None:
                        row.update({
                            "left_capability": capabilities[left_model],
                            "right_capability": capabilities[right_model],
                            "capability_difference": capabilities[left_model] - capabilities[right_model],
                            "capability_gap": abs(capabilities[left_model] - capabilities[right_model]),
                        })
                    rows.append(row)
                    seen_samples.add(sample_key)
                    accounting[judge]["attempts_kept"] += 1
        except (OSError, KeyError, ValueError, zipfile.BadZipFile) as error:
            raise ValueError(f"Failed to extract {path}: {error}") from error
    result = pd.DataFrame(rows)
    result.attrs["exclusions"] = _exclusion_rows(exclusions)
    return result, pd.DataFrame.from_dict(accounting, orient="index").fillna(0).astype(int)


def _eligible(
    data: pd.DataFrame, accounting: pd.DataFrame, *, min_pairs: int, min_samples: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    eligibility = data.groupby("judge").agg(pairs=("model_pair", "nunique"), samples=("judge", "size"))
    eligibility = eligibility.reindex(
        eligibility.index.union(accounting.index, sort=False), fill_value=0,
    )
    eligibility["eligible"] = (
        (eligibility["pairs"] > min_pairs) & (eligibility["samples"] >= min_samples)
    )
    judges = eligibility.index[eligibility["eligible"]]
    data = data[data["judge"].isin(judges)].copy()
    if data.empty:
        raise ValueError("No judges meet the pair and sample thresholds")
    return data, accounting, eligibility


def prepare_for_mc(
    log_root: Path, catalogue: pd.DataFrame, *, min_pairs: int = 3, min_samples: int = 500,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Prepare MC attempts on the catalogue's IRT scale, never raw accuracy."""
    data, accounting = extract_attempts(log_root)
    exclusions = list(data.attrs.get("exclusions", []))
    aliases = catalogue_aliases(catalogue)
    identity_columns = ("judge", "left_model", "right_model")
    unresolved = sorted({
        str(value)
        for column in identity_columns
        for value in data[column].unique()
        if normalized_name(value) not in aliases
    })
    if unresolved:
        raise ValueError(f"judgment logs contain identities absent from catalogue: {', '.join(unresolved)}")
    for column in identity_columns:
        data[column] = data[column].map(lambda value: aliases[normalized_name(value)])

    metadata = catalogue.set_index("model_id")
    complete = pd.Series(True, index=data.index)
    missing_by_row = defaultdict(list)
    models_by_row = defaultdict(set)
    for column in identity_columns:
        rows = metadata.reindex(data[column])
        theta_present = pd.to_numeric(rows["theta"], errors="coerce").notna().to_numpy()
        release_present = pd.to_datetime(rows["release"], errors="coerce").notna().to_numpy()
        for index in data.index[~theta_present]:
            missing_by_row[index].append(f"{column}.theta")
            models_by_row[index].add(str(data.at[index, column]))
        for index in data.index[~release_present]:
            missing_by_row[index].append(f"{column}.release")
            models_by_row[index].add(str(data.at[index, column]))
        complete &= theta_present
        complete &= release_present
    metadata_exclusions = Counter()
    for index in data.index[~complete]:
        row = data.loc[index]
        _exclude(
            metadata_exclusions, source="judgment", source_log=str(row.get("log_file", "")),
            judge=str(row["judge"]), model="|".join(sorted(models_by_row[index])),
            item=str(row.get("item", "")), reason="missing_catalogue_metadata",
            missing_fields=sorted(missing_by_row[index]),
        )
    exclusions.extend(_exclusion_rows(metadata_exclusions))
    skipped = data.loc[~complete, "judge"].value_counts()
    if not skipped.empty:
        accounting["attempts_skipped_missing_catalogue_metadata"] = skipped.reindex(
            accounting.index, fill_value=0,
        ).astype(int)
    data = data.loc[complete].copy()
    data, accounting, eligibility = _eligible(
        data, accounting, min_pairs=min_pairs, min_samples=min_samples,
    )
    result = data.sort_values(["judge", "model_pair", "item", "transcript_condition"]).reset_index(drop=True)
    result.attrs["exclusions"] = sorted(
        exclusions, key=lambda row: json.dumps(row, sort_keys=True, separators=(",", ":")),
    )
    return result, accounting, eligibility


def prepare(
    log_root: Path, models_path: Path, responses_path: Path, *,
    min_pairs: int = 3, min_samples: int = 500,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data, accounting = extract_attempts(log_root, load_capabilities(models_path, responses_path))
    data, accounting, eligibility = _eligible(
        data, accounting, min_pairs=min_pairs, min_samples=min_samples,
    )
    standardise(data, ("length_difference", "capability_difference"))
    return (
        data.sort_values(["judge", "model_pair", "item", "transcript_condition"]).reset_index(drop=True),
        accounting,
        eligibility,
    )
