#!/usr/bin/env python3
"""Extract attempt-level GPQA judge data directly from Inspect eval logs."""

import argparse
import json
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import zipfile_zstd  # noqa: F401 - registers ZIP_ZSTANDARD with zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import transcript_features  # noqa: E402 - sibling module, not an installed package


# Short pair-directory names whose transcripts come from the Epoch pool. Each
# judge log states its own sources in metadata.major_path / metadata.minor_path,
# so this table is only a fallback for logs written before that metadata existed;
# resolve_sources() derives the same mapping from the logs themselves and
# covers every Epoch model, not just the ten listed here.
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
# Transcripts were labelled A/B before commit 9d3e1db and 1/2 after it, and the
# 1/2 prompt interleaves an MCQ options block between the two transcripts. The
# middle section stays anchored to that literal block rather than a free `.*?`:
# transcripts often contain their own `---` rule, and only a strict separator
# forces the engine to backtrack past it to the real transcript boundary.
TRANSCRIPTS = re.compile(
    r"Transcript [A1]:\s*\n---\n(.*?)\n---\n\s*"
    r"(?:Multiple choice options as shown to Model 2:\n.*?\n\n)?"
    r"Transcript [B2]:\s*\n---\n(.*?)\n---\n\s*Which of",
    re.DOTALL,
)
LEFT_LABELS = {"A", "1"}
ANSWER_LABELS = {"A", "B", "1", "2"}


def model_name_family(name: str) -> str:
    """Leading alphabetic token used for the exploratory family split."""
    match = re.match(r"[a-z]+", name.lower())
    return match.group() if match else name.lower()


def resolve_sources(log_root: Path) -> dict[str, str]:
    """Map each pair-directory model name to the transcript log it came from.

    Every judge log records its own two sources, so the mapping does not have to
    be maintained by hand:

        metadata.major_path = external_data/epoch-gpqa/gpt-5.5_low.eval
        pair directory      = .../gpt-5.5-low-vs-qwen3.6-plus

    The value is the file stem, which for an Epoch log is the Epoch "Model
    version" and for one of our collection runs is that run's identifier.
    """
    resolved: dict[str, str] = {}
    for path in sorted(log_root.rglob("*.eval")):
        pair = next((p for p in path.parts if "-vs-" in p), None)
        if pair is None:
            continue
        try:
            with zipfile.ZipFile(path) as archive:
                if "header.json" not in archive.namelist():
                    continue
                header = json.loads(archive.read("header.json"))
                if header.get("status") != "success":
                    continue
                metadata = (header["eval"].get("metadata") or {})
        except (OSError, KeyError, ValueError, zipfile.BadZipFile):
            continue
        major, minor = pair.split("-vs-", 1)
        for name, key in ((major, "major_path"), (minor, "minor_path")):
            if not metadata.get(key):
                continue
            stem = Path(str(metadata[key])).stem
            if resolved.setdefault(name, stem) != stem:
                # Silently keeping whichever log sorted first would give this
                # model the wrong accuracy in every pair built from the other.
                raise ValueError(
                    f"{name} resolves to two different source transcripts: "
                    f"{resolved[name]} and {stem} (from {path}). The logs disagree "
                    "about which run produced this model, so its capability cannot "
                    "be determined."
                )
    return resolved


def collected_accuracies(collected_dir: Path) -> dict[str, float]:
    """Benchmark accuracy of each of our own collection runs, by file stem.

    Inspect stores the scorer's headline metric in the log header, so the whole
    sample set does not have to be read back.
    """
    accuracies: dict[str, float] = {}
    for path in sorted(collected_dir.glob("*.eval")):
        try:
            with zipfile.ZipFile(path) as archive:
                header = json.loads(archive.read("header.json"))
        except (OSError, KeyError, ValueError, zipfile.BadZipFile):
            continue
        if header.get("status") != "success":
            continue
        for score in (header.get("results") or {}).get("scores", []):
            value = (score.get("metrics") or {}).get("accuracy", {}).get("value")
            if value is not None:
                accuracies[path.stem] = float(value)
                break
    return accuracies


def load_capabilities(models_path: Path | None, responses_path: Path,
                      log_root: Path | None = None,
                      collected_dir: Path | None = None) -> dict[str, float]:
    """GPQA accuracy per pair-directory model name.

    Three sources: the Epoch response set for models from that pool, the header
    of our own collection logs for models we ran ourselves, and `models_path`.
    Resolved sources win over `models_path`, because they are read from the log
    that was actually judged; `models_path` supplies only the models no log
    resolves. All three are optional, and between them they need to cover every
    model that was judged.
    """
    capabilities: dict[str, float] = {}
    if models_path is not None and models_path.exists():
        capabilities.update(pd.read_csv(models_path).set_index("model")["accuracy"].to_dict())
    responses = pd.read_parquet(responses_path, columns=["model", "correct"])
    epoch_accuracy = responses.groupby("model")["correct"].mean()
    collected = collected_accuracies(collected_dir) if collected_dir is not None else {}

    mapping = dict(EXTERNAL_CAPABILITY_MODELS)
    resolved: set[str] = set()
    if log_root is not None:
        sources = resolve_sources(log_root)
        mapping.update(sources)
        resolved = set(sources)
    for short_name, stem in mapping.items():
        if stem in epoch_accuracy:
            capabilities[short_name] = float(epoch_accuracy[stem])
        elif stem in collected:
            capabilities[short_name] = collected[stem]
        elif short_name in resolved:
            # Metadata identifies the exact transcript source. A fallback value
            # for the same short name can describe a different run and must not
            # silently replace that missing source.
            capabilities.pop(short_name, None)
    return capabilities


def log_identity(path: Path, root: Path) -> tuple[str, str, str]:
    parts = path.relative_to(root).parts
    if parts[0].startswith("trunc-"):
        condition, judge, pair = parts[:3]
        tag = condition.removeprefix("trunc-")
        if tag.startswith("keepans-"):
            condition = f"truncated_keepans_{int(tag.removeprefix('keepans-')):02d}"
        else:
            condition = f"truncated_{int(tag):02d}"
    else:
        judge, pair = parts[:2]
        condition = "raw"
    if "-vs-" not in pair:
        raise ValueError(f"Cannot identify model pair from {path}")
    return judge, pair, condition


def prompt_text(sample: dict) -> str:
    prompt = sample.get("input", "")
    if not isinstance(prompt, str):
        prompt = next(
            (message.get("content", "") for message in sample.get("messages", []) if message.get("role") == "user"),
            "",
        )
    return prompt


def transcript_text(prompt: str) -> tuple[str, str]:
    match = TRANSCRIPTS.search(prompt)
    if not match:
        raise ValueError("Could not parse the two transcripts from the judge prompt")
    return match.group(1), match.group(2)


def judge_answer(sample: dict) -> tuple[str | None, float | None]:
    scores = sample.get("scores") or {}
    score = scores.get("judge_scorer")
    if score is None:
        if len(scores) > 1:
            raise ValueError(
                f"Sample {sample.get('id')} has multiple scorers and no judge_scorer"
            )
        score = next(iter(scores.values()), {})
    answer = score.get("answer")
    confidence = (score.get("metadata") or {}).get("confidence")
    return (answer if answer in ANSWER_LABELS else None), confidence


def token_usage(sample: dict) -> dict[str, int]:
    usage = list((sample.get("model_usage") or {}).values())
    return {
        key: int(sum(model.get(key, 0) or 0 for model in usage))
        for key in ("input_tokens", "output_tokens", "reasoning_tokens", "total_tokens")
    }


def check_capabilities(log_root: Path, capabilities: dict[str, float]) -> None:
    """Fail before extraction, naming every model that has no accuracy.

    Reporting them one at a time turns a single missing collection run into as
    many failed runs as it has pairs.
    """
    missing: dict[str, set[str]] = {}
    for path in sorted(log_root.rglob("*.eval")):
        pair = next((p for p in path.parts if "-vs-" in p), None)
        if pair is None:
            continue
        try:
            with zipfile.ZipFile(path) as archive:
                if "header.json" not in archive.namelist():
                    continue
                if json.loads(archive.read("header.json")).get("status") != "success":
                    continue
        except (OSError, ValueError, zipfile.BadZipFile):
            continue
        for name in pair.split("-vs-", 1):
            # A present-but-NaN accuracy is worse than an absent one: it passes a
            # membership test and then turns every gradient that touches it into
            # NaN, with the sampler failing far from the cause.
            if name not in capabilities or not np.isfinite(capabilities[name]):
                missing.setdefault(name, set()).add(pair)
    if not missing:
        return
    lines = "\n".join(
        f"  {name:26s} in {len(pairs)} pair(s), e.g. {sorted(pairs)[0]}"
        for name, pairs in sorted(missing.items())
    )
    raise ValueError(
        f"No GPQA accuracy for {len(missing)} model(s):\n{lines}\n"
        "These transcripts came from our own collection runs, not the Epoch pool, "
        "so their accuracy is not in the Epoch response set. Supply it with "
        "--models (a csv of model,accuracy) or point --logs at a run that uses "
        "Epoch transcripts only."
    )


def extract_attempts(log_root: Path, capabilities: dict[str, float],
                     with_features: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    check_capabilities(log_root, capabilities)
    rows = []
    seen_samples = set()
    accounting = defaultdict(Counter)
    for path in sorted(log_root.rglob("*.eval")):
        # Reverse-order samples in one log contain the same two transcripts.
        # Keep this cache local to the log so it halves repeat work without
        # retaining the full corpus in memory.
        feature_cache: dict[tuple[str, str, str], dict[str, float]] = {}
        judge, pair, condition = log_identity(path, log_root)
        accounting[judge]["eval_files_found"] += 1
        major, minor = pair.split("-vs-", 1)
        try:
            with zipfile.ZipFile(path) as archive:
                if "header.json" not in archive.namelist():
                    accounting[judge]["files_skipped_no_header"] += 1
                    continue
                header = json.loads(archive.read("header.json"))
                if header.get("status") != "success":
                    accounting[judge]["files_skipped_not_success"] += 1
                    continue
                eval_id = header["eval"].get("eval_id")
                eval_metadata = header["eval"].get("metadata") or {}
                include_reasoning = bool(eval_metadata.get("include_reasoning"))
                phase_match = re.search(r"(?:phase-|_p)([12])", str(path))
                for name in archive.namelist():
                    if not name.startswith("samples/") or not name.endswith(".json"):
                        continue
                    sample = json.loads(archive.read(name))
                    sample_key = sample.get("uuid") or (eval_id, sample.get("id"), sample.get("epoch"))
                    if sample_key in seen_samples:
                        accounting[judge]["duplicate_attempts_skipped"] += 1
                        continue
                    answer, confidence = judge_answer(sample)
                    if answer is None:
                        accounting[judge]["attempts_without_answer"] += 1
                        continue
                    prompt = prompt_text(sample)
                    left_text, right_text = transcript_text(prompt)
                    target = sample.get("target")
                    if target not in ANSWER_LABELS:
                        raise ValueError(
                            f"Sample {sample.get('id')} has invalid target {target!r}"
                        )
                    major_is_left = target in LEFT_LABELS
                    left_model, right_model = (major, minor) if major_is_left else (minor, major)
                    usage = token_usage(sample)
                    # gpqa_tfa's 138 features, differenced major minus minor, on
                    # the transcripts as the judge saw them. Same definitions as
                    # the capability analysis, so the feature names and signs align.
                    major_text, minor_text = (
                        (left_text, right_text) if major_is_left else (right_text, left_text)
                    )
                    question = transcript_features.question_of(prompt)
                    cache_key = (major_text, minor_text, question)
                    if with_features and cache_key not in feature_cache:
                        feature_cache[cache_key] = transcript_features.deltas(
                            major_text, minor_text, question
                        )
                    feature_deltas = feature_cache.get(cache_key, {})
                    rows.append(
                        {
                            **feature_deltas,
                            "chosen_left": int(answer in LEFT_LABELS),
                            "judge": judge,
                            "item": str(sample.get("id", "")).rsplit("_", 1)[0],
                            "model_pair": "|".join(sorted((major, minor))),
                            "left_model": left_model,
                            "right_model": right_model,
                            "left_capability": capabilities[left_model],
                            "right_capability": capabilities[right_model],
                            "transcript_condition": condition,
                            "length_difference": np.log1p(len(left_text)) - np.log1p(len(right_text)),
                            "reasoning_difference": int("<THOUGHT_START>" in left_text)
                            - int("<THOUGHT_START>" in right_text),
                            "contains_reasoning_and_the_other_does_not": int(
                                ("<THOUGHT_START>" in left_text) != ("<THOUGHT_START>" in right_text)
                            ),
                            "capability_difference": capabilities[left_model] - capabilities[right_model],
                            "capability_gap": abs(capabilities[left_model] - capabilities[right_model]),
                            # Correctness frame: "correct" means the judge picked the
                            # major slot, which is what the eval itself scored. That
                            # equals picking the stronger model only where
                            # capability_difference_major is positive.
                            "is_correct": int((answer in LEFT_LABELS) == major_is_left),
                            "truth_side": "left" if major_is_left else "right",
                            "major_is_right": int(not major_is_left),
                            "capability_difference_major": capabilities[major] - capabilities[minor],
                            "length_difference_major": (np.log1p(len(left_text)) - np.log1p(len(right_text)))
                            * (1 if major_is_left else -1),
                            "same_model_family": int(
                                left_model != right_model
                                and model_name_family(left_model) == model_name_family(right_model)
                            ),
                            "self_comparison": int(left_model == right_model),
                            "include_reasoning": include_reasoning,
                            "major_is_left": major_is_left,
                            "answer": answer,
                            "confidence": confidence,
                            "left_chars": len(left_text),
                            "right_chars": len(right_text),
                            "judge_input_tokens": usage["input_tokens"],
                            "judge_output_tokens": usage["output_tokens"],
                            "judge_reasoning_tokens": usage["reasoning_tokens"],
                            "judge_total_tokens": usage["total_tokens"],
                            "judge_model": header["eval"].get("model"),
                            "task_name": header["eval"].get("task"),
                            "experiment_name": eval_metadata.get("experiment_name"),
                            "phase": int(phase_match.group(1)) if phase_match else None,
                            "eval_id": eval_id,
                            "sample_uuid": sample.get("uuid"),
                            "log_file": str(path),
                        }
                    )
                    seen_samples.add(sample_key)
                    accounting[judge]["attempts_kept"] += 1
        except (OSError, KeyError, ValueError, zipfile.BadZipFile) as error:
            raise ValueError(f"Failed to extract {path}: {error}") from error
    return pd.DataFrame(rows), pd.DataFrame.from_dict(accounting, orient="index").fillna(0).astype(int)


def prepare(
    log_root: Path,
    models_path: Path | None,
    responses_path: Path,
    *,
    min_pairs: int = 3,
    min_samples: int = 500,
    with_features: bool = True,
    collected_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data, accounting = extract_attempts(
        log_root,
        load_capabilities(models_path, responses_path, log_root, collected_dir),
        with_features=with_features,
    )
    if data.empty:
        raise ValueError("No successful judge attempts were extracted")
    eligibility = data.groupby("judge").agg(pairs=("model_pair", "nunique"), samples=("judge", "size"))
    judges = eligibility[(eligibility["pairs"] > min_pairs) & (eligibility["samples"] >= min_samples)].index
    data = data[data["judge"].isin(judges)].copy()
    if data.empty:
        raise ValueError("No judges meet the pair and sample thresholds")
    reversed_pairs = data[
        (data["self_comparison"] == 0) & (data["capability_difference_major"] <= 0)
    ]["model_pair"].unique()
    if len(reversed_pairs):
        raise ValueError(
            "The configured major model is not the higher-accuracy model for "
            f"{len(reversed_pairs)} pair(s), including {sorted(reversed_pairs)[0]}. "
            "The GLM signs features against the major model, so reorder these pairs "
            "before fitting."
        )

    def scale_of(column: str) -> float:
        """SD over contested rows only; self-pairs have a gap of exactly zero
        and would deflate it. NaN when every pair is a self-pair."""
        value = data.loc[data["self_comparison"] == 0, column].std()
        return float("nan") if value is None else float(value)

    for column in ("length_difference", "capability_difference"):
        scale = data[column].std()
        if not np.isfinite(scale) or scale == 0:
            raise ValueError(f"Cannot standardise constant feature: {column}")
        data[f"{column}_sd"] = (data[column] - data[column].mean()) / scale
    # Correctness-model predictors are scaled but not centred, so that zero keeps its
    # meaning: no capability gap, no length advantage. Self-pairs are left out of the
    # scale because their gap is identically zero and would deflate it.
    for column in ("capability_difference_major", "length_difference_major"):
        scale = scale_of(column)
        if not np.isfinite(scale) or scale == 0:
            raise ValueError(f"Cannot scale constant feature: {column}")
        data[f"{column}_sd"] = data[column] / scale
    # The 138 transcript-feature differences. Scaled without centring for the
    # same reason, and split into the two parts that answer different questions:
    #   _between  the pair's average difference   -> "does the judge prefer models
    #                                                that write this way"
    #   _within   this item's deviation from it   -> "on the questions where this
    #                                                model wrote more of it than
    #                                                usual, does the judge switch"
    # Only the within part is immune to the judge simply recognising the models.
    # The pooled predictor is their sum, so it is not stored: entering both parts
    # gives the same information without forcing one coefficient on them.
    parts = {}
    for column in [c for c in data.columns if c.startswith("d_")]:
        scale = scale_of(column)
        if not np.isfinite(scale) or scale == 0:
            continue  # constant or all-NaN feature: no information, leave it out
        scaled = data[column] / scale
        pair_mean = scaled.groupby(data["model_pair"]).transform("mean")
        parts[f"{column}_between"] = pair_mean
        parts[f"{column}_within"] = scaled - pair_mean
    if parts:
        data = pd.concat([data, pd.DataFrame(parts, index=data.index)], axis=1)
    return data.sort_values(["judge", "model_pair", "item", "transcript_condition"]).reset_index(drop=True), accounting, eligibility


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", type=Path, default=Path("logs/imported_runs/gpqa_judging"))
    parser.add_argument("--models", type=Path,
                        default=Path("logs/imported_runs_2/gpqa-pilot-rd2/models.csv"),
                        help="fallback accuracy for models no log resolves; not needed "
                             "when every judged transcript resolves to the Epoch pool "
                             "or to a --collected run")
    parser.add_argument("--responses", type=Path, default=Path("external_data/epoch-gpqa/responses.parquet"))
    parser.add_argument("--collected", type=Path, default=Path("data/gpqa"),
                        help="directory of our own collection .eval logs; their headers "
                             "supply the accuracy for models outside the Epoch pool")
    parser.add_argument("--out", type=Path, default=Path("analysis/hibayes/gpqa_glm.csv"))
    parser.add_argument(
        "--out-correctness",
        type=Path,
        default=Path("analysis/hibayes/gpqa_correctness.csv"),
        help="Same attempts minus self-pairs, for the is_correct model",
    )
    parser.add_argument("--min-pairs", type=int, default=3, help="Require strictly more than this many model pairs")
    parser.add_argument("--min-samples", type=int, default=500)
    parser.add_argument("--no-transcript-features", action="store_true",
                        help="skip the 138 gpqa_tfa feature differences (much faster)")
    args = parser.parse_args(argv)
    data, accounting, eligibility = prepare(
        args.logs, args.models, args.responses, min_pairs=args.min_pairs,
        min_samples=args.min_samples, with_features=not args.no_transcript_features,
        collected_dir=args.collected if args.collected.exists() else None,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.out, index=False)
    contested = data[data["self_comparison"] == 0]
    args.out_correctness.parent.mkdir(parents=True, exist_ok=True)
    contested.to_csv(args.out_correctness, index=False)
    eligibility["included"] = (eligibility["pairs"] > args.min_pairs) & (eligibility["samples"] >= args.min_samples)
    summary = accounting.join(eligibility, how="left").fillna({"pairs": 0, "samples": 0, "included": False})
    print(
        f"Wrote {len(data):,} attempts to {args.out}\n"
        f"Wrote {len(contested):,} non-self attempts to {args.out_correctness}\n"
        f"{summary.to_string()}"
    )


if __name__ == "__main__":
    main()
