#!/usr/bin/env python3
"""Extract attempt-level GPQA judge data directly from Inspect eval logs."""

import argparse
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import zipfile_zstd  # noqa: F401 - registers ZIP_ZSTANDARD with zipfile


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
    r"Transcript A:\s*\n---\n(.*?)\n---\n\s*Transcript B:\s*\n---\n(.*?)\n---\n\s*Which of",
    re.DOTALL,
)


def model_family(name: str) -> str:
    match = re.match(r"[a-z]+", name.lower())
    return match.group() if match else name.lower()


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
            (message.get("content", "") for message in sample.get("messages", []) if message.get("role") == "user"),
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
    return (answer if answer in {"A", "B"} else None), confidence


def token_usage(sample: dict) -> dict[str, int]:
    usage = list((sample.get("model_usage") or {}).values())
    return {
        key: int(sum(model.get(key, 0) or 0 for model in usage))
        for key in ("input_tokens", "output_tokens", "reasoning_tokens", "total_tokens")
    }


def extract_attempts(log_root: Path, capabilities: dict[str, float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    seen_samples = set()
    accounting = defaultdict(Counter)
    for path in sorted(log_root.rglob("*.eval")):
        judge, pair, condition = log_identity(path, log_root)
        accounting[judge]["eval_files_found"] += 1
        major, minor = pair.split("-vs-", 1)
        if major not in capabilities or minor not in capabilities:
            raise ValueError(f"Missing capability estimate for pair {pair}")
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
                        continue
                    left_text, right_text = transcript_text(sample)
                    major_is_left = sample.get("target") == "A"
                    left_model, right_model = (major, minor) if major_is_left else (minor, major)
                    usage = token_usage(sample)
                    rows.append(
                        {
                            "chosen_left": int(answer == "A"),
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
                            "same_model_family": int(
                                left_model != right_model and model_family(left_model) == model_family(right_model)
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
    models_path: Path,
    responses_path: Path,
    *,
    min_pairs: int = 3,
    min_samples: int = 500,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data, accounting = extract_attempts(log_root, load_capabilities(models_path, responses_path))
    eligibility = data.groupby("judge").agg(pairs=("model_pair", "nunique"), samples=("judge", "size"))
    judges = eligibility[(eligibility["pairs"] > min_pairs) & (eligibility["samples"] >= min_samples)].index
    data = data[data["judge"].isin(judges)].copy()
    if data.empty:
        raise ValueError("No judges meet the pair and sample thresholds")
    for column in ("length_difference", "capability_difference"):
        scale = data[column].std()
        if not scale:
            raise ValueError(f"Cannot standardise constant feature: {column}")
        data[f"{column}_sd"] = (data[column] - data[column].mean()) / scale
    return data.sort_values(["judge", "model_pair", "item", "transcript_condition"]).reset_index(drop=True), accounting, eligibility


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", type=Path, default=Path("logs/imported_runs/gpqa_judging"))
    parser.add_argument("--models", type=Path, default=Path("logs/imported_runs_2/gpqa-pilot-rd2/models.csv"))
    parser.add_argument("--responses", type=Path, default=Path("external_data/epoch-gpqa/responses.parquet"))
    parser.add_argument("--out", type=Path, default=Path("analysis/hibayes/gpqa_glm.csv"))
    parser.add_argument("--min-pairs", type=int, default=3, help="Require strictly more than this many model pairs")
    parser.add_argument("--min-samples", type=int, default=500)
    args = parser.parse_args()
    data, accounting, eligibility = prepare(
        args.logs, args.models, args.responses, min_pairs=args.min_pairs, min_samples=args.min_samples
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.out, index=False)
    eligibility["included"] = (eligibility["pairs"] > args.min_pairs) & (eligibility["samples"] >= args.min_samples)
    summary = accounting.join(eligibility, how="left").fillna({"pairs": 0, "samples": 0, "included": False})
    print(f"Wrote {len(data):,} attempts to {args.out}\n{summary}")


if __name__ == "__main__":
    main()
