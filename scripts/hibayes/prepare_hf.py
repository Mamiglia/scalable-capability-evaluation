#!/usr/bin/env python3
"""Build the full-style judge-GLM frame from the 2026-08-17 HF GPQA snapshot.

GPQA only: MBPP directories are rejected at discovery and the frame is asserted
gpqa-only before it is written. Every row is one parsed verdict; each feature is
the left transcript minus the right, in the order the judge saw them. Style
rates and reasoning length come from the exact transcript the judge was shown
(recovered with the viewer's prompt parser), never an aggregate metric.

Feeds the existing hibayes model via config_hf.yaml; it does not fit anything.
"""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from inspect_ai.log import read_eval_log

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "analyses"))

from hf_judging_round import EXPECTED_MISSING, keep_judge, load_benchmark_scores
from hf_judging_viewer import parse_prompt

from judge.dataset import is_effort_variant, strip_answer
from mc_cost.gpqa import model_family, standardise
from style_features import feature_rates, load_lexicon

BENCH = "gpqa"
EVAL = BENCH.upper()  # capability scores are keyed by the upper-case benchmark
CONDITION = "hf_gpqa"  # one transcript condition; MBPP is never a condition here
# Style categories, in the fixed order their differences enter the model.
STYLE_CATEGORIES = (
    "personal_pronouns",
    "hedging",
    "backtracking",
    "uncertainty",
    "confidence",
    "questions",
)
# Continuous columns standardised on the final retained rows.
STANDARDISED = ("length_difference", "capability_difference") + tuple(
    f"{category}_difference" for category in STYLE_CATEGORIES
)
# The ten predictors, in the order config_hf.yaml and communicators.py expect.
PREDICTORS = (
    "length_difference_sd",
    "capability_difference_sd",
    "same_model_family",
    "self_comparison",
    "personal_pronouns_difference_sd",
    "hedging_difference_sd",
    "backtracking_difference_sd",
    "uncertainty_difference_sd",
    "confidence_difference_sd",
    "questions_difference_sd",
)
REASONING_MODES = ("none", "per_transcript", "per_model")

ROUND = ROOT / "data" / "hf-judging-2026-08-17"
LEXICON = Path(__file__).with_name("style_lexicon.yaml")
REASONING_CONFIG = ROOT / "configs" / "reasoning_filter.yaml"
OUT = ROOT / "analysis" / "hibayes" / "hf_gpqa_glm.csv"


def row_from_sample(sample, metadata, judge, capabilities, lexicon, epoch=""):
    """One parsed verdict -> a GLM row, left minus right in the shown order.

    Returns None if the judge's pick did not parse. Every continuous and style
    difference is left transcript minus right transcript.
    """
    picked, major_slot = sample.scores["judge_scorer"].answer, str(sample.target)
    if picked not in ("1", "2"):
        return None
    if major_slot not in ("1", "2"):
        raise ValueError(f"major slot must be 1 or 2, got {major_slot!r}")

    parsed = parse_prompt(str(sample.input))
    major = Path(metadata["major_path"]).parts[-3]
    minor = Path(metadata["minor_path"]).parts[-3]
    left, right = (major, minor) if major_slot == "1" else (minor, major)
    left_text, right_text = parsed["transcript_1"], parsed["transcript_2"]
    base_item = str(sample.id).rsplit("_", 1)[0]

    left_rates = feature_rates(left_text, lexicon)
    right_rates = feature_rates(right_text, lexicon)
    left_capability = capabilities[(EVAL, left)]
    right_capability = capabilities[(EVAL, right)]
    row = {
        "chosen_left": int(picked == "1"),
        "judge": judge,
        "item": f"{BENCH}:{base_item}",
        "transcript_condition": CONDITION,
        "bench": BENCH,
        "epoch": epoch,
        "model_pair": "|".join(sorted((left, right))),
        "left_model": left,
        "right_model": right,
        "left_chars": len(left_text),
        "right_chars": len(right_text),
        "left_reasoning_chars": len(strip_answer(left_text)),
        "right_reasoning_chars": len(strip_answer(right_text)),
        "left_capability": left_capability,
        "right_capability": right_capability,
        "length_difference": np.log1p(len(left_text)) - np.log1p(len(right_text)),
        "capability_difference": left_capability - right_capability,
        "same_model_family": int(left != right and model_family(left) == model_family(right)),
        "self_comparison": int(left == right),
    }
    for category in STYLE_CATEGORIES:
        row[f"{category}_difference"] = left_rates[category] - right_rates[category]
    return row


def reasoning_filter(
    frame: pd.DataFrame, mode: str, min_chars: int, effort_suffixes: Sequence[str]
) -> tuple[pd.Series, list[str], dict[str, int]]:
    """Row mask keeping comparisons whose two transcripts both show real work.

    Mirrors prepare_from_verdicts.keep_reasoning, but reads each row's own shown
    transcript rather than a per-(model, item) rates table, because HF keeps
    several response epochs per item. per_model drops an answer model whose
    median shown transcript is under the floor, keeping its models whole; effort
    variants are dropped in every mode.
    """
    if mode not in REASONING_MODES:
        raise ValueError(f"Unknown reasoning filter {mode}, have {list(REASONING_MODES)}")
    models = sorted(set(frame["left_model"]) | set(frame["right_model"]))
    dropped = {model for model in models if is_effort_variant(model, effort_suffixes)}
    counts = {"dropped_effort_variants": len(dropped)}
    if mode == "per_model":
        sides = pd.concat(
            [
                frame[["left_model", "left_reasoning_chars"]].set_axis(["model", "rc"], axis=1),
                frame[["right_model", "right_reasoning_chars"]].set_axis(["model", "rc"], axis=1),
            ]
        )
        median = sides.groupby("model")["rc"].median()
        dropped |= {model for model in models if median[model] < min_chars}
    counts["dropped_models"] = len(dropped)
    keep = ~(frame["left_model"].isin(dropped) | frame["right_model"].isin(dropped))
    if mode == "per_transcript":
        thin = (frame["left_reasoning_chars"] < min_chars) | (frame["right_reasoning_chars"] < min_chars)
        keep &= ~thin
        counts["dropped_rows_thin_transcript"] = int(thin.sum())
    counts["dropped_rows_reasoning_filter"] = int((~keep).sum())
    return keep, sorted(dropped), counts


def discover_gpqa_dirs(raw: Path) -> list[Path]:
    """logs.json paths for GPQA judging directories; MBPP is rejected here."""
    return [
        path
        for path in sorted(raw.glob("*/logs.json"))
        if f"-{BENCH}-" in path.parent.name and "-mbpp-" not in path.parent.name
    ]


def build_frame(
    round_dir=ROUND,
    lexicon_path=LEXICON,
    mode="per_model",
    min_reasoning_chars=100,
    effort_suffixes=("low", "high"),
):
    """HF GPQA verdicts -> the standardised full-style GLM frame plus audit counts."""
    capabilities = load_benchmark_scores(round_dir / "benchmark-scores")
    lexicon = load_lexicon(lexicon_path)
    rows, seen, missing = [], set(), set()
    counts = {
        "excluded_judges": 0,
        "duplicate_logs": 0,
        "logs_missing_capability": 0,
        "unparsed_verdicts": 0,
        "truncated_prompts": 0,
    }

    for summary_path in discover_gpqa_dirs(round_dir / "raw"):
        dirname = summary_path.parent.name
        judge = dirname.removeprefix("pilot-").split(f"-{BENCH}-", 1)[0]
        if not keep_judge(judge):
            counts["excluded_judges"] += 1
            continue
        for filename, summary in json.loads(summary_path.read_text()).items():
            if summary.get("status") != "success" or summary.get("invalidated"):
                continue
            key = (summary_path.parent, summary["eval"]["task_id"])
            if key in seen:
                counts["duplicate_logs"] += 1
                continue
            seen.add(key)
            log = read_eval_log(str(summary_path.parent / filename))
            metadata = log.eval.metadata or {}
            major_path, minor_path = Path(metadata["major_path"]), Path(metadata["minor_path"])
            models = {major_path.parts[-3], minor_path.parts[-3]}
            absent = {(EVAL, model) for model in models if (EVAL, model) not in capabilities}
            if absent:
                missing |= absent
                counts["logs_missing_capability"] += 1
                continue
            epoch = f"{major_path.parts[-2]}-{minor_path.parts[-2]}"
            for sample in log.samples or []:
                try:
                    row = row_from_sample(sample, metadata, judge, capabilities, lexicon, epoch)
                except ValueError as error:
                    if "unrecognised judge prompt" not in str(error):
                        raise
                    counts["truncated_prompts"] += 1
                    continue
                if row is None:
                    counts["unparsed_verdicts"] += 1
                else:
                    rows.append(row)

    unexpected = missing - EXPECTED_MISSING
    if unexpected:
        raise ValueError(f"Missing benchmark scores for unexpected models: {sorted(unexpected)}")
    if not rows:
        raise ValueError(f"No usable GPQA verdicts in {round_dir}")

    frame = pd.DataFrame(rows)
    benches = sorted(frame["bench"].unique())
    if benches != [BENCH]:
        raise ValueError(f"frame is not GPQA-only: {benches}")

    # Filter before standardising: the scales must come from the fitted rows.
    keep, dropped, filter_counts = reasoning_filter(frame, mode, min_reasoning_chars, effort_suffixes)
    counts |= filter_counts
    counts["dropped_model_names"] = dropped
    frame = frame[keep].reset_index(drop=True)
    if frame.empty:
        raise ValueError("The reasoning filter removed every comparison")

    standardise(frame, STANDARDISED)
    if not np.isfinite(frame[list(PREDICTORS)].to_numpy(float)).all():
        raise ValueError(f"Standardised predictors are not finite: {list(PREDICTORS)}")

    counts["model_rows"] = len(frame)
    frame = frame.sort_values(["judge", "model_pair", "item", "epoch"]).reset_index(drop=True)
    return frame, counts, missing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round", type=Path, default=ROUND)
    parser.add_argument("--lexicon", type=Path, default=LEXICON)
    parser.add_argument("--reasoning-config", type=Path, default=REASONING_CONFIG)
    parser.add_argument("--reasoning-filter", choices=REASONING_MODES)
    parser.add_argument("--min-reasoning-chars", type=int)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"{args.out} already exists; refusing to overwrite")

    config = yaml.safe_load(args.reasoning_config.read_text())
    mode = args.reasoning_filter or config["mode"]
    min_chars = args.min_reasoning_chars if args.min_reasoning_chars is not None else config["min_reasoning_chars"]
    suffixes = tuple(config["effort_suffixes"]) if config["exclude_effort_variants"] else ()
    print(f"Reasoning filter: {mode}, floor {min_chars} chars, effort suffixes {list(suffixes) or 'none'}")

    frame, counts, missing = build_frame(args.round, args.lexicon, mode, min_chars, suffixes)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)

    print(f"Wrote {len(frame):,} rows to {args.out}")
    names = counts.pop("dropped_model_names", [])
    for name, value in counts.items():
        print(f"  {name}: {value:,}")
    if names:
        print(f"  dropped answer models: {', '.join(names)}")
    if missing:
        print("  skipped (no capability score): " + ", ".join(f"{b}/{m}" for b, m in sorted(missing)))
    print(
        f"  gpqa items: {frame['item'].nunique()}, judges: {frame['judge'].nunique()}, "
        f"candidate models: {len(set(frame['left_model']) | set(frame['right_model']))}, "
        f"pairs: {frame['model_pair'].nunique()}, epochs: {frame['epoch'].nunique()}"
    )
    print(f"  chosen_left rate: {frame['chosen_left'].mean():.4f}")


if __name__ == "__main__":
    main()
