#!/usr/bin/env python3
"""Build the HiBayes judge-GLM frame for the 2026-08-17 HF snapshot."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from inspect_ai.log import read_eval_log

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "analyses"))

from hf_judging_round import EXPECTED_MISSING, keep_judge, load_benchmark_scores
from hf_judging_viewer import parse_prompt
from prepare_gpqa import model_family, standardise

ROUND = ROOT / "data" / "hf-judging-2026-08-17"
OUT = ROOT / "analysis" / "hibayes" / "hf_2026_08_17.csv"


def row_from_sample(sample, metadata, judge, bench, capabilities):
    """Convert one parsed verdict, preserving the prompt's left/right order."""
    picked, major_slot = sample.scores["judge_scorer"].answer, str(sample.target)
    if picked not in ("1", "2"):
        return None
    if major_slot not in ("1", "2"):
        raise ValueError(f"major slot must be 1 or 2, got {major_slot!r}")

    parsed = parse_prompt(str(sample.input))
    major = Path(metadata["major_path"]).parts[-3]
    minor = Path(metadata["minor_path"]).parts[-3]
    left, right = (major, minor) if major_slot == "1" else (minor, major)
    eval_name = bench.upper()
    base_item, order = str(sample.id).rsplit("_", 1)
    if order not in ("1", "2"):
        raise ValueError(f"sample id lacks an order suffix: {sample.id!r}")
    left_chars, right_chars = len(parsed["transcript_1"]), len(parsed["transcript_2"])
    left_capability, right_capability = capabilities[(eval_name, left)], capabilities[(eval_name, right)]
    return {
        "chosen_left": int(picked == "1"),
        "judge": judge,
        "item": f"{bench}:{base_item}",
        "transcript_condition": bench,
        "bench": bench,
        "model_pair": "|".join(sorted((left, right))),
        "left_model": left,
        "right_model": right,
        "left_chars": left_chars,
        "right_chars": right_chars,
        "left_capability": left_capability,
        "right_capability": right_capability,
        "length_difference": np.log1p(left_chars) - np.log1p(right_chars),
        "capability_difference": left_capability - right_capability,
        "same_model_family": int(left != right and model_family(left) == model_family(right)),
        "self_comparison": int(left == right),
    }


def build_frame(round_dir=ROUND):
    capabilities = load_benchmark_scores(round_dir / "benchmark-scores")
    rows, seen, missing = [], set(), set()
    counts = {"duplicate_logs": 0, "unparsed_verdicts": 0, "truncated_prompts": 0}

    for summary_path in sorted((round_dir / "raw").glob("*/logs.json")):
        dirname = summary_path.parent.name
        bench = next((name for name in ("gpqa", "mbpp") if f"-{name}-" in dirname), None)
        if bench is None:
            continue
        judge = dirname.removeprefix("pilot-").split(f"-{bench}-", 1)[0]
        if not keep_judge(judge):
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
            models = {Path(metadata[name]).parts[-3] for name in ("major_path", "minor_path")}
            absent = {(bench.upper(), model) for model in models if (bench.upper(), model) not in capabilities}
            if absent:
                missing |= absent
                continue
            for sample in log.samples or []:
                try:
                    row = row_from_sample(sample, metadata, judge, bench, capabilities)
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
        raise ValueError(f"No usable verdicts in {round_dir}")
    frame = pd.DataFrame(rows)
    standardise(frame, ("length_difference", "capability_difference"))
    counts.update(rows=len(frame), judges=frame["judge"].nunique(), items=frame["item"].nunique())
    return frame.sort_values(["judge", "bench", "model_pair", "item"]).reset_index(drop=True), counts, missing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round", type=Path, default=ROUND)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"{args.out} already exists; refusing to overwrite")
    frame, counts, missing = build_frame(args.round)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(f"Wrote {len(frame):,} rows to {args.out}")
    print("  " + ", ".join(f"{key}: {value:,}" for key, value in counts.items()))
    if missing:
        print("  skipped missing capabilities: " + ", ".join(f"{bench}/{model}" for bench, model in sorted(missing)))


if __name__ == "__main__":
    main()
