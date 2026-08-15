#!/usr/bin/env python3
"""Extract attempt-level GPQA judge data directly from Inspect eval logs."""

import argparse
from pathlib import Path

from mc_cost.gpqa import (
    EXTERNAL_CAPABILITY_MODELS,
    TRANSCRIPTS,
    extract_attempts,
    judge_answer,
    load_capabilities,
    log_identity,
    model_family,
    prepare,
    standardise,
    token_usage,
    transcript_text,
)


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
        args.logs, args.models, args.responses, min_pairs=args.min_pairs, min_samples=args.min_samples,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.out, index=False)
    eligibility["included"] = (
        (eligibility["pairs"] > args.min_pairs) & (eligibility["samples"] >= args.min_samples)
    )
    summary = accounting.join(eligibility, how="left").fillna({
        "pairs": 0, "samples": 0, "included": False,
    })
    print(f"Wrote {len(data):,} attempts to {args.out}\n{summary.to_string()}")


if __name__ == "__main__":
    main()
