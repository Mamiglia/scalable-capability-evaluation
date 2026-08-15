#!/usr/bin/env python3
"""Style-lexicon rates for every transcript a judging condition could show.

Re-extracts the same transcript the judge saw - first correct epoch, final text
only - from the eval logs the condition's config names, and writes one row of
per-1,000-word category rates per (model, item). n_chars travels with each row
so the frame builder can check the rates describe the text the judge was shown,
and reasoning_chars so it can filter on how much work the transcript holds.

Split from prepare_from_verdicts.py because this reads gigabytes of eval logs
and that reads a CSV.
"""

import argparse
import csv
from pathlib import Path

try:
    import zipfile_zstd  # noqa: F401  epoch .eval zips use zstd; py<3.14 zipfile needs this shim
except ImportError:
    pass

import yaml
from inspect_ai.log import read_eval_log

from judge.dataset import strip_answer
from judge.util import extract_transcript, first_correct_by_id
from style_features import feature_rates, load_lexicon

ROOT = Path(__file__).resolve().parents[2]
CONDITION_CONFIG = ROOT / "analyses" / "judge-variation" / "configs" / "gpqa_a_baseline.yaml"
LEXICON = Path(__file__).with_name("style_lexicon.yaml")
OUT = ROOT / "analysis" / "hibayes" / "style_rates.csv"


def rates_for_log(log_path: Path, model: str, lexicon: dict) -> list[dict]:
    """One row per item this model answered correctly, in log order."""
    log = read_eval_log(str(log_path))
    rows = []
    for item_id, samples in first_correct_by_id(log).items():
        transcript = extract_transcript(samples[0], include_reasoning=False)
        rows.append(
            {
                "model": model,
                "item": item_id,
                "n_chars": len(transcript),
                # what is left once the answer line is gone: the work a judge can assess
                "reasoning_chars": len(strip_answer(transcript)),
            }
            | feature_rates(transcript, lexicon)
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition-config", type=Path, default=CONDITION_CONFIG)
    parser.add_argument("--lexicon", type=Path, default=LEXICON)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    config = yaml.safe_load(args.condition_config.read_text())
    if config.get("include_reasoning"):
        raise ValueError(f"{args.condition_config} shows reasoning; these rates are final-text-only")
    lexicon = load_lexicon(args.lexicon)

    rows = []
    for model, relative in sorted(config["transcripts"].items()):
        rows.extend(rates_for_log(ROOT / relative, model, lexicon))
        print(f"  {model}: {len(rows):,} rows so far")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows):,} rows to {args.out}")


if __name__ == "__main__":
    main()
