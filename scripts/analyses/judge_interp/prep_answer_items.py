"""Benchmark items for the steered-answering round.

Takes the GPQA prompts exactly as the eval posed them (same wording,
same option order, same answer-format instruction), so answers
generated under steering are comparable to the transcripts the
directions were fit on. Keeps the ground-truth letter, so steering can
be scored against correctness as well as against judges.

Writes items.parquet: sample_id, prompt, target, options block.

Usage:
  uv run python scripts/analyses/judge_interp/prep_answer_items.py \
      --staging <dir> --n-items 60
"""

import argparse
import glob
import re
from pathlib import Path

import pandas as pd
import yaml
from inspect_ai.log import read_eval_log

ROOT = Path(__file__).resolve().parents[3]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "config.yaml")
    ap.add_argument("--staging", type=Path, required=True)
    ap.add_argument("--n-items", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())

    log = read_eval_log(sorted(glob.glob(str(ROOT / cfg["gpqa_glob"])))[0])
    rows = []
    for s in log.samples:
        user = next((m for m in s.messages if m.role == "user"), None)
        if user is None:
            continue
        prompt = str(user.content)
        # The options block, for the judge prompt's "as the model saw them" part.
        m = re.search(r"\n\n(A\).*)$", prompt, re.DOTALL)
        rows.append(dict(sample_id=str(s.id), prompt=prompt,
                         target=str(s.target),
                         options=m.group(1).strip() if m else ""))
    items = pd.DataFrame(rows).drop_duplicates("sample_id")
    items = items.sample(min(args.n_items, len(items)),
                         random_state=args.seed).reset_index(drop=True)
    args.staging.mkdir(parents=True, exist_ok=True)
    items.to_parquet(args.staging / "items.parquet", index=False)
    print(f"{len(items)} items (of {len(rows)}), targets "
          f"{items['target'].value_counts().to_dict()}")
    print(f"-> {args.staging}/items.parquet")


if __name__ == "__main__":
    main()
