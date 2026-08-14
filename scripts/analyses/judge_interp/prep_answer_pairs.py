"""Pair steered answers against unsteered ones for judging.

For each item and steering condition, pairs the steered answer with the
unsteered answer to the same item, keeping only pairs whose outcomes
match (both correct or both wrong), so no verdict can be driven by one
answer being right and the other wrong. Broken generations (repetition
loops, reasoning blocks) are dropped.

Emits the pair schema pod_judge_steer.py reads, so the same
forward-pass verdict read-out applies: text_id_1/2 carry the condition
labels, and the judge sees two answers to one question.

Usage:
  uv run python scripts/analyses/judge_interp/prep_answer_pairs.py \
      --answers judge_results/judge_interp/answers_260806 --staging <dir>
"""

import argparse
import glob
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]

OPTIONS_BLOCK = "Multiple choice options as shown to Model {n}:\n{options}\n\n"


def condition_name(axis, alpha):
    if axis == "none" or alpha == 0:
        return "unsteered" if axis == "none" else f"{axis}_a0"
    return f"{axis}_{'p' if alpha > 0 else 'm'}{abs(alpha):g}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--answers", type=Path, required=True)
    ap.add_argument("--staging", type=Path, required=True)
    ap.add_argument("--items", type=Path,
                    default=ROOT / "judge_results/judge_interp/staging_titration_260806/items.parquet")
    ap.add_argument("--out-name", default="answer_pairs.parquet")
    args = ap.parse_args()

    sys_path = ROOT / "scripts/analyses/judge_interp"
    import sys
    sys.path.insert(0, str(sys_path))
    from answer_effects import degeneration

    rows = []
    for f in glob.glob(str(args.answers / "*.jsonl")):
        rows += [json.loads(l) for l in open(f)]
    def usable(r):
        d = degeneration(r["answer"])
        return (r["parsed"] and not d["has_think"]
                and d["repeat_frac"] <= 0.5 and d["max_line_run"] < 5)
    rows = [r for r in rows if usable(r)]
    items = pd.read_parquet(args.items).set_index("sample_id")

    base = {}
    for r in rows:
        if r["axis"] == "none":
            base.setdefault(r["sample_id"], r)

    pairs = []
    for r in rows:
        if r["axis"] == "none":
            continue
        b = base.get(r["sample_id"])
        if b is None or b["correct"] != r["correct"]:
            continue                      # equal-outcome pairs only
        item = items.loc[r["sample_id"]]
        options = OPTIONS_BLOCK.format(n=1, options=item["options"])
        pairs.append(dict(
            item=r["sample_id"], condition=condition_name(r["axis"], r["alpha"]),
            axis=r["axis"], alpha=r["alpha"], both_correct=bool(r["correct"]),
            text_id_1=condition_name(r["axis"], r["alpha"]), text_id_2="unsteered",
            question=item["prompt"], options_1=options,
            options_2=OPTIONS_BLOCK.format(n=2, options=item["options"]),
            text_1=r["answer"], text_2=b["answer"]))
    df = pd.DataFrame(pairs)
    if df.empty:
        raise SystemExit("no equal-outcome pairs")
    args.staging.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.staging / args.out_name, index=False)
    print(f"{len(df)} pairs over {df['item'].nunique()} items, "
          f"{df['both_correct'].mean():.0%} both-correct")
    print(df.groupby(["axis", "alpha"]).size().to_string())
    print(f"-> {args.staging / args.out_name}")


if __name__ == "__main__":
    main()
