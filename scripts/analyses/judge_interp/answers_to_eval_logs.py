"""Turn steered answers into eval logs, one per steering condition.

The judging pipeline reads inspect eval logs, so each condition becomes
its own log: same GPQA items, same option order, the recorded answer as
the model's completion, scored against the real target. The logs then
go through the normal judge runner and its digest verification, with no
special-casing for synthetic text.

The model name encodes the condition (qwen3-32b-steer_v_pref_judge_p1.0),
so the judge config lists comparisons by name like any other pair.

Usage:
  uv run python scripts/analyses/judge_interp/answers_to_eval_logs.py \
      --answers judge_results/judge_interp/answers_260806 \
      --out-dir data/steered_answers --seed-index 0
"""

import argparse
import glob
import json
import re
from pathlib import Path

import pandas as pd
from inspect_ai import Task, eval
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ModelOutput, get_model
from inspect_ai.scorer import CORRECT, INCORRECT, Score, Target, accuracy, scorer
from inspect_ai.solver import generate


@scorer(metrics=[accuracy()])
def answer_letter():
    """Correct when the final ANSWER line matches the target.

    The judging pipeline reads correctness through util.extract_score,
    which wants a sample score of "C", so the replayed answers have to
    carry a real score rather than only the text.
    """
    async def score(state, target: Target):
        text = state.output.completion
        m = re.findall(r"ANSWER:\s*\**\s*([A-D])", text)
        given = m[-1] if m else None
        return Score(value=CORRECT if given == target.text else INCORRECT,
                     answer=given or "")
    return score

ROOT = Path(__file__).resolve().parents[3]


def condition_name(axis, alpha):
    if axis == "none":
        return "unsteered"
    sign = "p" if alpha >= 0 else "m"
    return f"{axis}_{sign}{abs(alpha):g}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "config.yaml")
    ap.add_argument("--answers", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seed-index", type=int, default=0,
                    help="which sampled generation per (item, condition) to judge")
    ap.add_argument("--items", type=Path,
                    default=ROOT / "judge_results/judge_interp/staging_titration_260806/items.parquet")
    args = ap.parse_args()

    rows = []
    for f in glob.glob(str(args.answers / "answers_shard*.jsonl")):
        rows += [json.loads(l) for l in open(f)]
    rows = [r for r in rows if r["seed"] == args.seed_index and r["parsed"]]
    if not rows:
        raise SystemExit(f"no parsed answers with seed {args.seed_index}")
    items = pd.read_parquet(args.items).set_index("sample_id")

    by_cond = {}
    for r in rows:
        by_cond.setdefault(condition_name(r["axis"], r["alpha"]), []).append(r)
    print(f"{len(rows)} answers, {len(by_cond)} conditions, "
          f"{len(set(r['sample_id'] for r in rows))} items")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for cond, recs in sorted(by_cond.items()):
        samples, outputs = [], []
        for r in sorted(recs, key=lambda r: r["sample_id"]):
            item = items.loc[r["sample_id"]]
            choices = [c.split(") ", 1)[1] if ") " in c else c
                       for c in str(item["options"]).splitlines() if c.strip()]
            samples.append(Sample(id=r["sample_id"], input=item["prompt"],
                                  choices=choices, target=r["target"]))
            outputs.append(ModelOutput.from_content("mockllm/model", r["answer"]))
        # One task name for every condition: the judge runner asserts the two
        # logs it compares come from the same task. Conditions are told apart
        # by their log directory, which the judge config maps by name.
        task = Task(dataset=MemoryDataset(samples), solver=generate(),
                    scorer=answer_letter(), name="gpqa_diamond_steered")
        eval(task, model=get_model("mockllm/model", custom_outputs=outputs),
             log_dir=str(args.out_dir / cond), display="none")
        print(f"{cond:28s} {len(samples):3d} answers -> {args.out_dir / cond}")


if __name__ == "__main__":
    main()
