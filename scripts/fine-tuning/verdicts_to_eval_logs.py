"""Replay judge-ft results as inspect eval logs under logs/judge_ft.

Two kinds: verdict runs (one log per judge variant x pair set; the
completion is the verdict token readout, "ANSWER: 1|2") and frontier
answer runs (one log per model x eval, real completions). The base/ft
GPQA answering runs are not converted: their raw completions were not
stored, and synthetic transcripts would masquerade as real ones.
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from inspect_ai import Task, eval as inspect_eval
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ModelOutput, get_model
from inspect_ai.scorer import CORRECT, INCORRECT, Score, Target, accuracy, scorer
from inspect_ai.solver import generate

import prompts

R = Path("judge_results/judge_ft")


@scorer(metrics=[accuracy()])
def verdict_scorer():
    async def score(state, target: Target):
        m = re.findall(r"ANSWER:\s*([12])", state.output.completion)
        return Score(value=CORRECT if m and m[-1] == target.text else INCORRECT,
                     answer=m[-1] if m else "")
    return score


@scorer(metrics=[accuracy()])
def letter_scorer():
    async def score(state, target: Target):
        m = re.findall(r"ANSWER:\s*\**\s*([A-Z])", state.output.completion)
        return Score(value=CORRECT if m and m[-1] == target.text else INCORRECT,
                     answer=m[-1] if m else "")
    return score


def replay(samples, outputs, task_name, log_dir, scorer_fn):
    task = Task(dataset=MemoryDataset(samples), solver=generate(),
                scorer=scorer_fn, name=task_name)
    inspect_eval(task, model=get_model("mockllm/model", custom_outputs=outputs),
                 log_dir=str(log_dir), display="none")


def convert_verdicts(jsonl, pairs_parquet, judge_name, set_name, out_root):
    v = pd.DataFrame([json.loads(l) for l in open(jsonl)])
    p = pd.read_parquet(pairs_parquet).set_index(["pair_uid", "order"])
    samples, outputs = [], []
    for r in v.itertuples():
        row = p.loc[(r.pair_uid, r.order)]
        samples.append(Sample(
            id=f"{r.pair_uid}#o{r.order}",
            input=prompts.user_text(row.question, row.options_1, row.options_2,
                                    row.text_1, row.text_2),
            target=str(r.label),
            metadata=dict(model_1=r.model_1, model_2=r.model_2,
                          logit_1=r.logit_1, logit_2=r.logit_2,
                          readout="verdict-token logits, not sampled text")))
        picked = "1" if r.logit_1 > r.logit_2 else "2"
        outputs.append(ModelOutput.from_content("mockllm/model", f"ANSWER: {picked}"))
    replay(samples, outputs, f"judge_ft_{set_name}", out_root / judge_name / set_name,
           verdict_scorer())
    print(f"{judge_name}/{set_name}: {len(samples)} verdicts")


def convert_frontier_answers(jsonl, out_root):
    df = pd.DataFrame([json.loads(l) for l in open(jsonl)])
    for (model, ev), g in df.groupby(["model", "eval"]):
        samples, outputs = [], []
        for r in g.itertuples():
            samples.append(Sample(id=f"{r.item}::e{r.epoch}", input="",
                                  target=r.target,
                                  metadata=dict(subtask=r.subtask, provider=r.provider,
                                                gen_id=r.gen_id)))
            outputs.append(ModelOutput.from_content("mockllm/model", r.text))
        replay(samples, outputs, f"frontier_{ev}", out_root / "frontier-answers" / model / ev,
               letter_scorer())
        print(f"frontier-answers/{model}/{ev}: {len(samples)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-root", type=Path, default=Path("logs/judge_ft"))
    args = ap.parse_args()
    sets = [
        ("out/verdicts_base_xl.jsonl", R / "pairs.parquet", "qwen3-8b-base", "eval_xl"),
        ("out/verdicts_ft_xl.jsonl", R / "pairs.parquet", "qwen3-8b-ft", "eval_xl"),
        ("out/selfpref_base.jsonl", R / "pairs_selfpref.parquet", "qwen3-8b-base", "selfpref"),
        ("out/selfpref_ft.jsonl", R / "pairs_selfpref.parquet", "qwen3-8b-ft", "selfpref"),
        ("out/self_base.jsonl", R / "pairs_self.parquet", "qwen3-8b-base", "selfpairs"),
        ("out/self_ft.jsonl", R / "pairs_self.parquet", "qwen3-8b-ft", "selfpairs"),
        ("out/verdicts_frontier_base.jsonl", R / "pairs_frontier.parquet", "qwen3-8b-base", "frontier_gen"),
        ("out/verdicts_frontier_ft.jsonl", R / "pairs_frontier.parquet", "qwen3-8b-ft", "frontier_gen"),
        ("out/verdicts_frontier_epoch_base.jsonl", R / "pairs_frontier_epoch.parquet", "qwen3-8b-base", "frontier_epoch"),
        ("out/verdicts_frontier_epoch_ft.jsonl", R / "pairs_frontier_epoch.parquet", "qwen3-8b-ft", "frontier_epoch"),
        ("out/verdicts_fvs_base.jsonl", R / "pairs_frontier_vs_strong.parquet", "qwen3-8b-base", "frontier_vs_strong"),
        ("out/verdicts_fvs_ft.jsonl", R / "pairs_frontier_vs_strong.parquet", "qwen3-8b-ft", "frontier_vs_strong"),
    ]
    for out_sub, judge_stem in [("out_qwen06", "qwen3-0.6b"), ("out_llama3b", "llama-3.2-3b")]:
        for tag in ["base", "ft"]:
            sets += [(f"{out_sub}/verdicts_{tag}.jsonl", R / "pairs.parquet",
                      f"{judge_stem}-{tag}", "eval"),
                     (f"{out_sub}/selfpref_{tag}.jsonl", R / "pairs_selfpref.parquet",
                      f"{judge_stem}-{tag}", "selfpref")]
    for jsonl, pairs, judge, name in sets:
        if not (args.out_root / judge / name).exists():
            convert_verdicts(R / jsonl, pairs, judge, name, args.out_root)
    if not (args.out_root / "frontier-answers").exists():
        convert_frontier_answers(R / "frontier_answers.jsonl", args.out_root)


if __name__ == "__main__":
    main()
