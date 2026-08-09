"""Turn answers.jsonl into one inspect eval log per condition, so the judge
runner + digest consume them unchanged (same pattern as
judge_interp/answers_to_eval_logs.py: mockllm replays the recorded answer,
scored by choice()).

One log dir per (target, label). Per item, use a correct + non-empty epoch's
answer if one exists, else any (empty/wrong scores INCORRECT and is dropped by
the judge's require_both_correct). All logs share the task name gpqa_diamond so
the judge's same-task assert passes; conditions are told apart by the config's
transcript shortnames, not the log model field.

Run (worktree root, main venv):
  ../small-judge/.venv/bin/python scripts/analyses/style_capability/answers_to_evals.py
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

# a few answers echo the prompt's "--- example solution ---" wrapper; strip it so
# the judge never sees a hint that the transcript was style-copied.
_LEAK = re.compile(r"^\s*(-{2,}\s*example solution\s*-{2,}\s*\n?)+", re.I)


def clean(ans):
    return _LEAK.sub("", ans or "")

from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ModelOutput, get_model
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice

HERE = Path(__file__).resolve().parent


def pick(rows):
    """Best answer for one (condition, item): a correct non-empty epoch if any,
    else the longest answer seen (keeps something to score/drop)."""
    good = [r for r in rows if r.get("correct") and (r.get("answer") or "").strip()]
    pool = good or rows
    return max(pool, key=lambda r: len(r.get("answer") or ""))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=HERE / "config.yaml")
    ap.add_argument("--answers", default="answers.jsonl", help="copy-condition answers file")
    ap.add_argument("--evals-name", default="evals", help="output subdir under out_dir")
    args = ap.parse_args()
    import yaml
    cfg = yaml.safe_load(args.config.read_text())
    out_root = Path(cfg["out_dir"])

    answers = [json.loads(l) for l in open(out_root / args.answers)]
    bp = out_root / "baseline_answers.jsonl"
    if bp.exists():
        answers += [json.loads(l) for l in open(bp)]
    items = {it["sample_id"]: it for it in json.load(open(out_root / "items.json"))}

    # (target,label) -> sample_id -> [rows]
    by_cond: dict = defaultdict(lambda: defaultdict(list))
    for r in answers:
        if "error" in r:
            continue
        by_cond[(r["target"], r["label"])][r["sample_id"]].append(r)

    evals_dir = Path(cfg["evals_dir"]) / args.evals_name
    for (target, label), sid_rows in sorted(by_cond.items()):
        samples, outputs = [], []
        for sid, rows in sorted(sid_rows.items()):
            it = items[sid]
            r = pick(rows)
            samples.append(Sample(id=sid, input=it["question"],
                                  choices=list(it["choices"]), target=it["target"]))
            outputs.append(ModelOutput.from_content("mockllm/model", clean(r.get("answer"))))
        task = Task(dataset=MemoryDataset(samples), solver=multiple_choice(),
                    scorer=choice(), name="gpqa_diamond")
        out = evals_dir / f"{target}__{label}"
        inspect_eval(task, model=get_model("mockllm/model", custom_outputs=outputs),
                     log_dir=str(out), display="none")
        print(f"{target}__{label:26s} {len(samples):3d} items -> {out}")
    print(f"\n{len(by_cond)} condition logs under {evals_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
