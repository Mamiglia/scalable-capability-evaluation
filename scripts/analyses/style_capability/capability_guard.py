"""Capability guard for the instruction variant: does telling a model to sound
`uncertain` actually lower its accuracy, or does it only sound weaker while
answering just as well? Per-attempt GPQA accuracy per (model, label), baseline
vs confident vs uncertain, on the same items. Run from the main checkout cwd.

  ../small-judge/.venv/bin/python .../capability_guard.py
"""

import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
OUT = Path(yaml.safe_load((HERE / "config.yaml").read_text())["out_dir"])
LABELS = ["baseline", "confident", "uncertain"]


def load(*files):
    rows = []
    for f in files:
        rows += [json.loads(l) for l in open(OUT / f)]
    return [r for r in rows if "error" not in r]


def main():
    rows = load("baseline_answers.jsonl", "instruct_answers.jsonl")
    # (model, label) -> [correct?]
    acc = defaultdict(list)
    for r in rows:
        if r["label"] in LABELS:
            acc[(r["target"], r["label"])].append(1.0 if r.get("correct") else 0.0)

    models = sorted({m for m, _ in acc})
    assert models, "no rows loaded"
    w = max(len(m) for m in models)
    head = f"{'model':<{w}}  " + "  ".join(f"{l:>10}" for l in LABELS) + "   d(unc-base)"
    print(head)
    print("-" * len(head))
    for m in models:
        cells, base, unc = "", None, None
        for l in LABELS:
            v = acc.get((m, l))
            if v:
                mean = st.mean(v)
                cells += f"  {mean:>7.3f}(n{len(v):>2})"
                if l == "baseline":
                    base = mean
                if l == "uncertain":
                    unc = mean
            else:
                cells += f"  {'-':>10}"
        d = f"{unc - base:+.3f}" if (base is not None and unc is not None) else "?"
        print(f"{m:<{w}}{cells}   {d:>10}")


if __name__ == "__main__":
    main()
