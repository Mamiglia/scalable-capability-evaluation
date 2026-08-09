"""Select GPQA items every author answered correctly, and freeze a canonical
question/options/target per item.

Reads the pilot eval logs in config.yaml, takes each author's items with at
least one correct epoch, intersects across all authors, samples n_items, and
writes items.parquet using reference_author's option order as canonical.
Intersecting on the originals means every re-generated condition starts from
items the tiers already get right, so both-correct comparisons exist.

Run (from the worktree root, main-checkout venv):
  ../small-judge/.venv/bin/python scripts/analyses/style_capability/select_items.py
"""

import argparse
import random
from pathlib import Path

import pandas as pd
import yaml
from inspect_ai.log import read_eval_log

from judge.util import extract_score

HERE = Path(__file__).resolve().parent


def correct_ids_and_refs(log_path: str):
    """Return (ids with >=1 correct epoch, {id: (question, options, target)})."""
    log = read_eval_log(log_path)
    correct: set = set()
    refs: dict = {}
    for s in log.samples or []:
        if not extract_score(s):
            continue
        correct.add(s.id)
        if s.id not in refs:
            choices = [str(c).strip() for c in (getattr(s, "choices", []) or [])]
            options = "\n".join(
                f"{chr(ord('A') + i)}) {c}" for i, c in enumerate(choices)
            )
            refs[s.id] = (s.input, choices, options, str(s.target))
    return correct, refs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=HERE / "config.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())

    authors = {**cfg["authors"]["strong"], **cfg["authors"]["weak"]}
    sel = cfg["select"]

    per_author: dict = {}
    refs_by_author: dict = {}
    for name, path in authors.items():
        ids, refs = correct_ids_and_refs(path)
        per_author[name] = ids
        refs_by_author[name] = refs
        print(f"{name:22s} correct on {len(ids):3d} items")

    all_correct = set.intersection(*per_author.values())
    strong = set.intersection(*(per_author[n] for n in cfg["authors"]["strong"]))
    weak = set.intersection(*(per_author[n] for n in cfg["authors"]["weak"]))
    print(f"\nstrong tier both-correct: {len(strong)}")
    print(f"weak tier all-correct:    {len(weak)}")
    print(f"all {len(authors)} authors correct: {len(all_correct)}")

    ref = refs_by_author[sel["reference_author"]]
    usable = sorted(i for i in all_correct if i in ref)
    rng = random.Random(sel["seed"])
    n = min(sel["n_items"], len(usable))
    chosen = sorted(rng.sample(usable, n))
    print(f"\nselecting {n} of {len(usable)} usable items (target {sel['n_items']})")

    rows = [
        {"sample_id": i, "question": ref[i][0], "choices": ref[i][1],
         "options": ref[i][2], "target": ref[i][3]}
        for i in chosen
    ]
    out = Path(cfg["out_dir"]) / "items.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out, index=False)
    print(f"wrote {len(rows)} items -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
