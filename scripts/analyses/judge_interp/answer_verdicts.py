"""Do steered answers sway a reader?

Each pair is one steered answer against the unsteered answer to the same
item, both with the same outcome, judged unsteered in both slot orders.
The measure is the share of verdicts going to the steered answer, with
slot order averaged out within a pair, so the slot bias cancels rather
than being fit. 50% means the steering left the reader indifferent.

Writes out_dir/answer_verdicts.csv.

Usage:
  uv run python scripts/analyses/judge_interp/answer_verdicts.py \
      --verdicts judge_results/judge_interp/answer_verdicts \
      --pairs judge_results/judge_interp/staging_titration_260806/answer_pairs.parquet
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "config.yaml")
    ap.add_argument("--verdicts", type=Path, required=True)
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--boots", type=int, default=5000)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    out = ROOT / cfg["out_dir"]

    rows = []
    for f in glob.glob(str(args.verdicts / "*.jsonl")):
        rows += [json.loads(l) for l in open(f)]
    pairs = pd.read_parquet(args.pairs)
    meta = {(r.item, r.text_id_1): r for r in pairs.itertuples()}
    print(f"{len(rows)} verdicts over {len({(r['item'], r['text_id_1']) for r in rows})} pairs")

    sc = [r for r in rows if r.get("full_answer")]
    if sc:
        agree = sum((r["logit_1"] > r["logit_2"]) == (r["full_answer"] == "1") for r in sc)
        print(f"read-out check: matches generated verdict on {agree}/{len(sc)}")

    # order 0 shows the steered answer in slot 1, order 1 swaps it.
    per_pair = {}
    for r in rows:
        key = (r["item"], r["text_id_1"])
        steered_wins = ((r["logit_1"] > r["logit_2"]) if r["order"] == 0
                        else (r["logit_2"] > r["logit_1"]))
        per_pair.setdefault(key, []).append(steered_wins)

    recs = []
    for key, wins in per_pair.items():
        m = meta.get(key)
        if m is None:
            continue
        recs.append(dict(axis=m.axis, alpha=m.alpha, both_correct=m.both_correct,
                         share=float(np.mean(wins)), n_orders=len(wins)))
    df = pd.DataFrame(recs)

    rng = np.random.default_rng(0)
    print(f"\n{'axis':14s} {'alpha':>6} {'pairs':>6} {'chosen over unsteered':>24}")
    table = []
    for (axis, alpha), g in df.groupby(["axis", "alpha"]):
        v = g["share"].to_numpy()
        boot = [rng.choice(v, len(v)).mean() for _ in range(args.boots)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        table.append(dict(axis=axis, alpha=alpha, pairs=len(v), share=v.mean(),
                          lo=lo, hi=hi))
        print(f"{axis:14s} {alpha:+6.2f} {len(v):6d} {v.mean():18.1%} "
              f"[{lo:.1%}, {hi:.1%}]")
    path = out / "answer_verdicts.csv"
    pd.DataFrame(table).to_csv(path, index=False)
    print(f"-> {path}")


if __name__ == "__main__":
    main()
