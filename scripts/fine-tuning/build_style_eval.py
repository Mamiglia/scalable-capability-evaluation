"""Reconstruct the adversarial style-swap eval from style_meta_<d>.parquet + the
restyled transcripts (rewrites_style_cross.jsonl). Each held-out pair yields a
`natural` version (original transcripts) and a `style_swap` version (stronger
author in the weak voice, weaker author in the strong voice), both counter-balanced
and same label. A style-robust judge keeps picking the stronger author.

Writes pairs_style_<domain>_<label>.parquet in the pod_verdicts format.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from prep_pairs import make_pair_rows


def load_rewrites(path):
    rw = {}
    for line in open(path):
        r = json.loads(line)
        if r.get("answer_preserved", True):
            rw[r["id"]] = r["rewrite"]
    return rw


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--meta", required=True, help="style_meta_<d>.parquet")
    ap.add_argument("--rewrites", required=True, help="rewrites_style_cross.jsonl")
    ap.add_argument("--prefix", required=True, help="id prefix used for this domain, e.g. 'gp'")
    ap.add_argument("--domain", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    meta = pd.read_parquet(args.meta)
    rw = load_rewrites(args.rewrites)

    rows = []
    kept = 0
    for r in meta.itertuples():
        s_id, w_id = f"{args.prefix}_{r.pair_uid}|S", f"{args.prefix}_{r.pair_uid}|W"
        if s_id not in rw or w_id not in rw:
            continue                                   # need both sides restyled
        kept += 1
        m1 = dict(model=r.model_1, cap=r.cap_1, options=r.options_1)
        m2 = dict(model=r.model_2, cap=r.cap_2, options=r.options_2)
        # natural: original transcripts
        for row in make_pair_rows(r.item, {**m1, "text": r.text_1},
                                  {**m2, "text": r.text_2}, r.question):
            rows.append({**row, "split": "natural", "domain": args.domain})
        # style_swap: stronger author -> weak voice (rw[S]); weaker -> strong voice (rw[W])
        t1 = rw[s_id] if r.strong_is_1 else rw[w_id]
        t2 = rw[w_id] if r.strong_is_1 else rw[s_id]
        for row in make_pair_rows(r.item, {**m1, "text": t1},
                                  {**m2, "text": t2}, r.question):
            rows.append({**row, "split": "style_swap", "domain": args.domain})

    df = pd.DataFrame(rows)
    df.to_parquet(args.out)
    print(f"{args.domain}/{args.label}: {kept} pairs kept -> {len(df)} rows "
          f"(splits {df.split.value_counts().to_dict()}, label-mean {df.label.mean():.3f})")


if __name__ == "__main__":
    main()
