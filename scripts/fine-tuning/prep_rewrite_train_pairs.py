"""Rewrite-augmented training pairs for the 3rd SFT set.

Takes existing train pairs and, where a side's transcript was restyled
(data/rewrite_style3.jsonl — neutral or another model's voice), emits a variant
with the rewritten transcript(s) swapped in, keeping the capability label. The
rewrite voice was assigned INDEPENDENTLY of the label, so writing style carries
no capability signal: the judge must read past it. Both counter-balanced orders
of a pair are restyled consistently (each order matches the rewrite, so the pair
stays balanced).
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from prep_pairs import check_counterbalance


def load_rewrites(path):
    rw = {}
    for line in open(path):
        r = json.loads(line)
        rw[(str(r["model"]), str(r["item"]))] = dict(text=r["rewrite"], voice=r["target_voice"])
    return rw


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", default="judge_results/judge_ft/pairs.parquet")
    ap.add_argument("--rewrites", default="data/rewrite_style3.jsonl")
    ap.add_argument("--out", default="judge_results/judge_ft/rewrite_train_pairs.parquet")
    ap.add_argument("--cap-per-transcript", type=int, default=8,
                    help="max augmented pair_uids seeded by one rewritten transcript")
    args = ap.parse_args()
    rw = load_rewrites(args.rewrites)
    tr = pd.read_parquet(args.pairs)
    tr = tr[tr.split == "train"].copy()
    tr["k1"] = list(zip(tr.model_1.astype(str), tr.item.astype(str)))
    tr["k2"] = list(zip(tr.model_2.astype(str), tr.item.astype(str)))
    sub = tr[tr.k1.isin(rw) | tr.k2.isin(rw)].copy()

    # cap augmented pair_uids per rewritten transcript (which key it hangs on)
    def cap_key(row):
        return row.k1 if row.k1 in rw else row.k2
    sub["capk"] = sub.apply(cap_key, axis=1)
    keep_uids, used = set(), {}
    for uid, ck in sub[["pair_uid", "capk"]].drop_duplicates("pair_uid").itertuples(index=False):
        if used.get(ck, 0) < args.cap_per_transcript:
            keep_uids.add(uid); used[ck] = used.get(ck, 0) + 1
    sub = sub[sub.pair_uid.isin(keep_uids)].copy()

    sub["text_1"] = [rw[k]["text"] if k in rw else t for k, t in zip(sub.k1, sub.text_1)]
    sub["text_2"] = [rw[k]["text"] if k in rw else t for k, t in zip(sub.k2, sub.text_2)]
    # sort voices so the uid suffix is order-independent (both counter-balanced
    # rows of a pair get the same new uid even when both sides are rewritten)
    vsuf = [("+".join(sorted(v for v in (rw.get(k1, {}).get("voice"), rw.get(k2, {}).get("voice")) if v)))
            for k1, k2 in zip(sub.k1, sub.k2)]
    sub["pair_uid"] = [f"{u}|rw:{s}" for u, s in zip(sub.pair_uid, vsuf)]
    sub["split"] = "train"
    aug = sub.drop(columns=["k1", "k2", "capk"])
    check_counterbalance(aug)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    aug.to_parquet(args.out)
    print(f"{aug.pair_uid.nunique()} rewrite-augmented pairs from {len(used)} rewritten "
          f"transcripts over {aug.item.nunique()} items (label balance {aug.label.mean():.3f})")


if __name__ == "__main__":
    main()
