"""Select a large, DIVERSE set of train transcripts to restyle with the local
rewriter (pod_rewrite.py), spread across all train items and many models.

Each source gets a target_voice assigned INDEPENDENTLY of capability (neutral, or
another model's style with a few of that model's transcripts as `examples`), so
the rewrite voice carries no capability signal. Writes sources.parquet.
"""

import argparse
import random
from collections import defaultdict

import pandas as pd


def strip_ans(t):
    i = t.rfind("ANSWER:")
    return t[:i].rstrip() if i != -1 else t


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", default="judge_results/judge_ft/pairs.parquet")
    ap.add_argument("--out", default="judge_results/judge_ft/rewrite_sources.parquet")
    ap.add_argument("--n", type=int, default=5000, help="transcripts to rewrite")
    ap.add_argument("--min-reasoning", type=int, default=200)
    ap.add_argument("--n-voices", type=int, default=20, help="distinct style-source models")
    ap.add_argument("--n-examples", type=int, default=2)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()
    tr = pd.read_parquet(args.pairs)
    tr = tr[tr.split == "train"]

    # distinct non-degenerate transcripts
    txt = {}
    for r in tr.itertuples():
        k = (r.model_1, str(r.item))
        if k not in txt and len(strip_ans(r.text_1)) >= args.min_reasoning:
            txt[k] = r.text_1
    keys = list(txt)
    rng = random.Random(args.seed)
    rng.shuffle(keys)  # random shuffle => spread across items/models
    picks = keys[:args.n]

    by_model = defaultdict(list)
    for k in txt:
        by_model[k[0]].append(txt[k])
    voices = [m for m, ts in by_model.items() if len(ts) >= 4]
    rng.shuffle(voices)
    voices = voices[:args.n_voices]
    bank = {m: [t for t in sorted(by_model[m], key=len) if 300 <= len(t) <= 1800][:args.n_examples]
               or by_model[m][:args.n_examples]
            for m in voices}

    rows = []
    for i, k in enumerate(picks):
        model, item = k
        if rng.random() < 0.5:
            voice, ex = "neutral", []
        else:
            cands = [m for m in voices if m != model]
            voice = rng.choice(cands) if cands else "neutral"
            ex = bank.get(voice, []) if voice != "neutral" else []
        rows.append(dict(id=f"g{i:05d}", model=model, item=item,
                         text=txt[k], target_voice=voice, examples=ex))
    df = pd.DataFrame(rows)
    df.to_parquet(args.out)
    n_neu = int((df.target_voice == "neutral").sum())
    print(f"{len(df)} sources over {df.item.nunique()} items, {df.model.nunique()} models | "
          f"neutral {n_neu} / style {len(df) - n_neu} | {len(voices)} voice models")


if __name__ == "__main__":
    main()
