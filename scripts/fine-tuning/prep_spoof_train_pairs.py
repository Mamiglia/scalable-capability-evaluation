"""Spoof-augmented training pairs + a held-out spoof eval set, from the full
198-item style-cap generations (baseline/confident/uncertain, 5 models).

Tone is assigned INDEPENDENTLY of the capability label, so tone carries no
information about which author is stronger — the judge must learn to ignore
confidence. Accuracy is held under the instruction (style-cap guard) so the
label (higher-accuracy author) is unchanged by the tone. Train vs held-out
items follow item_split.csv (shared with the epoch pool → no leakage).

Emits (in judge_results/judge_ft/):
  spoof_train_pairs.parquet   split="train", tone-crossed, train items only
  pairs_spoof_heldout.parquet split in {natural, spoof_uncertain}, held-out items
"""

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from prep_pairs import make_pair_rows, render_choices, check_counterbalance
from prep_spoof_pairs import load_answers, first_correct, model_accuracy

TONES = ["baseline", "confident", "uncertain"]
STRONG = ["qwen3.5-122b-a10b", "qwen3.6-35b-a3b"]  # told-uncertain side in the eval set


def tone_stores(sd: Path):
    base = load_answers(sd / "baseline.jsonl")
    conf = load_answers(sd / "instruct.jsonl", label="confident")
    unc = load_answers(sd / "instruct.jsonl", label="uncertain")
    return {"baseline": base, "confident": conf, "uncertain": unc}


def build_train(stores, acc, items, opts_by_item, questions, min_gap, n_combos, seed):
    """Tone-crossed cross-model both-correct pairs on train items."""
    rng = np.random.default_rng(seed)
    models = sorted(acc)
    rows = []
    for item in sorted(items):
        for m1, m2 in itertools.combinations(models, 2):
            if abs(acc[m1] - acc[m2]) < min_gap:
                continue
            strong, weak = (m1, m2) if acc[m1] > acc[m2] else (m2, m1)
            avail_s = [t for t in TONES if first_correct(stores[t], strong, item)]
            avail_w = [t for t in TONES if first_correct(stores[t], weak, item)]
            combos = [(ts, tw) for ts in avail_s for tw in avail_w]
            if not combos:
                continue
            pick = rng.choice(len(combos), size=min(n_combos, len(combos)), replace=False)
            for i in pick:
                ts, tw = combos[i]
                a = dict(model=f"{strong}|{ts}", cap=acc[strong],
                         text=first_correct(stores[ts], strong, item), options=opts_by_item[item])
                b = dict(model=f"{weak}|{tw}", cap=acc[weak],
                         text=first_correct(stores[tw], weak, item), options=opts_by_item[item])
                rows.extend(make_pair_rows(item, a, b, questions[item]))
    df = pd.DataFrame(rows)
    df["split"] = "train"
    return df


def build_heldout_eval(stores, acc, items, opts_by_item, questions):
    """Matched natural vs spoof_uncertain on held-out items (strong told uncertain
    vs weaker baseline) — same shape as prep_spoof_pairs, for the robustness re-test."""
    rows = []
    for s in STRONG:
        for o in [m for m in acc if acc[m] < acc[s]]:
            for item in sorted(items):
                s_base = first_correct(stores["baseline"], s, item)
                s_unc = first_correct(stores["uncertain"], s, item)
                o_base = first_correct(stores["baseline"], o, item)
                if o_base is None:
                    continue
                weak = dict(model=o, cap=acc[o], text=o_base, options=opts_by_item[item])
                if s_base:
                    strong = dict(model=s, cap=acc[s], text=s_base, options=opts_by_item[item])
                    rows += [{**r, "split": "natural"} for r in make_pair_rows(item, strong, weak, questions[item])]
                if s_unc:
                    strong = dict(model=f"{s}#uncertain", cap=acc[s], text=s_unc, options=opts_by_item[item])
                    rows += [{**r, "split": "spoof_uncertain"} for r in make_pair_rows(item, strong, weak, questions[item])]
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gen-dir", default="judge_results/judge_ft/spoof_gen")
    ap.add_argument("--items", default="judge_results/style_capability/items_full.json")
    ap.add_argument("--item-split", default="judge_results/judge_ft/item_split.csv")
    ap.add_argument("--out-dir", default="judge_results/judge_ft")
    ap.add_argument("--min-gap", type=float, default=0.05, help="min accuracy gap to pair")
    ap.add_argument("--n-combos", type=int, default=3, help="tone-combos sampled per model pair per item")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import json
    items = json.load(open(args.items))
    opts_by_item = {it["sample_id"]: render_choices(it["choices"]) for it in items}
    questions = {it["sample_id"]: it["question"] for it in items}
    split = pd.read_csv(args.item_split)
    held = set(split.item[split.holdout]); train_items = set(split.item[~split.holdout])
    all_sids = {it["sample_id"] for it in items}

    stores = tone_stores(Path(args.gen_dir))
    acc = model_accuracy(stores["baseline"])
    print("baseline accuracy:", {m: round(a, 3) for m, a in sorted(acc.items(), key=lambda x: -x[1])})

    tr = build_train(stores, acc, train_items & all_sids, opts_by_item, questions,
                     args.min_gap, args.n_combos, args.seed)
    check_counterbalance(tr)
    tr.to_parquet(Path(args.out_dir) / "spoof_train_pairs.parquet")
    print(f"train: {tr.pair_uid.nunique()} pairs over {tr.item.nunique()} items "
          f"(label balance {tr.label.mean():.3f})")

    ev = build_heldout_eval(stores, acc, held & all_sids, opts_by_item, questions)
    check_counterbalance(ev)
    ev.to_parquet(Path(args.out_dir) / "pairs_spoof_heldout.parquet")
    print(ev.groupby("split").pair_uid.nunique().rename("pairs").to_dict(),
          "over", ev.item.nunique(), "held-out items")


if __name__ == "__main__":
    main()
