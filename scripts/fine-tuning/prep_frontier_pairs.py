"""Build judge pairs from the frontier GPQA/LAB-Bench answers.

Both-correct cross-model pairs per (eval, item); capability = the model's
accuracy on that eval in this run; no gap filter — near-tie discrimination
is the point. Option renderings are reconstructed deterministically from
the generation seed.
"""

import argparse
import json
import random
import string
from pathlib import Path

import pandas as pd
import yaml

import prompts
from prep_pairs import check_counterbalance, make_pair_rows, token_filter

LETTERS = string.ascii_uppercase


def lab_options_body(record, seed, epoch):
    options = [record["ideal"], *record["distractors"]]
    random.Random((seed, str(record["id"]), epoch).__hash__()).shuffle(options)
    return "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(options))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(Path(__file__).parent / "frontier_config.yaml"))
    ap.add_argument("--main-config", default=str(Path(__file__).parent / "config.yaml"))
    ap.add_argument("--out", default="judge_results/judge_ft/pairs_frontier.parquet")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    prep = yaml.safe_load(open(args.main_config))["prep"]

    df = pd.DataFrame([json.loads(l) for l in open(cfg["out_jsonl"])])
    caps = df.groupby(["eval", "model"]).correct.mean()

    # question text + per-(item, epoch) option body for each eval
    gpqa_items = pd.read_parquet(cfg["gpqa"]["items_parquet"]).set_index("item")
    from datasets import load_dataset
    lab_records = {}
    lb = cfg["labbench"]
    for sub in lb["text_subtasks"] + lb["image_subtasks"]:
        for r in load_dataset(lb["source"], sub, split="train"):
            lab_records[str(r["id"])] = r

    first_correct = (df[df.correct].sort_values("epoch")
                     .drop_duplicates(["eval", "model", "item"]))
    rows = []
    for (ev, item), g in first_correct.groupby(["eval", "item"]):
        metas = {}
        for r in g.itertuples():
            if ev == "gpqa":
                question = gpqa_items.loc[item].question
                body = gpqa_items.loc[item].options_rendered
            else:
                rec = lab_records[item]
                question = rec["question"]
                body = lab_options_body(rec, cfg["seed"], r.epoch)
            metas[r.model] = dict(model=r.model, cap=float(caps[(ev, r.model)]),
                                  text=r.text[:prep["max_len_chars"]], options=body,
                                  question=question)
        models = sorted(metas)
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                a, b = metas[models[i]], metas[models[j]]
                pair = make_pair_rows(item, a, b, question=a["question"])
                for row in pair:
                    row["split"] = f"frontier_{ev}"
                rows.append(pair)
    pairs = pd.DataFrame([r for p in rows for r in p])

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(prep["tokenizer"])
    pairs, n_over = token_filter(pairs, tok, prep["max_input_tokens"])
    check_counterbalance(pairs)
    pairs.to_parquet(args.out)
    print(pairs.groupby("split").pair_uid.nunique().rename("pairs"))
    print(f"dropped for length: {n_over}")
    print("caps:", caps.round(3).to_dict())


if __name__ == "__main__":
    main()
