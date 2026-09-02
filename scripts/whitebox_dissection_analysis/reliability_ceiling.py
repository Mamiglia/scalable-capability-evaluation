"""Split-half reliability of sentence effects on the pooled 4096-draw campaign.

Draws are split at random into two halves; sentence betas are fit on each half
independently and correlated across all sentences of the sampled prompts.
Spearman-Brown then gives the reliability of betas fit on the full draw count,
the attainable R² ceiling for anything predicting sentence effects.
"""
import pathlib
import sys

import numpy as np
import pandas as pd

REPO = pathlib.Path("/Users/moriohamada/PycharmProjects/ai_safety/small-judge")
S = REPO / "tmp/whitebox"
sys.path.insert(0, str(REPO / "judge-whitebox"))
import os

os.environ["WB_PAIRS"] = str(S / "pairs_ceph.parquet")
import mask_regression as mr
import token_manifold_record
import wbconfig
import wbpairs
from transformers import AutoTokenizer

N_PROMPTS = 300
SEED = 0


def main():
    c = wbconfig.cfg()
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
    masks = mr.read_mask_parts(S / "maskcamp_llama/pooled")
    pairs = wbpairs.load_pairs(c, "train")
    keyed = {(str(r["pair_uid"]), int(r["order"])): r for _, r in pairs.iterrows()}

    rng = np.random.default_rng(SEED)
    prompts = masks[["pair_uid", "order"]].drop_duplicates().to_numpy()
    sample = prompts[rng.choice(len(prompts), N_PROMPTS, replace=False)]

    halves = {0: [], 1: []}
    for uid, order in sample:
        rows = masks[(masks.pair_uid == uid) & (masks.order == int(order))]
        meta = token_manifold_record.prompt_token_meta(keyed[(str(uid), int(order))], tokenizer, c)
        assert int(rows["n_tokens"].iloc[0]) == len(meta["input_ids"])
        unit_ids = mr.sentence_units(meta)
        X = mr.design_matrix(rows["masked_positions"].tolist(), unit_ids)
        y = rows["delta"].to_numpy(dtype=float)
        split = rng.permutation(len(y))
        for h, idx in ((0, split[: len(y) // 2]), (1, split[len(y) // 2 :])):
            fit = mr.fit_units(X[idx], y[idx])
            halves[h].append(fit["beta"])
    b0 = np.concatenate(halves[0])
    b1 = np.concatenate(halves[1])
    r = float(np.corrcoef(b0, b1)[0, 1])
    ceiling = 2 * r / (1 + r)
    n_draws = masks.groupby(["pair_uid", "order"]).size().median()
    print(f"{N_PROMPTS} prompts, {int(n_draws)} draws/prompt, half = {int(n_draws) // 2}")
    print(f"split-half beta correlation r = {r:.3f}")
    print(f"Spearman-Brown reliability at full draws (R² ceiling) = {ceiling:.3f}")


if __name__ == "__main__":
    main()
