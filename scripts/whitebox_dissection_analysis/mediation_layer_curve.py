"""First internals result: held-out R² per layer for predicting sentence effects.

Target: per-token sentence effect (beta_per_token). Features per layer: sentence
mean of h_L − h_0. Controls fitted identically: h_0 alone, position/size, their
combination, and shuffled-target floors. Item-grouped nested CV throughout.
"""
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/Users/moriohamada/PycharmProjects/ai_safety/small-judge/judge-whitebox")
import token_mediation as tm

S = pathlib.Path("/Users/moriohamada/PycharmProjects/ai_safety/small-judge/tmp/whitebox")
F = S / "feat_llama" / "Llama-3.2-3B-Instruct"

index = pd.read_parquet(F / "index.parquet")
scores = pd.read_parquet(S / "llama_sentence_scores.parquet")
merged = index.reset_index(drop=True).reset_index().merge(
    scores[["pair_uid", "order", "unit", "beta_per_token", "item"]],
    on=["pair_uid", "order", "unit"], how="inner")
assert len(merged) == len(index), (len(merged), len(index))
rows = merged["index"].to_numpy()
y = merged["beta_per_token"].to_numpy(float)
items = merged["item"].to_numpy()
print(f"{len(y)} sentences, {len(set(items))} items")

pos = tm.standardise(tm.control_features(merged))
h0 = tm.standardise(np.load(F / "feat_h0.npy")[rows])

import gc

results = {}
rng = np.random.default_rng(0)


def run(name, X, target=y):
    results[name] = tm.layer_fit(X, target, items)
    print(name, results[name]["heldout_r2"],
          "ENDPOINT" if results[name]["endpoint"] else "", flush=True)
    del X
    gc.collect()


run("position", pos.copy())
run("h0", h0.copy())
run("h0+position", np.column_stack([h0, pos]))

for layer in (7, 14, 21, 28):
    X = tm.standardise(np.load(F / f"feat_L{layer}.npy")[rows])
    if layer == 28:
        run("L28+controls", np.column_stack([X, h0, pos]))
        run("L28_shuffled", X.copy(), target=rng.permutation(y))
    run(f"L{layer}", X)
    del X
    gc.collect()

out = pd.DataFrame([{"features": k, "heldout_r2": v["heldout_r2"],
                     "endpoint": v["endpoint"], "penalties": str(v["penalties"])}
                    for k, v in results.items()])
out.to_csv(S / "mediation_layer_curve_llama.csv", index=False)
print(out.round(4).to_string(index=False))
