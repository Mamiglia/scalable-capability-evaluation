"""Fit and freeze the causal-stage candidate directions.

One ridge direction per frozen role, fit on unstandardised per-sentence mean
h_L14 - h_0 features predicting per-token sentence effects; penalty by
item-grouped cross-validation over the shared grid. Directions are saved
unit-normalised, in residual-stream coordinates.
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd

REPO = pathlib.Path("/Users/moriohamada/PycharmProjects/ai_safety/small-judge")
sys.path.insert(0, str(REPO / "judge-whitebox"))
import token_mediation as tm

S = REPO / "tmp/whitebox"
F = S / "feat_llama" / "Llama-3.2-3B-Instruct"
OUT = S / "causal"
FROZEN_ROLES = ["instr_format", "slot_labels", "answer_readout",
                "transcript_1_d9", "transcript_2_d8", "transcript_2_d9"]

index = pd.read_parquet(F / "index.parquet")
scores = pd.read_parquet(S / "llama_sentence_scores.parquet")
merged = index.reset_index(drop=True).reset_index().merge(
    scores[["pair_uid", "order", "unit", "beta_per_token", "item"]],
    on=["pair_uid", "order", "unit"], how="inner")
rows = merged["index"].to_numpy()
y = merged["beta_per_token"].to_numpy(float)
items = merged["item"].to_numpy()
roles, role_names = tm.role_ids(merged)
n_roles = len(role_names)

X = np.load(F / "feat_L14.npy")[rows].astype(np.float64)

errs = {lam: 0.0 for lam in tm.RIDGE_GRID}
for fold_items in tm._grouped_folds(items, 5):
    test = np.isin(items, fold_items)
    train = ~test
    for lam in tm.RIDGE_GRID:
        W, _ = tm._role_ridge(X[train], y[train], roles[train], lam, n_roles)
        pred = np.einsum("ij,ji->i", X[test], W[:, roles[test]])
        errs[lam] += float(np.sum((y[test] - pred) ** 2))
lam = min(errs, key=errs.get)
grid = list(tm.RIDGE_GRID)
assert lam not in (grid[0], grid[-1]), f"penalty at grid endpoint: {lam}"

W, counts = tm._role_ridge(X, y, roles, lam, n_roles)

meta = {"lam": lam, "layer": 14, "feature": "mean h_L14 - h_0 per sentence, unstandardised",
        "roles": {}}
directions = {}
for name in FROZEN_ROLES:
    r = role_names.index(name)
    w = W[:, r]
    norm = float(np.linalg.norm(w))
    assert norm > 0, name
    directions[name] = (w / norm).astype(np.float32)
    sel = roles == r
    pred = X[sel] @ w
    meta["roles"][name] = {"n_sentences": int(counts[r]), "raw_norm": norm,
                           "fit_r_within_role": float(np.corrcoef(pred, y[sel])[0, 1])}

OUT.mkdir(exist_ok=True)
np.savez(OUT / "directions_L14.npz", **directions)
(OUT / "directions_L14.json").write_text(json.dumps(meta, indent=1))
print(json.dumps(meta, indent=1))
