"""Rank-curve controls: h0-feature curve and random-subspace floors."""
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
rows = merged["index"].to_numpy()
y = merged["beta_per_token"].to_numpy(float)
items = merged["item"].to_numpy()
roles, _ = tm.role_ids(merged)

h0 = tm.standardise(np.load(F / "feat_h0.npy")[rows])
print("h0 control:", tm.rank_curve(h0, y, roles, items, ranks=(1, 3, 6)), flush=True)
del h0

X = tm.standardise(np.load(F / "feat_L14.npy")[rows])
rng = np.random.default_rng(5)
n_roles = int(roles.max()) + 1
for k in (1, 3, 6):
    r2s = []
    for seed in range(8):
        U = np.linalg.qr(rng.normal(size=(X.shape[1], k)))[0].astype(np.float32)
        proj = X @ U
        pred = np.full(len(y), np.nan)
        for test_items in tm._grouped_folds(items, 5):
            test = np.isin(items, test_items)
            train = ~test
            for r in range(n_roles):
                tr = train & (roles == r)
                te = test & (roles == r)
                if not te.any() or tr.sum() < k + 2:
                    continue
                v, *_ = np.linalg.lstsq(proj[tr], y[tr], rcond=None)
                pred[te] = proj[te] @ v
        r2s.append(1 - np.nansum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2))
    print(f"random rank-{k}: mean {np.mean(r2s):.4f} max {np.max(r2s):.4f}", flush=True)
