"""Held-out check of the additive model on the mask pilot, structural-span level.

For each prompt: X[r, g] = number of tokens of structural group g blanked in
draw r; y[r] = delta. Under an additive per-token model with group-constant
effects, y = X @ beta with no intercept. 5-fold CV over draws gives held-out R².
An intercept variant absorbs any constant non-additive offset.
"""
import numpy as np
import pandas as pd

ROOT = "/Users/moriohamada/PycharmProjects/ai_safety/small-judge/judge-whitebox/out/token_attribution"
GROUPS = ["instr_format", "slot_labels", "question", "transcript_1", "transcript_2", "answer_readout"]

occ = pd.concat([pd.read_parquet(f"{ROOT}/occlusion_v2/part_0000{i}.parquet") for i in (0, 1)])
mask = pd.read_parquet(f"{ROOT}/maskpilot/part_00000.parquet")


def position_groups(occ_prompt, n_tokens):
    """Group index per token position, from the 8-token occlusion spans."""
    gid = np.zeros(n_tokens, dtype=int)
    for _, r in occ_prompt.iterrows():
        gid[r["span_start"]:r["span_end"]] = GROUPS.index(r["group"])
    return gid


def cv_r2(X, y, folds=5, intercept=False):
    """Pooled held-out R² over CV folds, fit by least squares."""
    if intercept:
        X = np.column_stack([np.ones(len(X)), X])
    rng = np.random.default_rng(0)
    order = rng.permutation(len(y))
    pred = np.empty_like(y)
    for f in range(folds):
        test = order[f::folds]
        train = np.setdiff1d(order, test)
        beta, *_ = np.linalg.lstsq(X[train], y[train], rcond=None)
        pred[test] = X[test] @ beta
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1 - ss_res / ss_tot, pred


rows = []
for (uid, order), m in mask.groupby(["pair_uid", "order"]):
    n_tokens = int(m["n_tokens"].iloc[0])
    op = occ[(occ["pair_uid"] == uid) & (occ["order"] == order)]
    gid = position_groups(op, n_tokens)
    X = np.zeros((len(m), len(GROUPS)))
    for r, pos in enumerate(m["masked_positions"]):
        np.add.at(X[r], gid[np.asarray(pos)], 1)
    y = m["delta"].to_numpy()
    r2_no, _ = cv_r2(X, y)
    r2_int, _ = cv_r2(X, y, intercept=True)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    rows.append({"pair_uid": uid, "order": order, "n_tokens": n_tokens,
                 "heldout_r2": r2_no, "heldout_r2_intercept": r2_int,
                 "sd_delta": y.std(),
                 **{f"beta_{g}": b for g, b in zip(GROUPS, beta)}})

res = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(res[["heldout_r2", "heldout_r2_intercept", "sd_delta"]].describe().round(3))
print()
print("median per-token beta by group:")
print(res[[f"beta_{g}" for g in GROUPS]].median().round(4))
print()
print("per-prompt held-out R2 (no intercept):")
print(res[["pair_uid", "order", "n_tokens", "heldout_r2"]].round(3).to_string(index=False))
