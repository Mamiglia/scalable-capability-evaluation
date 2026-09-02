"""Mediating directions: do a layer's states carry the sentence effects?

Stage one, extraction: reduce the residual-stream recording to per-sentence
features — the mean over each sentence's tokens of h_L − h_0, the state at
layer L minus the token's own embedding. What the network computed about a
token beyond its identity, at sentence grain, joined by (pair_uid, order,
unit) to the sentence scores.
"""

import json
import pathlib

import numpy as np
import pandas as pd
import torch

import mask_regression
import token_manifold_record
import wbpairs


def sentence_features(states, unit_ids, layers):
    """Per-sentence mean of h_L − h_0 for every recorded layer above 0.

    states: [tokens, n_layers, d] with layer index 0 the embedding output.
    Returns {layer: [n_units, d] float32} plus the per-sentence h_0 means.
    """
    states = states.to(torch.float32)
    h0 = states[:, 0, :]
    n_units = int(unit_ids.max()) + 1
    counts = np.bincount(unit_ids, minlength=n_units).astype(np.float32)[:, None]
    out = {}
    idx = torch.from_numpy(np.asarray(unit_ids, dtype=np.int64))
    for li, layer in enumerate(layers):
        if li == 0:
            sums = torch.zeros(n_units, states.shape[2])
            sums.index_add_(0, idx, h0)
            out["h0"] = (sums / torch.from_numpy(counts)).numpy()
            continue
        diff = states[:, li, :] - h0
        sums = torch.zeros(n_units, states.shape[2])
        sums.index_add_(0, idx, diff)
        out[layer] = (sums / torch.from_numpy(counts)).numpy()
    return out


def extract(manifest_path, c, tokenizer, out_dir, limit=None):
    """Stream recording shards into per-layer sentence-feature files.

    Writes float16 matrices feat_L{layer}.npy and feat_h0.npy plus index.parquet
    with (pair_uid, order, unit, group, n_tokens, rel_start, rel_end) aligned
    row-for-row with every feature matrix.
    """
    manifest_path = pathlib.Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    layers = manifest["layers"]
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = wbpairs.load_pairs(c, manifest["split"])
    keyed = {(str(r["pair_uid"]), int(r["order"])): r for _, r in pairs.iterrows()}

    blocks = {layer: [] for layer in layers[1:]}
    blocks["h0"] = []
    index_rows = []
    done = 0
    for shard in manifest["shards"]:
        payload = torch.load(manifest_path.parent / shard["path"], map_location="cpu",
                             weights_only=False)
        prompt_idx = payload["prompt_idx"]
        for i, rec in enumerate(payload["prompts"]):
            uid, order = str(rec["pair_uid"]), int(rec["order"])
            row = keyed[(uid, order)]
            meta = token_manifold_record.prompt_token_meta(row, tokenizer, c)
            sel = prompt_idx == i
            states = payload["resid"][sel]
            assert payload["input_ids"][sel].tolist() == list(meta["input_ids"]), \
                f"recording and rendered prompt disagree for {uid}|{order}"
            unit_ids = mask_regression.sentence_units(meta)
            feats = sentence_features(states, unit_ids, layers)
            for key in blocks:
                blocks[key].append(feats[key].astype(np.float16))
            units = mask_regression.unit_table(meta, unit_ids)
            total = float(sum(units.n_tokens))
            starts = units.n_tokens.cumsum() - units.n_tokens
            index_rows.append(pd.DataFrame({
                "pair_uid": uid, "order": order, "unit": units.unit,
                "group": units.group, "n_tokens": units.n_tokens,
                "rel_start": (starts / total).astype(np.float32),
                "rel_end": (units.n_tokens.cumsum() / total).astype(np.float32)}))
            done += 1
            if limit and done >= limit:
                break
        if limit and done >= limit:
            break
        if done % 200 < len(payload["prompts"]):
            print(f"{done} prompts", flush=True)

    for key, mats in blocks.items():
        name = "feat_h0" if key == "h0" else f"feat_L{key}"
        np.save(out_dir / f"{name}.npy", np.concatenate(mats, axis=0))
    pd.concat(index_rows, ignore_index=True).to_parquet(out_dir / "index.parquet", index=False)
    print(f"extracted {done} prompts -> {out_dir}", flush=True)


RIDGE_GRID = tuple(10.0 ** e for e in range(-2, 7))


def _grouped_folds(items, n_folds, seed=0):
    """Deterministic item-grouped folds: lists of item labels per fold."""
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(set(items)))
    rng.shuffle(uniq)
    return [uniq[f::n_folds] for f in range(n_folds)]


def _ridge_path(X, y, ridge_grid):
    """Weights for every penalty from one eigendecomposition of the Gram."""
    gram = X.T @ X
    xty = X.T @ y
    vals, vecs = np.linalg.eigh(gram)
    proj = vecs.T @ xty
    return {lam: vecs @ (proj / (vals + lam)) for lam in ridge_grid}


def layer_fit(X, y, items, outer=5, inner=4, ridge_grid=RIDGE_GRID, seed=0):
    """Held-out R² of ridge predictions, folds grouped by item throughout.

    The penalty is chosen on inner folds of each outer training set. Returns
    pooled held-out R², the per-fold penalties, and an endpoint flag.
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float64)
    items = np.asarray(items)
    pred = np.full(len(y), np.nan)
    chosen = []
    for test_items in _grouped_folds(items, outer, seed):
        test = np.isin(items, test_items)
        train = ~test
        errs = {lam: 0.0 for lam in ridge_grid}
        for inner_items in _grouped_folds(items[train], inner, seed + 1):
            itest = train & np.isin(items, inner_items)
            itrain = train & ~np.isin(items, inner_items)
            for lam, w in _ridge_path(X[itrain], y[itrain], ridge_grid).items():
                errs[lam] += float(np.sum((y[itest] - X[itest] @ w) ** 2))
        lam = min(errs, key=errs.get)
        chosen.append(lam)
        w = _ridge_path(X[train], y[train], ridge_grid)[lam]
        pred[test] = X[test] @ w
    r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
    return {"heldout_r2": float(r2), "penalties": chosen,
            "endpoint": any(l in (ridge_grid[0], ridge_grid[-1]) for l in chosen)}


def control_features(index):
    """Position-and-size baseline: where the sentence sits and how big it is."""
    groups = sorted(index["group"].unique())
    onehot = np.stack([(index["group"] == g).to_numpy(float) for g in groups], axis=1)
    return np.column_stack([
        index["rel_start"].to_numpy(float), index["rel_end"].to_numpy(float),
        1.0 - index["rel_end"].to_numpy(float), np.log(index["n_tokens"].to_numpy(float)),
        onehot])


def standardise(X):
    """Zero-mean unit-variance columns; constant columns pass through as zeros."""
    X = np.asarray(X, dtype=np.float32)
    mu = X.mean(0)
    sd = X.std(0)
    sd[sd == 0] = 1.0
    return (X - mu) / sd


def main():
    import argparse

    from transformers import AutoTokenizer

    import wbconfig

    c = wbconfig.cfg()
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--tokenizer", required=True,
                        help="tokenizer name or path matching the recording's model")
    parser.add_argument("--limit", type=int, default=None,
                        help="prompts to extract; omit for all")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    extract(args.manifest, c, tokenizer, args.out, args.limit)


if __name__ == "__main__":
    main()


TRANSCRIPT_GROUPS = ("transcript_1", "transcript_2")


def role_ids(index, deciles=10):
    """Role per sentence: structural group, with transcripts split by position decile.

    Scaffolding groups are one role each; within each transcript, sentences are
    binned by their relative position inside that transcript span, per prompt.
    Returns integer roles and the role names.
    """
    frame = index.reset_index(drop=True).copy()
    names = []
    codes = pd.Series(index=frame.index, dtype=int)
    for group, g in frame.groupby("group"):
        if group not in TRANSCRIPT_GROUPS:
            names.append(group)
            codes[g.index] = len(names) - 1
            continue
        pos = g.groupby(["pair_uid", "order"]).cumcount()
        size = g.groupby(["pair_uid", "order"])["unit"].transform("size")
        dec = np.minimum((pos + 0.5) / size * deciles, deciles - 1).astype(int)
        base = len(names)
        names.extend(f"{group}_d{d}" for d in range(deciles))
        codes[g.index] = base + dec
    return codes.to_numpy(int), names


def _ridge_solve(X, y, lam):
    gram = X.T @ X + lam * np.eye(X.shape[1], dtype=np.float64)
    return np.linalg.solve(gram, X.T @ y)


def _role_ridge(X, y, roles, lam, n_roles):
    """One ridge direction per role; zero column for roles absent from the data."""
    W = np.zeros((X.shape[1], n_roles), dtype=np.float64)
    counts = np.zeros(n_roles)
    for r in range(n_roles):
        sel = roles == r
        if sel.sum() < 3:
            continue
        W[:, r] = _ridge_solve(X[sel], y[sel], lam)
        counts[r] = sel.sum()
    return W, counts


def _rank_predict(X, y, roles, train, test, lam, rank, n_roles):
    """Fit role maps on train, constrain to a rank-k shared subspace, predict test."""
    W, counts = _role_ridge(X[train], y[train], roles[train], lam, n_roles)
    weighted = W * np.sqrt(np.maximum(counts, 1.0))
    U = np.linalg.svd(weighted, full_matrices=False)[0][:, :rank]
    proj_train, proj_test = X[train] @ U, X[test] @ U
    pred = np.zeros(test.sum())
    roles_tr, roles_te = roles[train], roles[test]
    for r in range(n_roles):
        tr = roles_tr == r
        te = roles_te == r
        if not te.any():
            continue
        if tr.sum() < rank + 2:
            continue
        v, *_ = np.linalg.lstsq(proj_train[tr], y[train][tr], rcond=None)
        pred[te] = proj_test[te] @ v
    return pred


def rank_curve(X, y, roles, items, ranks, outer=5, inner=4,
               ridge_grid=RIDGE_GRID, seed=0):
    """Held-out R² per rank, role-shared subspace, item-grouped nested CV.

    The penalty is chosen by inner CV on the full-rank role maps; each rank is
    then evaluated on the outer test folds with that penalty. Rank 1 is the
    null: every role reads one shared direction.
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float64)
    n_roles = int(roles.max()) + 1
    preds = {k: np.full(len(y), np.nan) for k in ranks}
    for test_items in _grouped_folds(items, outer, seed):
        test = np.isin(items, test_items)
        train = ~test
        errs = {lam: 0.0 for lam in ridge_grid}
        for inner_items in _grouped_folds(items[train], inner, seed + 1):
            itest = train & np.isin(items, inner_items)
            itrain = train & ~np.isin(items, inner_items)
            for lam in ridge_grid:
                W, _ = _role_ridge(X[itrain], y[itrain], roles[itrain], lam, n_roles)
                pred = np.einsum("ij,ji->i", X[itest], W[:, roles[itest]])
                errs[lam] += float(np.sum((y[itest] - pred) ** 2))
        lam = min(errs, key=errs.get)
        for k in ranks:
            preds[k][test] = _rank_predict(X, y, roles, train, test, lam, k, n_roles)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return {k: float(1 - np.nansum((y - p) ** 2) / ss_tot) for k, p in preds.items()}
