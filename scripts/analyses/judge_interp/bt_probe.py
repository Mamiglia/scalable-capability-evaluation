"""Verdict-trained approval direction.

One-step Bradley-Terry: score(t) = w . z(t) with z the standardized
layer-26 last-token activation, P(slot1 picked) = sigmoid(score1 -
score2 + b). Logistic regression on activation differences over all
verdicts, C by inner validation, outer CV grouped by item. Add-ons:
per-judge gain/bias refit on the shared score, and a qwen-only fit
(angle to the shared direction). Control: label-shuffle null.

Writes out_dir/bt_direction.npz (w, feature stats, position term,
per-judge gain/bias) and prints the accuracy table.

Usage:
  uv run python scripts/analyses/judge_interp/bt_probe.py
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[3]


def load_layer(cfg, out: Path):
    """Activations at the best probe (layer, pooling), aligned entities."""
    meta = json.loads((out / "acts_qwen3-32b.meta.json").read_text())
    best = pd.read_csv(out / "probe_metrics.csv").sort_values("r").iloc[-1]
    acts = np.load(out / "acts_qwen3-32b.npy", mmap_mode="r")
    li = meta["layers"].index(int(best.layer))
    pi = meta["poolings"].index(best.pooling)
    features = np.asarray(acts[:, li, pi, :], dtype=np.float32)
    elo = pd.read_csv(out / "elo.csv")
    dropped = set(meta["dropped_entities"])
    elo = elo[[f"{t}_{i}" not in dropped
               for t, i in zip(elo["text_id"], elo["item"])]].reset_index(drop=True)
    index = {(t, i): k for k, (t, i) in enumerate(zip(elo["text_id"], elo["item"]))}
    return features, index, int(best.layer), best.pooling


def verdict_diffs(cfg, features, index):
    """(diff matrix, y, judge, item) for rows whose entities have acts."""
    rows = pd.read_parquet(ROOT / cfg["rows_parquet"])
    k1 = [index.get((t, i)) for t, i in zip(rows["text_id_slot1"], rows["item"])]
    k2 = [index.get((t, i)) for t, i in zip(rows["text_id_slot2"], rows["item"])]
    ok = np.array([a is not None and b is not None for a, b in zip(k1, k2)])
    rows = rows[ok].reset_index(drop=True)
    k1 = np.array([index[(t, i)] for t, i in zip(rows["text_id_slot1"], rows["item"])])
    k2 = np.array([index[(t, i)] for t, i in zip(rows["text_id_slot2"], rows["item"])])
    return rows, k1, k2


def cv_accuracy(rows, k1, k2, features, cfg_bt, rng=None):
    """Grouped-CV BT fit; held-out accuracy pooled, per judge, and with
    per-judge gain/bias. rng permutes labels within item (null runs)."""
    y = rows["y"].to_numpy().copy()
    if rng is not None:
        for item in rows["item"].unique():
            m = (rows["item"] == item).to_numpy()
            y[m] = rng.permutation(y[m])
    items = rows["item"].to_numpy()
    judges = rows["judge"].to_numpy()
    pred = np.full(len(y), np.nan)
    pred_gain = np.full(len(y), np.nan)
    directions = []
    for train, test in GroupKFold(cfg_bt["cv_folds"]).split(y, y, items):
        ent = np.unique(np.concatenate([k1[train], k2[train]]))
        mean = features[ent].mean(0)
        std = features[ent].std(0) + 1e-8
        def diffs(idx):
            return ((features[k1[idx]] - mean) - (features[k2[idx]] - mean)) / std
        d_train, d_test = diffs(train), diffs(test)
        inner_tr, inner_val = next(GroupKFold(cfg_bt["cv_folds"])
                                   .split(d_train, y[train], items[train]))
        losses = []
        for c in cfg_bt["c_grid"]:
            m = LogisticRegression(C=c, max_iter=cfg_bt["max_iter"]).fit(
                d_train[inner_tr], y[train][inner_tr])
            p = np.clip(m.predict_proba(d_train[inner_val])[:, 1], 1e-12, 1 - 1e-12)
            losses.append(-np.mean(np.where(y[train][inner_val] == 1,
                                            np.log(p), np.log(1 - p))))
        best_c = cfg_bt["c_grid"][int(np.nanargmin(losses))]
        model = LogisticRegression(C=best_c, max_iter=cfg_bt["max_iter"]).fit(
            d_train, y[train])
        pred[test] = model.predict_proba(d_test)[:, 1]
        directions.append(model.coef_[0] / np.linalg.norm(model.coef_[0]))
        # Per-judge gain/bias on the shared train scores, applied to test.
        s_train = d_train @ model.coef_[0]
        s_test = d_test @ model.coef_[0]
        for judge in np.unique(judges[test]):
            jtr = judges[train] == judge
            jte = judges[test] == judge
            if jtr.sum() < 2 or len(np.unique(y[train][jtr])) < 2:
                pred_gain[test[jte]] = pred[test[jte]]
                continue
            gj = LogisticRegression(C=1.0, max_iter=cfg_bt["max_iter"]).fit(
                s_train[jtr, None], y[train][jtr])
            pred_gain[test[jte]] = gj.predict_proba(s_test[jte, None])[:, 1]
    acc = float(((pred > .5) == (y == 1)).mean())
    acc_gain = float(((pred_gain > .5) == (y == 1)).mean())
    per_judge = {j: float(((pred[judges == j] > .5) == (y[judges == j] == 1)).mean())
                 for j in np.unique(judges)}
    per_judge_gain = {j: float(((pred_gain[judges == j] > .5) == (y[judges == j] == 1)).mean())
                      for j in np.unique(judges)}
    cos = [float(a @ b) for i, a in enumerate(directions) for b in directions[i + 1:]]
    return acc, acc_gain, per_judge, per_judge_gain, float(np.mean(cos))


def full_fit(rows, k1, k2, features, cfg_bt):
    mean = features.mean(0)
    std = features.std(0) + 1e-8
    d = ((features[k1] - mean) - (features[k2] - mean)) / std
    y = rows["y"].to_numpy()
    items = rows["item"].to_numpy()
    inner_tr, inner_val = next(GroupKFold(cfg_bt["cv_folds"]).split(d, y, items))
    losses = []
    for c in cfg_bt["c_grid"]:
        m = LogisticRegression(C=c, max_iter=cfg_bt["max_iter"]).fit(d[inner_tr], y[inner_tr])
        p = np.clip(m.predict_proba(d[inner_val])[:, 1], 1e-12, 1 - 1e-12)
        losses.append(-np.mean(np.where(y[inner_val] == 1, np.log(p), np.log(1 - p))))
    best_c = cfg_bt["c_grid"][int(np.nanargmin(losses))]
    model = LogisticRegression(C=best_c, max_iter=cfg_bt["max_iter"]).fit(d, y)
    return model, mean, std, best_c


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "config.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    out = ROOT / cfg["out_dir"]
    bt = cfg["bt_probe"]

    features, index, layer, pooling = load_layer(cfg, out)
    rows, k1, k2 = verdict_diffs(cfg, features, index)
    print(f"layer {layer} {pooling}, {len(rows)} verdicts, {len(index)} entities")

    acc, acc_gain, per_judge, per_judge_gain, cos = cv_accuracy(
        rows, k1, k2, features, bt)
    print(f"\nshared direction: held-out acc {acc:.3f}  "
          f"+judge gain/bias {acc_gain:.3f}  fold-cosine {cos:.3f}")
    print("per judge (shared / +gain):")
    for j in sorted(per_judge):
        print(f"  {j:>22}: {per_judge[j]:.3f} / {per_judge_gain[j]:.3f}")

    rng = np.random.default_rng(bt["seed"])
    nulls = [cv_accuracy(rows, k1, k2, features, bt, rng)[0]
             for _ in range(bt["n_shuffles"])]
    print(f"label-shuffle null: {[round(a, 3) for a in nulls]}")

    qrows = rows[rows["judge"] == "qwen3-32b"].reset_index(drop=True)
    qk1 = np.array([index[(t, i)] for t, i in zip(qrows["text_id_slot1"], qrows["item"])])
    qk2 = np.array([index[(t, i)] for t, i in zip(qrows["text_id_slot2"], qrows["item"])])
    qacc, _, _, _, qcos = cv_accuracy(qrows, qk1, qk2, features, bt)
    print(f"\nqwen-only fit: held-out acc {qacc:.3f}  fold-cosine {qcos:.3f}")

    model, mean, std, best_c = full_fit(rows, k1, k2, features, bt)
    qmodel, _, _, _ = full_fit(qrows, qk1, qk2, features, bt)
    w = model.coef_[0] / np.linalg.norm(model.coef_[0])
    qw = qmodel.coef_[0] / np.linalg.norm(qmodel.coef_[0])
    print(f"angle(shared, qwen-only): cosine {float(w @ qw):.3f}")

    elo_dir = np.load(out / "probe_directions.npz")
    meta = json.loads((out / "acts_qwen3-32b.meta.json").read_text())
    li = meta["layers"].index(layer)
    pi = meta["poolings"].index(pooling)
    w_elo = elo_dir["directions"][li, pi]
    w_elo = w_elo / np.linalg.norm(w_elo)
    print(f"angle(shared BT, elo probe): cosine {float(w @ w_elo):.3f}")

    np.savez(out / "bt_direction.npz", w=model.coef_[0], w_qwen=qmodel.coef_[0],
             mean=mean, std=std, intercept=model.intercept_[0],
             layer=layer, pooling=pooling, c=best_c)
    print(f"-> {out / 'bt_direction.npz'}")


if __name__ == "__main__":
    main()
