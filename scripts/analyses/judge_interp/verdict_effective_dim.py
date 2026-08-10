"""How many directions carry the verdict signal.

Removes verdict-predictive directions one at a time: fit the BT
direction on the training folds, project it out of the features, refit
on what is left, repeat. Held-out accuracy after each removal traces
how redundantly the signal is encoded - a signal confined to one
direction dies after one removal; a redundant one survives many.

Directions are refit inside each fold, so removal never uses test
verdicts.

Writes out_dir/verdict_effective_dim.csv.

Usage:
  uv run python scripts/analyses/judge_interp/verdict_effective_dim.py [--rounds 8]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from bt_probe import load_layer, verdict_diffs
from verdict_dimensionality import pick_c, unit

ROOT = Path(__file__).resolve().parents[3]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "config.yaml")
    ap.add_argument("--judge", default="qwen3-32b")
    ap.add_argument("--rounds", type=int, default=8, help="directions removed")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    out = ROOT / cfg["out_dir"]
    bt_cfg = cfg["bt_probe"]

    features, index, layer, pooling = load_layer(cfg, out)
    rows, k1, k2 = verdict_diffs(cfg, features, index)
    keep = (rows["judge"] == args.judge).to_numpy()
    rows_j = rows[keep].reset_index(drop=True)
    k1, k2 = k1[keep], k2[keep]
    y = rows_j["y"].to_numpy()
    items = rows_j["item"].to_numpy()
    print(f"layer {layer} {pooling}, judge {args.judge}, {len(y)} verdicts, "
          f"{args.rounds} removal rounds")

    preds = np.full((args.rounds + 1, len(y)), np.nan)
    for train, test in GroupKFold(bt_cfg["cv_folds"]).split(y, y, items):
        ent = np.unique(np.concatenate([k1[train], k2[train]]))
        mean, std = features[ent].mean(0), features[ent].std(0) + 1e-8
        z = (features - mean) / std
        d = z[k1] - z[k2]
        for r in range(args.rounds + 1):
            c = pick_c(d[train], y[train], items[train], bt_cfg["c_grid"],
                       bt_cfg["cv_folds"], bt_cfg["max_iter"])
            model = LogisticRegression(C=c, max_iter=bt_cfg["max_iter"]).fit(
                d[train], y[train])
            preds[r][test] = model.predict_proba(d[test])[:, 1]
            w = unit(model.coef_[0])           # this fold's best direction
            d = d - np.outer(d @ w, w)         # ... removed before the next round
    accs = [float(((p > .5) == (y == 1)).mean()) for p in preds]
    for r, a in enumerate(accs):
        print(f"{r} directions removed: {a:.3f}")
    pd.DataFrame(dict(removed=range(len(accs)), accuracy=accs)).to_csv(
        out / "verdict_effective_dim.csv", index=False)
    print(f"-> {out / 'verdict_effective_dim.csv'}")


if __name__ == "__main__":
    main()
