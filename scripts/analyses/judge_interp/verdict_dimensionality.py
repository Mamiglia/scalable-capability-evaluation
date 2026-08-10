"""How much of a judge's verdict is the axis, and how much is the rest?

Predicts verdicts from the activation difference of the two compared
transcripts, three ways at the same layer: the projection onto one
axis alone (1 feature), the full difference vector, and the difference
with that axis projected out. If the axis carried everything the judge
uses, the orthogonal complement would predict at chance; whatever it
predicts above chance is verdict information the axis misses.

Every axis is refit inside each training fold - a direction fit on all
the data would have seen the test verdicts, and its 1-D model then
beats the full model.

Writes out_dir/verdict_dimensionality.csv.

Usage:
  uv run python scripts/analyses/judge_interp/verdict_dimensionality.py [--judge qwen3-32b]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold

from bt_probe import load_layer, verdict_diffs

ROOT = Path(__file__).resolve().parents[3]


def unit(v):
    return v / np.linalg.norm(v)


def pick_c(x, y, items, c_grid, folds, max_iter):
    inner_tr, inner_val = next(GroupKFold(folds).split(x, y, items))
    losses = []
    for c in c_grid:
        m = LogisticRegression(C=c, max_iter=max_iter).fit(x[inner_tr], y[inner_tr])
        p = np.clip(m.predict_proba(x[inner_val])[:, 1], 1e-9, 1 - 1e-9)
        losses.append(-np.mean(np.where(y[inner_val] == 1, np.log(p), np.log1p(-p))))
    return c_grid[int(np.nanargmin(losses))]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "config.yaml")
    ap.add_argument("--judge", default="qwen3-32b", help="judge whose verdicts are modelled")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    out = ROOT / cfg["out_dir"]
    bt_cfg, pr_cfg = cfg["bt_probe"], cfg["probes"]

    features, index, layer, pooling = load_layer(cfg, out)
    rows, k1, k2 = verdict_diffs(cfg, features, index)
    keep = (rows["judge"] == args.judge).to_numpy()
    rows_j = rows[keep].reset_index(drop=True)
    k1, k2 = k1[keep], k2[keep]
    y = rows_j["y"].to_numpy()
    items = rows_j["item"].to_numpy()
    print(f"layer {layer} {pooling}, judge {args.judge}, {len(y)} verdicts")

    # Capability target for the fold-wise v_cap fit (item-centered logit).
    elo = pd.read_csv(out / "elo.csv")
    capability = pd.read_csv(ROOT / cfg["models_csv"]).set_index("model")["logit"]
    cap_all = elo["model"].map(capability).to_numpy()
    cap_all = cap_all - pd.Series(cap_all).groupby(elo["item"]).transform("mean").to_numpy()
    cap_by_entity = {(t, i): c for t, i, c in zip(elo["text_id"], elo["item"], cap_all)}
    cap = np.array([cap_by_entity[(t, i)] for t, i in
                    zip(elo["text_id"], elo["item"])])[:len(features)]

    axis_names = ["approval_judge", "v_pref_judge", "v_cap"]
    preds = {("full", ""): np.full(len(y), np.nan)}
    for name in axis_names:
        preds[("axis_only", name)] = np.full(len(y), np.nan)
        preds[("axis_removed", name)] = np.full(len(y), np.nan)

    for train, test in GroupKFold(bt_cfg["cv_folds"]).split(y, y, items):
        ent = np.unique(np.concatenate([k1[train], k2[train]]))
        mean, std = features[ent].mean(0), features[ent].std(0) + 1e-8
        z = (features - mean) / std
        d = z[k1] - z[k2]

        # Directions from the training verdicts / training entities only.
        c = pick_c(d[train], y[train], items[train], bt_cfg["c_grid"],
                   bt_cfg["cv_folds"], bt_cfg["max_iter"])
        w_judge = unit(LogisticRegression(C=c, max_iter=bt_cfg["max_iter"])
                       .fit(d[train], y[train]).coef_[0])
        v_cap = unit(Ridge(pr_cfg["alpha_grid"][-1]).fit(z[ent], cap[ent]).coef_)
        axes = {"approval_judge": w_judge,
                "v_pref_judge": unit(w_judge - (v_cap @ w_judge) * v_cap),
                "v_cap": v_cap}

        def fit_predict(x):
            cc = pick_c(x[train], y[train], items[train], bt_cfg["c_grid"],
                        bt_cfg["cv_folds"], bt_cfg["max_iter"])
            m = LogisticRegression(C=cc, max_iter=bt_cfg["max_iter"]).fit(
                x[train], y[train])
            return m.predict_proba(x[test])[:, 1]

        preds[("full", "")][test] = fit_predict(d)
        for name, ax in axes.items():
            preds[("axis_only", name)][test] = fit_predict((d @ ax)[:, None])
            preds[("axis_removed", name)][test] = fit_predict(d - np.outer(d @ ax, ax))

    acc = {k: float(((v > .5) == (y == 1)).mean()) for k, v in preds.items()}
    full = acc[("full", "")]
    print(f"\nfull difference vector ({features.shape[1]} dims): {full:.3f}")
    results = [dict(model="full", axis="", accuracy=full)]
    for name in axis_names:
        one, rest = acc[("axis_only", name)], acc[("axis_removed", name)]
        print(f"{name:16s} alone {one:.3f} | complement {rest:.3f} | "
              f"axis captures {(one - .5) / (full - .5):.0%} of the full edge, "
              f"complement keeps {(rest - .5) / (full - .5):.0%}")
        results += [dict(model="axis_only", axis=name, accuracy=one),
                    dict(model="axis_removed", axis=name, accuracy=rest)]

    path = out / "verdict_dimensionality.csv"
    pd.DataFrame(results).to_csv(path, index=False)
    print(f"-> {path}")


if __name__ == "__main__":
    main()
