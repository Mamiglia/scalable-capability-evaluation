"""Second probe direction on the first one's residuals, as a
dimensionality control.

At the best (layer, pooling) from probe_metrics.csv: per outer fold,
fit direction 1 on elo, then direction 2 on the training residuals with
features projected orthogonal to direction 1. Reports how much held-out
correlation the second direction adds, its stability across folds, and
the verdict-ordering agreement of the combined score. Appends the
full-data second direction to probe_directions.npz as a separate file.

Usage:
  uv run python scripts/analyses/judge_interp/residual_probe.py
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[3]


def pick_alpha(features, y, groups, grid, n_folds):
    fit_idx, val_idx = next(GroupKFold(n_folds).split(features, y, groups))
    losses = [np.mean((Ridge(a).fit(features[fit_idx], y[fit_idx])
                       .predict(features[val_idx]) - y[val_idx]) ** 2)
              for a in grid]
    return grid[int(np.nanargmin(losses))]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "config.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    out = ROOT / cfg["out_dir"]
    pc = cfg["probes"]

    metrics = pd.read_csv(out / "probe_metrics.csv")
    best = metrics.loc[metrics["r"].idxmax()]
    meta = json.loads((out / "acts_qwen3-32b.meta.json").read_text())
    li = meta["layers"].index(int(best.layer))
    pi = meta["poolings"].index(best.pooling)
    acts = np.load(out / "acts_qwen3-32b.npy", mmap_mode="r")
    features_raw = np.asarray(acts[:, li, pi, :], dtype=np.float32)

    elo = pd.read_csv(out / "elo.csv")
    dropped = set(meta["dropped_entities"])
    elo = elo[[f"{t}_{i}" not in dropped
               for t, i in zip(elo["text_id"], elo["item"])]].reset_index(drop=True)
    y = elo["elo"].to_numpy()
    groups = elo["item"].to_numpy()
    print(f"layer {int(best.layer)} {best.pooling}, {len(y)} entities")

    preds1 = np.full(len(y), np.nan)
    preds12 = np.full(len(y), np.nan)
    second_dirs = []
    for train, test in GroupKFold(pc["cv_folds"]).split(features_raw, y, groups):
        mean = features_raw[train].mean(0)
        std = features_raw[train].std(0) + 1e-8
        ftr = (features_raw[train] - mean) / std
        fte = (features_raw[test] - mean) / std
        a1 = pick_alpha(ftr, y[train], groups[train], pc["alpha_grid"], pc["cv_folds"])
        m1 = Ridge(a1).fit(ftr, y[train])
        w1 = m1.coef_ / np.linalg.norm(m1.coef_)
        preds1[test] = m1.predict(fte)
        # Orthogonalize features to direction 1, fit on its residuals.
        resid = y[train] - m1.predict(ftr)
        ftr_o = ftr - np.outer(ftr @ w1, w1)
        fte_o = fte - np.outer(fte @ w1, w1)
        a2 = pick_alpha(ftr_o, resid, groups[train], pc["alpha_grid"], pc["cv_folds"])
        m2 = Ridge(a2).fit(ftr_o, resid)
        preds12[test] = preds1[test] + m2.predict(fte_o)
        second_dirs.append(m2.coef_ / np.linalg.norm(m2.coef_))

    r1 = float(np.corrcoef(preds1, y)[0, 1])
    r12 = float(np.corrcoef(preds12, y)[0, 1])
    cosines = [float(a @ b) for k, a in enumerate(second_dirs)
               for b in second_dirs[k + 1:]]
    print(f"direction 1 alone: r {r1:.3f}")
    print(f"1 + residual direction: r {r12:.3f}  (delta {r12 - r1:+.3f})")
    print(f"second-direction fold-cosine {np.mean(cosines):.3f} "
          f"(direction 1 managed {best.fold_cosine:.3f})")

    # Verdict ordering with the combined held-out score.
    score_of = {(t, i): s for t, i, s in zip(elo["text_id"], elo["item"], preds12)}
    verdicts = pd.read_parquet(ROOT / cfg["rows_parquet"])
    for judges, label in ((("qwen3-32b",), "within-judge"), (None, "all judges")):
        sub = verdicts if judges is None else verdicts[verdicts["judge"].isin(judges)]
        s1 = np.array([score_of.get((t, i), np.nan)
                       for t, i in zip(sub["text_id_slot1"], sub["item"])])
        s2 = np.array([score_of.get((t, i), np.nan)
                       for t, i in zip(sub["text_id_slot2"], sub["item"])])
        ok = ~np.isnan(s1) & ~np.isnan(s2) & (s1 != s2)
        acc = float(((s1 > s2)[ok] == (sub["y"].to_numpy() == 1)[ok]).mean())
        print(f"{label} combined-score ordering (held-out): {acc:.3f} of {ok.sum()}")

    # Full-data second direction for downstream use.
    mean, std = features_raw.mean(0), features_raw.std(0) + 1e-8
    f_all = (features_raw - mean) / std
    a1 = pick_alpha(f_all, y, groups, pc["alpha_grid"], pc["cv_folds"])
    m1 = Ridge(a1).fit(f_all, y)
    w1 = m1.coef_ / np.linalg.norm(m1.coef_)
    resid = y - m1.predict(f_all)
    f_o = f_all - np.outer(f_all @ w1, w1)
    a2 = pick_alpha(f_o, resid, groups, pc["alpha_grid"], pc["cv_folds"])
    m2 = Ridge(a2).fit(f_o, resid)
    np.savez(out / "probe_second_direction.npz",
             layer=int(best.layer), pooling=best.pooling,
             w1=w1, w2=m2.coef_, mean=mean, std=std)
    print(f"-> {out / 'probe_second_direction.npz'}")


if __name__ == "__main__":
    main()
