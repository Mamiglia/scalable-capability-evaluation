"""Ridge probes from judge activations to elo.

For every (layer, pooling) in the consolidated array: z-score features
on the training folds, ridge-regress onto elo, CV grouped by item with
alpha picked on an inner grouped split. Reports held-out correlation
per depth (the where-is-preference-computed curve), the direction
stability gate (cosine of per-fold directions), and the within-judge
gate (probe-score difference predicting qwen3-32b's own verdicts,
benchmark: pooled elo itself manages 0.77).

Writes out_dir/probe_metrics.csv and out_dir/probe_directions.npz
(full-data direction + feature stats per combination, standardized
space).

Usage:
  uv run python scripts/analyses/judge_interp/probes.py
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


def load_aligned(cfg, out: Path):
    """(acts memmap, elo rows filtered to recorded entities, meta)."""
    meta = json.loads((out / "acts_qwen3-32b.meta.json").read_text())
    acts = np.load(out / "acts_qwen3-32b.npy", mmap_mode="r")
    elo = pd.read_csv(out / "elo.csv")
    dropped = set(meta["dropped_entities"])
    keep = [f"{t}_{i}" not in dropped
            for t, i in zip(elo["text_id"], elo["item"])]
    elo = elo[keep].reset_index(drop=True)
    assert len(elo) == acts.shape[0], "acts rows do not match filtered elo"
    return acts, elo, meta


def fit_one(features, y, groups, alpha_grid, n_folds, seed):
    """Grouped-CV ridge: held-out predictions, per-fold directions."""
    preds = np.full(len(y), np.nan)
    directions, alphas = [], []
    for train, test in GroupKFold(n_folds).split(features, y, groups):
        inner = GroupKFold(n_folds).split(features[train], y[train], groups[train])
        fit_idx, val_idx = next(inner)
        mean = features[train][fit_idx].mean(0)
        std = features[train][fit_idx].std(0) + 1e-8
        losses = []
        for alpha in alpha_grid:
            model = Ridge(alpha).fit((features[train][fit_idx] - mean) / std,
                                     y[train][fit_idx])
            pred = model.predict((features[train][val_idx] - mean) / std)
            losses.append(np.mean((pred - y[train][val_idx]) ** 2))
        best = alpha_grid[int(np.nanargmin(losses))]
        mean = features[train].mean(0)
        std = features[train].std(0) + 1e-8
        model = Ridge(best).fit((features[train] - mean) / std, y[train])
        preds[test] = model.predict((features[test] - mean) / std)
        directions.append(model.coef_ / np.linalg.norm(model.coef_))
        alphas.append(best)
    cosines = [float(a @ b) for k, a in enumerate(directions)
               for b in directions[k + 1:]]
    return preds, float(np.mean(cosines)), alphas


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "config.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    out = ROOT / cfg["out_dir"]
    pc = cfg["probes"]

    acts, elo, meta = load_aligned(cfg, out)
    y = elo["elo"].to_numpy()
    groups = elo["item"].to_numpy()
    layers, poolings = meta["layers"], meta["poolings"]
    print(f"{acts.shape[0]} entities, {len(layers)} layers x {len(poolings)} poolings")

    rows = []
    full_directions = np.zeros((len(layers), len(poolings), acts.shape[3]), dtype=np.float32)
    full_stats = np.zeros((len(layers), len(poolings), 2, acts.shape[3]), dtype=np.float32)
    for li, layer in enumerate(layers):
        for pi, pooling in enumerate(poolings):
            features = np.asarray(acts[:, li, pi, :], dtype=np.float32)
            preds, cos, alphas = fit_one(features, y, groups, pc["alpha_grid"],
                                         pc["cv_folds"], pc["seed"])
            r = float(np.corrcoef(preds, y)[0, 1])
            r2 = 1 - float(np.mean((preds - y) ** 2)) / float(np.var(y))
            rows.append(dict(layer=layer, pooling=pooling, r=r, r2=r2,
                             fold_cosine=cos,
                             alpha=max(set(alphas), key=alphas.count)))
            # Full-data direction for downstream use (attribution, steering).
            mean, std = features.mean(0), features.std(0) + 1e-8
            model = Ridge(rows[-1]["alpha"]).fit((features - mean) / std, y)
            full_directions[li, pi] = model.coef_
            full_stats[li, pi] = np.stack([mean, std])
            print(f"layer {layer:>2} {pooling:>15}: r {r:.3f}  r2 {r2:.3f}  "
                  f"fold-cosine {cos:.3f}  alpha {rows[-1]['alpha']}", flush=True)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / "probe_metrics.csv", index=False)
    np.savez(out / "probe_directions.npz", directions=full_directions,
             stats=full_stats, layers=layers, poolings=poolings)

    best = metrics.loc[metrics["r"].idxmax()]
    print(f"\nbest: layer {int(best.layer)} {best.pooling} "
          f"(r {best.r:.3f}, fold-cosine {best.fold_cosine:.3f})")

    # Within-judge gate: does the probe score order pairs the way the
    # studied judge itself voted? Benchmark: pooled elo reaches 0.770.
    li = layers.index(int(best.layer))
    pi = poolings.index(best.pooling)
    features = np.asarray(acts[:, li, pi, :], dtype=np.float32)
    mean, std = full_stats[li, pi]
    score = (features - mean) / std @ full_directions[li, pi]
    score_of = {(t, i): s for t, i, s in zip(elo["text_id"], elo["item"], score)}
    verdicts = pd.read_parquet(ROOT / cfg["rows_parquet"])
    for judge_set, label in ((("qwen3-32b",), "within-judge (qwen3-32b)"),
                             (None, "all judges")):
        sub = verdicts if judge_set is None else verdicts[verdicts["judge"].isin(judge_set)]
        s1 = np.array([score_of.get((t, i), np.nan)
                       for t, i in zip(sub["text_id_slot1"], sub["item"])])
        s2 = np.array([score_of.get((t, i), np.nan)
                       for t, i in zip(sub["text_id_slot2"], sub["item"])])
        ok = ~np.isnan(s1) & ~np.isnan(s2) & (s1 != s2)
        acc = float(((s1 > s2)[ok] == (sub["y"].to_numpy() == 1)[ok]).mean())
        print(f"{label}: probe-score ordering matches {acc:.3f} of {ok.sum()} verdicts")
    print(f"-> {out / 'probe_metrics.csv'}, {out / 'probe_directions.npz'}")


if __name__ == "__main__":
    main()
