"""Simple verdict predictors on the embeddings: logistic regression + GBM.

For each CV regime and fold, fits on out-of-fold rows and writes held-out
predictions to preds/{regime}_{fold}.parquet (same format the MLP will
use), plus a console summary per regime.

Feature per row: delta = embedding(slot 1) - embedding(slot 2), plus a
judge one-hot. Logistic regression on this is the linear twin model:
theta(t) = w.e(t) with per-judge position offsets.

Usage:
  uv run python scripts/analyses/score_prediction/ml_suite.py
"""

import argparse
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from cv_folds import grouped_folds

ROOT = Path(__file__).resolve().parents[3]


def load_features(out_dir: Path):
    """(rows, e1, e2, judge_names, judge_idx, y), row-aligned."""
    rows = pd.read_parquet(out_dir / "rows.parquet")
    emb = pd.read_parquet(out_dir / "embeddings.parquet")
    vectors = np.stack(emb["embedding"].to_numpy()).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)  # MPS norm drift
    index = {tid: i for i, tid in enumerate(emb["text_id"])}
    missing = (set(rows["text_id_slot1"]) | set(rows["text_id_slot2"])) - index.keys()
    assert not missing, f"{len(missing)} text_ids lack embeddings"

    # One vector per unique transcript; each verdict row looks up two.
    e1 = vectors[rows["text_id_slot1"].map(index).to_numpy()]
    e2 = vectors[rows["text_id_slot2"].map(index).to_numpy()]
    judge_names = sorted(rows["judge"].unique())
    judge_idx = rows["judge"].map({j: i for i, j in enumerate(judge_names)}).to_numpy()
    y = rows["y"].to_numpy()
    print(f"{len(rows)} rows, embeddings {e1.shape}, {len(judge_names)} judges")
    return rows, e1, e2, judge_names, judge_idx, y


def merge_preds(path: Path, frames: list[pd.DataFrame]):
    """Write these models' rows, replacing prior rows of the same models."""
    new = pd.concat(frames, ignore_index=True)
    if path.exists():
        old = pd.read_parquet(path)
        new = pd.concat([old[~old["model"].isin(new["model"].unique())], new],
                        ignore_index=True)
    new.to_parquet(path, index=False)


def grid_options(grid: dict) -> list[dict]:
    """Config block -> list of param dicts: the product of *_grid entries,
    fixed keys carried into every option."""
    fixed = {k: v for k, v in grid.items() if not k.endswith("_grid")}
    swept = {k[: -len("_grid")]: v for k, v in grid.items() if k.endswith("_grid")}
    return [dict(fixed, **dict(zip(swept, combo)))
            for combo in product(*swept.values())]


def fit_logreg(features, y, params: dict, seed: int):
    """Linear twin model: one weight per embedding dim, judge dummies as
    the only intercepts (each dummy coefficient is that judge's b_j)."""
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=params["l2_c"], max_iter=params["max_iter"],
                              fit_intercept=False, random_state=seed).fit(features, y)


def fit_gbm(features, y, params: dict, seed: int):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(
        learning_rate=params["learning_rate"], max_leaf_nodes=params["max_leaf_nodes"],
        max_iter=params["max_iter"], early_stopping=True,
        random_state=seed).fit(features, y)


def log_loss(y, p) -> float:
    # float64: in float32 the clip bound rounds to 1.0 and a saturated
    # prediction becomes 0 * -inf = NaN.
    y = np.asarray(y, dtype=np.float64)
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def tune_and_fit(fit_fn, grid: dict, features, y, val_mask, seed: int):
    """Pick the grid point by inner-validation log-loss, refit on all rows."""
    options = grid_options(grid)
    losses = []
    for params in options:
        model = fit_fn(features[~val_mask], y[~val_mask], params, seed)
        losses.append(log_loss(y[val_mask], model.predict_proba(features[val_mask])[:, 1]))
    # nanargmin: a diverged fit's NaN loss must not win the selection.
    best = options[int(np.nanargmin(losses))]
    # Refit the winning option on all rows (train + inner val).
    return fit_fn(features, y, best, seed), best, float(min(losses))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "config.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    out = ROOT / cfg["out_dir"]

    rows, e1, e2, judge_names, judge_idx, y = load_features(out)
    # Feature matrix: embedding difference (content) + judge one-hot (in
    # the logreg, each dummy's coefficient is that judge's position bias).
    delta = e1 - e2
    judge_onehot = np.eye(len(judge_names), dtype=np.float32)[judge_idx]
    features = np.hstack([delta, judge_onehot])
    folds = pd.read_parquet(out / "cv_folds.parquet")
    fold_of_row = folds.pivot(index="row_id", columns="regime", values="fold")
    preds_dir = out / "preds"
    preds_dir.mkdir(exist_ok=True)
    seed = cfg["seeds"][0]
    group_cols = {"items": "item", "pairs": "pair_key", "judges": "judge"}
    fitters = {"logreg": (fit_logreg, cfg["ml_suite"]["logreg"]),
               "gbm": (fit_gbm, cfg["ml_suite"]["gbm"])}

    for regime, group_col in group_cols.items():
        regime_fold = rows["row_id"].map(fold_of_row[regime]).to_numpy()
        for fold in sorted(np.unique(regime_fold)):
            test_mask = regime_fold == fold
            train_rows = rows[~test_mask]
            inner = grouped_folds(train_rows, group_col, cfg["inner_folds"], seed)
            val_mask = (inner == 0).to_numpy()
            frames = []
            for name, (fit_fn, grid) in fitters.items():
                model, best, _ = tune_and_fit(fit_fn, grid, features[~test_mask],
                                              y[~test_mask], val_mask, seed)
                p = model.predict_proba(features[test_mask])[:, 1]
                frames.append(pd.DataFrame(dict(
                    row_id=rows.loc[test_mask, "row_id"].values, model=name,
                    regime=regime, fold=fold, p_slot1=p, y=y[test_mask])))
                print(f"{regime} fold {fold} {name}: "
                      f"test nll {log_loss(y[test_mask], p):.4f} "
                      f"acc {(np.round(p) == y[test_mask]).mean():.3f} chose {best}",
                      flush=True)
            merge_preds(preds_dir / f"{regime}_{fold}.parquet", frames)

    all_preds = pd.concat([pd.read_parquet(f)
                           for f in sorted(preds_dir.glob("*.parquet"))])
    summary = (all_preds.groupby(["regime", "model"])
               .apply(lambda g: pd.Series(dict(
                   nll=log_loss(g["y"], g["p_slot1"]),
                   acc=float((g["p_slot1"].round() == g["y"]).mean()),
                   n=len(g)))))
    print(f"\npooled held-out metrics (always-0.5 reference nll {np.log(2):.4f}):")
    print(summary.round(4).to_string())
    print(f"-> {preds_dir}/")


if __name__ == "__main__":
    main()