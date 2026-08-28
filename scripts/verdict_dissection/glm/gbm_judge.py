"""Gradient-boosted judge-verdict model on each side's own features, with a real hyperparameter search.

    uv run python scripts/verdict_dissection/glm/gbm_judge.py <inputs_dir> <fits_dir> --designs ... [--per-judge] [--gap]

Features: <feature>_left and <feature>_right for the feature set (waves.py), plus judge and benchmark as
categorical inputs (pooled model) or one model per judge (--per-judge). --gap adds the true accuracy
difference. Hyperparameters: random search over learning rate, leaves, minimum leaf size, L2,
feature subsampling and number of trees, scored by grouped (by model) inner CV on the training rows.
Writes held_out_/predictions_ files like screen.py, best_params.csv, and importance_<model>.csv:
permutation importance on the held-out rows (both sides of a feature shuffled together), per judge.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).parent))
from evaluate import report  # noqa: E402
from waves import waves  # noqa: E402

HERE = Path(__file__).resolve().parent
MODEL = "judge_verdict_from_pair"
SEARCH = {"learning_rate": [0.03, 0.05, 0.1], "max_leaf_nodes": [7, 15, 31, 63], "min_samples_leaf": [20, 50, 100, 200],
          "l2_regularization": [0.0, 1.0, 10.0], "max_features": [0.5, 0.8, 1.0], "max_iter": [100, 300, 1000]}
N_CONFIGS = 30
INNER_FOLDS = 3


def matrix(table, features, categorical):
    x = table[features].to_numpy(np.float32)
    for column in categorical:                              # appended as integer codes
        x = np.column_stack([x, pd.Categorical(table[column]).codes.astype(np.float32)])
    return x


def fit(x, y, params, n_categorical, seed=0):
    categorical = np.zeros(x.shape[1], bool)
    if n_categorical:
        categorical[-n_categorical:] = True
    return HistGradientBoostingClassifier(**params, categorical_features=categorical, random_state=seed).fit(x, y)


def search(x, y, groups, n_categorical, rng):
    """Best configuration by grouped inner CV accuracy."""
    best, best_score = None, -np.inf
    for _ in range(N_CONFIGS):
        params = {k: rng.choice(v).item() for k, v in SEARCH.items()}
        scores = []
        for train, check in GroupKFold(INNER_FOLDS).split(x, y, groups):
            scores.append(np.mean(fit(x[train], y[train], params, n_categorical).predict(x[check]) == y[check]))
        if np.mean(scores) > best_score:
            best, best_score = params, np.mean(scores)
    return best, best_score


def side_columns(table, feature_names):
    """<feature>_left and _right where the table has them; pair-level features (no sides) enter once."""
    sided = [f for f in feature_names if f"{f}_left" in table]
    return [f"{f}_{side}" for side in ("left", "right") for f in sided] + [f for f in feature_names if f not in sided]


def permutation_importance(model, x, y, judges, feature_names, features, rng):
    """Accuracy lost per feature when its columns (left and right together, or the pair-level column) are shuffled, per judge and overall."""
    base = model.predict(x) == y
    rows = []
    for name in feature_names:
        columns = [features.index(c) for c in (f"{name}_left", f"{name}_right", name) if c in features]
        shuffled = x.copy()
        order = rng.permutation(len(x))
        shuffled[:, columns] = x[order][:, columns]
        hit = model.predict(shuffled) == y
        rows.append({"feature": name, "all": base.mean() - hit.mean(),
                     **{j: base[judges == j].mean() - hit[judges == j].mean() for j in np.unique(judges)}})
    return pd.DataFrame(rows).sort_values("all", ascending=False)


def run(table, feature_names, out_dir, per_judge, use_gap, seed=0):
    rng = np.random.default_rng(seed)
    features = side_columns(table, feature_names) + (["accuracy_difference"] if use_gap else [])
    categorical = ["bench"] if per_judge else ["judge", "bench"]
    train, test = table[~table.test], table[table.test]
    predicted, params_rows = np.zeros(len(test)), []
    for judge in (sorted(table.judge.unique()) if per_judge else [None]):
        tr = train if judge is None else train[train.judge == judge]
        te_mask = np.ones(len(test), bool) if judge is None else (test.judge == judge).to_numpy()
        x, y = matrix(tr, features, categorical), tr.pick_left.to_numpy()
        params, cv = search(x, y, tr.model_left.to_numpy(), len(categorical), rng)
        model = fit(x, y, params, len(categorical))
        x_test = matrix(test[te_mask], features, categorical)
        predicted[te_mask] = model.predict_proba(x_test)[:, 1]
        params_rows.append({"judge": judge or "pooled", "inner_cv_accuracy": cv, **params})
        if judge is None:
            importance = permutation_importance(model, x_test, test.pick_left.to_numpy(), test.judge.to_numpy(), feature_names, features, rng)
    observed = test.pick_left.to_numpy()
    out_dir.mkdir(parents=True, exist_ok=True)
    groups = {label: (test.judge == label).to_numpy() for label in sorted(test.judge.unique())}
    scores = report(observed, predicted, groups, gaussian=False)
    scores.to_csv(out_dir / f"held_out_{MODEL}.csv", index=False)
    test[["bench", "item", "judge", "model_left", "model_right"]].assign(observed=observed, predicted=predicted).to_csv(
        out_dir / f"predictions_{MODEL}.csv", index=False)
    pd.DataFrame(params_rows).to_csv(out_dir / "best_params.csv", index=False)
    if not per_judge:
        importance.to_csv(out_dir / f"importance_{MODEL}.csv", index=False)
    return scores.loc[0, "accuracy"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs_dir", type=Path)
    parser.add_argument("fits_dir", type=Path)
    parser.add_argument("--designs", nargs="+", required=True)
    parser.add_argument("--wave", default="surface+sae+surprisal")
    parser.add_argument("--per-judge", action="store_true")
    parser.add_argument("--gap", action="store_true")
    parser.add_argument("--predictors", type=Path, default=HERE / "predictors.yaml", help="predictor list (predictors-structure.yaml for the structure waves)")
    args = parser.parse_args()
    feature_names = waves(yaml.safe_load(args.predictors.read_text()))[args.wave]
    name = args.wave + ("+accuracy-gap" if args.gap else "") + "+sides+gbm" + ("+per-judge" if args.per_judge else "")
    for design in args.designs:
        table = pd.read_parquet(args.inputs_dir / design / f"{MODEL}.parquet")
        score = run(table, feature_names, args.fits_dir / MODEL / name / design, args.per_judge, args.gap)
        print(f"{MODEL} {name} {design}: {score:.3f}", flush=True)


if __name__ == "__main__":
    main()
