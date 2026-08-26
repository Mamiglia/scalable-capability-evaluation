"""Gradient-boosted trees for the stronger-model task, to test whether the feature GLMs lose anything
to linearity. Same inputs as glm/screen.py (glm_inputs/<design>/stronger_model_from_pair.parquet:
left - right features, left_stronger, test) and the same feature sets as glm/waves.py, one pooled
fit per design with the benchmark as a categorical feature. Antisymmetry is enforced at prediction:
P(left stronger) = (f(x) + 1 - f(-x)) / 2. Leaves and learning rate are chosen on inner folds
grouped by item.

    uv run python scripts/baselines/gbm.py <inputs_dir> <output_dir> [--waves ...]

Writes <output_dir>/stronger_model_from_pair/gbm-<wave>/<design>/ in the GLM fits' layout (see pairs.py).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

from pairs import MODEL, ROOT, write_design

sys.path.insert(0, str(ROOT / "scripts/verdict_dissection/glm"))
from waves import waves  # noqa: E402

DEFAULT_WAVES = ("surface", "surface+sae", "surface+sae+surprisal")
LEAVES = (7, 31)
LEARNING_RATES = (0.05, 0.1)
MAX_ITER = 300
INNER_FOLDS = 3


def features(table, predictors):
    """Predictor columns plus the benchmark code as the last (categorical) column."""
    bench = pd.Categorical(table.bench).codes.astype(np.float32)[:, None]
    return np.hstack([table[predictors].to_numpy(np.float32), bench])


def negated(x):
    """The pair shown the other way round: features flipped, benchmark unchanged."""
    out = -x
    out[:, -1] = x[:, -1]
    return out


def fit(x, y, leaves, learning_rate, seed=0):
    categorical = np.zeros(x.shape[1], bool)
    categorical[-1] = True
    return HistGradientBoostingClassifier(max_leaf_nodes=leaves, learning_rate=learning_rate, max_iter=MAX_ITER,
                                          categorical_features=categorical, random_state=seed).fit(x, y)


def predict(model, x):
    return (model.predict_proba(x)[:, 1] + 1 - model.predict_proba(negated(x))[:, 1]) / 2


def fit_predict(x_train, y_train, groups_train, x_test, seed=0):
    """Antisymmetric P(left stronger) for the test rows; leaves and learning rate by inner grouped CV."""
    best, best_score = None, -1.0
    for leaves in LEAVES:
        for learning_rate in LEARNING_RATES:
            score = 0.0
            for train, check in GroupKFold(INNER_FOLDS, shuffle=True, random_state=seed).split(x_train, y_train, groups_train):
                model = fit(x_train[train], y_train[train], leaves, learning_rate, seed)
                score += np.mean((predict(model, x_train[check]) > 0.5) == y_train[check])
            if score > best_score:
                best, best_score = (leaves, learning_rate), score
    return predict(fit(x_train, y_train, *best, seed), x_test), best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--waves", nargs="+", default=list(DEFAULT_WAVES))
    args = parser.parse_args()
    predictors = yaml.safe_load((ROOT / "scripts/verdict_dissection/glm/predictors.yaml").read_text())
    feature_sets = waves(predictors)
    for design_dir in sorted(p for p in args.inputs_dir.iterdir() if p.is_dir()):
        table = pd.read_parquet(design_dir / f"{MODEL}.parquet")
        train, test = table[~table.test], table[table.test]
        for wave in args.waves:
            x = features(table, feature_sets[wave])
            predicted, chosen = fit_predict(x[~table.test], train.left_stronger.to_numpy(), train.item.to_numpy(), x[table.test])
            predictions = test[["bench", "item", "model_left", "model_right"]].assign(observed=test.left_stronger, predicted=predicted)
            scores = write_design(args.output_dir, f"gbm-{wave}", design_dir.name, predictions)
            print(f"{design_dir.name} gbm-{wave}: accuracy {scores.accuracy.iloc[0]:.3f}, leaves {chosen[0]}, "
                  f"learning rate {chosen[1]}", flush=True)


if __name__ == "__main__":
    main()
