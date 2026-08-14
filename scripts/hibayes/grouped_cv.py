"""Does out-of-sample accuracy hold up when whole groups are held out?

The leave-one-out figures reported by the fitting package hold out a single
comparison row. A question appears many times across judges, presentation orders
and length conditions, so near-duplicates of a held-out row stay in the training
set and the estimate can flatter itself.

This cross-validates each fit's own specification three ways: random row folds,
folds holding out whole questions, and folds holding out whole model pairs. The
last asks whether the relationship transfers to a model comparison never seen
during fitting.

Each specification is reproduced exactly as the corresponding Bayesian fit
defines it, because they do not share a target or a row set:

  headline    predicts chosen_left on gpqa_glm.csv, position term major_is_left
  marginal    predicts is_correct on the features_within complete cases,
              position term major_is_right

An earlier version of this script used chosen_left for both and drew from
gpqa_correctness.csv throughout. Those numbers were not comparable to either fit
and are not reported.

The estimator here is a plain logistic regression with judge indicators, not the
hierarchical Bayesian model. Absolute accuracy is therefore not comparable to
the package's PSIS-LOO figures. The comparison between the three
cross-validation schemes within one specification is the point.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, KFold

from plotting import HIB

SEED = 0
FOLDS = 5

SPECS = {
    "headline: four predictors": dict(
        csv=HIB / "gpqa_glm.csv",
        target="chosen_left",
        position="major_is_left",
        feats=["length_difference_sd", "capability_difference_sd",
               "same_model_family", "self_comparison"],
    ),
    "marginal: log word count": dict(
        csv=HIB / "features_within" / "complete_cases.csv",
        target="is_correct",
        position="major_is_right",
        feats=["d_log_n_words_within"],
    ),
}


def evaluate(spec, name):
    need = ["judge", "item", "model_pair", spec["target"], spec["position"]] + spec["feats"]
    head = pd.read_csv(spec["csv"], nrows=0).columns
    missing = [c for c in need if c not in head]
    if missing:
        print(f"\n{name}: skipped, {spec['csv'].name} is missing {missing}")
        return
    d = pd.read_csv(spec["csv"], usecols=need, low_memory=False).dropna()

    X = pd.get_dummies(d["judge"], prefix="j").astype(float)
    X[spec["position"]] = d[spec["position"]].astype(float)
    for f in spec["feats"]:
        X[f] = d[f].astype(float)
    X = X.to_numpy()
    y = d[spec["target"]].astype(int).to_numpy()

    schemes = [
        ("random rows", KFold(FOLDS, shuffle=True, random_state=SEED), None),
        ("hold out questions", GroupKFold(FOLDS), d["item"].to_numpy()),
        ("hold out model pairs", GroupKFold(FOLDS), d["model_pair"].to_numpy()),
    ]
    print(f"\n{name}")
    print(f"    target {spec['target']}   rows {len(d):,}   "
          f"pairs {d['model_pair'].nunique()}   questions {d['item'].nunique():,}")
    for label, cv, groups in schemes:
        accs = []
        split = cv.split(X, y, groups) if groups is not None else cv.split(X, y)
        for tr, te in split:
            m = LogisticRegression(max_iter=5000, C=1.0)
            m.fit(X[tr], y[tr])
            accs.append(float((m.predict(X[te]) == y[te]).mean()))
        print(f"    {label:<22} {np.mean(accs):.4f}  (fold sd {np.std(accs):.4f})")


def main():
    for name, spec in SPECS.items():
        evaluate(spec, name)


if __name__ == "__main__":
    main()
