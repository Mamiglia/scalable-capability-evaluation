"""Item-grouped post-hoc probes for capability, length, position, and errors."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from token_manifold_discover import grouped_code_predictions


def _binary_predictions(X_train, y_train, item_train, X_test, cv_folds):
    splits = list(GroupKFold(
        n_splits=min(cv_folds, len(np.unique(item_train)))).split(
            X_train, y_train, item_train))
    best_c, best_score = 0.01, -np.inf
    for c in (0.01, 0.1, 1, 10):
        pred = np.full(len(y_train), np.nan)
        for train, valid in splits:
            scale = StandardScaler().fit(X_train[train])
            model = LogisticRegression(C=c, class_weight="balanced", max_iter=1000).fit(
                scale.transform(X_train[train]), y_train[train])
            pred[valid] = model.predict_proba(scale.transform(X_train[valid]))[:, 1]
        score = roc_auc_score(y_train, pred)
        if score > best_score:
            best_c, best_score = c, score
    scale = StandardScaler().fit(X_train)
    model = LogisticRegression(C=best_c, class_weight="balanced", max_iter=1000).fit(
        scale.transform(X_train), y_train)
    return model.predict_proba(scale.transform(X_test))[:, 1]


def probe_table(code, frame, cv_folds=5):
    """Decode external covariates from frozen anonymous codes, always out of item."""
    code = np.asarray(code)
    item = np.asarray(frame["item"])
    outer = list(GroupKFold(n_splits=min(cv_folds, len(np.unique(item)))).split(
        code, groups=item))
    rows = []
    for target in ("capability_gap", "token_length_gap"):
        if target not in frame:
            continue
        y = np.asarray(frame[target], dtype=float)
        pred = np.full(len(y), np.nan)
        for train, test in outer:
            pred[test], _ = grouped_code_predictions(
                code[train], y[train], item[train], code[test], cv_folds=min(4, cv_folds))
        rows.append({"target": target, "metric": "r2", "score": r2_score(y, pred),
                     "n": len(y)})
    for target in ("picked_slot_1", "correct", "picked_longer_weaker"):
        if target not in frame:
            continue
        y = np.asarray(frame[target], dtype=int)
        if len(np.unique(y)) < 2:
            rows.append({"target": target, "metric": "unavailable", "score": np.nan,
                         "n": len(y)})
            continue
        pred = np.full(len(y), np.nan)
        for train, test in outer:
            pred[test] = _binary_predictions(
                code[train], y[train], item[train], code[test], min(4, cv_folds))
        rows.append({"target": target, "metric": "roc_auc",
                     "score": roc_auc_score(y, pred), "n": len(y)})
    return pd.DataFrame(rows)
