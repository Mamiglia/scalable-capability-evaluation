"""Held-out decomposition of inherited versus new cross-layer information."""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from token_manifold_discover import grouped_code_predictions


def decompose(early, late, gap, item, cv_folds=5, bridge_alpha=1.0):
    """Map early→late, then decode gap from inherited and residual late codes."""
    rows = []
    splits = GroupKFold(n_splits=min(cv_folds, len(np.unique(item)))).split(
        early, gap, item)
    for fold, (train, test) in enumerate(splits):
        early_scale = StandardScaler().fit(early[train])
        late_scale = StandardScaler().fit(late[train])
        x_train, x_test = early_scale.transform(early[train]), early_scale.transform(early[test])
        y_train, y_test = late_scale.transform(late[train]), late_scale.transform(late[test])
        bridge = Ridge(alpha=bridge_alpha).fit(x_train, y_train)
        inherited_train, inherited_test = bridge.predict(x_train), bridge.predict(x_test)
        novel_train, novel_test = y_train - inherited_train, y_test - inherited_test
        representations = {
            "early": (x_train, x_test),
            "late": (y_train, y_test),
            "inherited": (inherited_train, inherited_test),
            "novel_residual": (novel_train, novel_test),
            "early_plus_novel": (np.column_stack([x_train, novel_train]),
                                  np.column_stack([x_test, novel_test])),
        }
        metrics = {}
        for name, (train_code, test_code) in representations.items():
            pred, _ = grouped_code_predictions(
                train_code, gap[train], item[train], test_code,
                alphas=(0.001, 0.01, 0.1, 1, 10, 100), cv_folds=min(4, cv_folds))
            metrics[f"gap_{name}"] = r2_score(gap[test], pred)
        rows.append({"fold": fold,
                     "map_late_r2": r2_score(
                         y_test, inherited_test, multioutput="variance_weighted"),
                     "novel_gain": (metrics["gap_early_plus_novel"]
                                    - metrics["gap_early"]),
                     **metrics})
    return pd.DataFrame(rows)
