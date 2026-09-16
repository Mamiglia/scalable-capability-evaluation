"""protocols.accuracy: accuracy over the task's EASY (or all) frozen items, linearly mapped to theta*."""

import numpy as np
import pandas as pd

from capability import task
from protocols import accuracy
from reference import tables


def frozen_domain(root, semester="s"):
    """items 0,1 easy and 2 hard for task s; models a,b,c,t answer them with rising accuracy."""
    directory = root / "dom"
    directory.mkdir(parents=True)
    pd.DataFrame({"bench": "b", "item": ["0", "1", "2"], f"{semester}_set": ["easy", "easy", "hard"]}).to_csv(directory / "items.csv", index=False)
    pd.DataFrame({"model": list("abct"), "path_b": "b/h/x/0/log.eval"}).to_csv(directory / "models.csv", index=False)
    rows = []
    for model, easy_right, hard_right in [("a", 0, 0), ("b", 1, 0), ("c", 2, 1), ("t", 2, 1)]:
        for i in range(3):
            right = i < easy_right if i < 2 else hard_right
            rows += [("b", str(i), model, repeat, bool(right)) for repeat in (0, 1)]
    pd.DataFrame(rows, columns=["bench", "item", "model", "repeat", "correct"]).to_csv(directory / "responses.csv", index=False)


def test_accuracy_regression_maps_easy_accuracy_to_theta(tmp_path, monkeypatch):
    monkeypatch.setattr(tables, "META", tmp_path)
    accuracy._frozen.cache_clear()
    frozen_domain(tmp_path)
    items = pd.DataFrame({"bench": "b", "item": ["0", "1"], "guess": 0.0, "a": 1.0, "b": 0.0})
    train = task.Cohort(pd.DataFrame({"theta": [0.0, 1.0, 2.0], "path_b": "b/h/x/0/log.eval"},
                                     index=pd.Index(list("abc"), name="model")), items, "s")
    test = task.Cohort(pd.DataFrame({"theta": np.nan, "path_b": "b/h/x/0/log.eval"},
                                    index=pd.Index(list("tu"), name="model")), items, "s")
    t = task.Task("s", train, task.Cohort(train.models.iloc[:0], items, "s"), test, pd.Series([2.0, 3.0], index=test.models.index))

    easy = accuracy.accuracy_table(train, "easy")
    assert np.allclose(easy.accuracy, [0.0, 0.5, 1.0])  # theta = 2 * accuracy
    assert np.allclose(accuracy.accuracy_table(train, "all").accuracy, [0.0, 1 / 3, 1.0])
    theta_hat = accuracy.PROTOCOLS["acc_easy"](t)
    assert np.isclose(theta_hat["t"], 2.0) and np.isnan(theta_hat["u"])  # u is in no frozen table: unpredicted

    only_u = task.Cohort(test.models.loc[["u"]], items, "s")
    assert accuracy.frozen_dir(only_u) is None  # no frozen table lists any of its models: no directory
