import numpy as np
import pandas as pd
import pytest

from capability import evaluate, semesters, task
from capability.semesters import load_semesters, local_fit, score, split_items


def _items(n=8):
    return pd.DataFrame({"bench": "b", "item": [str(i) for i in range(n)], "guess": 0.0, "a": 1.0, "b": np.arange(float(n))})


def _resp():
    # x solves everything; y and z solve items 0-3. Item 0: y fails repeat 1 (stays easy); item 1: y fails repeat 0 (hard).
    rows = [dict(bench="b", model=m, item=str(i), repeat=r, correct=i < 4 or m == "x", guess=0.0)
            for m in "xyz" for i in range(8) for r in (0, 1)]
    rows.append(dict(bench="b", model="y", item="0", repeat=1, correct=False, guess=0.0))
    rows.append(dict(bench="b", model="y", item="1", repeat=0, correct=False, guess=0.0))
    return pd.DataFrame(rows)


def test_split_repeat0_rule_and_fixed_train():
    split = split_items(_resp(), _items(), top=["x", "y"], train=1)
    assert split.set.tolist() == ["easy", "hard", "easy", "easy"] + ["hard"] * 4
    assert split.train.sum() == 3 and split[split.train].set.eq("easy").all()  # 3 easy items, 4 quartiles -> min(1, n)
    assert split.train.equals(split_items(_resp(), _items(), top=["x", "y"], train=1).train)  # seeded: fixed subset
    assert split_items(_resp(), _items(), top=["x", "y"], train=0).train.sum() == 3
    # an item a top model never answered on repeat 0 is out of the task (neither easy nor hard)
    assert split_items(_resp()[lambda d: ~((d.model == "x") & (d.item == "2"))], _items(), ["x", "y"], 0).set[2] == ""


def test_local_fit_orders_models_and_rejects_unseen():
    resp, items = _resp(), _items()
    hard = items[split_items(resp, items, top=["x", "y"], train=0).set == "hard"]
    local = local_fit(resp, hard, ["x", "y", "z"], warmup=100, samples=100)
    assert local.theta["x"] > local.theta["y"] and local.theta["x"] > local.theta["z"]
    with pytest.raises(ValueError, match="no HARD responses"):
        local_fit(resp, hard, ["x", "w"], warmup=10, samples=10)


def test_score_and_roster():
    truth = pd.Series([0.0, 1, 2, 3], index=list("abcd"))
    s = score(2 * truth + 5, truth)
    assert s["r"] > 0.999 and s["affine_mse"] < 1e-9 and s["mse"] > 1
    sem = load_semesters()
    assert len(sem) == 6 and all(len(v) == 8 for v in sem.values())


def test_evaluate_on_frozen_tables(tmp_path, monkeypatch):
    monkeypatch.setattr(semesters, "META", tmp_path)
    d = tmp_path / "toy"
    d.mkdir()
    models = pd.DataFrame({"model": list("abcdefgh"), "semester": ["s1"] * 4 + ["s2"] * 4, "global_theta": np.arange(8.0),
                           "path_b": np.nan, "s2_role": ["reference"] * 4 + ["target"] * 4, "s2_theta": np.arange(8.0)})
    models.to_csv(d / "models.csv", index=False)
    _items().assign(s2_set=["easy"] * 4 + ["hard"] * 4, s2_train=[True] * 4 + [False] * 4).to_csv(d / "items.csv", index=False)
    domain = type("D", (), {"name": "toy", "source": ""})()
    ts = list(task.tasks(domain))
    assert [t.target for t in ts] == list("efgh") and ts[0].references == list("abcd") and len(ts[0].items) == 4
    assert ts[0].transcripts("a") == {}  # never run on bench b
    truth = models.set_index("model").s2_theta
    table = evaluate.evaluate(domain, lambda t: 2 * truth[t.target] + 1, "toy")
    assert table.semester.tolist() == ["s2", "all"] and table.r.gt(0.99).all() and table.affine_mse.lt(1e-9).all()
    assert (d / "eval_toy.csv").exists()
    with pytest.raises(FileExistsError):
        semesters.build(domain)


def test_pairwise_judge_protocol():
    from protocols.judge import PairwiseJudge
    items = pd.DataFrame({"bench": "b", "item": ["0", "1"], "guess": 0.0, "a": 1.0, "b": 0.0})
    t = task.Task("s", ["a", "b", "c"], pd.Series({"a": 0.0, "b": 1.0, "c": 2.0}), "t", items, pd.DataFrame())
    # c beats b beats a on both items and orders; t beats everyone: ability(t) > ability(c) -> theta-hat > 2
    rows = [dict(bench="b", judge="j", item=i, model_left=w, model_right=l, pick_left=1.0)
            for i in ("0", "1") for w, l in [("t", "a"), ("t", "b"), ("t", "c"), ("c", "b"), ("c", "a"), ("b", "a")]]
    p = PairwiseJudge(["unused"], name="j")
    p._verdicts = pd.DataFrame(rows)
    assert p(t) > 2
    assert np.isnan(p(task.Task("s", ["a", "b", "c"], t.theta_ref, "u", items, pd.DataFrame())))  # target never judged
    assert np.isnan(semesters.score(pd.Series({"a": np.nan, "b": 1.0}), pd.Series({"a": 0.0, "b": 1.0}))["r"])
