import numpy as np
import pandas as pd
import pytest

from capability import evaluate, task
from evals.responses import guess_rate
from capability.evaluate import score
from reference import freeze, tables
from reference.freeze import local_fit, split_items
from reference.tables import load_semesters
from types import SimpleNamespace


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
    assert score(truth + 1, truth)["mse"] == 1 and score(truth, truth)["n"] == 4
    sem = load_semesters()
    assert len(sem) == 6 and all(len(v) == 8 for v in sem.values())


def test_evaluate_on_frozen_tables(tmp_path, monkeypatch):
    monkeypatch.setattr(tables, "META", tmp_path)
    monkeypatch.setattr(tables, "load_semesters", lambda: {"s1": list("abcd"), "s2": list("efgh")})
    d = tmp_path / "toy"
    d.mkdir()
    models = pd.DataFrame({"model": list("abcdefgh"), "semester": ["s1"] * 4 + ["s2"] * 4, "global_theta": np.arange(8.0),
                           "global_theta_sd": 0.1, "path_b": np.nan, "s2_role": ["reference"] * 4 + ["target"] * 4,
                           "s2_theta": np.arange(8.0), "s2_theta_sd": 0.1})
    items = _items().assign(a_sd=0.1, b_sd=0.1, s2_set=["easy"] * 4 + ["hard"] * 4, s2_train=[True] * 4 + [False] * 4)
    no_logs = pd.DataFrame(columns=["bench", "model", "repeat", "path"])
    for table, rows in tables.to_long(models.set_index("model"), items, _resp().iloc[:0], no_logs).items():
        tables.write(table, "toy", rows)
    domain = type("D", (), {"name": "toy", "source": ""})()
    assert evaluate.semester_names(domain) == ["s2"]
    t = task.Task.from_semester("s2", domain)
    assert t.references.models.index.tolist() == list("abcd") and t.test.models.index.tolist() == list("efgh")
    assert t.references.theta.tolist() == [0, 1, 2, 3] and t.test.theta.isna().all()  # targets' theta hidden
    assert t.truth.tolist() == [4, 5, 6, 7] and len(t.references.items) == 4
    assert t.references.transcripts == {m: {} for m in "abcd"}  # never run on bench b
    truth = models.set_index("model").s2_theta
    monkeypatch.setattr(evaluate, "RESULTS", tmp_path)
    table = evaluate.evaluate(domain, lambda t: truth[t.test.models.index], "toy")
    assert table.semester.tolist() == ["s2", "all"] and table.mse.lt(1e-9).all()
    assert (d / "eval_toy.csv").exists()


def test_guess_rate():
    assert guess_rate(SimpleNamespace(choices=["a", "b", "c", "d"], input="q")) == 0.25
    hle = "Which?\n\nAnswer Choices:\nA. x\nB. y\nC. z\nD. w\nE. v"
    assert guess_rate(SimpleNamespace(choices=None, input=hle)) == 0.2
    assert guess_rate(SimpleNamespace(choices=None, input="What is 2+2?")) == 0.0


def test_build_writes_the_frozen_tables(tmp_path, monkeypatch):
    """End to end on a toy domain: two semesters, one task, every frozen column present."""
    # x solves all; y items < 9; w, z items < 6 -> top = x, y: EASY = items 0-8, HARD = 9-11
    rows = [dict(bench="b", model=m, item=str(i), repeat=r, correct=(i < 6) or (m == "x") or (m == "y" and i < 9), guess=0.0)
            for m in "wxyz" for i in range(12) for r in (0, 1)]
    logs = pd.DataFrame([dict(bench="b", model=m, repeat=r, path=f"data/b/h/{m}/{r}/run.eval") for m in "wxyz" for r in (0, 1)])
    monkeypatch.setattr(tables, "META", tmp_path)
    monkeypatch.setattr(freeze, "read_responses", lambda domain: pd.DataFrame(rows))
    monkeypatch.setattr(freeze, "eval_logs", lambda domain: logs)
    monkeypatch.setattr(freeze, "load_semesters", lambda: {"s1": ["x", "y"], "s2": ["w", "z", "missing"]})
    domain = type("D", (), {"name": "toy", "source": ""})()
    freeze.build(domain, train=1, warmup=50, samples=50)

    models, items = tables.load(domain)
    assert models.s2_role.to_dict() == {"w": "target", "x": "reference", "y": "reference", "z": "target"}
    assert models.path_b.tolist() == ["b/h/w/0/run.eval", "b/h/x/0/run.eval", "b/h/y/0/run.eval", "b/h/z/0/run.eval"]
    assert set(items.s2_set) == {"easy", "hard"} and items.s2_train.sum() <= 4 and items[items.s2_train].s2_set.eq("easy").all()
    assert {"semester", "global_theta", "s2_theta", "s2_theta_sd"} <= set(models.columns)
    assert (tmp_path / "samples.csv").exists() and (tmp_path / "toy" / "fit.yml").exists()
    assert tables.load_responses(domain).sort_values(["model", "item", "repeat"]).correct.tolist() == \
        pd.DataFrame(rows).sort_values(["model", "item", "repeat"]).correct.tolist()
    # frozen and unchanged: a second build is a no-op; a new log or a roster edit demands --force
    freeze.build(domain)
    assert freeze.changes(domain) == []
    monkeypatch.setattr(freeze, "eval_logs", lambda domain: pd.concat([logs, pd.DataFrame([dict(bench="b", model="w", repeat=2, path="data/b/h/w/2/run.eval")])]))
    assert freeze.changes(domain) == ["new logs on (bench, model): [('b', 'w')]"]
    monkeypatch.setattr(freeze, "load_semesters", lambda: {"s1": ["x", "y", "z"], "s2": ["w", "v"]})  # v uncollected: ignored
    assert freeze.changes(domain)[-1] == "roster (configs/semesters.yml) changed for: ['z']"
    with pytest.raises(FileExistsError):
        freeze.build(domain)
