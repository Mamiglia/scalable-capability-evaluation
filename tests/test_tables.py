import numpy as np
import pandas as pd

from reference import tables


def test_long_tables_round_trip_and_cohort_view(tmp_path, monkeypatch):
    """Wide -> long -> wide is the identity; a cohort domain swaps in its targets, blanking the roster's."""
    monkeypatch.setattr(tables, "META", tmp_path)
    monkeypatch.setitem(tables.COHORTS, "toy_extra", ("toy", "extra"))
    models = pd.DataFrame({"semester": ["s1", "s1", "s2"], "global_theta": [0., 1., 2.], "global_theta_sd": .1,
                           "path_b": ["b/r/0.eval", "b/q/0.eval", "b/t/0.eval"], "s2_role": ["reference", "reference", "target"],
                           "s2_theta": [0., 1., 2.], "s2_theta_sd": .2}, index=pd.Index(["r", "q", "t"], name="model"))
    items = pd.DataFrame({"bench": "b", "item": list("abcdef"), "guess": 0., "a": 1., "a_sd": .1, "b": np.arange(6.), "b_sd": .1,
                          "s2_set": ["easy"] * 4 + ["hard", np.nan], "s2_train": [True, False, True, False, False, False]})
    responses = pd.DataFrame([dict(bench="b", item=i, model=m, repeat=r, correct=True) for m in "rqt" for i in "abcde" for r in (0, 1)])
    logs = pd.DataFrame([dict(bench="b", model=m, repeat=1, path=f"b/{m}/1.eval") for m in "rqt"])
    long = tables.to_long(models, items, responses, logs)
    assert long["task_items"].quartile.tolist() == [0, 1, 2, 3, pd.NA]  # quartiles among EASY only
    assert long["samples"].log.notna().all()
    long["tasks"] = pd.concat([long["tasks"], pd.DataFrame([dict(semester="s2", model="x", role="target", cohort="extra", theta=9., theta_sd=.3)])])
    long["models"] = pd.concat([long["models"], pd.DataFrame([dict(model="x", semester=None, theta_global=np.nan, theta_global_sd=np.nan)])])
    for table, rows in long.items():
        tables.write(table, "toy", rows)

    wide, back = tables.load("toy")
    pd.testing.assert_frame_equal(wide.loc[models.index, models.columns], models, check_dtype=False)
    pd.testing.assert_frame_equal(back[items.columns], items, check_dtype=False)
    assert len(tables.load_responses("toy")) == len(responses)
    extra, _ = tables.load("toy_extra")
    assert extra.s2_role.dropna().to_dict() == {"q": "reference", "r": "reference", "x": "target"}
    assert extra.loc["t", "s2_theta"] == 2.  # the roster target keeps its theta*, only its role is blanked


def test_write_paper_keeps_roster_and_listed_cohort_targets(tmp_path, monkeypatch):
    monkeypatch.setattr(tables, "META", tmp_path)
    monkeypatch.setattr(tables, "PAPER", {"extra": ["x"]})
    rows = dict(semester="s", role="target", theta=0., theta_sd=.1)
    tables.write("tasks", "d", pd.DataFrame([dict(model="t", cohort="roster", **rows), dict(model="x", cohort="extra", **rows),
                                              dict(model="y", cohort="extra", **rows)]))
    tables.write("models", "d", pd.DataFrame(dict(model=["t", "x", "y", "off"], semester="s", theta_global=0., theta_global_sd=.1)))
    tables.write("samples", "d", pd.DataFrame(dict(bench="b", item="i", model=["t", "x", "y", "off"], repeat=0, correct=True, log="l")))
    tables.write("items", "d", pd.DataFrame(dict(bench="b", item=["i"], a=1., a_sd=.1, b=0., b_sd=.1, c=0.)))
    tables.write("task_items", "d", pd.DataFrame(dict(semester="s", bench="b", item=["i"], split="easy", train=True, quartile=0)))
    tables.write_paper()
    paper = {t: pd.read_csv(tmp_path / f"{t}.csv") for t in tables.KEYS}
    assert paper["tasks"].model.tolist() == ["t", "x"]
    assert paper["models"].model.tolist() == paper["samples"].model.tolist() == ["t", "x"]  # off-roster "off" stays in full/
    assert len(paper["items"]) == len(paper["task_items"]) == 1
