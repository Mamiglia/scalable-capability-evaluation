"""Historical labels and protocol routing, with no paid judging or experiment data."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from inspect_ai.log import EvalConfig, EvalDataset, EvalLog, EvalSample, EvalSpec, write_eval_log
from inspect_ai.model import ChatMessageAssistant
from inspect_ai.scorer import Score
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from capability.task import Task
from configs.domain import DomainBench, DomainConfig
from configs.judge import DirectJudgeConfig, JudgeConfig
from configs.model import ModelConfig
from protocols.direct_judge import JudgeDirect
from protocols.features import TabularRegression
from protocols.pairwise import JudgeBT
from reference import freeze, tables


def test_freeze_extends_labels_without_changing_reference_fit(tmp_path, monkeypatch, caplog):
    # Deliberately not lexical order: semester chronology is the roster's file order.
    semesters = {"old": ["h0", "h1", "missing"], "previous": ["r0", "r1"],
                 "current": ["t0", "t1"], "future": ["f0", "f1"]}
    names = [m for roster in semesters.values() for m in roster] + ["unrostered"]
    rows = [dict(bench="b", model=m, item=i, repeat=0, correct=(i == "easy" and m not in {"h0", "h1"}), guess=0.)
            for m in names for i in ("easy", "hard") if m != "missing" or i == "easy"]
    responses = pd.DataFrame(rows)
    items = pd.DataFrame({"bench": "b", "item": ["easy", "hard"], "a": [1., 2.], "b": [3., 4.], "guess": [0., .25]})
    models = pd.DataFrame({"global_theta": [100., 90., 80., 2., 1., 3., 4., 5., 6., 1000.]}, index=names)
    models.index.name = "model"
    logs = pd.DataFrame([dict(bench="b", model=m, repeat=0, path=f"data/b/{m}/0/run.eval") for m in names])
    monkeypatch.setattr(tables, "META", tmp_path)
    monkeypatch.setattr(freeze, "read_responses", lambda domain: responses)
    monkeypatch.setattr(freeze, "eval_logs", lambda domain: logs)
    monkeypatch.setattr(freeze, "load_semesters", lambda: semesters)
    monkeypatch.setattr(tables, "load_semesters", lambda: semesters)
    monkeypatch.setattr(freeze, "global_fit", lambda *a: (models.copy(), items.copy()))
    fits = []

    def posterior(data, model, **kwargs):
        assert model == "fixed_2pl"
        fits.append((data.models, data.items, kwargs))
        # Distinguish labels derived from different HARD sets without sampling NUTS.
        return {"theta": np.full((2, len(data.models)), sum(kwargs["b"]))}

    monkeypatch.setattr(freeze, "fit", posterior)
    freeze.build(DomainConfig(name="toy", benches=[]), train=0)
    frozen_models, frozen_items = tables.load(DomainConfig(name="toy", benches=[]))
    assert frozen_items.current_set.tolist() == ["easy", "hard"]
    assert frozen_items.current_train.tolist() == [True, False]
    assert frozen_models.current_role.dropna().to_dict() == {
        "h0": "historical", "h1": "historical", "missing": "historical",
        "r0": "reference", "r1": "reference", "t0": "target", "t1": "target"}
    assert frozen_models.loc[["h0", "h1"], "current_theta"].tolist() == [4., 4.]
    assert frozen_models.loc["h0", "previous_theta"] == 3.  # never borrowed for this task
    assert np.isnan(frozen_models.loc["missing", "current_theta"])
    assert frozen_models.loc[["f0", "f1", "unrostered"], "current_theta"].isna().all()
    assert "missing" in caplog.text and "HARD responses" in caplog.text
    current_fits = [entry for entry in fits if entry[1] == [("b", "hard")]]
    assert current_fits[0][0] == ["r0", "r1", "t0", "t1"]  # original fit remains separate
    assert current_fits[1][0] == ["h0", "h1"]
    for _, _, params in current_fits:
        assert params["a"].tolist() == [2.] and params["b"].tolist() == [4.]
        assert params["guess"].tolist() == [.25]


@pytest.fixture
def historical_task(tmp_path, monkeypatch):
    names = ["h", "missing", "r0", "r1", "t", "future", "unrostered"]
    rows = []
    for model, chars in zip(names, [63, 63, 127, 255, 511, 1023, 2047]):
        path = tmp_path / f"{model}.eval"
        samples = [EvalSample(id=item, epoch=epoch, input="Question?", target="", messages=[
            ChatMessageAssistant(content="x" * (chars - 13))], scores={"correct": Score(value=score)})
            for item, epoch, score in [("easy", 1, "C"), ("wrong", 1, "I"), ("wrong", 2, "C"),
                                       ("hard", 1, "C"), ("unselected", 1, "C")]]
        write_eval_log(EvalLog(status="success", eval=EvalSpec(created="2026-09-14T12:00:00Z", task="b",
            dataset=EvalDataset(name="b", samples=len(samples)), model=model, config=EvalConfig()), samples=samples), path)
        rows.append(str(path))
    models = pd.DataFrame({"semester": ["old", "old", "previous", "previous", "current", "future", "old"],
        "current_role": ["historical", "historical", "reference", "reference", "target", "", ""],
        "current_theta": [0., np.nan, 1., 2., 999., 1000., 1000.], "previous_theta": -1000., "path_b": rows}, index=names)
    items = pd.DataFrame({"bench": "b", "item": ["easy", "wrong", "hard", "unselected"],
                         "guess": 0., "a": 1., "b": 0., "current_train": [True, True, False, False]})
    monkeypatch.setattr(tables, "load", lambda domain: (models, items))
    monkeypatch.setattr(tables, "load_semesters", lambda: {
        "old": ["h", "missing"], "previous": ["r0", "r1"], "current": ["t"], "future": ["future"]})
    return Task.from_semester("current", DomainConfig(name="toy", benches=[]))


def test_task_unites_only_labelled_past_models_on_shared_first_attempts(historical_task, caplog, monkeypatch):
    task = historical_task
    assert task.references.models.index.tolist() == ["r0", "r1"]
    assert task.historical.models.index.tolist() == ["h"]
    assert task.train.theta.to_dict() == {"r0": 1., "r1": 2., "h": 0.}
    assert task.test.models.index.tolist() == ["t"] and task.test.theta.isna().all()
    assert task.truth.to_dict() == {"t": 999.}
    assert any("missing" in record.message for record in caplog.get_records("setup"))
    for cohort in (task.references, task.historical, task.train, task.test):
        assert cohort.semester == "current"
        assert cohort.items is task.references.items
        assert cohort.items.item.tolist() == ["easy", "wrong"]
        for transcripts in cohort.transcripts.values():
            assert set(transcripts) == {("b", "easy")}
            assert transcripts[("b", "easy")].epoch == 1
    # Stale/overlapping roster metadata must never expose a target label through train.
    models, _ = tables.load(None)
    models.loc[["t", "r0"], "semester"] = "old"
    monkeypatch.setattr(tables, "load_semesters", lambda: {
        "old": ["h", "missing", "t", "r0"], "previous": ["r0", "r1"], "current": ["t"], "future": ["future"]})
    rebuilt = Task.from_semester("current", DomainConfig(name="toy", benches=[]))
    assert rebuilt.train.theta.to_dict() == {"r0": 1., "r1": 2., "h": 0.}


def test_regression_fits_preprocessing_and_coefficients_on_references_only(historical_task):
    task = historical_task
    task.historical.models.loc["no-transcripts"] = {"theta": 42., "path_b": np.nan}
    regression = TabularRegression("chars", ["chars"], make_pipeline(StandardScaler(), Ridge(alpha=1.)))
    prediction = regression(task)
    # Rendered lengths include the 13-character "[Assistant]: " prefix.
    # references x = log(2) * [7, 8], y = [1, 2] (h at 6 is history, not fitted); ridge slope 1/3 on z = +-1,
    # and the target at 9 log(2) sits at z = 3.
    assert regression.estimator[0].mean_[0] == pytest.approx(7.5 * np.log(2))
    assert regression.estimator[0].n_samples_seen_ == 2
    assert prediction["t"] == pytest.approx(2.5)
    task.historical.models.theta = 1000.  # history never enters the fit
    pd.testing.assert_series_equal(regression(task), prediction)
    task.truth[:] = -1000
    task.test.models.theta = -1000
    pd.testing.assert_series_equal(regression(task), prediction)


def test_direct_ratings_cover_references_and_calibrate_on_their_labels(historical_task, tmp_path, monkeypatch):
    domain = DomainConfig(name="toy", benches=[DomainBench(bench="b")])
    domain.task = lambda bench: SimpleNamespace(name="b")
    judge = JudgeDirect(DirectJudgeConfig(name="j", judge=ModelConfig(name="j", slug="mockllm/model"), log_dir=tmp_path), domain)
    built = []

    def scores(train, test):
        # Keep the real task builder; replace only paid judging with fixed ratings.
        built.extend(judge.tasks(train, test))
        return pd.DataFrame({"bench": "b", "item": "easy", "semester": "current",
                             "model": ["h", "r0", "r1", "t", "future"], "score": [1., 2., 2., 3., 10.]})

    monkeypatch.setattr(judge, "scores", scores)
    prediction = judge(historical_task)
    assert [t.name for t in built] == ["b/r0", "b/r1", "b/t"]
    assert all([sample.id for sample in t.dataset] == ["easy"] for t in built)
    assert prediction.to_dict() == {"t": pytest.approx(1.5)}  # a fit on the union would predict 3


def test_pairwise_uses_only_references_when_history_expands(historical_task, tmp_path, monkeypatch):
    domain = DomainConfig(name="toy", benches=[DomainBench(bench="b")])
    domain.task = lambda bench: SimpleNamespace(name="b")
    judge = JudgeBT(JudgeConfig(name="j", judge=ModelConfig(name="j", slug="mockllm/model"), log_dir=tmp_path), domain)
    built = []

    def verdicts(references, test):
        built.extend(judge.tasks(references, test))
        return pd.DataFrame({"bench": "b", "judge": "j", "item": "easy", "model_left": ["r0", "r1"],
                             "model_right": "t", "pick_left": [.5, .5]})

    monkeypatch.setattr(judge, "verdicts", verdicts)
    prediction = judge(historical_task)
    assert [t.name for t in built] == ["b/r0|t", "b/r1|t", "b/r0|r1"]
    assert prediction["t"] == pytest.approx(1.5, abs=1e-5)
    historical_task.historical.models.theta = 1000.
    pd.testing.assert_series_equal(judge(historical_task), prediction)
