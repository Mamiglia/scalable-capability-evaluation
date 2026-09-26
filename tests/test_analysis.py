"""Baselines, judge conversions, scores and judge summaries on toy data (the toy task from test_protocols)."""

import numpy as np
import pandas as pd

from analysis.baselines import AccuracyOracle, EasyItemPrediction
from test_protocols import toy_task


def test_easy_item_prediction_and_oracle(monkeypatch):
    t = toy_task()  # references a, b, c with theta 0, 1, 2 on train items "0", "1"; targets t, u
    assert EasyItemPrediction()(t).eq(1.0).all()
    # a HARD item "9" over two repeats: a 0/2, b 1/2, c 2/2, t 2/2, u 0/2 -> theta = 2 * accuracy on the toy
    rows = [dict(bench="b", item="9", model=m, repeat=r, correct=int(r < k))
            for m, k in {"a": 0, "b": 1, "c": 2, "t": 2, "u": 0}.items() for r in (0, 1)]
    rows += [dict(bench="b", item="0", model=m, repeat=0, correct=1) for m in "abctu"]  # a train item: ignored
    items = pd.DataFrame({"bench": "b", "item": ["0", "1", "9"], "s_set": ["easy", "easy", "hard"]})
    monkeypatch.setattr("analysis.baselines.tables.load_responses", lambda dom: pd.DataFrame(rows))
    monkeypatch.setattr("analysis.baselines.tables.load", lambda dom: (None, items))
    oracle = AccuracyOracle(None)
    assert oracle.name == "accuracy_hard"
    np.testing.assert_allclose(oracle(t).to_numpy(), [2.0, 0.0], atol=1e-9)


from analysis import judges


def toy_ratings():
    """One bench, two items; references a, b, c (theta 0, 1, 2) rated 2/4/6 on an easy item and 1/2/3
    on a hard one; targets t (rated like c) and u (rated like a)."""
    base = {"a": 0, "b": 1, "c": 2, "t": 2, "u": 0}
    rows = []
    for model, level in base.items():
        for item, scale, offset in [("0", 2, 2), ("1", 1, 1)]:
            rows.append(dict(bench="b", model=model, item=item, score=offset + scale * level, semester="s"))
    return pd.DataFrame(rows)


def test_direct_theta_raw_and_item_z():
    ratings, refs = toy_ratings(), pd.Series({"a": 0.0, "b": 1.0, "c": 2.0})
    raw = judges.direct_theta(ratings, refs, ["t", "u"], "raw")
    np.testing.assert_allclose(raw.to_numpy(), [2.0, 0.0], atol=1e-9)
    z = judges.direct_theta(ratings, refs, ["t", "u"], "item_z")
    np.testing.assert_allclose(z.to_numpy(), [2.0, 0.0], atol=1e-9)
    # a constant item contributes z = 0, not NaN
    flat = pd.concat([ratings, pd.DataFrame(dict(bench="b", model=list("abctu"), item="9", score=5.0, semester="s"))])
    assert judges.direct_theta(flat, refs, ["t"], "item_z").notna().all()
    # the leave-one-out error on the references is exact here: each reference is on the others' line
    loo = judges.direct_reference_loo(ratings, refs, "raw")
    np.testing.assert_allclose(loo.to_numpy(), refs.to_numpy(), atol=1e-9)


def test_pairwise_reference_loo_recovers_references():
    rng = np.random.default_rng(0)
    # 8 references as in the real task: a leave-one-out fold keeps 21 pairs, enough for the per-bench slope
    theta = {f"r{i}": float(t) for i, t in enumerate(np.linspace(-2.5, 2.5, 8), 1)}
    refs = pd.Series(theta)
    rows = []
    for a in theta:
        for b in theta:
            if a >= b:
                continue
            for item in range(80):
                p = 1 / (1 + np.exp(-(theta[a] - theta[b])))
                rows.append(("g", "j", str(item), a, b, float(rng.random() < p), "s", False))
    verdicts = pd.DataFrame(rows, columns=["bench", "judge", "item", "model_left", "model_right", "pick_left", "semester", "imported"])
    for conversion in judges.PAIRWISE:
        loo = judges.pairwise_reference_loo(verdicts, refs, conversion)
        assert loo.index.equals(refs.index) and ((loo - refs).abs() < 0.8).all(), conversion


from analysis import stats


def toy_forecasts():
    """Two semesters, two targets each (theta* 1): easy-item prediction forecasts 3 for a and 2 for b (errors 4 and
    1, MSE 2.5), the oracle is exact, glm is off by 1, and a direct judge protocol (variant raw) has two
    judges: "weak" exact with a poor reference LOO, "strong" off by 1 with the better LOO; "strong" also
    has a reasoning-hidden run off by 4."""
    base = dict(domain="d", variant="-", judge="-", condition="-", judge_loo_mse=np.nan, missing=0, theta=1.0)
    rows = []
    for sem in ("s1", "s2"):
        for model, baseline in (("a", 3.0), ("b", 2.0)):
            rows += [dict(base, semester=sem, model=model, protocol=stats.BASELINE, theta_hat=baseline),
                     dict(base, semester=sem, model=model, protocol=stats.ORACLE, theta_hat=1.0),
                     dict(base, semester=sem, model=model, protocol="glm_regex", theta_hat=2.0),
                     dict(base, semester=sem, model=model, protocol="direct", variant="raw", judge="weak",
                          judge_loo_mse=0.9, condition="reasoning-shown", theta_hat=1.0),
                     dict(base, semester=sem, model=model, protocol="direct", variant="raw", judge="strong",
                          judge_loo_mse=0.1, condition="reasoning-shown", theta_hat=2.0),
                     dict(base, semester=sem, model=model, protocol="direct", variant="raw", judge="strong",
                          judge_loo_mse=0.1, condition="reasoning-hidden", theta_hat=5.0)]
    return pd.DataFrame(rows)


def test_scores_pool_and_summary():
    s = stats.scores(toy_forecasts())
    pooled = s[(s.semester == "all") & (s.domain == "d") & (s.judge == "-")].set_index("protocol").mse
    assert pooled[stats.BASELINE] == 2.5 and pooled[stats.ORACLE] == 0.0 and pooled["glm_regex"] == 1.0
    assert s[(s.semester == "s1") & (s.protocol == stats.BASELINE)].n.item() == 2
    # best-judge rows are one forecast per target (interval kept); mean-over-judges pools judges (interval blanked)
    direct = s[(s.semester == "all") & (s.domain == "d") & (s.protocol == "direct")].set_index("judge")
    assert np.isfinite(direct.loc[stats.BEST, "lo"]) and np.isfinite(direct.loc[stats.BEST, "hi"])
    assert np.isnan(direct.loc[stats.MEAN, "lo"]) and np.isnan(direct.loc[stats.MEAN, "hi"])
    r = stats.summary(s)
    # the core rows only: per-condition judge rows would make (protocol, judge) non-unique
    r = r[(r.domain == "d") & r.condition.isin(["-", "core"])].set_index(["protocol", "judge"])
    assert r.skill[("glm_regex", "-")] == 0.6 and r.skill[(stats.ORACLE, "-")] == 1.0
    # the best judge on references is "strong" (lowest LOO), off by 1; the mean over judges is off by 0.5
    assert r.mse[("direct", stats.BEST)] == 1.0 and r.mse[("direct", stats.MEAN)] == 0.5


def test_core_condition_prefers_a_smaller_train_set_over_hiding_the_reasoning():
    f = toy_forecasts()
    shown = stats.core_condition(f)
    assert set(shown[shown.protocol == "direct"].condition) == {"reasoning-shown"}
    # where no judge has a full-train shown run, the cell falls to the smaller-train-set shown run, not the hidden one
    smaller = f[f.condition != "reasoning-shown"].assign(
        condition=lambda d: d.condition.where(d.judge != "strong", "reasoning-shown-10pq"))
    picked = stats.core_condition(smaller)
    picked = picked[picked.protocol == "direct"]
    assert set(picked.condition) == {"reasoning-shown-10pq"} and set(picked.judge) == {"strong"}
    # each judge keeps its own closest run: strong on its smaller-train shown run, weak on the full one, both stay
    both = f.assign(condition=lambda d: d.condition.where(d.judge != "strong", "reasoning-shown-10pq"))
    each = stats.core_condition(both)
    each = each[each.protocol == "direct"].drop_duplicates(["judge", "condition"]).set_index("judge").condition
    assert each.to_dict() == {"strong": "reasoning-shown-10pq", "weak": "reasoning-shown"}
    # only where nothing shown exists does the hidden run stand in
    hidden = stats.core_condition(f[f.condition == "reasoning-hidden"])
    assert set(hidden[hidden.protocol == "direct"].condition) == {"reasoning-hidden"}


def test_task_metrics_are_paired_against_easy_item_prediction():
    m = stats.task_metrics(toy_forecasts())
    s1 = m[(m.domain == "d") & (m.semester == "s1")].set_index(["protocol", "judge"])
    # glm: squared errors 1, 1 against easy-item prediction's 4, 1 -> delta -1.5; the paired deltas (-3, 0) vary, so z is finite
    assert s1.delta[("glm_regex", "-")] == -1.5 and s1.baseline_mse[("glm_regex", "-")] == 2.5 and np.isfinite(s1.z[("glm_regex", "-")])
    # easy-item prediction against itself: delta 0, no spread -> z NaN
    assert s1.delta[(stats.BASELINE, "-")] == 0.0 and np.isnan(s1.z[(stats.BASELINE, "-")])
    # judge rows enter as the best judge on references only
    assert s1.delta[("direct", stats.BEST)] == -1.5 and (m.judge != stats.MEAN).all()
    pooled = m[(m.domain == "all") & (m.semester == "all") & (m.protocol == "glm_regex")]
    assert pooled.n.item() == 4 and pooled.delta.item() == -1.5


from types import SimpleNamespace

from analysis import calibration
from analysis.loading import JudgeRun


def test_ensemble_runs_pool_the_core_condition_of_every_judge():
    def run(protocol, judge, condition, n, missing=0):
        table = pd.DataFrame({"bench": "b", "item": [str(i) for i in range(n)], "judge": judge})
        return JudgeRun(protocol, judge, condition, SimpleNamespace(domain=None), table, missing)
    runs = [run("pairwise", "j1", "reasoning-shown", 3, 1), run("pairwise", "j1", "reasoning-hidden", 9),
            run("pairwise", "j2", "reasoning-hidden", 4, 2), run("pairwise", "j3", "reasoning-shown-10pq", 7),
            run("direct", "j1", "reasoning-shown", 5)]
    pooled = calibration.ensemble_runs(runs)
    # pairwise: j1's shown run, j2's hidden run and j3's 10pq shown run (its only one) pooled; direct has one judge
    assert [(r.protocol, r.judge, r.condition, len(r.table), r.missing) for r in pooled] == [("pairwise", "ensemble", "core", 14, 3)]
    assert sorted(pooled[0].table.judge.unique()) == ["j1", "j2", "j3"]


from analysis import loading


def test_direct_runs_restricts_to_the_top3_roster(tmp_path, monkeypatch):
    judges_dir = tmp_path / "judges"
    judges_dir.mkdir()
    for name in "abcd":
        (judges_dir / f"{name}.yml").write_text("")
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    (frozen / "fit.yml").write_text("top: {s: [a, b, c]}")

    class FakeJudgeDirect:
        def __init__(self, config, domain, include_reasoning=True):
            pass

        def log_dirs(self, semester):
            return [tmp_path]  # exists for every judge; fetched() finds no .eval stubs there

    monkeypatch.setattr(loading, "JUDGES", judges_dir)
    monkeypatch.setattr(loading.DirectJudgeConfig, "from_cli", staticmethod(lambda path: path))
    monkeypatch.setattr(loading, "JudgeDirect", FakeJudgeDirect)
    monkeypatch.setattr(loading, "direct_from_logs", lambda run_dir: pd.DataFrame())
    monkeypatch.setattr(loading.tables, "frozen_dir", lambda dom: frozen)

    task, dom = SimpleNamespace(semester="s"), SimpleNamespace(name="dom")
    runs = loading.direct_runs(dom, task)
    assert sorted({r.judge for r in runs}) == ["a", "b", "c"]
