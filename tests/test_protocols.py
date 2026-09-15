from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from types import SimpleNamespace

from capability import task
from configs.domain import DomainBench, DomainConfig
from configs.judge import JudgeConfig
from configs.model import ModelConfig
from protocols import features
from protocols.pairwise import JudgeBT, judge as judge_module


def toy_task():
    items = pd.DataFrame({"bench": "b", "item": ["0", "1"], "guess": 0.0, "a": 1.0, "b": 0.0})
    train = task.Cohort(pd.DataFrame({"theta": [0.0, 1.0, 2.0], "path_b": "b/h/0.eval"}, index=pd.Index(list("abc"), name="model")), items, "s")
    test = task.Cohort(pd.DataFrame({"theta": np.nan, "path_b": "b/h/0.eval"}, index=pd.Index(list("tu"), name="model")), items, "s")
    historical = task.Cohort(train.models.iloc[:0], items, "s")
    return task.Task("s", train, historical, test, pd.Series([3.0, 4.0], index=test.models.index))


def toy_judge(tmp_path):
    domain = DomainConfig(name="dom", benches=[DomainBench(bench="b"), DomainBench(bench="b_gold")])  # two entries, one bench
    domain.task = lambda bench: SimpleNamespace(name="b")
    config = JudgeConfig(name="j", judge=ModelConfig(name="p/j", slug="j"), log_dir=tmp_path)
    return JudgeBT(config, domain)


def test_judge_bt_places_targets_among_pinned_references(tmp_path, monkeypatch):
    t = toy_task()
    p = toy_judge(tmp_path)
    # c beats b beats a on both items and orders; t beats everyone; u was never judged.
    rows = [dict(bench="b", judge="j", item=i, model_left=w, model_right=l, pick_left=1.0, semester="s", imported=False)
            for i in ("0", "1") for w, l in [("t", "a"), ("t", "b"), ("t", "c"), ("c", "b"), ("c", "a"), ("b", "a")]]
    # an imported verdict that disagrees on a judged (item, pair, order) loses to the run's own; one on a new item counts
    rows += [dict(rows[0], pick_left=0.0, imported=True), dict(rows[0], item="9", pick_left=0.0, imported=True)]
    monkeypatch.setattr(JudgeBT, "judge", lambda self, train, test: "run-dir")
    monkeypatch.setattr(judge_module, "from_logs", lambda log_dir: pd.DataFrame(rows))
    theta_hat = p(t)
    assert p.name == "judge_j" and theta_hat["t"] > 2 and np.isnan(theta_hat["u"])
    kept = p.verdicts(t.references, t.test)
    assert len(kept) == len(rows) - 1 and kept[kept.item == "0"].pick_left.eq(1.0).all()
    # verdicts among targets only leave the scale unpinned
    monkeypatch.setattr(judge_module, "from_logs", lambda log_dir: pd.DataFrame(columns=judge_module.VERDICT_COLUMNS))
    assert p(t).isna().all()


def test_judge_bt_judges_target_vs_reference_pairs_on_shared_items_and_skips_finished_runs(tmp_path, monkeypatch):
    t = toy_task()
    p = toy_judge(tmp_path)
    # transcripts (cached_property): references a, b solved both items, c none; target t solved item 0, u none
    t.references.__dict__["transcripts"] = {"a": {("b", "0"): 1, ("b", "1"): 1}, "b": {("b", "0"): 1, ("b", "1"): 1}, "c": {}}
    t.test.__dict__["transcripts"] = {"t": {("b", "0"): 1}, "u": {}}
    built = []
    monkeypatch.setattr(judge_module, "judge_task", lambda *a, **k: built.append(k) or SimpleNamespace(name=k["task_name"]))
    tasks = p.tasks(t.references, t.test)
    # 2 refs x 1 target on item 0, then the reference pair a|b on both items; no t|u, no c
    assert [x.name for x in tasks] == ["b/a|t", "b/b|t", "b/a|b"]
    assert [k["items"] for k in built] == [{"0"}, {"0"}, {"0", "1"}]
    assert built[0]["metadata"]["semester"] == "s" and built[0]["metadata"]["major"] == "a"

    ran = []
    monkeypatch.setattr(judge_module, "run_tasks", lambda tasks, config, log_dir: ran.append(log_dir) or True)
    log_dir = p.judge(t.references, t.test)
    assert log_dir == tmp_path / "dom" / "s" / "j" / p.config.condition_hash and ran == [log_dir]
    # every task finished in the dir -> read only
    monkeypatch.setattr(judge_module, "is_successful_eval", lambda path: True)
    monkeypatch.setattr(judge_module, "read_header", lambda path: SimpleNamespace(eval=SimpleNamespace(task=path.stem)))
    log_dir.mkdir(parents=True)
    for name in ("b_a|t", "b_b|t", "b_a|b"):
        (log_dir / f"{name}.eval").touch()
    monkeypatch.setattr(judge_module, "read_header", lambda path: SimpleNamespace(eval=SimpleNamespace(task=path.stem.replace("_", "/", 1))))
    judged = {"b/a|t": ["0"], "b/b|t": ["0"], "b/a|b": ["0", "1"]}
    monkeypatch.setattr(judge_module, "read_eval_log_sample_summaries",
                        lambda path: [SimpleNamespace(id=f"{i}_1") for i in judged[Path(path).stem.replace("_", "/", 1)]])
    p.judge(t.references, t.test)
    assert ran == [log_dir]
    # a train item the dir has no verdict on (item 1 of a|b, after a refreeze) is judged on its own
    judged["b/a|b"] = ["0"]
    built.clear()
    p.judge(t.references, t.test)
    assert ran == [log_dir, log_dir] and [(k["task_name"], k["items"]) for k in built] == [("b/a|b", {"1"})]


def test_tabular_regression(monkeypatch):
    t = toy_task()
    chars = {"a": 1.0, "b": 2.0, "c": 3.0, "t": 4.0, "u": 0.0}  # theta = chars - 1
    monkeypatch.setattr(features, "feature_table",
                        lambda cohort: pd.DataFrame({"chars": [chars[m] for m in cohort.models.index]}, index=cohort.models.index))
    theta_hat = features.TabularRegression("chars", ["chars"], LinearRegression())(t)
    assert np.allclose(theta_hat[["t", "u"]], [3.0, -1.0])


def test_stratified_items_is_seeded_per_quartile_and_off_by_default():
    import pandas as pd

    from configs.judge import JudgeConfig
    from protocols.pairwise.judge import stratified_items

    items = pd.DataFrame({"bench": "b", "item": [str(i) for i in range(100)], "b": range(100)})
    assert stratified_items(items, 0) == set(items.item)
    picked = stratified_items(items, 10)
    assert len(picked) == 40 and picked == stratified_items(items, 10)
    assert sum(int(i) < 25 for i in picked) == 10  # 10 from the easiest quartile
    cfg = dict(name="x", judge="configs/models/zai/glm-4.7.yml", include_reasoning=False)
    assert JudgeConfig.from_cli(cfg).condition_hash == "7ea04f2b"  # unchanged with the default
    assert JudgeConfig.from_cli(cfg | {"train_per_quartile": 10}).condition_hash != "7ea04f2b"


def test_mean_reference_baselines():
    from protocols.baseline import MeanReference
    t = toy_task()
    assert MeanReference("m")(t).eq(1.0).all()  # mean of 0, 1, 2
    assert MeanReference("f", 2)(t).eq(1.5).all()  # top 2: 1, 2


def test_calibrated_recovers_theta_from_synthetic_verdicts():
    from protocols.pairwise import calibrated
    rng = np.random.default_rng(0)
    refs = pd.Series(np.linspace(-2, 2, 8), index=[f"r{i}" for i in range(8)])
    truth = {"t0": -1.0, "t1": 1.5}
    theta = pd.concat([refs, pd.Series(truth)])
    alpha, beta = {"gpqa": 1.5, "arc": 0.3}, 0.4
    rows = []
    for a in theta.index:
        for b in theta.index:
            if a >= b or (a in truth and b in truth):
                continue
            for bench in alpha:
                for item in range(60):
                    for left, right in ((a, b), (b, a)):
                        p = 1 / (1 + np.exp(-(alpha[bench] * (theta[left] - theta[right]) + beta)))
                        rows.append(dict(bench=bench, judge="j", item=str(item), model_left=left, model_right=right,
                                         pick_left=float(rng.random() < p)))
    verdicts = pd.DataFrame(rows)
    a_hat, b_hat = calibrated.calibrate(verdicts, refs)
    assert abs(a_hat["gpqa"] - 1.5) < 0.3 and abs(a_hat["arc"] - 0.3) < 0.15 and abs(b_hat - 0.4) < 0.15
    theta_hat = calibrated.estimate_all(verdicts, refs, list(truth))
    assert all(abs(theta_hat[t] - truth[t]) < 0.3 for t in truth)
    sigma2 = ((calibrated.loo(verdicts, refs) - refs) ** 2).mean()
    assert sigma2 < 0.1
    # a small held-out error keeps the estimate; a model without evidence gets the prior mean
    shrunk = calibrated.shrink(theta_hat, sigma2, (0.0, 1.0))
    assert abs(shrunk["t1"] - truth["t1"]) < 0.4 and calibrated.shrink(pd.Series({"x": np.nan}), sigma2, (0.5, 1.0))["x"] == 0.5
