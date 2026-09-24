from types import SimpleNamespace

import numpy as np
import pandas as pd

from protocols.pairwise import judge as judge_module
from protocols.pairwise.bradley_terry import rank


def verdicts(rows):
    """rows of (judge, item, model_left, model_right, pick_left), one benchmark."""
    frame = pd.DataFrame(rows, columns=["judge", "item", "model_left", "model_right", "pick_left"])
    return frame.assign(bench="synthetic")


def simulated_verdicts(abilities, n_items=200, seed=0):
    """Every pair judged in both orders on every item, picks drawn from the Bradley-Terry probabilities."""
    rng = np.random.default_rng(seed)
    models = list(abilities)
    rows = []
    for item in range(n_items):
        for i, left in enumerate(models):
            for right in models[i + 1:]:
                p_left = 1 / (1 + np.exp(-(abilities[left] - abilities[right])))
                rows.append(("judge", item, left, right, int(rng.random() < p_left)))
                rows.append(("judge", item, right, left, int(rng.random() < 1 - p_left)))
    return verdicts(rows)


def test_rank_recovers_generating_order():
    abilities = {"strong": 2.0, "middle": 0.5, "weak": -1.0, "weakest": -2.5}
    ranking = rank(simulated_verdicts(abilities))
    assert list(ranking.sort_values("rank").model) == ["strong", "middle", "weak", "weakest"]
    assert ranking.set_index("model").loc["strong", "rank"] == 1


def test_anchors_pin_references_and_place_the_rest_on_their_scale():
    abilities = {"strong": 2.0, "middle": 0.5, "weak": -1.0, "weakest": -2.5}
    anchored = rank(simulated_verdicts(abilities), anchors=pd.Series({"strong": 2.0, "weakest": -2.5}))
    table = anchored.set_index("model").ability
    assert table["strong"] == 2.0 and table["weakest"] == -2.5
    assert abs(table["middle"] - 0.5) < 0.3 and abs(table["weak"] + 1.0) < 0.3


def test_disagreeing_orders_match_half_rows():
    disagreeing = verdicts([("j", 0, "a", "b", 1), ("j", 0, "b", "a", 1), ("j", 0, "a", "c", 1), ("j", 0, "c", "a", 0)])
    half_rows = verdicts([("j", 0, "a", "b", 0.5), ("j", 0, "b", "a", 0.5), ("j", 0, "a", "c", 1), ("j", 0, "c", "a", 0)])
    abilities = rank(disagreeing).set_index("model").ability
    abilities_half = rank(half_rows).set_index("model").ability
    assert abilities.sub(abilities_half).abs().max() < 1e-4


def test_unbeaten_model_is_finite_and_flagged():
    ranking = rank(verdicts([("j", 0, "a", "b", 1), ("j", 0, "b", "a", 0), ("j", 1, "a", "b", 1), ("j", 1, "b", "a", 0)]))
    a = ranking.set_index("model").loc["a"]
    assert np.isfinite(a.ability) and a.ability > 0
    assert a.n_wins == a.n_comparisons == 4


def test_drop_ties_removes_only_disagreeing_groups():
    frame = verdicts([
        ("j", 0, "a", "b", 1), ("j", 0, "b", "a", 1),   # disagree: tie
        ("j", 1, "a", "b", 1), ("j", 1, "b", "a", 0),   # agree
        ("j", 2, "a", "b", 0),                          # single order
    ])
    kept = rank(frame, drop_ties=False).set_index("model")
    dropped = rank(frame, drop_ties=True).set_index("model")
    assert kept.loc["a", "n_comparisons"] == 5 and kept.loc["a", "n_ties"] == 1
    assert dropped.loc["a", "n_comparisons"] == 3 and dropped.loc["a", "n_ties"] == 0
    assert kept.loc["a", "n_wins"] == 3 and dropped.loc["a", "n_wins"] == 2


def test_verdicts_from_logs(monkeypatch, tmp_path):
    (tmp_path / "x.eval").touch()
    header = SimpleNamespace(status="success", eval=SimpleNamespace(
        model="judge", task="gpqa/big|small", metadata={"bench": "gpqa", "major": "big", "minor": "small", "semester": "s"}))
    summaries = [
        SimpleNamespace(id="q1_1", scores={"s": SimpleNamespace(answer="1")}, metadata={"chars_1": 900, "chars_2": 900}),  # big shown left, picked
        SimpleNamespace(id="q1_2", scores={"s": SimpleNamespace(answer="1")}, metadata={"chars_1": 900, "chars_2": 900}),  # small shown left, picked
        SimpleNamespace(id="q2_1", scores={"s": SimpleNamespace(answer=None)}, metadata={}),  # no verdict
    ]
    monkeypatch.setattr(judge_module, "read_eval_log", lambda *a, **k: header)
    monkeypatch.setattr(judge_module, "read_eval_log_sample_summaries", lambda *a, **k: summaries)
    table = judge_module.from_logs(tmp_path)
    assert table.to_dict("records") == [
        dict(bench="gpqa", judge="judge", item="q1", model_left="big", model_right="small", pick_left=1.0, semester="s", imported=False, chars_left=900, chars_right=900),
        dict(bench="gpqa", judge="judge", item="q1", model_left="small", model_right="big", pick_left=1.0, semester="s", imported=False, chars_left=900, chars_right=900),
    ]
