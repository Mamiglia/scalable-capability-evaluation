import pytest
import numpy as np
import pandas as pd

from ranking import bootstrap, rank, score
from ranking.cli import main as cli_main


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


def correctness(rows):
    """rows of (model, item, correct), one benchmark."""
    return pd.DataFrame(rows, columns=["model", "item", "correct"]).assign(bench="synthetic")


def test_score_is_kendall_tau_against_accuracy():
    ranking = pd.DataFrame({"model": ["a", "b", "c"], "ability": [1.0, 0.0, -1.0]})
    judged = verdicts([("j", 0, "a", "b", 1), ("j", 1, "b", "c", 1), ("j", 1, "a", "c", 1)])
    agreeing = correctness([("a", 0, 1), ("b", 0, 1), ("c", 0, 0), ("a", 1, 1), ("b", 1, 0), ("c", 1, 0)])  # a 1, b .5, c 0
    reversed_ = agreeing.assign(correct=1 - agreeing.correct)
    assert score(ranking, agreeing, judged) == {"correlation": 1.0, "method": "kendall", "n_models": 3, "n_items": 2}
    assert score(ranking, reversed_, judged)["correlation"] == -1.0


def test_score_can_use_spearman():
    ranking = pd.DataFrame({"model": ["a", "b", "c", "d"], "ability": [3.0, 2.0, 1.0, 0.0]})
    judged = verdicts([("j", 0, "a", "b", 1), ("j", 1, "c", "d", 1), ("j", 2, "a", "d", 1)])
    # accuracy over items 0-2: b 1, a 2/3, c 1/3, d 0 -> one adjacent swap against the ability order
    table = correctness([(m, item, int(item < n_right)) for m, n_right in [("a", 2), ("b", 3), ("c", 1), ("d", 0)]
                         for item in range(3)])
    assert score(ranking, table, judged, correlation="kendall")["correlation"] == pytest.approx(2 / 3)
    assert score(ranking, table, judged, correlation="spearman")["correlation"] == pytest.approx(0.8)


@pytest.mark.parametrize("items, expected, n_items", [("all", -1.0, 3), ("judged", 1.0, 1), ("held_out", -1.0, 2)])
def test_score_truth_items(items, expected, n_items):
    ranking = pd.DataFrame({"model": ["a", "b"], "ability": [1.0, 0.0]})
    judged = verdicts([("j", 0, "a", "b", 1)])
    # a beats b on item 0, but loses on the unjudged items 1 and 2
    table = correctness([("a", 0, 1), ("b", 0, 0), ("a", 1, 0), ("b", 1, 1), ("a", 2, 0), ("b", 2, 1)])
    result = score(ranking, table, judged, items=items)
    assert (result["correlation"], result["n_items"]) == (expected, n_items)


def simulated_correctness(abilities, n_items=200, seed=0):
    """Each model correct on an item with probability sigmoid(ability)."""
    rng = np.random.default_rng(seed)
    return correctness([(model, item, int(rng.random() < 1 / (1 + np.exp(-ability))))
                        for model, ability in abilities.items() for item in range(n_items)])


def test_bootstrap_brackets_point_estimate_and_is_seeded():
    abilities = {"strong": 2.0, "middle": 0.5, "weak": -1.0, "weakest": -2.5}
    judged, table = simulated_verdicts(abilities), simulated_correctness(abilities)
    point = rank(judged)
    result = bootstrap(judged, table, n_resamples=20, seed=0)
    ability = result["ability"].set_index("model")
    assert list(ability.columns) == ["p5", "p50", "p95"]
    assert (ability.p5 <= point.set_index("model").ability[ability.index] + 1e-9).all()
    assert (ability.p95 >= point.set_index("model").ability[ability.index] - 1e-9).all()
    assert result["correlation"].p5 <= score(point, table, judged)["correlation"] <= result["correlation"].p95
    again = bootstrap(judged, table, n_resamples=20, seed=0)
    assert again["ability"].equals(result["ability"])
    assert not bootstrap(judged, table, n_resamples=20, seed=1)["ability"].equals(result["ability"])


def test_cli_filters_and_writes_per_benchmark(monkeypatch, tmp_path):
    abilities = {"strong": 2.0, "middle": 0.5, "weak": -1.0}
    judged = pd.concat([simulated_verdicts(abilities, n_items=30).assign(bench=b, judge=j)
                        for b in ("gpqa", "aime") for j in ("judge-1", "judge-2")])
    table = pd.concat([simulated_correctness(abilities, n_items=30).assign(bench=b) for b in ("gpqa", "aime")])
    judged.to_parquet(tmp_path / "verdicts.parquet")
    table.to_parquet(tmp_path / "correctness.parquet")
    monkeypatch.setattr("sys.argv", ["rank", "--verdicts", str(tmp_path / "verdicts.parquet"),
                                     "--correctness", str(tmp_path / "correctness.parquet"),
                                     "--out_dir", str(tmp_path / "out"), "--judges", "[judge-1]",
                                     "--models", "[strong,weak]", "--n_resamples", "5"])
    assert cli_main() == 0
    scores = pd.read_csv(tmp_path / "out" / "scores.csv")
    assert sorted(scores.bench) == ["aime", "gpqa"] and (scores.n_models == 2).all()
    ranking = pd.read_csv(tmp_path / "out" / "ranking_gpqa.csv")
    assert list(ranking.model) == ["strong", "weak"]
    assert ranking.n_comparisons.sum() == 2 * 30 * 2  # one judge, one pair, both orders, 30 items, two sides
    assert set(pd.read_csv(tmp_path / "out" / "bootstrap.csv").bench) == {"aime", "gpqa"}
