import numpy as np
import pandas as pd

from ranking import rank, score


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
    assert score(ranking, agreeing, judged) == {"kendall_tau": 1.0, "n_models": 3, "n_items": 2}
    assert score(ranking, reversed_, judged)["kendall_tau"] == -1.0


def test_score_truth_uses_only_judged_items():
    ranking = pd.DataFrame({"model": ["a", "b"], "ability": [1.0, 0.0]})
    judged = verdicts([("j", 0, "a", "b", 1)])
    # a beats b on item 0, but loses badly on the unjudged items 1 and 2
    table = correctness([("a", 0, 1), ("b", 0, 0), ("a", 1, 0), ("b", 1, 1), ("a", 2, 0), ("b", 2, 1)])
    assert score(ranking, table, judged)["kendall_tau"] == 1.0
