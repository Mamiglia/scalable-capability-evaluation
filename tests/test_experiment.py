import pandas as pd
import pytest

from configs.domain import DomainBench, DomainConfig
from configs.experiment import ExperimentConfig
from configs.model import ModelConfig


def domain(items, responses):
    d = DomainConfig(name="dom", benches=[DomainBench(bench="b1")])
    d.load_items = lambda: items
    d.load_responses = lambda: responses
    return d


ITEMS = pd.DataFrame({"bench": "b1", "item": ["1", "2", "3", "4"], "quartile": [1, 2, 3, 4]})
# Q4 (item 4): a > b > c. Q1 (item 1): b = c > a. Two repeats of b on item 4 average to 0.5.
RESPONSES = pd.DataFrame(
    [("a", "4", 0, 1), ("b", "4", 0, 1), ("b", "4", 1, 0), ("c", "4", 0, 0),
     ("a", "1", 0, 0), ("b", "1", 0, 1), ("c", "1", 0, 1)],
    columns=["model", "item", "repeat", "correct"]).assign(bench="b1")
MODELS = [ModelConfig(name=n, slug=n) for n in "abc"]


def experiment(**kwargs):
    kwargs = dict(name="exp", domain=domain(ITEMS, RESPONSES), judge=ModelConfig(name="j", slug="j"),
                  models=list(MODELS), q_src=1) | kwargs
    return ExperimentConfig(**kwargs)


def test_pairs_are_ordered_by_q_dst_accuracy():
    names = lambda pairs: [(M.name, m.name) for M, m in pairs]
    assert names(experiment().ordered_pairs()) == [("a", "b"), ("a", "c"), ("b", "c")]
    # Ground truth moves with q_dst; a tie is broken by name.
    assert names(experiment(q_dst=1).ordered_pairs()) == [("b", "a"), ("c", "a"), ("b", "c")]


def test_explicit_pairs_replace_the_combinations():
    exp = experiment(models=[], pairs=[[MODELS[2], MODELS[0]]])
    assert [(M.name, m.name) for M, m in exp.ordered_pairs()] == [("a", "c")]
    assert [m.name for m in exp.models] == ["c", "a"]
    with pytest.raises(ValueError, match="not both"):
        experiment(pairs=[[MODELS[0], MODELS[1]]])


def test_items_are_the_quartile_per_bench():
    assert experiment().items(1) == {"b1": {"1"}}
    assert experiment().items(4) == {"b1": {"4"}}


def test_model_without_responses_raises():
    exp = experiment(models=MODELS[:2] + [ModelConfig(name="zzz", slug="zzz")])
    with pytest.raises(ValueError, match="zzz"):
        exp.ordered_pairs()


def test_log_dir_separates_conditions():
    plain, other = experiment(), experiment(q_src=2)
    assert plain.log_dir.parent == other.log_dir.parent
    assert plain.log_dir != other.log_dir
    assert experiment(include_reasoning=False).log_dir != plain.log_dir


def test_bad_quartile_and_model_count_raise():
    with pytest.raises(ValueError):
        experiment(q_src=5)
    with pytest.raises(ValueError):
        experiment(models=MODELS[:1])


def test_eval_kwargs_only_carry_limit():
    assert experiment(limit=5).eval_kwargs() == {"limit": 5}
    assert experiment().eval_kwargs() == {}
