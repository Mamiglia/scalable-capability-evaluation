from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.judge_verdict_analysis.config import CONFIG
from scripts.judge_verdict_analysis.hierarchy import outer_roles

from .pairwise_wrapper import (
    CAPABILITY_TASK,
    _capability_design,
    chance_predictions,
    fit_fold,
    overall_model_accuracy,
    prepare_capability_fold,
    prepare_capability_rows,
)


def _responses() -> pd.DataFrame:
    return pd.DataFrame([
        {"bench": "b", "model": model, "item": f"i{item:02d}",
         "correct": int(item < solved), "is_canonical": True}
        for model, solved in (("m0", 8), ("m1", 6), ("m2", 4), ("m3", 2))
        for item in range(8)
    ])


def _pairs() -> pd.DataFrame:
    ability = {"m0": 3.0, "m1": 2.0, "m2": 1.0, "m3": 0.0}
    rows = []
    for item in range(8):
        for pair_index, (first, second) in enumerate(
            (("m0", "m1"), ("m0", "m2"), ("m0", "m3"),
             ("m1", "m2"), ("m1", "m3"), ("m2", "m3"))
        ):
            left, right = ((first, second) if (item + pair_index) % 2
                           else (second, first))
            for judge in ("j0", "j1"):
                difference = ability[left] - ability[right]
                rows.append({
                    "bench": "b", "item": f"i{item:02d}", "judge": judge,
                    "judge_log": judge, "pick_left": int(difference > 0),
                    "model_left": left, "model_right": right,
                    "log_left": left, "log_right": right,
                    "signal": difference, "other": difference * 0.5,
                })
    return pd.DataFrame(rows)


def _config(tmp_path: Path, feature_space: str = "raw", penalty: str = "ridge"):
    return replace(
        CONFIG, results_dir=tmp_path, benchmarks=("b",), judges=("j0", "j1"),
        model_folds=tuple((f"model-fold-{i}", (f"m{i}",)) for i in range(4)),
        item_folds=4, inner_folds=2, fit_n_jobs=1, fit_pca_components=2,
        elastic_net_cs=(1.0,), ridge_cs=(1.0,), glm_max_iter=1_000,
        feature_space=feature_space, penalty=penalty, execution_mode="prepare",
    )


def test_overall_accuracy_is_canonical_and_equal_item_weighted() -> None:
    responses = pd.DataFrame([
        {"bench": "b", "model": "a", "item": "i0", "correct": 1, "is_canonical": True},
        {"bench": "b", "model": "a", "item": "i0", "correct": 1, "is_canonical": True},
        {"bench": "b", "model": "a", "item": "i1", "correct": 0, "is_canonical": True},
        {"bench": "b", "model": "a", "item": "i1", "correct": 1, "is_canonical": False},
        {"bench": "b", "model": "b", "item": "i0", "correct": 1, "is_canonical": True},
        {"bench": "b", "model": "b", "item": "i1", "correct": 1, "is_canonical": True},
    ])

    assert overall_model_accuracy(responses).to_dict("records") == [
        {"bench": "b", "model": "a", "accuracy": 0.5},
        {"bench": "b", "model": "b", "accuracy": 1.0},
    ]


def test_preparation_deduplicates_judges_checks_support_and_uses_shared_roles(tmp_path) -> None:
    pairs = _pairs()
    dictionary = pd.DataFrame({"feature": ["signal", "other"], "block": "surface"})
    config = _config(tmp_path)

    prepared = prepare_capability_rows(pairs, _responses(), dictionary)
    roles = outer_roles(prepared, {"m0"}, 2, config.item_folds)
    train, test = prepare_capability_fold(
        pairs, _responses(), dictionary, {"m0"}, 2, config,
    )

    assert len(prepared) == len(pairs) // 2
    assert {"judge", "judge_log", "pick_left"}.isdisjoint(prepared)
    assert set(train.item) | set(test.item) == set(prepared.item)
    assert train.item.tolist() == prepared.loc[roles == "train", "item"].tolist()
    assert test.item.tolist() == prepared.loc[roles == "test", "item"].tolist()
    assert not (prepared.accuracy_left == prepared.accuracy_right).any()
    assert set(prepared.left_stronger) == {0, 1}

    missing = pairs.iloc[[0]].assign(model_left="unsupported")
    with pytest.raises(ValueError, match="missing overall accuracy for left"):
        prepare_capability_rows(missing, _responses(), dictionary)


def test_design_and_null_are_exactly_antisymmetric(tmp_path) -> None:
    frame = pd.DataFrame({
        "bench": ["b"], "item": ["i"], "model_left": ["l"],
        "model_right": ["r"], "signal": [2.0], "left_stronger": [1],
    })
    reversed_frame = frame.assign(
        model_left="r", model_right="l", signal=-frame.signal, left_stronger=0,
    )
    levels = ("b",)

    design, _ = _capability_design(frame, ["signal"], levels, _config(tmp_path))
    reversed_design, _ = _capability_design(
        reversed_frame, ["signal"], levels, _config(tmp_path),
    )
    np.testing.assert_array_equal(reversed_design.toarray(), -design.toarray())
    assert CAPABILITY_TASK.outcome == "left_stronger"
    assert CAPABILITY_TASK.group_columns == ("bench", "item")
    assert set(_capability_design(
        frame, ["signal"], levels, _config(tmp_path),
    )[1].block) == {"population_slope", "benchmark_slope"}
    assert chance_predictions(frame, ["signal"]).probability.tolist() == [0.5]


@pytest.mark.parametrize("feature_space", ("raw", "pc"))
@pytest.mark.parametrize("penalty", ("l1", "ridge"))
def test_fit_fold_runs_every_supported_variant(tmp_path, feature_space, penalty) -> None:
    dictionary = pd.DataFrame({
        "feature": ["signal", "other"], "block": ["surface", "surface"],
    })
    config = _config(tmp_path, feature_space, penalty)

    output = fit_fold(_pairs(), _responses(), dictionary, 0, config)

    assert output == config.fit_dir / f"{feature_space}-{penalty}" / "model-fold-0__item-0"
    predictions = pd.read_parquet(output / "predictions.parquet")
    assert predictions.probability.between(0, 1).all()
    assert {"judge", "pick_left"}.isdisjoint(predictions)
