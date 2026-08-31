from dataclasses import replace

import numpy as np
import pandas as pd

from .config import CONFIG
from .fit_frequentist import fit_sparse, tune_sparse
from .hierarchy import HierarchyLevels, hierarchical_design
from .pooling import POOLING_PARAMETERS, estimate_pooling_ratios, penalty_multiplier


def test_relative_penalties_use_family_specific_scaling_and_keep_global_intercept() -> None:
    frame = pd.DataFrame([
        {"judge": judge, "bench": bench, "item": item, "feature": 1.0}
        for judge in ("j0", "j1")
        for bench in ("b0", "b1")
        for item in ("i0", "i1")
    ])
    penalties = {
        "judge_deviation_penalty": 4.0,
        "benchmark_deviation_penalty": 9.0,
        "cell_deviation_penalty": 16.0,
        "item_intercept_penalty": 25.0,
    }
    expected = {
        "l1": {"judge_intercept": 1 / 4, "benchmark_intercept": 1 / 9,
               "cell_intercept": 1 / 16, "item_intercept": 1 / 25},
        "ridge": {"judge_intercept": 1 / 2, "benchmark_intercept": 1 / 3,
                  "cell_intercept": 1 / 4, "item_intercept": 1 / 5},
    }

    assert penalty_multiplier(4, "l1") == 1 / 4
    assert penalty_multiplier(4, "ridge") == 1 / 2
    for penalty, wanted in expected.items():
        design, columns = hierarchical_design(
            frame, ("feature",), HierarchyLevels.from_training(frame), penalties,
            penalty=penalty,
        )
        intercept = columns[columns.block == "global_intercept"].iloc[0]
        np.testing.assert_array_equal(design[:, int(intercept.column)].toarray(), 1)
        assert intercept.multiplier == 1
        for block, multiplier in wanted.items():
            assert columns.loc[columns.block == block, "multiplier"].unique().tolist() == [multiplier]


def test_pooling_ratios_are_tuned_once_and_frozen_for_lesions() -> None:
    rows = [
        {
            "judge": judge, "bench": f"b{item_index % 2}", "item": f"i{item_index}",
            "fold": "outer-0", "pick_left": int(sign > 0),
            "feature": sign * (1 + judge_index),
        }
        for item_index in range(8)
        for judge_index, judge in enumerate(("j0", "j1"))
        for sign in (-1.0, 1.0)
    ]
    frame = pd.DataFrame(rows)
    train = frame[~frame.item.isin(("i6", "i7"))].reset_index(drop=True)
    test = frame[frame.item.isin(("i6", "i7"))].reset_index(drop=True)
    config = replace(
        CONFIG, inner_folds=2, fit_n_jobs=1, elastic_net_cs=(0.1, 1.0),
        pooling_svi_steps=150, pooling_svi_learning_rate=0.03,
        glm_max_iter=1_000,
    )

    estimated, variance_fit = estimate_pooling_ratios(
        train, ["feature"], "l1", config,
    )
    frozen, tuning = tune_sparse(train, ["feature"], "l1", config)
    full = fit_sparse(train, test, ["feature"], "l1", frozen, config, tuning)
    lesion = fit_sparse(train, test, [], "l1", frozen, config)

    assert set(estimated) == set(POOLING_PARAMETERS)
    assert all(np.isfinite(value) and value > 0 for value in estimated.values())
    assert variance_fit.record_type.item() == "variance_component"
    assert set(frozen) == {"C", "l1_ratio", *POOLING_PARAMETERS}
    assert set(tuning.record_type) == {"variance_component", "tuning"}
    selected = tuning[tuning.selected]
    assert len(selected) == 1
    assert selected.iloc[0].C == frozen["C"]
    for name in POOLING_PARAMETERS:
        assert selected.iloc[0][name] == frozen[name]
        assert full.diagnostics[name] == lesion.diagnostics[name] == frozen[name]
