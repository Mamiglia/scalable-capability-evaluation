from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from .config import CONFIG
from .fit_frequentist import fit_frequentist


pytestmark = pytest.mark.filterwarnings("error::FutureWarning")


def test_sparse_signal_is_selected_and_refit_with_the_full_hierarchy() -> None:
    rows = []
    for bench_index, bench in enumerate(("a", "b")):
        for item_index in range(18):
            signal = -2.0 if item_index % 2 == 0 else 2.0
            for judge in ("j1", "j2"):
                rows.append({
                    "bench": bench,
                    "judge": judge,
                    "item": f"item-{item_index}",
                    "fold": "synthetic",
                    "pick_left": int(signal > 0),
                    "signal": signal,
                    "noise": float((item_index + bench_index) % 3 - 1),
                })
    frame = pd.DataFrame(rows)
    train = frame[~frame.item.isin(("item-16", "item-17"))].reset_index(drop=True)
    test = frame[frame.item.isin(("item-16", "item-17"))].reset_index(drop=True)
    config = replace(
        CONFIG,
        inner_folds=3,
        elastic_net_cs=(0.1, 1.0),
        elastic_net_l1_ratios=(1.0,),
        selection_min_fold_share=2 / 3,
        ridge_cs=(0.1, 1.0),
        glm_max_iter=2_000,
        random_seed=4,
    )

    serial = fit_frequentist(
        train, test, ["signal", "noise"], replace(config, fit_n_jobs=1)
    )
    result = fit_frequentist(
        train, test, ["signal", "noise"], replace(config, fit_n_jobs=2)
    )

    assert result.backend == "frequentist"
    assert result.fold == "synthetic"
    assert result.selected_features == ("signal",)
    assert {"population", "judge_deviation", "benchmark_deviation",
            "cell_deviation", "effective"} <= set(result.coefficients)
    assert set(result.coefficients.feature) == {"signal"}
    assert np.isfinite(result.coefficients.select_dtypes(include="number")).all().all()
    assert np.isfinite(result.predictions.probability).all()
    assert result.predictions.observed.tolist() == test.pick_left.tolist()
    assert set(result.tuning.stage) == {"elastic_net", "ridge"}
    assert set(result.metrics.metric) == {"accuracy", "log_loss"}
    assert np.isfinite(result.metrics.value).all()
    assert result.selected_features == serial.selected_features
    np.testing.assert_allclose(result.predictions.probability, serial.predictions.probability)
    pd.testing.assert_frame_equal(result.tuning, serial.tuning)
