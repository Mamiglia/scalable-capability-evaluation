from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from . import fit, lesioning
from .config import CONFIG
from .fit_common import FitResult
from .fit_frequentist import select_features
from .hierarchy import HierarchyLevels
from .lesioning import _bootstrap_summary, lesion_fold


def _rows() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for bench in ("a", "b"):
        for item_index in range(12):
            signal = -2.0 if item_index % 2 == 0 else 2.0
            for judge in ("j1", "j2"):
                rows.append({
                    "bench": bench, "judge": judge, "item": f"item-{item_index}",
                    "fold": "synthetic", "pick_left": int(signal > 0),
                    "signal": signal, "noise": float((item_index + (bench == "b")) % 3 - 1),
                    "dropped": 0.0,
                })
    frame = pd.DataFrame(rows)
    return (
        frame[~frame.item.isin(("item-10", "item-11"))].reset_index(drop=True),
        frame[frame.item.isin(("item-10", "item-11"))].reset_index(drop=True),
    )


def test_fixed_selection_lesion_pairs_each_refit_with_the_full_predictions() -> None:
    """Fails if lesions retune, misalign rows, or leave derived feature terms behind."""
    train, test = _rows()
    full_predictions = test.drop(columns=["signal", "noise", "dropped", "pick_left"]).copy()
    full_predictions["observed"] = test.pick_left
    full_predictions["probability"] = np.where(test.pick_left == 1, 0.99, 0.01)
    full_fit = FitResult(
        backend="frequentist", fold="synthetic", selected_features=("signal", "noise", "dropped"),
        predictions=full_predictions, coefficients=pd.DataFrame(), tuning=pd.DataFrame(),
        metrics=pd.DataFrame(), diagnostics={"ridge_C": 1.0},
    )
    config = replace(CONFIG, glm_max_iter=2_000, random_seed=4)

    lesions = lesion_fold(full_fit, train, test, config)

    assert set(lesions.feature) == {"signal", "noise", "dropped"}
    assert lesions.groupby("feature").size().to_dict() == {
        "dropped": len(test), "noise": len(test), "signal": len(test),
    }
    assert lesions.groupby("feature").delta_log_loss.mean()["signal"] > lesions.groupby("feature").delta_log_loss.mean()["noise"]
    assert lesions.groupby("feature").delta_log_loss.mean()["signal"] > 0
    for _, lesion in lesions.groupby("feature"):
        pd.testing.assert_frame_equal(
            lesion.loc[:, full_predictions.columns].reset_index(drop=True),
            full_predictions.reset_index(drop=True),
        )


def test_lesions_keep_all_regressors_when_elastic_net_drops_one(tmp_path, monkeypatch) -> None:
    """Fails if lesions use saved elastic-net selection rather than every regressor."""
    train, test = _rows()
    config = replace(
        CONFIG, results_dir=tmp_path / "results", execution_mode="execute", execution_target="local_test",
        inner_folds=3, elastic_net_cs=(0.1,), elastic_net_l1_ratios=(1.0,), ridge_cs=(1.0,),
        bootstrap_iterations=2,
    )
    features = ["signal", "noise", "dropped"]
    selected, _ = select_features(train, features, config, HierarchyLevels.from_training(train))
    assert "dropped" not in selected
    calls: list[tuple[str, ...]] = []
    ridge_calls: list[float] = []
    tuning_calls: list[tuple[str, ...]] = []
    captured = []

    def fixed_ridge(train_rows, test_rows, names, _, ridge_c):
        calls.append(tuple(names))
        ridge_calls.append(ridge_c)
        predictions = test_rows.drop(columns=features + ["pick_left"]).copy()
        predictions["observed"] = test_rows.pick_left
        predictions["probability"] = np.where(test_rows.pick_left == 1, 0.99, 0.01)
        return FitResult(
            backend="frequentist", fold="synthetic", selected_features=tuple(names),
            predictions=predictions, coefficients=pd.DataFrame(), tuning=pd.DataFrame(),
            metrics=pd.DataFrame(), diagnostics={"ridge_C": ridge_c},
        )

    def tune(rows, names, *_):
        tuning_calls.append(tuple(names))
        return 1.0, pd.DataFrame()

    monkeypatch.setattr(fit, "_guard_execution", lambda _: None)
    monkeypatch.setattr(fit, "_load_rows", lambda _: (pd.concat([train, test], ignore_index=True), features))
    monkeypatch.setattr(fit, "_outer_folds", lambda *_: [("synthetic", train, test)])
    monkeypatch.setattr(lesioning, "tune_ridge", tune)
    monkeypatch.setattr(lesioning, "fit_fixed_ridge", fixed_ridge)
    monkeypatch.setattr(lesioning, "_write_lesions", lambda rows, *_: captured.append(rows))

    lesioning.run_lesions(config)

    assert tuning_calls == [tuple(features)]
    assert calls[0] == tuple(features)
    assert set(calls[1:]) == {
        ("noise", "dropped"), ("signal", "dropped"), ("signal", "noise"),
    }
    assert ridge_calls == [1.0] * (len(features) + 1)
    assert set(captured[0].feature) == set(features)
    assert len(captured[0]) == len(test) * len(features)


def test_clustered_summary_reports_both_deltas_and_the_configured_primary_metric() -> None:
    """Fails if bootstrap samples rows instead of benchmark/item clusters or drops accuracy."""
    rows = pd.DataFrame({
        "backend": ["frequentist"] * 4,
        "feature": ["signal"] * 4,
        "bench": ["a", "a", "b", "b"],
        "item": ["i0", "i0", "i1", "i1"],
        "fold": ["f0", "f1", "f0", "f1"],
        "delta_log_loss": [0.1, 0.5, 0.2, 0.2],
        "delta_accuracy": [0.0, 0.4, 0.2, 0.2],
    })
    summary = _bootstrap_summary(
        rows, replace(CONFIG, lesion_metric="accuracy", bootstrap_iterations=20, random_seed=7)
    ).iloc[0]

    assert summary.lesion_metric == "accuracy"
    assert summary.estimate == summary.delta_accuracy == 0.2
    assert summary.delta_log_loss == pytest.approx(0.25)
    assert summary.lower == summary.delta_log_loss_lower
    assert summary.upper == summary.delta_log_loss_upper
    assert summary.fold_count == 2
    assert summary.row_count == 4
    assert summary.delta_accuracy_lower == summary.delta_accuracy_upper == 0.2
    assert summary.delta_log_loss_lower <= summary.delta_log_loss <= summary.delta_log_loss_upper


def test_remote_lesion_guard_rejects_before_loading_rows(monkeypatch) -> None:
    """Fails if lesions bypass the fit facade's unavailable-target guard."""
    config = replace(CONFIG, execution_mode="execute", execution_target="swc_hpc")
    monkeypatch.setattr(fit, "_active_remote_target", lambda _: False)
    monkeypatch.setattr(fit, "_load_rows", lambda _: pytest.fail("guard loaded rows"))

    with pytest.raises(RuntimeError, match="refusing heavy local fit"):
        lesioning.run_lesions(config)
