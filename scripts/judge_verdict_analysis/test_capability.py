from dataclasses import replace
import hashlib
import json

import numpy as np
import pandas as pd

from .capability import fit_capability_companion, prepare_capability_difference
from .config import CONFIG
from .final_report import VARIANTS, capability_companion_results, ceiling_by_judge
from .pooling import POOLING_PARAMETERS


def _rows() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.DataFrame([
        {"bench": "a", "item": "i", "pair_id": "p1", "judge": "j1",
         "model_left": "m3", "model_right": "m1", "accuracy_left": 1.0, "accuracy_right": 0.0},
        {"bench": "a", "item": "i", "pair_id": "p1", "judge": "j2",
         "model_left": "m3", "model_right": "m1", "accuracy_left": 1.0, "accuracy_right": 0.0},
        {"bench": "a", "item": "i", "pair_id": "p2", "judge": "j1",
         "model_left": "m3", "model_right": "m2", "accuracy_left": 1.0, "accuracy_right": 0.5},
        {"bench": "b", "item": "i", "pair_id": "p3", "judge": "j1",
         "model_left": "m5", "model_right": "m4", "accuracy_left": 0.75, "accuracy_right": 0.25},
        {"bench": "b", "item": "i", "pair_id": "p4", "judge": "j1",
         "model_left": "m4", "model_right": "m5", "accuracy_left": 0.25, "accuracy_right": 0.75},
    ])
    test = pd.DataFrame([
        {"bench": "a", "item": "held", "pair_id": "t1", "judge": "j1",
         "model_left": "m6", "model_right": "m1", "accuracy_left": 1.0, "accuracy_right": 0.0},
        {"bench": "a", "item": "held", "pair_id": "t2", "judge": "j1",
         "model_left": "m1", "model_right": "m6", "accuracy_left": 0.0, "accuracy_right": 1.0},
    ])
    return train, test


def test_capability_scaling_uses_unique_outer_training_side_scores_by_benchmark() -> None:
    train, test = _rows()

    _, _, scaling = prepare_capability_difference(train, test)

    a, b = scaling.set_index("bench").loc[["a", "b"]].itertuples(index=False)
    assert (a.score_count, a.score_mean, a.score_sd) == (3, 0.5, 0.5)
    assert b.score_count == 2 and b.score_mean == 0.5
    assert np.isclose(b.score_sd, np.sqrt(0.125))


def test_capability_difference_deduplicates_judges_and_keeps_reversals_antisymmetric() -> None:
    train, test = _rows()

    prepared_train, prepared_test, scaling = prepare_capability_difference(train, test)

    np.testing.assert_allclose(scaling.delta_sd, 1.5)
    np.testing.assert_allclose(
        prepared_train.capability_difference,
        [4 / 3, 4 / 3, 2 / 3, np.sqrt(8) / 3, -np.sqrt(8) / 3],
    )
    np.testing.assert_allclose(prepared_test.capability_difference, [4 / 3, -4 / 3])
    assert prepared_train.capability_difference.mean() != 0


def test_capability_companion_reuses_the_frozen_full_model() -> None:
    train, test = _rows()
    for frame in (train, test):
        frame["fold"] = "outer-0"
        frame["feature"] = np.linspace(-1, 1, len(frame))
    train["pick_left"] = [1, 0, 1, 0, 1]
    test["pick_left"] = [1, 0]
    frozen = {"C": 1.0, "l1_ratio": 1.0, **dict.fromkeys(POOLING_PARAMETERS, 1.0)}
    config = replace(CONFIG, feature_space="raw", penalty="l1", glm_max_iter=1_000)

    result, scaling = fit_capability_companion(
        train, test, ["feature"], frozen, config,
    )

    assert result.backend == "raw-l1-capability"
    assert set(result.coefficients.feature) == {"feature", "capability_difference"}
    assert all(result.diagnostics[name] == frozen[name] for name in frozen)
    assert set(scaling) == {"bench", "score_mean", "score_sd", "score_count", "delta_sd"}


def test_capability_companion_reports_increment_and_coefficient_shift(tmp_path) -> None:
    cells = []
    for variant in ("raw-l1", "pc-l1"):
        root = tmp_path / VARIANTS[variant]
        (root / "fit").mkdir(parents=True)
        (root / "capability-fit").mkdir()
        cells.append({
            "variant": variant, "judge": "j0", "bench": "b0",
            "accuracy": 0.7, "log_loss": 0.6, "row_count": 10,
        })
        pd.DataFrame([{
            "judge": "j0", "bench": "b0", "accuracy": 0.72,
            "log_loss": 0.55, "row_count": 10,
        }]).to_csv(root / "capability-fit/judge-benchmark.csv", index=False)
        predictions = pd.DataFrame({
            "fold": ["f0"] * 10, "judge": ["j0"] * 10,
            "bench": ["b0"] * 10, "item": [f"i{i}" for i in range(10)],
            "observed": [1, 0] * 5, "probability": [0.7, 0.3] * 5,
        })
        predictions.to_parquet(root / "fit/predictions.parquet", index=False)
        predictions.assign(probability=[0.8, 0.2] * 5).to_parquet(
            root / "capability-fit/predictions.parquet", index=False)
        coefficient = pd.DataFrame([{
            "fold": "f0", "feature": "x", "judge": "j0", "benchmark": "b0",
            "effective": 0.2,
        }])
        coefficient.to_csv(root / "fit/coefficients.csv", index=False)
        coefficient.assign(effective=0.25).to_csv(
            root / "capability-fit/coefficients.csv", index=False)
        artifacts = {
            name: hashlib.sha256((root / "capability-fit" / name).read_bytes()).hexdigest()
            for name in ("predictions.parquet", "coefficients.csv")
        }
        (root / "capability-fit/manifest.json").write_text(json.dumps({
            "state": "completed", "artifact_sha256": artifacts,
        }))

    increments, shifts = capability_companion_results(tmp_path, pd.DataFrame(cells))

    overall = increments[increments.scope == "overall"]
    np.testing.assert_allclose(overall.delta_accuracy, 0)
    assert (overall.delta_log_loss_gain > 0).all()
    np.testing.assert_allclose(shifts.delta_effective, 0.05)


def test_judge_ceiling_summary_uses_direct_accuracy_ratio(tmp_path) -> None:
    cells = pd.DataFrame([
        {"variant": variant, "judge": "j", "bench": "b", "accuracy": 0.72,
         "row_count": 10}
        for variant in VARIANTS
    ])
    ceiling = tmp_path / "ceiling.csv"
    pd.DataFrame([{
        "judge": "j", "benchmark": "b", "wilson_95_low": 0.75,
        "raw_agreement": 0.8,
    }]).to_csv(ceiling, index=False)

    summary = ceiling_by_judge(tmp_path, cells, ceiling).iloc[0]

    assert np.isclose(summary.raw_l1_accuracy_over_raw_ceiling, 0.9)
    assert np.isclose(summary.raw_l1_accuracy_over_conservative_ceiling, 0.96)
    assert "raw_l1_recovery_vs_raw" not in summary.index
