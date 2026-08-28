from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
import yaml
from numpyro import handlers

from . import fit_hibayes
from .config import AnalysisConfig
from .fit_hibayes import evaluate_hibayes, judge_verdict_hibayes, write_hibayes_job


PRIOR_SCALES = {
    "population": 0.5,
    "judge": 0.5,
    "benchmark": 0.5,
    "cell": 0.5,
    "item": 0.5,
}


def test_hibayes_trace_has_crossed_slopes_and_item_intercepts_only() -> None:
    features = {
        "obs": jnp.array([1, 0, 1]),
        "judge_index": jnp.array([0, 1, 0]),
        "num_judge": 2,
        "bench_index": jnp.array([0, 1, 1]),
        "num_bench": 2,
        "cell_index": jnp.array([0, 3, 1]),
        "num_cell": 4,
        "item_key_index": jnp.array([0, 1, 2]),
        "num_item_key": 3,
        "signal": jnp.array([1.0, -1.0, 0.5]),
        "noise": jnp.array([0.0, 0.5, -0.5]),
    }

    trace = handlers.trace(handlers.seed(
        judge_verdict_hibayes(predictors=("signal", "noise"), prior_scales=PRIOR_SCALES), rng_seed=0
    )).get_trace(features)

    assert trace["population_slopes"]["value"].shape == (2,)
    assert trace["judge_slope_deviations"]["value"].shape == (2, 2)
    assert trace["benchmark_slope_deviations"]["value"].shape == (2, 2)
    assert trace["cell_slope_deviations"]["value"].shape == (4, 2)
    assert trace["item_intercepts"]["value"].shape == (3,)
    assert trace["obs"]["value"].shape == (3,)
    assert not any(name.startswith("item_slope") for name in trace)


def test_hibayes_prior_scales_are_explicit_positive_config() -> None:
    config = AnalysisConfig()

    assert {
        config.hibayes_population_prior_scale,
        config.hibayes_judge_prior_scale,
        config.hibayes_benchmark_prior_scale,
        config.hibayes_cell_prior_scale,
        config.hibayes_item_prior_scale,
    } == {0.5}
    with pytest.raises(ValueError, match="hibayes_judge_prior_scale"):
        AnalysisConfig(hibayes_judge_prior_scale=0).validate()
    with pytest.raises(ValueError, match="hibayes_item_prior_scale"):
        AnalysisConfig(hibayes_item_prior_scale=float("nan")).validate()
    with pytest.raises(ValueError, match="positive values"):
        judge_verdict_hibayes(
            predictors=("signal",), prior_scales=PRIOR_SCALES | {"item": float("nan")}
        )


def test_write_hibayes_job_uses_selected_features_and_fold_inputs(tmp_path: Path) -> None:
    train_path = tmp_path / "train.parquet"
    test_path = tmp_path / "test.parquet"
    pd.DataFrame({
        "judge": ["j0", "j1"],
        "bench": ["b0", "b1"],
        "item": ["i0", "i1"],
        "pick_left": [1, 0],
        "signal": [1.0, -1.0],
        "discarded": [9.0, 9.0],
    }).to_parquet(train_path, index=False)
    pd.DataFrame({
        "judge": ["j0"],
        "bench": ["b0"],
        "item": ["i2"],
        "pick_left": [1],
        "signal": [0.5],
        "discarded": [9.0],
    }).to_parquet(test_path, index=False)
    config = AnalysisConfig(
        results_dir=tmp_path / "results", mcmc_warmup=3, mcmc_samples=4, mcmc_chains=1
    ).validate()

    job_dir = write_hibayes_job(
        train_path, test_path, ["signal"], "model-0__item-0", config
    )

    assert job_dir == config.fit_dir / "hibayes" / "model-0__item-0"
    frame = pd.concat([
        pd.read_parquet(job_dir / "train.parquet"),
        pd.read_parquet(job_dir / "test.parquet"),
    ], ignore_index=True)
    assert frame.test.tolist() == [False, False, True]
    assert frame.columns.tolist() == [
        "judge", "bench", "item", "pick_left", "signal", "test", "cell", "item_key"
    ]
    job = yaml.safe_load((job_dir / "config.yaml").read_text())
    model_config = job["model"]["models"][0]["config"]
    assert job["data_loader"]["paths"]["extracted_data"] == [
        str(job_dir / "train.parquet"), str(job_dir / "test.parquet")
    ]
    assert job["model"]["path"] == "scripts/judge_verdict_analysis/fit_hibayes.py"
    assert model_config["predictors"] == ["signal"]
    assert model_config["prior_scales"] == PRIOR_SCALES
    assert model_config["fit"] == {"samples": 4, "warmup": 3, "chains": 1, "seed": 0}


def test_evaluate_hibayes_uses_saved_draws_without_sampling(tmp_path: Path, monkeypatch) -> None:
    train_path = tmp_path / "train.parquet"
    test_path = tmp_path / "test.parquet"
    train = pd.DataFrame({
        "judge": ["j0", "j1"],
        "bench": ["b0", "b1"],
        "item": ["i0", "i1"],
        "pick_left": [1, 0],
        "signal": [1.0, -1.0],
    })
    test = pd.DataFrame({
        "judge": ["j0", "j1"],
        "bench": ["b0", "b1"],
        "item": ["i2", "i2"],
        "pick_left": [1, 0],
        "signal": [1.0, -1.0],
    })
    train.to_parquet(train_path, index=False)
    test.to_parquet(test_path, index=False)
    config = AnalysisConfig(results_dir=tmp_path / "results").validate()
    job_dir = write_hibayes_job(train_path, test_path, ["signal"], "fold-0", config)
    draws = 4
    posterior = {
        "population_intercept": np.zeros(draws),
        "population_slopes": np.arange(draws, dtype=float)[:, None],
        "judge_intercepts": np.zeros((draws, 2)),
        "judge_slope_deviations": np.zeros((draws, 2, 1)),
        "benchmark_intercepts": np.zeros((draws, 2)),
        "benchmark_slope_deviations": np.zeros((draws, 2, 1)),
        "cell_intercepts": np.zeros((draws, 4)),
        "cell_slope_deviations": np.zeros((draws, 4, 1)),
        "item_intercepts": np.zeros((draws, 4)),
    }
    monkeypatch.setattr(fit_hibayes, "_load_posterior", lambda _: posterior)

    result = evaluate_hibayes(job_dir, test, config)

    assert result.backend == "hibayes"
    assert result.fold == "fold-0"
    assert result.selected_features == ("signal",)
    np.testing.assert_allclose(result.predictions.probability, [0.76610745, 0.23389255])
    assert result.predictions.observed.tolist() == [1, 0]
    assert len(result.coefficients) == 4
    np.testing.assert_allclose(result.coefficients.effective, 1.5)
    np.testing.assert_allclose(result.coefficients.lower, 0.075)
    np.testing.assert_allclose(result.coefficients.upper, 2.925)
    assert result.metrics.metric.tolist() == ["accuracy", "log_loss"]
    assert result.metrics.value.tolist() == pytest.approx([1.0, -np.log(0.76610745)])
    assert result.diagnostics["posterior_draws"] == 4
