import numpy as np
import pandas as pd
import pytest


def _calibration(draws=1):
    return {
        "judges": ["j"], "sources": ["s"], "conditions": ["c"],
        "features": [], "feature_mean": np.array([]), "feature_scale": np.array([]),
        "alpha": np.zeros((draws, 1)), "source": np.zeros((draws, 1, 1)),
        "condition": np.zeros((draws, 1, 1)),
        "discrimination": np.ones((draws, 1)), "beta": np.zeros((draws, 1, 0)),
    }


def _capture_target_mcmc(monkeypatch, module):
    captured = {}

    class FakeMCMC:
        def __init__(self, kernel, **kwargs):
            pass

        def run(self, key, **kwargs):
            captured.update(kwargs)

        def get_samples(self):
            return {"theta": np.zeros((2, captured["n_targets"]))}

    monkeypatch.setattr(module, "NUTS", lambda model, **_kwargs: model)
    monkeypatch.setattr(module, "MCMC", FakeMCMC)
    return captured


def _infer_one_target(module, calibration):
    frame = pd.DataFrame([{
        "judge": "j", "source": "s", "condition": "c",
        "left_model": "target", "right_model": "anchor", "chosen_left": 1.0,
        "feature": 0.0,
    }])
    anchors = pd.DataFrame({"model": ["anchor"], "theta": [0.0]})
    return module.infer_targets(
        frame, anchors, calibration, warmup=1, samples=1, seed=0,
    )


def _forbid_mcmc(*args, **kwargs):
    pytest.fail("MCMC must not run")


def test_linear_predictor_reverses_capability_and_features_not_position():
    from bias_model import linear_predictor

    forward = linear_predictor(
        alpha=np.array([0.3]), source=np.array([[0.1]]),
        condition=np.array([[0.0]]), discrimination=np.array([2.0]),
        delta_theta=np.array([0.5]), beta=np.array([[1.5]]),
        design=np.array([[0.2]]), judge=np.array([0]),
        source_index=np.array([0]), condition_index=np.array([0]),
    )[0]
    reverse = linear_predictor(
        alpha=np.array([0.3]), source=np.array([[0.1]]),
        condition=np.array([[0.0]]), discrimination=np.array([2.0]),
        delta_theta=np.array([-0.5]), beta=np.array([[1.5]]),
        design=np.array([[-0.2]]), judge=np.array([0]),
        source_index=np.array([0]), condition_index=np.array([0]),
    )[0]

    assert np.isclose(forward + reverse, 0.8)


def test_model_sites_have_expected_shapes():
    from numpyro import handlers
    from bias_model import judge_bias_model

    values = handlers.trace(handlers.seed(judge_bias_model, rng_seed=0)).get_trace(
        obs=np.array([1.0, 0.0]), judge=np.array([0, 1]),
        source_index=np.array([0, 2]), condition_index=np.array([0, 1]),
        delta_theta=np.array([1.0, -1.0]), design=np.ones((2, 2)),
        n_judges=2, n_sources=3, n_conditions=2,
    )

    assert values["judge_beta"]["value"].shape == (2, 2)
    assert values["judge_discrimination"]["value"].shape == (2,)
    assert values["source_mean"]["value"].shape == (2,)
    assert values["source_scale"]["value"].shape == (2,)
    assert values["source_raw"]["value"].shape == (2, 2)
    assert values["condition_mean"]["value"].shape == (1,)
    assert values["condition_scale"]["value"].shape == (1,)
    assert values["condition_raw"]["value"].shape == (2, 1)
    assert np.allclose(values["judge_source"]["value"][:, 0], 0)
    assert np.allclose(values["judge_condition"]["value"][:, 0], 0)


def test_model_supports_baseline_only_categories_and_no_features():
    from numpyro import handlers
    from bias_model import judge_bias_model

    values = handlers.trace(handlers.seed(judge_bias_model, rng_seed=0)).get_trace(
        obs=np.array([1.0]), judge=np.array([0]), source_index=np.array([0]),
        condition_index=np.array([0]), delta_theta=np.array([0.0]),
        design=np.empty((1, 0)), n_judges=1, n_sources=1, n_conditions=1,
    )

    assert values["judge_beta"]["value"].shape == (1, 0)
    assert values["judge_source"]["value"].shape == (1, 1)
    assert values["judge_condition"]["value"].shape == (1, 1)
    assert "source_mean" not in values
    assert "condition_mean" not in values


def test_fit_calibration_normalizes_artifact_and_persists_chain_diagnostics(monkeypatch):
    import bias_model

    captured = {}
    flat = {
        "alpha_mean": np.zeros(8), "alpha_scale": np.full(8, 0.5),
        "alpha_raw": np.tile([-1.0, 1.0], (8, 1)),
        "d_mean": np.ones(8), "d_scale": np.full(8, 0.25),
        "d_raw": np.tile([-1.0, 1.0], (8, 1)),
        "beta_mean": np.zeros((8, 1)), "beta_scale": np.ones((8, 1)),
        "beta_raw": np.tile([[[-2.0], [2.0]]], (8, 1, 1)),
        "source_mean": np.full((8, 1), 0.2),
        "source_scale": np.full((8, 1), 0.5),
        "source_raw": np.tile([[[-1.0], [1.0]]], (8, 1, 1)),
        "condition_mean": np.full((8, 1), -0.2),
        "condition_scale": np.full((8, 1), 0.25),
        "condition_raw": np.tile([[[-2.0], [2.0]]], (8, 1, 1)),
    }
    chained = {key: value.reshape((4, 2) + value.shape[1:]) for key, value in flat.items()}

    class FakeMCMC:
        def __init__(self, kernel, **kwargs):
            captured.update(kernel=kernel, **kwargs)

        def run(self, key, **kwargs):
            captured["run"] = kwargs

        def get_samples(self, group_by_chain=False):
            captured.setdefault("grouped", []).append(group_by_chain)
            return chained if group_by_chain else flat

        def get_extra_fields(self, group_by_chain=False):
            captured["extra_grouped"] = group_by_chain
            return {"diverging": np.array([
                [False, True], [False, False], [True, False], [False, False],
            ])}

    monkeypatch.setattr(bias_model, "NUTS", lambda model, **_kwargs: model)
    monkeypatch.setattr(bias_model, "MCMC", FakeMCMC)
    monkeypatch.setattr(
        bias_model.numpyro.diagnostics, "summary",
        lambda samples, group_by_chain: {
            "alpha_mean": {"r_hat": np.array(1.0), "n_eff": np.array(8.0)},
        },
    )
    frame = pd.DataFrame({
        "judge": ["z", "a"], "source": ["z", "a"],
        "condition": ["z", "a"], "chosen_left": [1.0, 0.0],
        "delta_theta": [1.0, -1.0], "delta_log_chars": [3.0, 1.0],
    })

    got = bias_model.fit_calibration(
        frame, ["delta_log_chars"], warmup=3, samples=2, seed=7,
    )

    assert got["judges"] == ["a", "z"]
    assert got["sources"] == ["a", "z"]
    assert got["conditions"] == ["a", "z"]
    assert got["features"] == ["delta_log_chars"]
    assert np.array_equal(got["feature_mean"], [2.0])
    assert np.array_equal(got["feature_scale"], [1.0])
    assert {key for key in got if key in bias_model.POSTERIOR_KEYS} == set(
        bias_model.POSTERIOR_KEYS
    )
    assert not any(key.startswith("judge_") for key in got)
    assert np.array_equal(got["alpha"][0], [-0.5, 0.5])
    assert np.array_equal(got["discrimination"][0], [0.75, 1.25])
    assert np.array_equal(got["source"][0, :, 0], [0.0, 0.0])
    assert np.array_equal(got["condition"][0, :, 0], [0.0, 0.0])
    assert np.allclose(got["source"][0, :, 1], [-0.3, 0.7])
    assert np.allclose(got["condition"][0, :, 1], [-0.7, 0.3])
    assert np.array_equal(got["beta"][0, :, 0], [-2.0, 2.0])
    assert got["divergences"] == 2
    assert got["diagnostics"]["alpha_mean"]["r_hat"] == 1.0
    assert captured["num_chains"] == 4
    assert captured["chain_method"] == "parallel"
    assert captured["num_warmup"] == 3
    assert captured["num_samples"] == 2
    assert captured["grouped"] == [False, True]
    assert captured["extra_grouped"] is True
    assert np.array_equal(captured["run"]["judge"], [1, 0])
    assert np.array_equal(captured["run"]["source_index"], [1, 0])
    assert np.array_equal(captured["run"]["condition_index"], [1, 0])
    assert np.array_equal(captured["run"]["design"], [[1.0], [-1.0]])


def test_fit_calibration_rejects_zero_variance_features():
    from bias_model import fit_calibration

    frame = pd.DataFrame({
        "judge": ["j", "j"], "source": ["s", "s"],
        "condition": ["c", "c"], "chosen_left": [1.0, 0.0],
        "delta_theta": [1.0, -1.0], "constant": [2.0, 2.0],
    })

    with pytest.raises(ValueError, match="zero variance.*constant"):
        fit_calibration(frame, ["constant"], warmup=1, samples=1, seed=0)


@pytest.mark.filterwarnings("ignore:There are not enough devices to run parallel chains")
def test_fit_calibration_without_features_collects_diagnostics():
    from bias_model import fit_calibration

    frame = pd.DataFrame({
        "judge": ["j"] * 8, "source": ["s"] * 8,
        "condition": ["c"] * 8, "chosen_left": [0.0, 1.0] * 4,
        "delta_theta": [-1.0, 1.0] * 4,
    })

    got = fit_calibration(frame, [], warmup=5, samples=5, seed=0)

    assert got["beta"].shape == (20, 1, 0)
    assert got["diagnostics"]


def test_simulate_calibration_returns_fixed_truth_and_reproducible_rows():
    from bias_model import simulate_calibration

    first, truth = simulate_calibration(seed=4, n_judges=3, n_rows=20)
    second, _ = simulate_calibration(seed=4, n_judges=3, n_rows=20)

    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 20
    assert set(first) == {
        "judge", "source", "condition", "chosen_left", "delta_theta",
        "delta_log_chars", "delta_personal_pronouns",
    }
    assert truth["alpha"].tolist() == [-0.3, 0.0, 0.3]
    assert truth["discrimination"].tolist() == [0.5, 1.0, 1.5]
    assert truth["delta_log_chars"].tolist() == [0.8, 0.8, 0.8]
    assert truth["delta_personal_pronouns"].tolist() == [-0.5, -0.5, -0.5]
    assert set(first["chosen_left"]) <= {0.0, 1.0}


def test_target_inference_uses_frozen_bias_offset():
    from bias_model import infer_targets

    frame = pd.DataFrame([
        {"judge": "j", "source": "s", "condition": "c", "left_model": "t",
         "right_model": anchor, "chosen_left": 1.0, "delta_log_chars": 1.0}
        for anchor in ("a", "b") for _ in range(10)
    ])
    anchors = pd.DataFrame({"model": ["a", "b"], "theta": [0.0, 0.0]})

    def calibration(beta):
        return {
            "judges": ["j"], "sources": ["s"], "conditions": ["c"],
            "features": ["delta_log_chars"], "feature_mean": np.array([0.0]),
            "feature_scale": np.array([1.0]), "alpha": np.zeros((2, 1)),
            "source": np.zeros((2, 1, 1)), "condition": np.zeros((2, 1, 1)),
            "discrimination": np.ones((2, 1)),
            "beta": np.full((2, 1, 1), beta),
        }

    neutral = infer_targets(frame, anchors, calibration(0.0), warmup=20, samples=20, seed=0)
    offset = infer_targets(frame, anchors, calibration(1.0), warmup=20, samples=20, seed=0)

    assert neutral.loc[0, "theta"] > 0
    assert offset.loc[0, "theta"] < neutral.loc[0, "theta"]


def test_cut_likelihood_marginalizes_one_global_calibration_draw():
    from numpyro import handlers
    from bias_model import cut_target_model

    calibration = {
        "alpha": np.array([[-2.0], [2.0]]),
        "source": np.zeros((2, 1, 1)),
        "condition": np.zeros((2, 1, 1)),
        "discrimination": np.zeros((2, 1)),
        "beta": np.zeros((2, 1, 0)),
    }
    conditioned = handlers.condition(
        cut_target_model, data={"theta": np.array([0.0])},
    )
    trace = handlers.trace(handlers.seed(conditioned, rng_seed=0)).get_trace(
        obs=np.ones(2), left_target=np.zeros(2, dtype=int),
        right_target=np.full(2, -1, dtype=int), left_anchor=np.zeros(2),
        right_anchor=np.zeros(2), judge=np.zeros(2, dtype=int),
        source_index=np.zeros(2, dtype=int), condition_index=np.zeros(2, dtype=int),
        design=np.empty((2, 0)), calibration=calibration, n_targets=1,
    )

    log_likelihood = np.asarray(
        trace["cut_obs"]["fn"].log_prob(trace["cut_obs"]["value"]),
    ).sum()
    assert np.isclose(log_likelihood, -0.9288532747280807)
    assert set(trace) == {"theta", "cut_obs"}


def test_target_inference_uses_saved_calibration_feature_scaling(monkeypatch):
    import bias_model

    captured = _capture_target_mcmc(monkeypatch, bias_model)
    frame = pd.DataFrame([{
        "judge": "j", "source": "s", "condition": "c",
        "left_model": "target", "right_model": "anchor",
        "chosen_left": 1.0, "delta_log_chars": 12.0,
    }])
    anchors = pd.DataFrame({"model": ["anchor"], "theta": [0.0]})
    calibration = _calibration()
    calibration.update({
        "features": ["delta_log_chars"], "feature_mean": np.array([10.0]),
        "feature_scale": np.array([2.0]), "beta": np.zeros((1, 1, 1)),
    })

    bias_model.infer_targets(
        frame, anchors, calibration, warmup=1, samples=2, seed=0,
    )

    assert np.array_equal(captured["design"], [[1.0]])
    assert captured["n_targets"] == 1


def test_target_inference_subsamples_calibration_draws_deterministically(monkeypatch):
    import bias_model

    captured = _capture_target_mcmc(monkeypatch, bias_model)
    frame = pd.DataFrame([{
        "judge": "j", "source": "s", "condition": "c",
        "left_model": "target", "right_model": "anchor", "chosen_left": 1.0,
    }])
    anchors = pd.DataFrame({"model": ["anchor"], "theta": [0.0]})
    calibration = _calibration(draws=5)
    calibration["alpha"] = np.arange(5.0)[:, None]

    bias_model.infer_targets(
        frame, anchors, calibration, warmup=1, samples=2, seed=0,
        max_calibration_draws=3,
    )

    assert np.array_equal(
        captured["calibration"]["alpha"][:, 0], [0.0, 2.0, 4.0],
    )


@pytest.mark.parametrize("level", ["judge", "source", "condition"])
def test_target_inference_rejects_unseen_levels_with_coverage_reason(
    monkeypatch, level,
):
    import bias_model

    _capture_target_mcmc(monkeypatch, bias_model)
    row = {
        "judge": "j", "source": "s", "condition": "c",
        "left_model": "target", "right_model": "anchor", "chosen_left": 1.0,
    }
    row[level] = f"new-{level}"

    with pytest.raises(
        ValueError, match=rf"coverage failure.*unseen {level}.*new-{level}",
    ):
        bias_model.infer_targets(
            pd.DataFrame([row]),
            pd.DataFrame({"model": ["anchor"], "theta": [0.0]}),
            _calibration(), warmup=1, samples=1, seed=0,
        )


def test_target_inference_excludes_non_gpqa_rows(monkeypatch):
    import bias_model

    captured = _capture_target_mcmc(monkeypatch, bias_model)
    frame = pd.DataFrame([
        {"benchmark": benchmark, "judge": "j", "source": "s", "condition": "c",
         "left_model": f"target-{benchmark}", "right_model": "anchor",
         "chosen_left": choice}
        for benchmark, choice in (("gpqa", 1.0), ("other", 0.0))
    ])
    got = bias_model.infer_targets(
        frame, pd.DataFrame({"model": ["anchor"], "theta": [0.0]}),
        _calibration(), warmup=1, samples=2, seed=0,
    )

    assert got["model"].tolist() == ["target-gpqa"]
    assert np.array_equal(captured["obs"], [1.0])


@pytest.mark.parametrize(
    ("frame", "anchors"),
    [
        (
            pd.DataFrame([{
                "benchmark": "other", "judge": "j", "source": "s",
                "condition": "c", "left_model": "target",
                "right_model": "anchor", "chosen_left": 1.0,
            }]),
            pd.DataFrame({"model": ["anchor"], "theta": [0.0]}),
        ),
        (
            pd.DataFrame([{
                "benchmark": "gpqa", "judge": "j", "source": "s",
                "condition": "c", "left_model": "a", "right_model": "b",
                "chosen_left": 1.0,
            }]),
            pd.DataFrame({"model": ["a", "b"], "theta": [0.0, 1.0]}),
        ),
    ],
    ids=["non-gpqa-only", "all-anchors"],
)
def test_target_inference_returns_empty_without_targets_or_mcmc(
    monkeypatch, frame, anchors,
):
    import bias_model

    monkeypatch.setattr(bias_model, "MCMC", _forbid_mcmc)

    got = bias_model.infer_targets(
        frame, anchors, _calibration(), warmup=1, samples=1, seed=0,
    )

    assert got.empty
    assert got.columns.tolist() == ["model", "theta", "theta_p5", "theta_p95"]


def test_target_inference_rejects_zero_calibration_draws(monkeypatch):
    import bias_model

    monkeypatch.setattr(bias_model, "MCMC", _forbid_mcmc)

    with pytest.raises(ValueError, match="calibration has zero posterior draws"):
        _infer_one_target(bias_model, _calibration(draws=0))


def test_target_inference_defaults_to_one_sequential_chain(monkeypatch):
    import bias_model

    captured = {}

    class FakeMCMC:
        def __init__(self, kernel, **kwargs):
            captured.update(kwargs)

        def run(self, key, **kwargs):
            captured.update(kwargs)

        def get_samples(self):
            return {"theta": np.zeros((2, captured["n_targets"]))}

    monkeypatch.setattr(bias_model, "NUTS", lambda model, **_kwargs: model)
    monkeypatch.setattr(bias_model, "MCMC", FakeMCMC)
    _infer_one_target(bias_model, _calibration())

    assert captured["num_chains"] == 1
    assert captured["chain_method"] == "sequential"


def test_target_inference_accepts_four_parallel_production_chains(monkeypatch):
    import bias_model

    captured = {}

    class FakeMCMC:
        def __init__(self, kernel, **kwargs):
            captured.update(kwargs)

        def run(self, key, **kwargs):
            captured.update(kwargs)

        def get_samples(self):
            return {"theta": np.zeros((2, captured["n_targets"]))}

    monkeypatch.setattr(bias_model, "NUTS", lambda model, **_kwargs: model)
    monkeypatch.setattr(bias_model, "MCMC", FakeMCMC)
    frame = pd.DataFrame([{
        "judge": "j", "source": "s", "condition": "c",
        "left_model": "target", "right_model": "anchor", "chosen_left": 1.0,
    }])
    bias_model.infer_targets(
        frame, pd.DataFrame({"model": ["anchor"], "theta": [0.0]}),
        _calibration(), warmup=1, samples=1, seed=0, chains=4,
    )

    assert captured["num_chains"] == 4
    assert captured["chain_method"] == "parallel"


def test_target_inference_can_attach_chain_diagnostics(monkeypatch):
    import bias_model

    captured = {}

    class FakeMCMC:
        def __init__(self, kernel, **kwargs):
            pass

        def run(self, key, **kwargs):
            captured.update(kwargs)

        def get_samples(self, group_by_chain=False):
            if group_by_chain:
                return {"theta": np.zeros((4, 2, captured["n_targets"]))}
            return {"theta": np.zeros((8, captured["n_targets"]))}

        def get_extra_fields(self, group_by_chain=False):
            assert group_by_chain
            return {"diverging": np.zeros((4, 2), dtype=bool)}

    monkeypatch.setattr(bias_model, "NUTS", lambda model, **_kwargs: model)
    monkeypatch.setattr(bias_model, "MCMC", FakeMCMC)
    monkeypatch.setattr(
        bias_model.numpyro.diagnostics, "summary",
        lambda samples, group_by_chain: {
            "theta": {"r_hat": np.array([1.0]), "n_eff": np.array([8.0])},
        },
    )
    frame = pd.DataFrame([{
        "judge": "j", "source": "s", "condition": "c",
        "left_model": "target", "right_model": "anchor", "chosen_left": 1.0,
    }])

    got = bias_model.infer_targets(
        frame, pd.DataFrame({"model": ["anchor"], "theta": [0.0]}),
        _calibration(), warmup=1, samples=2, seed=0,
        collect_diagnostics=True,
    )

    assert got.attrs["divergences"] == 0
    assert got.attrs["diagnostics"]["theta"]["r_hat"] == 1.0


@pytest.mark.parametrize(
    ("key", "shape"),
    [
        ("source", (2, 1, 1)),
        ("condition", (2, 1, 1)),
        ("discrimination", (2, 1)),
        ("beta", (2, 1, 0)),
        ("alpha", (1, 2)),
        ("source", (1, 2, 1)),
        ("condition", (1, 2, 1)),
        ("discrimination", (1, 2)),
        ("beta", (1, 2, 0)),
        ("source", (1, 1, 2)),
        ("condition", (1, 1, 2)),
        ("beta", (1, 1, 1)),
    ],
    ids=[
        "source-draws", "condition-draws", "discrimination-draws", "beta-draws",
        "alpha-judges", "source-judges", "condition-judges",
        "discrimination-judges", "beta-judges", "sources", "conditions", "features",
    ],
)
def test_target_inference_rejects_malformed_posterior_shapes(
    monkeypatch, key, shape,
):
    import bias_model

    monkeypatch.setattr(bias_model, "MCMC", _forbid_mcmc)
    calibration = _calibration()
    calibration[key] = np.zeros(shape)

    with pytest.raises(
        ValueError, match=rf"calibration {key} shape.*expected.*got",
    ):
        _infer_one_target(bias_model, calibration)


@pytest.mark.parametrize("key", ["feature_mean", "feature_scale"])
def test_target_inference_rejects_malformed_feature_stat_shapes(
    monkeypatch, key,
):
    import bias_model

    monkeypatch.setattr(bias_model, "MCMC", _forbid_mcmc)
    calibration = _calibration()
    calibration.update({
        "features": ["feature"], "feature_mean": np.array([0.0]),
        "feature_scale": np.array([1.0]), "beta": np.zeros((1, 1, 1)),
    })
    calibration[key] = np.array([])

    with pytest.raises(
        ValueError, match=rf"calibration {key} shape.*expected.*got",
    ):
        _infer_one_target(bias_model, calibration)


def test_target_inference_requires_positive_calibration_draw_limit(monkeypatch):
    import bias_model

    _capture_target_mcmc(monkeypatch, bias_model)
    frame = pd.DataFrame([{
        "judge": "j", "source": "s", "condition": "c", "left_model": "target",
        "right_model": "anchor", "chosen_left": 1.0,
    }])
    with pytest.raises(ValueError, match="max_calibration_draws must be positive"):
        bias_model.infer_targets(
            frame, pd.DataFrame({"model": ["anchor"], "theta": [0.0]}),
            _calibration(), warmup=1, samples=1, seed=0,
            max_calibration_draws=0,
        )


def test_target_inference_rejects_unanchored_nonheldout_endpoint(monkeypatch):
    import bias_model

    monkeypatch.setattr(bias_model, "MCMC", _forbid_mcmc)
    frame = pd.DataFrame([{
        "judge": "j", "source": "s", "condition": "c",
        "left_model": "held", "right_model": "not-held-or-anchor",
        "chosen_left": 1.0,
    }])

    with pytest.raises(ValueError, match="unanchored non-held-out.*not-held-or-anchor"):
        bias_model.infer_targets(
            frame, pd.DataFrame({"model": ["anchor"], "theta": [0.0]}),
            _calibration(), warmup=1, samples=1, seed=0,
            target_models={"held"},
        )
