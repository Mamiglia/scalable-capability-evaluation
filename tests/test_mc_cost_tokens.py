import numpy as np
import pandas as pd
import pytest

from mc_cost.tokens import draw_token_usage, draw_transcript_lengths, fit_token_model


def test_reasoning_is_a_subset_of_billed_output():
    artifact = {
        "role": "judge", "models": ["seen"], "reasoning_classes": ["reasoning"],
        "posterior": {
            "input_mean": np.array([[np.log1p(500)]]),
            "output_mean": np.array([[np.log1p(100)]]),
            "reasoning_logit": np.array([[0.0]]),
            "input_sigma": np.array([0.0]), "output_sigma": np.array([0.0]),
        },
    }
    rows = pd.DataFrame([{"model": "seen", "reasoning_class": "reasoning", "transcript_chars": 500}])
    got = draw_token_usage(artifact, rows, np.random.default_rng(1), posterior_index=0)
    assert got.loc[0, "output_tokens"] == 100
    assert got.loc[0, "reasoning_tokens"] == 50
    assert got.loc[0, "ordinary_output_tokens"] == 50


def test_non_reasoning_has_structural_zero_reasoning_tokens():
    artifact = {
        "role": "candidate", "models": [], "reasoning_classes": ["non_reasoning"],
        "posterior": {
            "class_input_mean": np.array([[5.0]]), "class_output_mean": np.array([[4.0]]),
            "input_sigma": np.array([0.0]), "output_sigma": np.array([0.0]),
        },
    }
    rows = pd.DataFrame([{"model": "unseen", "reasoning_class": "non_reasoning"}])
    got = draw_token_usage(artifact, rows, np.random.default_rng(2), posterior_index=0)
    assert got.loc[0, "reasoning_tokens"] == 0


def test_unseen_transcript_length_uses_reasoning_class_population():
    artifact = {
        "role": "candidate", "models": [],
        "reasoning_classes": ["non_reasoning", "reasoning"],
        "length_features": [
            "theta_sd", "release_day_sd", "difficulty_sd", "correct", "visible_reasoning",
        ],
        "posterior": {
            "length_class_mean": np.array([[np.log1p(100), np.log1p(200)]]),
            "length_beta": np.array([[0.0, 0.0, 0.0, 0.0, 0.0]]),
            "length_model_scale": np.array([0.0]), "length_sigma": np.array([0.0]),
        },
    }
    rows = pd.DataFrame([{
        "model": "unseen", "reasoning_class": "reasoning",
        "theta_sd": 1.0, "release_day_sd": 1.0, "difficulty_sd": 1.0,
        "correct": 0, "visible_reasoning": 1,
    }])
    got = draw_transcript_lengths(
        artifact, rows, np.random.default_rng(4), posterior_index=0, latent_cache={},
    )
    assert got.tolist() == [200]


def test_unseen_token_effect_is_reused_within_trial():
    artifact = {
        "role": "judge", "models": [], "reasoning_classes": ["reasoning"],
        "posterior": {
            "class_input_mean": np.array([[np.log1p(100)]]),
            "class_output_mean": np.array([[np.log1p(40)]]),
            "class_reasoning_logit": np.array([[0.0]]),
            "input_model_scale": np.array([0.7]), "output_model_scale": np.array([0.7]),
            "reasoning_model_scale": np.array([0.7]),
            "input_sigma": np.array([0.0]), "output_sigma": np.array([0.0]),
            "reasoning_sigma": np.array([0.0]),
        },
    }
    rows = pd.DataFrame([
        {"model": "new", "reasoning_class": "reasoning"},
        {"model": "new", "reasoning_class": "reasoning"},
    ])
    cache = {}

    got = draw_token_usage(
        artifact, rows, np.random.default_rng(5), posterior_index=0, latent_cache=cache,
    )

    assert got.iloc[0].tolist() == got.iloc[1].tolist()
    assert len(cache) == 3


def test_request_token_residuals_are_independent():
    artifact = {
        "role": "candidate", "models": [], "reasoning_classes": ["non_reasoning"],
        "posterior": {
            "class_input_mean": np.array([[np.log1p(100)]]),
            "class_output_mean": np.array([[np.log1p(40)]]),
            "input_model_scale": np.array([0.0]), "output_model_scale": np.array([0.0]),
            "input_sigma": np.array([0.5]), "output_sigma": np.array([0.5]),
        },
    }
    rows = pd.DataFrame([{"model": "new", "reasoning_class": "non_reasoning"}] * 12)

    got = draw_token_usage(artifact, rows, np.random.default_rng(6), posterior_index=0)

    assert got.input_tokens.nunique() > 1
    assert got.output_tokens.nunique() > 1


def test_unseen_transcript_effect_is_reused_within_trial():
    artifact = {
        "role": "candidate", "models": [], "reasoning_classes": ["reasoning"],
        "length_features": [],
        "posterior": {
            "length_class_mean": np.array([[np.log1p(100)]]),
            "length_beta": np.empty((1, 0)), "length_model_scale": np.array([0.8]),
            "length_sigma": np.array([0.0]),
        },
    }
    rows = pd.DataFrame([
        {"model": "new", "reasoning_class": "reasoning"},
        {"model": "new", "reasoning_class": "reasoning"},
    ])
    cache = {}

    got = draw_transcript_lengths(
        artifact, rows, np.random.default_rng(7), posterior_index=0, latent_cache=cache,
    )

    assert got[0] == got[1]
    assert len(cache) == 1


def test_judge_fit_requires_catalogue_reasoning_class_coverage():
    frame = pd.DataFrame([{
        "role": "judge", "model": "missing", "input_tokens": 10, "output_tokens": 5,
        "reasoning_tokens": 0, "transcript_chars": 20,
    }])

    with pytest.raises(ValueError, match="judge models missing reasoning class: missing"):
        fit_token_model(
            frame, "judge", reasoning_classes={}, source_hashes={"judge_tokens": "abc"},
            warmup=1, samples=1,
        )


@pytest.mark.parametrize("column, value, message", [
    ("input_tokens", -1, "input_tokens"),
    ("output_tokens", np.inf, "output_tokens"),
    ("reasoning_tokens", 6, "reasoning_tokens.*output_tokens"),
])
def test_fit_rejects_invalid_token_accounting(column, value, message):
    row = {
        "role": "candidate", "model": "m", "reasoning_class": "reasoning",
        "input_tokens": 10, "output_tokens": 5, "reasoning_tokens": 2,
        "response_chars": 30, "theta": 0.1, "release_day": 100,
        "difficulty": 0.2, "correct": 1, "visible_reasoning": 1,
    }
    row[column] = value

    with pytest.raises(ValueError, match=message):
        fit_token_model(
            pd.DataFrame([row]), "candidate", source_hashes={"candidate_tokens": "abc"},
            warmup=1, samples=1,
        )


def test_fit_requires_nonempty_source_hashes():
    frame = _candidate_training_rows()
    with pytest.raises(ValueError, match="source hashes"):
        fit_token_model(frame, "candidate", source_hashes={"candidate_tokens": ""})


def test_tiny_candidate_fit_records_hierarchy_provenance_and_coverage():
    artifact = fit_token_model(
        _candidate_training_rows(), "candidate", source_hashes={"candidate_tokens": "abc"},
        warmup=4, samples=4, chains=1, seed=8,
    )

    assert artifact.metadata["role"] == "candidate"
    assert artifact.metadata["models"] == ["plain", "think"]
    assert artifact.metadata["reasoning_classes"] == ["non_reasoning", "reasoning"]
    assert artifact.metadata["length_features"] == [
        "theta_sd", "release_day_sd", "difficulty_sd", "correct", "visible_reasoning",
    ]
    assert artifact.metadata["input_features"] == ["difficulty_sd", "correct"]
    assert artifact.metadata["output_features"] == ["difficulty_sd", "correct"]
    assert artifact.metadata["source_hashes"] == {"candidate_tokens": "abc"}
    assert set(artifact.metadata["coverage_90"]) == {"input", "output", "reasoning", "length"}
    assert all(0.0 <= value <= 1.0 for value in artifact.metadata["coverage_90"].values())
    assert artifact.posterior["input_mean"].shape == (4, 2)
    assert artifact.posterior["output_mean"].shape == (4, 2)
    assert artifact.posterior["reasoning_logit"].shape == (4, 2)
    assert artifact.posterior["length_mean"].shape == (4, 2)
    assert artifact.diagnostics["draws"] == 4


def _candidate_training_rows():
    return pd.DataFrame([
        {
            "role": "candidate", "model": "plain", "reasoning_class": "non_reasoning",
            "input_tokens": 90, "output_tokens": 20, "reasoning_tokens": 0,
            "response_chars": 100, "theta": -0.2, "release_day": 100,
            "difficulty": -0.5, "correct": 1, "visible_reasoning": 0,
        },
        {
            "role": "candidate", "model": "plain", "reasoning_class": "non_reasoning",
            "input_tokens": 100, "output_tokens": 25, "reasoning_tokens": 0,
            "response_chars": 120, "theta": -0.2, "release_day": 100,
            "difficulty": 0.5, "correct": 0, "visible_reasoning": 0,
        },
        {
            "role": "candidate", "model": "think", "reasoning_class": "reasoning",
            "input_tokens": 110, "output_tokens": 80, "reasoning_tokens": 50,
            "response_chars": 300, "theta": 0.8, "release_day": 400,
            "difficulty": -0.5, "correct": 1, "visible_reasoning": 1,
        },
        {
            "role": "candidate", "model": "think", "reasoning_class": "reasoning",
            "input_tokens": 120, "output_tokens": 90, "reasoning_tokens": 60,
            "response_chars": 340, "theta": 0.8, "release_day": 400,
            "difficulty": 0.5, "correct": 0, "visible_reasoning": 1,
        },
    ])
