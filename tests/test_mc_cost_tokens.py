import numpy as np
import pandas as pd
import pytest

from mc_cost.data import judge_token_rows
from mc_cost.tokens import (
    _coverage, _prepare_frame, draw_token_usage, draw_transcript_lengths, fit_token_model,
)


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
        artifact, rows, np.random.default_rng(4), posterior_index=0,
        latent_cache={}, trial_id="trial-0",
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
        artifact, rows, np.random.default_rng(5), posterior_index=0,
        latent_cache=cache, trial_id="trial-0",
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
        artifact, rows, np.random.default_rng(7), posterior_index=0,
        latent_cache=cache, trial_id="trial-0",
    )

    assert got[0] == got[1]
    assert len(cache) == 1


def test_external_latent_cache_is_scoped_by_trial_not_posterior_index():
    artifact = {
        "role": "candidate", "models": [], "reasoning_classes": ["non_reasoning"],
        "posterior": {
            "class_input_mean": np.array([[np.log1p(100)]]),
            "class_output_mean": np.array([[np.log1p(40)]]),
            "input_model_scale": np.array([1.5]), "output_model_scale": np.array([1.5]),
            "input_sigma": np.array([0.0]), "output_sigma": np.array([0.0]),
        },
    }
    rows = pd.DataFrame([{"model": "new", "reasoning_class": "non_reasoning"}])
    cache, rng = {}, np.random.default_rng(12)

    first = draw_token_usage(
        artifact, rows, rng, posterior_index=0, latent_cache=cache, trial_id=0,
    )
    repeated = draw_token_usage(
        artifact, rows, rng, posterior_index=0, latent_cache=cache, trial_id=0,
    )
    next_trial = draw_token_usage(
        artifact, rows, rng, posterior_index=0, latent_cache=cache, trial_id=1,
    )

    assert first.equals(repeated)
    assert not first.equals(next_trial)
    assert len(cache) == 4


def test_external_latent_cache_requires_explicit_trial_id():
    artifact = {
        "role": "candidate", "models": ["seen"], "reasoning_classes": ["non_reasoning"],
        "posterior": {
            "input_mean": np.array([[1.0]]), "output_mean": np.array([[1.0]]),
            "input_sigma": np.array([0.0]), "output_sigma": np.array([0.0]),
        },
    }
    rows = pd.DataFrame([{"model": "seen", "reasoning_class": "non_reasoning"}])

    with pytest.raises(ValueError, match="trial_id"):
        draw_token_usage(
            artifact, rows, np.random.default_rng(1), posterior_index=0, latent_cache={},
        )


def test_artifact_scales_raw_sources_instead_of_trusting_foreign_sd_columns():
    artifact = {
        "role": "candidate", "models": ["seen"], "reasoning_classes": ["non_reasoning"],
        "input_features": ["difficulty_sd"], "output_features": [],
        "training_scales": {
            "difficulty_sd": {
                "source": "difficulty", "transform": "identity", "mean": 10.0, "scale": 2.0,
            },
        },
        "posterior": {
            "input_mean": np.array([[np.log1p(100)]]),
            "output_mean": np.array([[np.log1p(40)]]),
            "input_beta": np.array([[1.0]]),
            "input_sigma": np.array([0.0]), "output_sigma": np.array([0.0]),
        },
    }
    rows = pd.DataFrame([{
        "model": "seen", "reasoning_class": "non_reasoning",
        "difficulty": 12.0, "difficulty_sd": 0.0,
    }])

    got = draw_token_usage(artifact, rows, np.random.default_rng(1), posterior_index=0)

    assert got.input_tokens.tolist() == [274]


def test_coverage_uses_seeded_posterior_predictive_draws():
    actual = np.array([1.6])
    mean = np.zeros((20, 1))
    sigma = np.ones(20)

    first = _coverage(actual, mean, sigma, np.random.default_rng(0))
    repeated = _coverage(actual, mean, sigma, np.random.default_rng(0))

    assert first == repeated == 0.0


def test_judge_fit_includes_preserved_difficulty_for_input_and_output():
    rows = judge_token_rows(pd.DataFrame([
        {
            "judge": "j", "item": f"i{index}", "chosen_left": index % 2,
            "left_chars": 100 + index * 10, "right_chars": 80 + index * 10,
            "difficulty": difficulty, "judge_input_tokens": 200 + index * 10,
            "judge_output_tokens": 40 + index, "judge_reasoning_tokens": 10 + index,
            "judge_total_tokens": 240 + index * 11,
        }
        for index, difficulty in enumerate([-1.0, -0.5, 0.5, 1.0])
    ]))

    artifact = fit_token_model(
        rows, "judge", reasoning_classes={"j": "reasoning"},
        source_hashes={"judge_tokens": "abc"}, warmup=4, samples=4, seed=13,
    )

    assert artifact.metadata["input_features"] == ["difficulty_sd", "transcript_chars_log_sd"]
    assert artifact.metadata["output_features"] == ["difficulty_sd"]


@pytest.mark.parametrize("column", ["model", "reasoning_class"])
def test_prediction_rejects_null_identity_fields_before_string_conversion(column):
    artifact = {
        "role": "candidate", "models": [], "reasoning_classes": ["non_reasoning"],
        "posterior": {
            "class_input_mean": np.array([[1.0]]), "class_output_mean": np.array([[1.0]]),
            "input_model_scale": np.array([0.0]), "output_model_scale": np.array([0.0]),
            "input_sigma": np.array([0.0]), "output_sigma": np.array([0.0]),
        },
    }
    row = {"model": "m", "reasoning_class": "non_reasoning"}
    row[column] = None

    with pytest.raises(ValueError, match=f"null {column}"):
        draw_token_usage(artifact, pd.DataFrame([row]), np.random.default_rng(1), 0)


def test_fit_rejects_null_model_before_string_conversion():
    frame = _candidate_training_rows().iloc[[0]].copy()
    frame.loc[frame.index[0], "model"] = None

    with pytest.raises(ValueError, match="null model"):
        fit_token_model(frame, "candidate", source_hashes={"candidate_tokens": "abc"})


def test_candidate_reasoning_class_must_agree_with_catalogue_mapping():
    frame = _candidate_training_rows().iloc[[2]].copy()

    with pytest.raises(ValueError, match="candidate reasoning class conflicts.*think"):
        fit_token_model(
            frame, "candidate", reasoning_classes={"think": "non_reasoning"},
            source_hashes={"candidate_tokens": "abc"}, warmup=1, samples=4,
        )


def test_unused_catalogue_model_may_lack_reasoning_class():
    frame = pd.DataFrame([{
        "role": "candidate", "model": "used", "reasoning_class": "non_reasoning",
    }])
    catalogue = pd.DataFrame([
        {"model_id": "used", "reasoning_class": "non_reasoning"},
        {"model_id": "unused", "reasoning_class": pd.NA},
    ])

    prepared = _prepare_frame(frame, "candidate", catalogue)

    assert prepared["reasoning_class"].tolist() == ["non_reasoning"]


def test_prediction_row_role_must_match_artifact_role():
    artifact = {
        "role": "judge", "models": ["seen"], "reasoning_classes": ["non_reasoning"],
        "posterior": {
            "input_mean": np.array([[1.0]]), "output_mean": np.array([[1.0]]),
            "input_sigma": np.array([0.0]), "output_sigma": np.array([0.0]),
        },
    }
    rows = pd.DataFrame([{
        "role": "candidate", "model": "seen", "reasoning_class": "non_reasoning",
    }])

    with pytest.raises(ValueError, match="role.*judge"):
        draw_token_usage(artifact, rows, np.random.default_rng(1), 0)


def test_nonfinite_reasoning_logit_is_rejected():
    artifact = {
        "role": "judge", "models": ["seen"], "reasoning_classes": ["reasoning"],
        "posterior": {
            "input_mean": np.array([[1.0]]), "output_mean": np.array([[1.0]]),
            "reasoning_logit": np.array([[np.nan]]),
            "input_sigma": np.array([0.0]), "output_sigma": np.array([0.0]),
        },
    }
    rows = pd.DataFrame([{"model": "seen", "reasoning_class": "reasoning"}])

    with pytest.raises(ValueError, match="non-finite.*reasoning logits"):
        draw_token_usage(artifact, rows, np.random.default_rng(1), 0)


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
    assert artifact.metadata["coverage_seed"] == 8
    assert artifact.metadata["coverage_method"] == "seeded_posterior_predictive"
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
