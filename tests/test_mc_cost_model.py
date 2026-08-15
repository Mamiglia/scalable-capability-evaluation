import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from numpyro import handlers

from mc_cost.model import (
    CORE_FEATURES,
    GLOBAL_FEATURES,
    JUDGMENT_SPECS,
    JudgmentData,
    JudgmentSpec,
    PosteriorArtifact,
    encode_judgments,
    fit_judgment_model,
    judgment_model,
    left_logits,
    load_artifact,
    save_artifact,
)


TRAINING_SCALES = {
    feature: {"mean": 0.0, "scale": 1.0}
    for feature in (
        "theta_gap", "length_difference", "oversight", "difficulty",
        "theta_gap_x_oversight", "theta_gap_x_difficulty",
        "same_family_x_theta_gap", "reasoning_difference",
    )
}


def test_judgment_spec_declares_exact_sanctioned_feature_order():
    assert {
        name: (spec.global_features, spec.include_length)
        for name, spec in JUDGMENT_SPECS.items()
    } == {
        "baseline": ((), False),
        "core": ((
            "oversight_sd", "difficulty_sd", "theta_gap_x_oversight_sd",
            "theta_gap_x_difficulty_sd", "same_family_x_theta_gap_sd",
        ), True),
        "core_release": ((
            "oversight_sd", "difficulty_sd", "theta_gap_x_oversight_sd",
            "theta_gap_x_difficulty_sd", "same_family_x_theta_gap_sd",
            "release_gap_days_sd", "relative_release_days_sd",
        ), True),
    }


def tiny_data():
    return JudgmentData(
        y=jnp.array([1, 0]), judge=jnp.array([0, 0]), item=jnp.array([0, 0]),
        left_model=jnp.array([0, 1]), right_model=jnp.array([1, 0]),
        theta_gap=jnp.array([1.0, -1.0]),
        global_x=jnp.zeros((2, 7)), length_difference=jnp.array([0.2, -0.2]),
        judge_covariates=jnp.array([[0.0, 0.0]]), include_length=True,
        num_judges=1, num_items=1, num_models=2,
    )


def test_judgment_model_declares_pooled_judge_and_model_effects():
    trace = handlers.trace(handlers.seed(judgment_model, 2)).get_trace(tiny_data(), True)
    assert trace["judge_effects"]["value"].shape == (1, 2)
    assert trace["candidate_effects"]["value"].shape == (2,)
    assert np.allclose(trace["item_effects"]["value"], 0.0)
    assert trace["obs"]["value"].shape == (2,)


def test_swapping_candidates_negates_content_but_keeps_position_term():
    posterior = {
        "judge_effects": np.array([[[0.3, 1.5]]]),
        "global_beta": np.array([[0.2] + [0.0] * 6]),
        "length_mean": np.array([0.4]), "length_deviation": np.array([[0.1]]),
        "candidate_effects": np.array([[0.2, -0.1]]), "item_effects": np.array([[0.0]]),
    }
    forward = left_logits(posterior, tiny_data())[0, 0]
    swapped = left_logits(posterior, tiny_data())[0, 1]
    assert np.isclose((forward + swapped) / 2, 0.3)


def test_single_unbatched_draw_keeps_draw_axis_with_length_enabled():
    posterior = {
        "judge_effects": np.array([[[0.3, 1.5]]]),
        "global_beta": np.array([[0.2] + [0.0] * 6]),
        "length_mean": np.array([0.4]), "length_deviation": np.array([[0.1]]),
        "candidate_effects": np.array([[0.2, -0.1]]), "item_effects": np.array([[0.0]]),
    }

    got = left_logits({name: values[0] for name, values in posterior.items()}, tiny_data())

    assert got.shape == (1, 2)
    assert np.allclose(got, left_logits(posterior, tiny_data()))


def test_baseline_omits_global_and_length_sites():
    data = tiny_data()._replace(global_x=jnp.zeros((2, 0)), include_length=False)
    trace = handlers.trace(handlers.seed(judgment_model, 2)).get_trace(data, False)
    assert "global_beta" not in trace
    assert "length_mean" not in trace
    assert "length_scale" not in trace
    assert "length_raw" not in trace
    assert "length_deviation" not in trace


@pytest.mark.parametrize("features, include_length", [
    ((), False),
    (CORE_FEATURES, True),
    (GLOBAL_FEATURES, True),
])
def test_judgment_spec_accepts_only_the_three_sanctioned_models(features, include_length):
    assert JudgmentSpec(features, include_length).global_features == features


@pytest.mark.parametrize("features, include_length", [
    (("oversight_sd",), True),
    (tuple(reversed(GLOBAL_FEATURES)), True),
    ((), True),
    (CORE_FEATURES, False),
])
def test_judgment_spec_rejects_arbitrary_subsets_reordering_and_length_settings(features, include_length):
    with pytest.raises(ValueError, match="sanctioned"):
        JudgmentSpec(features, include_length)


def test_encode_judgments_uses_declared_features_and_theta_release_judge_covariates():
    frame = pd.DataFrame([
        {
            "chosen_left": 1, "judge": "j2", "item": "i2", "left_model": "b", "right_model": "a",
            "theta_gap": 4.0, "length_difference": 0.5,
            "oversight": 1.0, "difficulty": 2.0, "theta_gap_x_oversight": 3.0,
            "theta_gap_x_difficulty": 4.0, "same_family_x_theta_gap": 5.0,
            "self_comparison": 0.0, "reasoning_difference": 0.0,
            "include_reasoning": False,
            "presentation_max_chars": 65536, "presentation_truncate_frac": None,
            "presentation_truncate_keep": "first",
        },
        {
            "chosen_left": 0, "judge": "j1", "item": "i1", "left_model": "a", "right_model": "b",
            "theta_gap": -4.0, "length_difference": -0.5,
            "oversight": -1.0, "difficulty": -2.0, "theta_gap_x_oversight": -3.0,
            "theta_gap_x_difficulty": -4.0, "same_family_x_theta_gap": -5.0,
            "self_comparison": 0.0, "reasoning_difference": 0.0,
            "include_reasoning": False,
            "presentation_max_chars": 65536, "presentation_truncate_frac": None,
            "presentation_truncate_keep": "first",
        },
    ])
    catalogue = pd.DataFrame([
        {"model_id": "j1", "theta": -1.0, "release": "2024-01-01"},
        {"model_id": "j2", "theta": 1.0, "release": "2025-01-01"},
    ])
    spec = JudgmentSpec(CORE_FEATURES, include_length=True)

    got = encode_judgments(
        frame, catalogue, spec, training_scales=TRAINING_SCALES,
        source_hashes={"judgments": "abc", "catalogue": "def"},
    )

    assert np.asarray(got.global_x).tolist() == [
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [-1.0, -2.0, -3.0, -4.0, -5.0],
    ]
    assert np.asarray(got.theta_gap).tolist() == [4.0, -4.0]
    assert got.global_features == spec.global_features
    assert got.category_maps == {
        "judges": {"j1": 0, "j2": 1},
        "items": {"i1": 0, "i2": 1},
        "models": {"a": 0, "b": 1},
    }
    assert np.allclose(np.asarray(got.judge_covariates), [[-1.0, -1.0], [1.0, 1.0]])
    assert got.training_scales == {
        **TRAINING_SCALES,
        "judge_theta": {"mean": 0.0, "scale": 1.0},
        "judge_release_days": {"mean": 1644.0, "scale": 183.0},
    }
    assert got.training_ranges == {
        "theta_gap": [-4.0, 4.0], "oversight_sd": [-1.0, 1.0], "difficulty_sd": [-2.0, 2.0],
        "theta_gap_x_oversight_sd": [-3.0, 3.0], "theta_gap_x_difficulty_sd": [-4.0, 4.0],
        "same_family_x_theta_gap_sd": [-5.0, 5.0], "length_difference": [-0.5, 0.5],
        "judge_theta": [-1.0, 1.0], "judge_release_days": [1461.0, 1827.0],
    }
    assert got.source_hashes == {"judgments": "abc", "catalogue": "def"}
    assert got.presentation_include_reasoning is False
    assert got.presentation_max_chars == 65536
    assert got.presentation_truncate_frac is None
    assert got.presentation_truncate_keep == "first"


@pytest.mark.parametrize("condition", [True, None, "false"])
def test_encode_judgments_rejects_unsupported_or_undeclared_presentation(condition):
    frame = pd.DataFrame([{
        "chosen_left": 1, "judge": "j", "item": "i", "left_model": "a", "right_model": "b",
        "theta_gap": 0.0, "include_reasoning": condition,
        "presentation_max_chars": 65536, "presentation_truncate_frac": None,
        "presentation_truncate_keep": "first",
    }])
    catalogue = pd.DataFrame([{"model_id": "j", "theta": 0.0, "release": "2024-01-01"}])

    with pytest.raises(ValueError, match="include_reasoning=false"):
        encode_judgments(
            frame, catalogue, JudgmentSpec((), False), training_scales=TRAINING_SCALES,
            source_hashes={"judgments": "abc"},
        )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"presentation_max_chars": [100, 200]}, "presentation_max_chars"),
        ({"presentation_truncate_frac": [None, 0.5]}, "paired truncation"),
        ({"presentation_truncate_keep": ["first", "last"]}, "presentation_truncate_keep"),
    ],
)
def test_encode_judgments_rejects_mixed_or_paired_truncation(changes, message):
    frame = pd.DataFrame([
        {
            "chosen_left": 1, "judge": "j", "item": "i1", "left_model": "a",
            "right_model": "b", "theta_gap": 0.0, "include_reasoning": False,
            "presentation_max_chars": 65536, "presentation_truncate_frac": None,
            "presentation_truncate_keep": "first",
        },
        {
            "chosen_left": 0, "judge": "j", "item": "i2", "left_model": "b",
            "right_model": "a", "theta_gap": 0.0, "include_reasoning": False,
            "presentation_max_chars": 65536, "presentation_truncate_frac": None,
            "presentation_truncate_keep": "first",
        },
    ])
    for column, values in changes.items():
        frame[column] = values
    catalogue = pd.DataFrame([{"model_id": "j", "theta": 0.0, "release": "2024-01-01"}])

    with pytest.raises(ValueError, match=message):
        encode_judgments(
            frame, catalogue, JudgmentSpec((), False), training_scales=TRAINING_SCALES,
            source_hashes={"judgments": "abc"},
        )


def test_encode_judgments_rejects_empty_or_incompatible_provenance():
    frame = pd.DataFrame([{
        "chosen_left": 1, "judge": "j", "item": "i", "left_model": "a", "right_model": "b",
        "theta_gap": 0.0, "include_reasoning": False,
        "presentation_max_chars": 65536, "presentation_truncate_frac": None,
        "presentation_truncate_keep": "first",
    }])
    catalogue = pd.DataFrame([{"model_id": "j", "theta": 0.0, "release": "2024-01-01"}])
    spec = JudgmentSpec((), False)

    for invalid_hashes in ({}, {"judgments": ""}, {"judgments": None}):
        with pytest.raises(ValueError, match="source hashes"):
            encode_judgments(
                frame, catalogue, spec, training_scales=TRAINING_SCALES,
                source_hashes=invalid_hashes,
            )
    with pytest.raises(ValueError, match="theta_gap.*scale"):
        encode_judgments(
            frame, catalogue, spec, training_scales={"difficulty": {"mean": 0.0, "scale": 1.0}},
            source_hashes={"judgments": "abc"},
        )


def test_artifact_round_trips_without_pickle(tmp_path):
    artifact = PosteriorArtifact(
        posterior={"judge_effects": np.array([[[0.1, 0.2]]]), "labels": np.array(["j"], dtype=str)},
        metadata={
            "feature_order": list(GLOBAL_FEATURES),
            "category_maps": {"judges": {"j": 0}},
            "training_scales": {"theta_gap": {"mean": 0.0, "scale": 1.0}},
            "training_ranges": {"theta_gap": [-1.0, 1.0]},
            "source_hashes": {"judgments": "abc"},
            "conditional_judges": True,
        },
        diagnostics={"divergences": 0, "summary": {"x": {"mean": 0.0}}},
    )
    path = tmp_path / "judgment.npz"

    save_artifact(artifact, path)
    stored = np.load(path, allow_pickle=False)
    loaded = load_artifact(path)

    assert stored["labels"].tolist() == ["j"]
    assert loaded.metadata == artifact.metadata
    assert loaded.diagnostics == artifact.diagnostics
    assert np.array_equal(loaded.posterior["judge_effects"], artifact.posterior["judge_effects"])


def test_tiny_fit_records_configured_diagnostics_and_metadata(tmp_path):
    data = tiny_data()._replace(
        global_x=jnp.zeros((2, 0)), include_length=False, global_features=(),
        category_maps={"judges": {"j": 0}, "items": {"i": 0}, "models": {"a": 0, "b": 1}},
        training_scales={"theta_gap": {"mean": 0.0, "scale": 1.0}},
        training_ranges={"theta_gap": [-1.0, 1.0]}, source_hashes={"judgments": "abc"},
    )

    artifact = fit_judgment_model(data, warmup=4, samples=4, chains=1, seed=3, conditional_judges=False)

    assert artifact.posterior["judge_effects"].shape == (4, 1, 2)
    assert artifact.metadata == {
        "feature_order": [],
        "category_maps": data.category_maps,
        "training_scales": data.training_scales,
        "training_ranges": data.training_ranges,
        "source_hashes": data.source_hashes,
        "conditional_judges": False,
        "presentation_include_reasoning": False,
        "presentation_max_chars": 65536,
        "presentation_truncate_frac": None,
        "presentation_truncate_keep": "first",
    }
    assert isinstance(artifact.diagnostics["divergences"], int)
    assert artifact.diagnostics["draws"] == 4
    assert "summary" in artifact.diagnostics
    save_artifact(artifact, tmp_path / "fitted.npz")
    assert load_artifact(tmp_path / "fitted.npz").diagnostics == artifact.diagnostics
