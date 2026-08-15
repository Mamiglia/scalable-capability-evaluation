import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from mc_cost.catalogue import (
    build_capability_draws, build_catalogue, pool_enabled, resolve_model_id,
    theta_draws_from_accuracy, validate_catalogue,
)


def test_catalogue_joins_normalized_names_and_keeps_missing_metadata_visible(tmp_path):
    canonical = tmp_path / "canonical.md"
    canonical.write_text("- model-a\n- model-missing\n")
    pool = tmp_path / "pool.json"
    pool.write_text(json.dumps([{
        "name": "Model A", "release": "2025-01-02", "price_in": 1.0,
        "price_out": 4.0, "reasoning": False,
    }]))
    capabilities = pd.DataFrame([{
        "irt_model": "fixed_3pl", "fit": "gpqa", "model": "model-a",
        "theta": 1.2, "theta_p5": 1.0, "theta_p95": 1.4,
    }])

    got = build_catalogue(canonical, pool, capabilities)

    assert got.loc[got.model_id == "model-a", "release"].item() == "2025-01-02"
    assert got.loc[got.model_id == "model-a", "theta"].item() == 1.2
    assert pd.isna(got.loc[got.model_id == "model-missing", "release"]).item()
    assert pd.isna(got.loc[got.model_id == "model-missing", "judge_pool"]).item()
    assert pd.isna(got.loc[got.model_id == "model-missing", "candidate_pool"]).item()


def test_catalogue_unions_fitted_models_and_applies_explicit_overrides(tmp_path):
    canonical = tmp_path / "canonical.md"
    canonical.write_text("- Canonical Model\n- missing-model\n")
    pool = tmp_path / "pool.json"
    pool.write_text(json.dumps([
        {"name": "Canonical Model", "release": "2024-01-01", "reasoning": False},
        {"name": "Future Model", "release": "2026-01-01", "reasoning": True},
    ]))
    capabilities = pd.DataFrame([
        {
            "irt_model": "fixed_3pl", "fit": "gpqa", "model": "canonical-model",
            "theta": 0.1, "theta_p5": 0.0, "theta_p95": 0.2,
        },
        {
            "irt_model": "fixed_3pl", "fit": "gpqa", "model": "future-model",
            "theta": 1.1, "theta_p5": 0.9, "theta_p95": 1.3,
        },
    ])
    overrides = tmp_path / "overrides.json"
    overrides.write_text(json.dumps({"models": [
        {"model_id": "Canonical Model", "aliases": ["canonical-model"], "judge_pool": True},
        {"model_id": "future-model", "candidate_pool": True},
    ]}))

    got = build_catalogue(canonical, pool, capabilities, overrides)

    assert got.model_id.tolist() == ["Canonical Model", "missing-model", "future-model"]
    assert got.model_id.map(lambda value: value.lower().replace(" ", "")).is_unique
    assert resolve_model_id("canonical-model", got) == "Canonical Model"
    assert resolve_model_id("FUTURE MODEL", got) == "future-model"
    assert pool_enabled(got.loc[got.model_id == "Canonical Model", "judge_pool"].item())
    assert pool_enabled(got.loc[got.model_id == "future-model", "candidate_pool"].item())
    assert pd.isna(got.loc[got.model_id == "missing-model", "theta"]).item()
    validate_catalogue(got)


def test_catalogue_requires_pool_name_to_disambiguate_distinct_normalized_matches(tmp_path):
    canonical = tmp_path / "canonical.md"
    canonical.write_text("- Shared Model\n")
    pool = tmp_path / "pool.json"
    pool.write_text(json.dumps([
        {"name": "Shared Model", "release": "2024-01-01"},
        {"name": "Other", "or_id": "provider/shared-model", "release": "2025-01-01"},
    ]))
    capabilities = pd.DataFrame([{
        "irt_model": "fixed_3pl", "fit": "gpqa", "model": "shared-model", "theta": 0.0,
    }])

    with pytest.raises(ValueError, match=r"Shared Model.*sharedmodel.*pool_name"):
        build_catalogue(canonical, pool, capabilities)

    overrides = [{"model_id": "Shared Model", "pool_name": "Other"}]
    got = build_catalogue(canonical, pool, capabilities, overrides)
    assert got.loc[0, "release"] == "2025-01-01"


def test_catalogue_allows_duplicate_aliases_from_one_pool_record(tmp_path):
    canonical = tmp_path / "canonical.md"
    canonical.write_text("- Same\n")
    pool = tmp_path / "pool.json"
    pool.write_text(json.dumps([{
        "name": "Same", "model": "provider/same", "or_id": "same", "release": "2024-01-01",
    }]))
    capabilities = pd.DataFrame([{
        "irt_model": "fixed_3pl", "fit": "gpqa", "model": "same", "theta": 0.0,
    }])

    assert build_catalogue(canonical, pool, capabilities).loc[0, "release"] == "2024-01-01"


def test_catalogue_rejects_aliases_reaching_distinct_pool_records_without_pool_name(tmp_path):
    canonical = tmp_path / "canonical.md"
    canonical.write_text("- Canonical\n")
    pool = tmp_path / "pool.json"
    pool.write_text(json.dumps([
        {"name": "Canonical", "release": "2024-01-01"},
        {"name": "Alias", "release": "2025-01-01"},
    ]))
    capabilities = pd.DataFrame([{
        "irt_model": "fixed_3pl", "fit": "gpqa", "model": "canonical", "theta": 0.0,
    }])
    override = {"model_id": "Canonical", "aliases": ["Alias"]}

    with pytest.raises(ValueError, match=r"Canonical.*canonical.*alias.*pool_name"):
        build_catalogue(canonical, pool, capabilities, [override])

    got = build_catalogue(canonical, pool, capabilities, [{**override, "pool_name": "Alias"}])
    assert got.loc[0, "release"] == "2025-01-01"


def test_default_catalogue_resolves_observed_logs_and_has_explicit_eligible_pools():
    config = json.loads(Path("configs/mc_cost/default.json").read_text())
    paths = config["paths"]
    got = build_catalogue(
        paths["canonical_models"], paths["model_pool"], paths["irt_capabilities"],
        paths["model_overrides"],
    )
    observed = {
        model
        for path in Path(paths["judge_logs"]).rglob("*.eval")
        for model in path.relative_to(paths["judge_logs"]).parts[1].split("-vs-", 1)
    }

    assert observed
    assert not [model for model in observed if resolve_model_id(model, got) is None]
    assert got.loc[got["judge_pool"].map(pool_enabled), "model_id"].tolist()
    assert got.loc[got["candidate_pool"].map(pool_enabled), "model_id"].tolist()
    missing = got.loc[got.model_id == "gpt-oss-20b"].iloc[0]
    assert pd.isna(missing.theta)
    assert not pool_enabled(missing.judge_pool)
    validate_catalogue(got)


def test_accuracy_to_theta_inverts_each_fixed_3pl_draw():
    draws = {"a": np.ones((3, 2)), "b": np.zeros((3, 2)), "floor": np.full(2, 0.25)}
    got = theta_draws_from_accuracy(75, 100, draws, np.random.default_rng(3))
    assert got.shape == (3,)
    assert np.all(np.isfinite(got))


def test_capability_draws_reuse_irt_and_do_not_impute_bare_point_estimate():
    catalogue = pd.DataFrame([
        {"model_id": "fitted", "theta": 1.0, "theta_source": "fixed_3pl"},
        {"model_id": "counted", "gpqa_correct": 7, "gpqa_total": 10,
         "theta_source": "gpqa_counts"},
        {"model_id": "bare", "theta": 0.0, "theta_source": "manual_point"},
    ])
    irt = {
        "models": ["fitted"], "theta": np.array([[0.9], [1.0], [1.1]]),
        "a": np.ones((3, 2)), "b": np.zeros((3, 2)), "floor": np.full(2, 0.25),
    }
    models, draws = build_capability_draws(catalogue, irt, np.random.default_rng(5))
    assert models == ["fitted", "counted"]
    assert draws.shape == (3, 2)
    assert np.allclose(draws[:, 0], [0.9, 1.0, 1.1])


def test_catalogue_rejects_duplicate_ids_and_bad_reasoning_class():
    frame = pd.DataFrame([
        {"model_id": "m", "reasoning_class": "reasoning", "release": "2025-01-01", "theta": 0.0},
        {"model_id": "m", "reasoning_class": "sometimes", "release": "2025-01-01", "theta": 0.0},
    ])
    with pytest.raises(ValueError, match="duplicate model_id|reasoning_class"):
        validate_catalogue(frame)


@pytest.mark.parametrize("column, value", [("price_in", "unknown"), ("price_out", np.inf), ("price_in", -1.0)])
def test_catalogue_rejects_present_invalid_prices(column, value):
    frame = pd.DataFrame([{"model_id": "m", column: value}])
    with pytest.raises(ValueError, match="price"):
        validate_catalogue(frame)


def test_accuracy_to_theta_rejects_fractional_gpqa_counts():
    draws = {"a": np.ones((2, 1)), "b": np.zeros((2, 1)), "floor": np.array([0.25])}
    with pytest.raises(ValueError, match="integer|counts"):
        theta_draws_from_accuracy(7.5, 10, draws, np.random.default_rng(3))


@pytest.mark.parametrize("counts", [(7.5, 10), (7, "unknown"), (7, None)])
def test_catalogue_rejects_present_invalid_gpqa_counts_outside_pools(counts):
    correct, total = counts
    values = {"model_id": "m", "gpqa_correct": correct, "gpqa_total": total}
    if total is None:
        values.pop("gpqa_total")
    with pytest.raises(ValueError, match="GPQA|counts"):
        validate_catalogue(pd.DataFrame([values]))
