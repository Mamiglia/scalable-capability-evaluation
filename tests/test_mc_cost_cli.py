import json
import hashlib
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from inspect_ai.log import (
    EvalConfig,
    EvalDataset,
    EvalLog,
    EvalSample,
    EvalSpec,
    write_eval_log,
)
from inspect_ai.model import ChatMessageAssistant, ModelUsage
from inspect_ai.scorer import Score

import mc_cost.cli as cli
from mc_cost.cli import main, write_run_config
from mc_cost.data import JUDGMENT_COLUMNS
from mc_cost.model import JUDGMENT_SPECS, PosteriorArtifact, load_artifact, save_artifact
from mc_cost.simulation import (
    SimulationResult,
    _trial_rows,
    summarize_trials,
    validate_dashboard_payload,
)
from mc_cost.validation import require_decision


def config_file(tmp_path, **updates):
    config = {
        "paths": {"output": str(tmp_path / "artifacts")},
        "fit": {"warmup": 10, "samples": 10, "chains": 1, "seed": 1},
        "simulation": {
            "trials": 10,
            "attempted_items": 4,
            "presentations": 2,
            "reuse_responses": True,
            "accuracy_target": "higher_theta",
            "seed": 1,
        },
        "dashboard": {},
    }
    for section, values in updates.items():
        config.setdefault(section, {}).update(values)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    return path


def write_generation(root, manifest, names):
    files = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in names
    }
    path = root / manifest
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema_version": 1, "files": files}, sort_keys=True))


CATALOGUE_FILES = ["catalogue.csv", "capability_draws.npz"]
PREPARE_FILES = [
    "judgments.parquet", "token_candidate.parquet", "token_judge.parquet",
    "judgment_standardization.json", "scales.json", "prepare_accounting.json",
]
MODEL_FILES = [
    f"model/{name}.{suffix}"
    for name in ("baseline", "core", "core_release", "candidate_token", "judge_token")
    for suffix in ("npz", "json")
]


@pytest.fixture
def tiny_mc_config(tmp_path):
    def build(*, seed=9, warmup=1, samples=4, trials=2):
        models = ["j1", "j2", "ca", "cb", "cc"]
        canonical = tmp_path / "canonical.md"
        canonical.write_text("".join(f"- {model}\n" for model in models))
        pool = tmp_path / "pool.json"
        pool.write_text(json.dumps([
            {
                "name": model, "family": model[0], "release": f"{year}-01-01",
                "reasoning": model.startswith("c"),
                "judge_pool": model.startswith("j"), "candidate_pool": model.startswith("c"),
                "observed_judge": model.startswith("j"), "observed_candidate": True,
                "price_in": 1.0, "price_out": 2.0,
            }
            for model, year in zip(models, (2022, 2023, 2024, 2025, 2026))
        ]))
        theta = np.array([[-0.7, -0.2, 0.3, 0.8, 1.3], [-0.6, -0.1, 0.4, 0.9, 1.4]])
        capabilities = tmp_path / "capabilities.csv"
        pd.DataFrame([
            {
                "irt_model": "fixed_3pl", "fit": "gpqa", "model": model,
                "theta": float(theta[:, index].mean()), "theta_p5": float(theta[:, index].min()),
                "theta_p95": float(theta[:, index].max()),
            }
            for index, model in enumerate(models)
        ]).to_csv(capabilities, index=False)
        items = [f"item{index}" for index in range(4)]
        difficulties = np.array([-0.6, -0.2, 0.2, 0.6])
        item_path = tmp_path / "items.csv"
        pd.DataFrame([
            {
                "irt_model": "fixed_3pl", "fit": "gpqa", "benchmark": "gpqa",
                "item": item, "difficulty": float(difficulty),
            }
            for item, difficulty in zip(items, difficulties)
        ]).to_csv(item_path, index=False)
        posterior = tmp_path / "irt.npz"
        np.savez_compressed(
            posterior,
            fixed_3pl__gpqa__models=np.array(models), fixed_3pl__gpqa__theta=theta,
            fixed_3pl__gpqa__a=np.full((2, 4), 1.2),
            fixed_3pl__gpqa__b=np.vstack([difficulties, difficulties + 0.05]),
            fixed_3pl__gpqa__floor=np.full(4, 0.25),
            fixed_3pl__gpqa__items=np.array([f"gpqa:{item}" for item in items]),
        )

        judge_logs = tmp_path / "judge_logs"
        candidate_logs = tmp_path / "candidate_logs"

        def write_log(path, model, samples_for_log, *, include_reasoning=False):
            path.parent.mkdir(parents=True, exist_ok=True)
            write_eval_log(EvalLog(
                status="success",
                eval=EvalSpec(
                    created="2026-08-15T00:00:00Z", task="gpqa",
                    dataset=EvalDataset(name="tiny", samples=len(samples_for_log)),
                    model=f"openrouter/{model}", config=EvalConfig(),
                    metadata={"include_reasoning": include_reasoning},
                ),
                samples=samples_for_log,
            ), path)

        for model in models:
            candidate_samples = []
            reasoning = model.startswith("c")
            for item_index, item in enumerate(items):
                output_tokens = 8 + item_index
                reasoning_tokens = 3 if reasoning else 0
                candidate_samples.append(EvalSample(
                    id=f"{item}_1", epoch=1, input="question", target="answer",
                    messages=[ChatMessageAssistant(content=(
                        "<think>reasoning</think> answer" if reasoning else "answer"
                    ))],
                    scores={"accuracy": Score(value="C")},
                    model_usage={"candidate": ModelUsage(
                        input_tokens=12 + item_index, output_tokens=output_tokens,
                        reasoning_tokens=reasoning_tokens,
                        total_tokens=12 + item_index + output_tokens,
                    )},
                    uuid=f"candidate-{model}-{item}",
                ))
            write_log(candidate_logs / f"{model}.eval", model, candidate_samples)

        pairs = (("ca", "cb"), ("ca", "cc"), ("cb", "cc"), ("ca", "ca"))
        for judge_index, judge in enumerate(("j1", "j2")):
            for pair_index, (major, minor) in enumerate(pairs):
                judge_samples = []
                for row in range(125):
                    item = items[row % len(items)]
                    epoch = row // len(items) + 1
                    major_left = (row + pair_index) % 2 == 0
                    answer = "A" if (row + judge_index) % 2 == 0 else "B"
                    left = "L" * (8 + row % 5 + pair_index)
                    right = "R" * (5 + (row + judge_index) % 7)
                    judge_samples.append(EvalSample(
                        id=f"{item}_{epoch}", epoch=epoch,
                        input=(
                            f"Transcript A:\n---\n{left}\n---\n"
                            f"Transcript B:\n---\n{right}\n---\nWhich of"
                        ),
                        target="A" if major_left else "B",
                        messages=[ChatMessageAssistant(content=answer)],
                        scores={"judge": Score(
                            value="C", answer=answer, metadata={"confidence": 0.8},
                        )},
                        model_usage={"judge": ModelUsage(
                            input_tokens=30 + row % 3, output_tokens=4,
                            reasoning_tokens=0, total_tokens=34 + row % 3,
                        )},
                        uuid=f"judge-{judge}-{pair_index}-{row}",
                    ))
                write_log(
                    judge_logs / judge / f"{major}-vs-{minor}" / "samples.eval",
                    judge, judge_samples,
                )

        return config_file(
            tmp_path,
            paths={
                "output": str(tmp_path / "artifacts"), "canonical_models": str(canonical),
                "model_pool": str(pool), "irt_capabilities": str(capabilities),
                "irt_items": str(item_path), "irt_posterior": str(posterior),
                "judge_logs": str(judge_logs),
                "candidate_log_roots": [str(candidate_logs)],
            },
            fit={"warmup": warmup, "samples": samples, "chains": 1, "seed": seed},
            simulation={
                "trials": trials, "attempted_items": 1, "presentations": 1,
                "reuse_responses": True, "accuracy_target": "higher_theta", "seed": seed,
                "judges": 1, "anchors": 1, "comparators": 1, "require_stronger": True,
            },
            dashboard={
                "judge_counts": [1], "anchor_counts": [1], "comparator_counts": [1],
                "attempted_item_counts": [1], "presentation_counts": [1],
                "strength_filters": [True], "manual_triples": [], "manual_triple_limit": 0,
                "out": str(tmp_path / "viewer/data/mc_cost.json"),
            },
        )

    return build


def test_tiny_pipeline_is_seed_reproducible(tmp_path, tiny_mc_config):
    config = tiny_mc_config(seed=9, warmup=1, samples=4, trials=2)
    for command in ("catalogue", "prepare", "fit", "validate", "simulate", "export-dashboard"):
        assert main([command, "--config", str(config)]) == 0

    output = tmp_path / "artifacts"
    cli._verify_generation(output, "catalogue.manifest.json", CATALOGUE_FILES, "catalogue", "catalogue")
    cli._verify_generation(output, "prepare.manifest.json", PREPARE_FILES, "prepare", "prepared-data")
    cli._verify_generation(output, "model/manifest.json", MODEL_FILES, "fit", "model")
    cli._verify_generation(
        output, "validation/manifest.json", ["validation/decision.json"], "validate", "validation",
    )
    hashes = current_data_hashes(output)
    require_decision(output / "validation/decision.json", hashes)
    for name in ("baseline", "core", "core_release", "candidate_token", "judge_token"):
        assert load_artifact(output / f"model/{name}").metadata["source_hashes"] == hashes
    run = next((output / "runs").iterdir())
    cli._verify_generation(
        run, "manifest.json", ["trials.parquet", "summary.csv", "warnings.csv", "config.json"],
        "simulate", "simulation",
    )

    dashboard = tmp_path / "viewer/data/mc_cost.json"
    first = dashboard.read_bytes()
    validate_dashboard_payload(json.loads(first))
    assert main(["export-dashboard", "--config", str(config)]) == 0
    assert dashboard.read_bytes() == first


def current_data_hashes(output):
    paths = {
        "catalogue": output / "catalogue.csv",
        "capability_draws": output / "capability_draws.npz",
        "judgments": output / "judgments.parquet",
        "candidate_tokens": output / "token_candidate.parquet",
        "judge_tokens": output / "token_judge.parquet",
        "scales": output / "scales.json",
    }
    return {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}


def refresh_artifact_provenance(output):
    hashes = current_data_hashes(output)
    for name in ("baseline", "core", "core_release", "candidate_token", "judge_token"):
        artifact = load_artifact(output / f"model/{name}")
        save_artifact(
            PosteriorArtifact(
                artifact.posterior, {**artifact.metadata, "source_hashes": hashes}, artifact.diagnostics,
            ),
            output / f"model/{name}",
        )
    write_generation(output, "model/manifest.json", MODEL_FILES)
    return hashes


def write_valid_decision(output, hashes, *, selected="baseline", gate=False):
    validation = output / "validation"
    validation.mkdir(exist_ok=True)
    (validation / "decision.json").write_text(json.dumps({
        "selected_judgment_spec": selected,
        "judgment_spec": {
            "global_features": list(JUDGMENT_SPECS[selected].global_features),
            "include_length": JUDGMENT_SPECS[selected].include_length,
        },
        "data_hashes": hashes,
        "judge_covariate_gate": {"enabled": gate},
        "validation_timestamp": "2026-08-15T00:00:00Z",
    }))
    write_generation(output, "validation/manifest.json", ["validation/decision.json"])


def test_cli_exposes_six_commands(capsys):
    with pytest.raises(SystemExit) as stopped:
        main(["--help"])
    assert stopped.value.code == 0
    help_text = capsys.readouterr().out
    for command in ("catalogue", "prepare", "fit", "validate", "simulate", "export-dashboard"):
        assert command in help_text


def test_manual_simulation_requires_exact_columns_before_upstream_artifacts(tmp_path, capsys):
    manual = tmp_path / "manual.csv"
    manual.write_text("judge,m1\nj,a\n")

    assert main(["simulate", "--manual", str(manual), "--config", str(config_file(tmp_path))]) == 2

    error = capsys.readouterr().err
    assert "judge,m1,m2" in error
    assert "catalogue" not in error


def test_saved_config_records_seed_and_source_hashes(tmp_path):
    path = tmp_path / "config.json"
    write_run_config(path, {"seed": 7}, {"catalogue": "abc", "model": "def"})
    assert json.loads(path.read_text()) == {
        "config": {"seed": 7}, "source_hashes": {"catalogue": "abc", "model": "def"},
    }


def test_response_cost_flags_are_mutually_exclusive(tmp_path):
    with pytest.raises(SystemExit) as stopped:
        main([
            "simulate", "--config", str(config_file(tmp_path)),
            "--reuse-responses", "--generate-responses",
        ])
    assert stopped.value.code == 2


def test_missing_prerequisite_names_the_producing_command(tmp_path, capsys):
    assert main(["prepare", "--config", str(config_file(tmp_path))]) == 2
    assert "mc-cost catalogue" in capsys.readouterr().err


def test_missing_irt_product_names_the_irt_proxy_producer(tmp_path, capsys):
    canonical = tmp_path / "canonical.md"
    canonical.write_text("- model\n")
    pool = tmp_path / "pool.json"
    pool.write_text("[]\n")
    capabilities = tmp_path / "capabilities.csv"
    capabilities.write_text("model,theta\n")
    config = config_file(tmp_path, paths={
        "output": str(tmp_path / "artifacts"),
        "canonical_models": str(canonical), "model_pool": str(pool),
        "irt_capabilities": str(capabilities),
        "irt_posterior": str(tmp_path / "missing-irt.npz"),
    })

    assert main(["catalogue", "--config", str(config)]) == 2
    assert "scripts/irt_proxy/run.py" in capsys.readouterr().err


def test_default_config_points_at_current_producers_and_observed_logs():
    paths = json.loads(Path("configs/mc_cost/default.json").read_text())["paths"]
    assert paths["irt_capabilities"] == "irt_results/easy_item_prediction/tables/fit_capabilities.csv"
    assert paths["irt_items"] == "irt_results/easy_item_prediction/tables/fit_difficulties.csv"
    assert paths["irt_posterior"] == "irt_results/easy_item_prediction/posterior/irt_posterior.npz"
    assert paths["judge_logs"] == "logs/judging-variations/gpqa/a_baseline"
    assert paths["model_overrides"] == "configs/mc_cost/model_overrides.json"
    assert "judge_models" not in paths
    assert "epoch_responses" not in paths


def test_catalogue_builds_validated_table_and_aligned_draws_atomically(tmp_path):
    canonical = tmp_path / "canonical.md"
    canonical.write_text("- fitted\n")
    pool = tmp_path / "pool.json"
    pool.write_text(json.dumps([{
        "name": "fitted", "release": "2025-01-01", "reasoning": False,
        "judge_pool": False, "candidate_pool": False,
    }]))
    capabilities = tmp_path / "capabilities.csv"
    pd.DataFrame([{
        "irt_model": "fixed_3pl", "fit": "gpqa", "model": "fitted",
        "theta": 0.5, "theta_p5": 0.2, "theta_p95": 0.8,
    }]).to_csv(capabilities, index=False)
    posterior = tmp_path / "irt.npz"
    np.savez_compressed(
        posterior,
        fixed_3pl__gpqa__models=np.array(["fitted"]),
        fixed_3pl__gpqa__theta=np.array([[0.4], [0.6]]),
        fixed_3pl__gpqa__a=np.ones((2, 1)),
        fixed_3pl__gpqa__b=np.zeros((2, 1)),
        fixed_3pl__gpqa__floor=np.array([0.25]),
        fixed_3pl__gpqa__items=np.array(["gpqa:i"]),
    )
    overrides = tmp_path / "overrides.json"
    overrides.write_text('{"models": []}\n')
    config = config_file(tmp_path, paths={
        "output": str(tmp_path / "artifacts"),
        "canonical_models": str(canonical), "model_pool": str(pool),
        "model_overrides": str(overrides), "irt_capabilities": str(capabilities),
        "irt_posterior": str(posterior),
    })

    assert main(["catalogue", "--config", str(config), "--seed", "7"]) == 0

    stored = pd.read_csv(tmp_path / "artifacts/catalogue.csv")
    assert stored["model_id"].tolist() == ["fitted"]
    with np.load(tmp_path / "artifacts/capability_draws.npz", allow_pickle=False) as draws:
        assert draws["models"].tolist() == ["fitted"]
        assert draws["draws"].tolist() == [[0.4], [0.6]]
        assert set(draws["source_names"].tolist()) == {
            "canonical_models", "model_pool", "model_overrides", "irt_capabilities", "irt_posterior",
        }
    manifest = json.loads((tmp_path / "artifacts/catalogue.manifest.json").read_text())
    assert set(manifest["files"]) == set(CATALOGUE_FILES)


def test_prepare_calls_raw_boundaries_and_writes_declared_tables(tmp_path, monkeypatch):
    output = tmp_path / "artifacts"
    output.mkdir()
    catalogue = pd.DataFrame([
        {"model_id": "j", "theta": 0.0, "release": "2024-01-01", "family": "j"},
        {"model_id": "a", "theta": 1.0, "release": "2025-01-01", "family": "a"},
        {"model_id": "b", "theta": -1.0, "release": "2023-01-01", "family": "b"},
    ])
    catalogue.to_csv(output / "catalogue.csv", index=False)
    (output / "capability_draws.npz").write_bytes(b"draws")
    write_generation(output, "catalogue.manifest.json", CATALOGUE_FILES)
    items = tmp_path / "items.csv"
    pd.DataFrame([{
        "irt_model": "fixed_3pl", "fit": "gpqa", "benchmark": "gpqa",
        "item": "i", "difficulty": 0.2,
    }]).to_csv(items, index=False)
    logs = tmp_path / "logs"
    logs.mkdir()
    attempts = pd.DataFrame([{
        "chosen_left": 1, "judge": "j", "item": "i", "left_model": "a", "right_model": "b",
        "length_difference": 0.2, "same_model_family": 0, "self_comparison": 0,
        "reasoning_difference": 0, "left_chars": 10, "right_chars": 8,
        "include_reasoning": False,
        "judge_input_tokens": 20, "judge_output_tokens": 4,
        "judge_reasoning_tokens": 0, "judge_total_tokens": 24,
    }])
    judgment_exclusion = {
        "source": "judgment", "source_log": "bad.eval", "judge": "j", "model": "a|b",
        "item": "bad", "reason": "invalid_answer", "missing_fields": [], "count": 2,
    }
    attempts.attrs["exclusions"] = [judgment_exclusion]
    judgment_counts = pd.DataFrame({"attempts_kept": [1], "attempts_skipped_invalid_answer": [2]}, index=["j"])
    judgment_eligibility = pd.DataFrame({
        "pairs": [1], "samples": [1], "eligible": [True],
    }, index=["j"])
    candidate_exclusion = {
        "source": "candidate", "source_log": "candidate.eval", "judge": None,
        "model": "missing", "item": None, "reason": "unknown_model",
        "missing_fields": ["catalogue"], "count": 3,
    }
    candidates = pd.DataFrame()
    candidates.attrs["accounting"] = {"zero_usage_or_error": 0, "missing_catalogue_metadata": 3}
    candidates.attrs["exclusions"] = [candidate_exclusion]
    calls = []

    def fake_prepare(*args, **kwargs):
        calls.append((args, kwargs))
        return attempts, judgment_counts, judgment_eligibility

    monkeypatch.setattr(cli, "prepare_mc_gpqa", fake_prepare)
    monkeypatch.setattr(cli, "candidate_token_rows", lambda *_, **__: candidates)
    config = config_file(tmp_path, paths={
        "output": str(output), "irt_items": str(items), "judge_logs": str(logs),
        "candidate_log_roots": [],
    })

    assert main(["prepare", "--config", str(config), "--seed", "9"]) == 0

    assert calls and calls[0][0][0] == logs
    assert calls[0][0][1].equals(catalogue)
    assert pd.read_parquet(output / "judgments.parquet").shape[0] == 1
    assert (output / "token_candidate.parquet").is_file()
    assert (output / "token_judge.parquet").is_file()
    assert (output / "scales.json").is_file()
    assert (output / "judgment_standardization.json").is_file()
    assert json.loads((output / "prepare_accounting.json").read_text()) == {
        "schema_version": 1,
        "judgment_counts": [{
            "judge": "j", "attempts_kept": 1, "attempts_skipped_invalid_answer": 2,
        }],
        "judgment_eligibility": [{"judge": "j", "pairs": 1, "samples": 1, "eligible": True}],
        "candidate_counts": {"missing_catalogue_metadata": 3, "zero_usage_or_error": 0},
        "exclusions": [judgment_exclusion, candidate_exclusion],
    }
    manifest = json.loads((output / "prepare.manifest.json").read_text())
    assert set(manifest["files"]) == set(PREPARE_FILES)


def test_prepare_rejects_missing_configured_candidate_root_before_raw_loading(
    tmp_path, monkeypatch, capsys,
):
    output = tmp_path / "artifacts"
    output.mkdir()
    pd.DataFrame([{"model_id": "m"}]).to_csv(output / "catalogue.csv", index=False)
    (output / "capability_draws.npz").write_bytes(b"draws")
    write_generation(output, "catalogue.manifest.json", CATALOGUE_FILES)
    items = tmp_path / "items.csv"
    items.write_text("irt_model,fit,benchmark,item,difficulty\n")
    logs = tmp_path / "logs"
    logs.mkdir()
    missing = tmp_path / "missing-candidate-logs"
    monkeypatch.setattr(cli, "prepare_mc_gpqa", lambda *_: pytest.fail("raw preparation ran"))
    config = config_file(tmp_path, paths={
        "output": str(output), "irt_items": str(items), "judge_logs": str(logs),
        "candidate_log_roots": [str(missing)],
    })

    assert main(["prepare", "--config", str(config)]) == 2
    error = capsys.readouterr().err
    assert "candidate log root" in error
    assert str(missing) in error


def test_fit_writes_three_judgment_and_two_token_artifacts(tmp_path, monkeypatch):
    output = tmp_path / "artifacts"
    output.mkdir()
    pd.DataFrame([{"model_id": "m", "theta": 0.0, "release": "2024-01-01"}]).to_csv(
        output / "catalogue.csv", index=False,
    )
    pd.DataFrame([{column: 0 for column in JUDGMENT_COLUMNS}]).to_parquet(
        output / "judgments.parquet", index=False,
    )
    pd.DataFrame([{"model": "m"}]).to_parquet(output / "token_candidate.parquet", index=False)
    pd.DataFrame([{"model": "m"}]).to_parquet(output / "token_judge.parquet", index=False)
    scale_payload = json.dumps({
        "mean": {"theta_gap": 0}, "scale": {"theta_gap": 1},
    })
    (output / "scales.json").write_text(scale_payload)
    (output / "judgment_standardization.json").write_text(scale_payload)
    (output / "prepare_accounting.json").write_text("{}\n")
    (output / "capability_draws.npz").write_bytes(b"draws")
    write_generation(output, "catalogue.manifest.json", CATALOGUE_FILES)
    write_generation(output, "prepare.manifest.json", PREPARE_FILES)
    encoded = []
    fit_calls = []

    def fake_encode(frame, catalogue, spec, **kwargs):
        encoded.append((spec, kwargs))
        return spec

    def fake_judgment(data, **kwargs):
        fit_calls.append(("judgment", kwargs))
        return PosteriorArtifact({"draw": np.array([1.0])}, {"kind": "judgment"}, {})

    def fake_token(frame, role, **kwargs):
        fit_calls.append((role, kwargs))
        return PosteriorArtifact({"draw": np.array([1.0])}, {"role": role}, {})

    monkeypatch.setattr(cli, "encode_judgments", fake_encode)
    monkeypatch.setattr(cli, "fit_judgment_model", fake_judgment)
    monkeypatch.setattr(cli, "fit_token_model", fake_token)

    assert main([
        "fit", "--config", str(config_file(tmp_path)), "--seed", "8",
        "--warmup", "2", "--samples", "3", "--chains", "1",
    ]) == 0

    assert len(encoded) == 3
    assert encoded[0][1]["training_scales"] == {"theta_gap": {"mean": 0, "scale": 1}}
    assert [call[1]["seed"] for call in fit_calls] == [8] * 5
    assert [call[1]["warmup"] for call in fit_calls] == [2] * 5
    for name in ("baseline", "core", "core_release", "candidate_token", "judge_token"):
        assert (output / f"model/{name}.npz").is_file()
        assert (output / f"model/{name}.json").is_file()


def pipeline_artifacts(tmp_path, *, dashboard=None):
    output = tmp_path / "artifacts"
    output.mkdir()
    pd.DataFrame([
        {
            "model_id": "j", "theta": 0.0, "release": "2024-01-01", "family": "j",
            "reasoning_class": "non_reasoning", "judge_pool": True, "candidate_pool": False,
            "price_in": 1.0, "price_out": 2.0,
        },
        {
            "model_id": "a", "theta": 1.0, "release": "2025-01-01", "family": "a",
            "reasoning_class": "non_reasoning", "judge_pool": False, "candidate_pool": True,
            "price_in": 1.0, "price_out": 2.0,
        },
        {
            "model_id": "b", "theta": -1.0, "release": "2023-01-01", "family": "b",
            "reasoning_class": "non_reasoning", "judge_pool": False, "candidate_pool": True,
            "price_in": 1.0, "price_out": 2.0,
        },
    ]).to_csv(output / "catalogue.csv", index=False)
    judgments = pd.DataFrame([{column: 0 for column in JUDGMENT_COLUMNS} for _ in range(2)])
    judgments[["judge", "item", "left_model", "right_model"]] = ["j", "i", "a", "b"]
    judgments["chosen_left"] = [0, 1]
    judgments["include_reasoning"] = False
    judgments["major_is_left"] = [False, True]
    judgments.loc[1, [
        "theta_gap", "oversight", "release_gap_days", "relative_release_days", "difficulty",
        "theta_gap_x_oversight", "theta_gap_x_difficulty", "length_difference",
        "same_family_x_theta_gap", "reasoning_difference",
        "self_comparison",
    ]] = 1
    judgments.to_parquet(output / "judgments.parquet", index=False)
    tokens = pd.DataFrame([{
        "role": "candidate", "model": "a", "item": "i", "reasoning_class": "non_reasoning",
        "correct": 1, "response_chars": 10, "visible_reasoning": 0,
        "theta": 1.0, "release_day": 1827, "difficulty": 0.0,
        "input_tokens": 2, "output_tokens": 2, "reasoning_tokens": 0,
        "ordinary_output_tokens": 2,
    }])
    tokens.to_parquet(output / "token_candidate.parquet", index=False)
    tokens.assign(role="judge", model="j", transcript_chars=20).drop(columns="response_chars").to_parquet(
        output / "token_judge.parquet", index=False,
    )
    scales = json.dumps({
        column: {"mean": 0.0, "scale": 1.0}
        for column in (
            "theta_gap", "oversight", "difficulty", "theta_gap_x_oversight",
            "theta_gap_x_difficulty", "same_family_x_theta_gap", "reasoning_difference",
            "release_gap_days", "relative_release_days", "length_difference",
        )
    })
    (output / "scales.json").write_text(scales)
    (output / "judgment_standardization.json").write_text(scales)
    (output / "prepare_accounting.json").write_text("{}\n")
    irt = tmp_path / "irt.npz"
    np.savez_compressed(
        irt, fixed_3pl__gpqa__a=np.ones((2, 1)), fixed_3pl__gpqa__b=np.zeros((2, 1)),
        fixed_3pl__gpqa__floor=np.array([0.25]), fixed_3pl__gpqa__items=np.array(["gpqa:i"]),
    )
    upstream = {
        "canonical_models": tmp_path / "canonical.md",
        "model_pool": tmp_path / "pool.json",
        "irt_capabilities": tmp_path / "capabilities.csv",
        "irt_posterior": irt,
    }
    upstream["canonical_models"].write_text("- j\n- a\n- b\n")
    upstream["model_pool"].write_text("[]\n")
    upstream["irt_capabilities"].write_text("model,theta\n")
    upstream_hashes = {
        name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in upstream.items()
    }
    np.savez_compressed(
        output / "capability_draws.npz", models=np.array(["j", "a", "b"]),
        draws=np.array([[0.0, 1.0, -1.0], [0.1, 1.1, -0.9]]),
        source_names=np.array(list(upstream_hashes)),
        source_hashes=np.array(list(upstream_hashes.values())),
    )
    names = {
        "catalogue": output / "catalogue.csv", "capability_draws": output / "capability_draws.npz",
        "judgments": output / "judgments.parquet", "candidate_tokens": output / "token_candidate.parquet",
        "judge_tokens": output / "token_judge.parquet", "scales": output / "scales.json",
    }
    hashes = {
        name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in names.items()
    }
    for name in ("baseline", "core", "core_release", "candidate_token", "judge_token"):
        metadata = {"source_hashes": hashes}
        if name in JUDGMENT_SPECS:
            metadata.update({
                "feature_order": list(JUDGMENT_SPECS[name].global_features),
                "include_length": JUDGMENT_SPECS[name].include_length,
                "conditional_judges": False,
                "presentation_include_reasoning": False,
            })
        else:
            metadata["role"] = name.removesuffix("_token")
        save_artifact(
            PosteriorArtifact({"draw": np.array([1.0])}, metadata, {}),
            output / f"model/{name}",
        )
    write_generation(output, "catalogue.manifest.json", CATALOGUE_FILES)
    write_generation(output, "prepare.manifest.json", PREPARE_FILES)
    write_generation(output, "model/manifest.json", MODEL_FILES)
    config = config_file(
        tmp_path,
        paths={"output": str(output), **{name: str(path) for name, path in upstream.items()}},
        simulation={
            "trials": 2, "attempted_items": 1, "presentations": 1,
            "reuse_responses": True, "accuracy_target": "higher_theta", "seed": 3,
            "judges": 1, "anchors": 1, "comparators": 1,
        },
        dashboard=dashboard or {},
    )
    return config, output, hashes


def test_validate_writes_held_out_tables_and_hash_bound_decision(tmp_path, monkeypatch):
    config, output, hashes = pipeline_artifacts(tmp_path)
    calls = []
    folds = pd.DataFrame([
        {
            "spec": spec, "population": population, "validation": "judge",
            "fold": "judge:j", "held_out": "j",
            "log_loss": loss + (0.2 if spec == "core" and population == "pooled" else 0),
        }
        for spec, loss in (("baseline", 0.7), ("core", 0.5), ("core_release", 0.6))
        for population in ("conditional", "pooled")
    ])

    def fake_judgments(frame, callback):
        calls.append("judgments")
        return pd.DataFrame([
            {
                "spec": spec, "population": population, "validation": "judge",
                "fold": "judge:j", "held_out": "j", "source_row": 0,
                "draw": draw, "observed": 0, "probability": probability,
            }
            for spec in ("baseline", "core", "core_release")
            for population in ("conditional", "pooled")
            for draw, probability in enumerate((0.2, 0.8))
        ]), folds

    def fake_tokens(frame, role, callback):
        calls.append(role)
        predictions = pd.DataFrame([{"role": role, "outcome": "input_tokens"}])
        summary = pd.DataFrame([{
            "role": role, "reasoning_class": "non_reasoning", "outcome": "input_tokens",
            "coverage_90": 0.9,
        }])
        return predictions, summary, pd.DataFrame()

    monkeypatch.setattr(cli, "cross_validate_judgments", fake_judgments)
    monkeypatch.setattr(cli, "validate_tokens", fake_tokens)
    monkeypatch.setattr(cli, "encode_judgments", lambda *args, **kwargs: object())
    monkeypatch.setattr(cli, "fit_judgment_model", lambda data, **kwargs: PosteriorArtifact(
        {"draw": np.array([1.0])},
        {"source_hashes": hashes, "conditional_judges": kwargs["conditional_judges"]}, {},
    ))

    assert main(["validate", "--config", str(config), "--seed", "11"]) == 0

    decision = json.loads((output / "validation/decision.json").read_text())
    assert decision["selected_judgment_spec"] == "core"
    assert decision["judge_covariate_gate"]["enabled"] is True
    assert decision["data_hashes"] == hashes
    selected_artifact = load_artifact(output / "model/core")
    assert selected_artifact.metadata["conditional_judges"] is True
    assert selected_artifact.metadata["feature_order"] == list(JUDGMENT_SPECS["core"].global_features)
    assert selected_artifact.metadata["include_length"] is True
    assert calls == ["judgments", "candidate", "judge"]
    for name in (
        "judgment_predictions.csv", "judgment_folds.csv", "token_candidate_predictions.csv",
        "token_candidate_summary.csv", "token_judge_predictions.csv", "token_judge_summary.csv",
    ):
        assert (output / f"validation/{name}").is_file()
    replay = pd.read_csv(output / "validation/posterior_predictive_replay.csv")
    assert replay.loc[replay["check"].eq("position_effect"), "valid"].all()
    assert replay.loc[replay["check"].eq("aggregate_choice_rate"), "n_draws"].tolist() == [2]


def test_validate_refuses_stale_full_token_artifact_before_decision(tmp_path, monkeypatch, capsys):
    config, output, _hashes = pipeline_artifacts(tmp_path)
    artifact = load_artifact(output / "model/candidate_token")
    save_artifact(
        PosteriorArtifact(
            artifact.posterior, {**artifact.metadata, "source_hashes": {"stale": "hash"}},
            artifact.diagnostics,
        ),
        output / "model/candidate_token",
    )
    write_generation(output, "model/manifest.json", MODEL_FILES)
    monkeypatch.setattr(
        cli, "cross_validate_judgments", lambda *_: pytest.fail("validation ran on stale artifacts"),
    )

    assert main(["validate", "--config", str(config)]) == 2
    assert "source hash mismatch" in capsys.readouterr().err
    assert not (output / "validation/decision.json").exists()


def test_real_judgment_callback_encodes_target_free_held_out_rows(monkeypatch):
    catalogue = pd.DataFrame([
        {"model_id": "j1", "theta": 0.0, "release": "2024-01-01"},
        {"model_id": "j2", "theta": 0.1, "release": "2024-02-01"},
        {"model_id": "a", "theta": 1.0, "release": "2025-01-01"},
        {"model_id": "b", "theta": -1.0, "release": "2023-01-01"},
    ])
    rows = pd.DataFrame([{column: 0 for column in JUDGMENT_COLUMNS} for _ in range(2)])
    rows["include_reasoning"] = False
    rows[["judge", "item", "left_model", "right_model"]] = ["j1", "i", "a", "b"]
    rows.loc[1, "judge"] = "j2"
    rows.loc[1, [
        "theta_gap", "oversight", "release_gap_days", "relative_release_days",
        "difficulty", "theta_gap_x_oversight", "theta_gap_x_difficulty",
        "length_difference", "same_family_x_theta_gap", "reasoning_difference",
    ]] = 1.0

    def cheap_fit(data, **kwargs):
        draws = 2
        posterior = {
            "judge_effects": np.zeros((draws, data.num_judges, 2)),
            "judge_population": np.zeros((draws, 2, 1)),
            "judge_corr": np.tile(np.eye(2), (draws, 1, 1)),
            "judge_scales": np.zeros((draws, 2)),
            "candidate_effects": np.zeros((draws, data.num_models)),
            "candidate_scale": np.zeros(draws),
            "item_effects": np.zeros((draws, data.num_items)),
            "item_scale": np.zeros(draws),
        }
        return PosteriorArtifact(
            posterior,
            {
                "training_scales": data.training_scales,
                "source_hashes": data.source_hashes,
                "conditional_judges": kwargs["conditional_judges"],
            },
            {},
        )

    monkeypatch.setattr(cli, "fit_judgment_model", cheap_fit)
    callback = cli._judgment_validator(
        catalogue, {"judgments": "hash"},
        {"warmup": 1, "samples": 2, "chains": 1, "seed": 1},
        np.random.default_rng(1),
    )

    probabilities = callback(
        rows.iloc[:1], rows.iloc[1:].drop(columns="chosen_left"),
        JUDGMENT_SPECS["baseline"], False,
    )

    assert probabilities.shape == (2, 1)
    assert np.allclose(probabilities, 0.5)


def fake_result(configuration="test"):
    trials = pd.DataFrame([{
        "trial": 0, "configuration": configuration, "mode": "automatic",
        "comparison_id": None, "m1": None, "m2": None,
        "metric": "cost", "value": 1.0,
    }])
    summary = pd.DataFrame([{
        "configuration": configuration, "mode": "automatic", "metric": "cost",
        "p5": 1.0, "p50": 1.0, "p95": 1.0, "n_trials": 1,
    }])
    return SimulationResult(trials, summary, pd.DataFrame(columns=["code", "message"]))


def test_simulate_validates_hashes_and_publishes_deterministic_run(tmp_path, monkeypatch):
    config, output, hashes = pipeline_artifacts(tmp_path)
    validation = output / "validation"
    write_valid_decision(output, hashes)
    seen = []

    def fake_simulate(comparisons, resolved, **kwargs):
        seen.append((comparisons, resolved, kwargs))
        return fake_result(resolved.configuration)

    monkeypatch.setattr(cli, "simulate", fake_simulate)
    command = [
        "simulate", "--config", str(config), "--judges", "1", "--anchors", "1",
        "--comparators", "1", "--items", "1", "--presentations", "1",
        "--generate-responses", "--accuracy-target", "higher_theta", "--seed", "17",
    ]

    assert main(command) == 0
    existing_run = next((output / "runs").iterdir())
    original_replace = Path.replace

    def keep_existing_directory(source, target):
        if source == existing_run and Path(target).name == f".{existing_run.name}.previous":
            pytest.fail("existing deterministic run directory was temporarily removed")
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", keep_existing_directory)
    decision = json.loads((validation / "decision.json").read_text())
    decision["validation_timestamp"] = "2026-08-16T00:00:00Z"
    (validation / "decision.json").write_text(json.dumps(decision))
    write_generation(output, "validation/manifest.json", ["validation/decision.json"])
    assert main(command) == 0

    runs = list((output / "runs").iterdir())
    assert len(runs) == 1
    saved = json.loads((runs[0] / "config.json").read_text())
    assert saved["config"] == {
        "accuracy_target": "higher_theta", "anchors": 1, "comparators": 1,
        "items": 1, "judges": 1, "mode": "automatic", "presentations": 1,
        "require_stronger": True, "reuse_responses": False, "seed": 17,
        "selected_judges": [], "trials": 2,
    }
    assert "decision" in saved["source_hashes"]
    assert (runs[0] / "trials.parquet").is_file()
    assert (runs[0] / "summary.csv").is_file()
    assert (runs[0] / "warnings.csv").is_file()
    assert (runs[0] / "manifest.json").is_file()
    assert len(seen) == 2


def test_simulate_refuses_mixed_model_generation(tmp_path, monkeypatch, capsys):
    config, output, hashes = pipeline_artifacts(tmp_path)
    write_valid_decision(output, hashes)
    with (output / "model/candidate_token.npz").open("ab") as handle:
        handle.write(b"mixed-generation")
    monkeypatch.setattr(cli, "simulate", lambda *_args, **_kwargs: pytest.fail("simulation ran"))

    assert main(["simulate", "--config", str(config)]) == 2
    assert "model generation" in capsys.readouterr().err


def test_simulate_enforces_selected_artifact_spec_metadata(tmp_path, monkeypatch, capsys):
    config, output, hashes = pipeline_artifacts(tmp_path)
    artifact = load_artifact(output / "model/baseline")
    save_artifact(
        PosteriorArtifact(
            artifact.posterior, {**artifact.metadata, "feature_order": ["oversight_sd"]},
            artifact.diagnostics,
        ),
        output / "model/baseline",
    )
    write_generation(output, "model/manifest.json", MODEL_FILES)
    write_valid_decision(output, hashes)
    monkeypatch.setattr(cli, "simulate", lambda *_args, **_kwargs: pytest.fail("simulation ran"))

    assert main(["simulate", "--config", str(config)]) == 2
    assert "selected judgment specification" in capsys.readouterr().err


def test_simulate_rejects_decision_with_tampered_selected_spec(tmp_path, monkeypatch, capsys):
    config, output, hashes = pipeline_artifacts(tmp_path)
    write_valid_decision(output, hashes)
    decision_path = output / "validation/decision.json"
    decision = json.loads(decision_path.read_text())
    decision["judgment_spec"]["global_features"] = ["oversight_sd"]
    decision_path.write_text(json.dumps(decision))
    write_generation(output, "validation/manifest.json", ["validation/decision.json"])
    monkeypatch.setattr(cli, "simulate", lambda *_args, **_kwargs: pytest.fail("simulation ran"))

    assert main(["simulate", "--config", str(config)]) == 2
    assert "decision judgment specification" in capsys.readouterr().err


def test_simulate_rejects_capability_draws_from_changed_irt(tmp_path, monkeypatch, capsys):
    config, output, hashes = pipeline_artifacts(tmp_path)
    write_valid_decision(output, hashes)
    irt = Path(json.loads(Path(config).read_text())["paths"]["irt_posterior"])
    with irt.open("ab") as handle:
        handle.write(b"new-irt-generation")
    monkeypatch.setattr(cli, "simulate", lambda *_args, **_kwargs: pytest.fail("simulation ran"))

    assert main(["simulate", "--config", str(config)]) == 2
    assert "IRT posterior hash" in capsys.readouterr().err


def test_simulate_rejects_misaligned_capability_models(tmp_path, monkeypatch, capsys):
    config, output, _hashes = pipeline_artifacts(tmp_path)
    with np.load(output / "capability_draws.npz", allow_pickle=False) as stored:
        sources = {name: stored[name].copy() for name in ("source_names", "source_hashes")}
    np.savez_compressed(
        output / "capability_draws.npz", models=np.array(["j", "a", "unknown"]),
        draws=np.array([[0.0, 1.0, -1.0], [0.1, 1.1, -0.9]]), **sources,
    )
    write_generation(output, "catalogue.manifest.json", CATALOGUE_FILES)
    hashes = refresh_artifact_provenance(output)
    write_valid_decision(output, hashes)
    monkeypatch.setattr(cli, "simulate", lambda *_args, **_kwargs: pytest.fail("simulation ran"))

    assert main(["simulate", "--config", str(config)]) == 2
    assert "capability model alignment" in capsys.readouterr().err


def test_simulate_refuses_stale_decision_before_calling_engine(tmp_path, monkeypatch, capsys):
    config, output, hashes = pipeline_artifacts(tmp_path)
    write_valid_decision(output, hashes)
    with (output / "judgments.parquet").open("ab") as handle:
        handle.write(b"stale")
    monkeypatch.setattr(cli, "simulate", lambda *args, **kwargs: pytest.fail("simulation ran"))

    assert main(["simulate", "--config", str(config)]) == 2
    assert "hash mismatch" in capsys.readouterr().err


def test_manual_run_config_keeps_exact_rows(tmp_path, monkeypatch):
    config, output, hashes = pipeline_artifacts(tmp_path)
    write_valid_decision(output, hashes)
    monkeypatch.setattr(cli, "simulate", lambda *args, **kwargs: fake_result())
    manual = tmp_path / "manual.csv"
    manual.write_text("judge,m1,m2\nj,a,b\nj,a,b\n")

    assert main(["simulate", "--config", str(config), "--manual", str(manual)]) == 0

    run = next((output / "runs").iterdir())
    saved = json.loads((run / "config.json").read_text())["config"]
    assert saved["manual_comparisons"] == [
        {"judge": "j", "m1": "a", "m2": "b"},
        {"judge": "j", "m1": "a", "m2": "b"},
    ]


def test_simulate_rejects_artifact_that_disagrees_with_judge_gate(tmp_path, monkeypatch, capsys):
    config, output, hashes = pipeline_artifacts(tmp_path)
    write_valid_decision(output, hashes, gate=True)
    monkeypatch.setattr(cli, "simulate", lambda *args, **kwargs: pytest.fail("simulation ran"))

    assert main(["simulate", "--config", str(config)]) == 2
    assert "judge-covariate gate" in capsys.readouterr().err


def test_export_dashboard_runs_grid_and_manual_bank_then_replaces_json(tmp_path, monkeypatch):
    dashboard = {
        "judge_counts": [1], "anchor_counts": [1], "comparator_counts": [1],
        "attempted_item_counts": [1], "presentation_counts": [1],
        "strength_filters": [True, False], "manual_triple_limit": 0,
        "manual_triples": [{"judge": "j", "m1": "a", "m2": "b"}],
    }
    config, output, hashes = pipeline_artifacts(tmp_path, dashboard=dashboard)
    write_valid_decision(output, hashes)
    calls = []

    def fake_simulate(comparisons, resolved, **kwargs):
        calls.append(resolved.mode)
        result = fake_result(resolved.configuration)
        return SimulationResult(
            result.trials.assign(mode=resolved.mode), result.summary.assign(mode=resolved.mode), result.warnings,
        )

    monkeypatch.setattr(cli, "simulate", fake_simulate)
    target = tmp_path / "dashboard.json"

    assert main([
        "export-dashboard", "--config", str(config), "--out", str(target),
        "--trials", "3", "--seed", "19",
    ]) == 0

    payload = json.loads(target.read_text())
    assert calls == ["automatic", "automatic", "manual"]
    assert set(payload) == {
        "schema_version", "generated_at", "source_hashes", "quantiles", "catalogue",
        "automatic", "manual", "scaling", "ranking_recovery", "warnings",
    }
    assert payload["generated_at"] == "2026-08-15T00:00:00Z"
    assert {row["strength_filter"] for row in payload["automatic"]} == {True, False}
    assert {row["mode"] for row in payload["automatic"]} == {"automatic"}
    assert {row["mode"] for row in payload["manual"]} == {"manual"}
    assert payload["ranking_recovery"]
    automatic_configurations = {row["configuration"] for row in payload["automatic"]}
    assert {
        row["configuration"] for row in payload["ranking_recovery"]
        if row["mode"] == "automatic"
    } == automatic_configurations
    identity = {
        key: payload["manual"][0][key]
        for key in (
            "judge", "m1", "m2", "judge_reasoning_class",
            "m1_reasoning_class", "m2_reasoning_class",
        )
    }
    assert identity == {
        "judge": "j", "m1": "a", "m2": "b",
        "judge_reasoning_class": "non_reasoning",
        "m1_reasoning_class": "non_reasoning",
        "m2_reasoning_class": "non_reasoning",
    }
    assert payload["manual"][0]["manual_source"] == "configured"
    assert payload["manual"][0]["resolved_configuration"]["manual"] == {
        "identity": identity, "limit": 0, "source": "configured",
    }
    assert payload["source_hashes"]["decision"]


def test_export_dashboard_keeps_unavailable_metrics_and_warnings(tmp_path, monkeypatch):
    dashboard = {
        "judge_counts": [1], "anchor_counts": [1], "comparator_counts": [1],
        "attempted_item_counts": [1], "presentation_counts": [1],
        "strength_filters": [True], "manual_triple_limit": 0,
        "manual_triples": [
            {"judge": "j", "m1": "a", "m2": "a"},
            {"judge": "j", "m1": "a", "m2": "missing"},
        ],
    }
    config, output, _ = pipeline_artifacts(tmp_path, dashboard=dashboard)
    catalogue = pd.read_csv(output / "catalogue.csv")
    catalogue.loc[catalogue.model_id.eq("a"), "price_out"] = np.nan
    catalogue = pd.concat([catalogue, pd.DataFrame([{
        "model_id": "missing", "theta": np.nan, "release": None, "family": "missing",
        "reasoning_class": "non_reasoning", "judge_pool": False, "candidate_pool": False,
        "price_in": 1.0, "price_out": 2.0,
    }])], ignore_index=True)
    catalogue.to_csv(output / "catalogue.csv", index=False)
    write_generation(output, "catalogue.manifest.json", CATALOGUE_FILES)
    hashes = refresh_artifact_provenance(output)
    write_valid_decision(output, hashes)

    def unavailable_result(comparisons, simulation_config, **kwargs):
        selected = comparisons(np.random.default_rng(0)) if callable(comparisons) else comparisons
        if isinstance(selected, tuple):
            selected, warnings = selected
        else:
            warnings = pd.DataFrame(columns=["code", "message"])
        selected = selected.reset_index(drop=True).copy()
        selected.insert(0, "comparison_id", np.arange(len(selected)))
        presented = pd.DataFrame(columns=[
            "comparison_id", "m1_theta", "m2_theta", "chose_m1", "p_choose_m1",
        ])
        usage = pd.DataFrame(columns=[
            "role", "model", "reasoning_class", "calls", "input_tokens", "output_tokens",
            "ordinary_output_tokens", "reasoning_tokens",
        ])
        priced = usage.assign(cost=pd.Series(dtype=float))
        missing_prices = set()
        if simulation_config.mode == "automatic":
            usage = pd.DataFrame([{
                "role": "candidate", "model": "a", "reasoning_class": "non_reasoning",
                "calls": 1, "input_tokens": 2, "output_tokens": 3,
                "ordinary_output_tokens": 3, "reasoning_tokens": 0,
            }])
            priced = usage.assign(cost=np.nan)
            missing_prices = {"a"}
            warnings = pd.DataFrame([{
                "code": "missing_price", "model": "a",
                "message": "missing input or output price for a",
            }])
        trials = pd.DataFrame(_trial_rows(
            simulation_config, 0, selected, presented, usage, priced, missing_prices,
        ))
        return SimulationResult(trials, summarize_trials(trials), warnings)

    monkeypatch.setattr(cli, "simulate", unavailable_result)
    target = tmp_path / "dashboard.json"

    assert main(["export-dashboard", "--config", str(config), "--out", str(target)]) == 0

    payload = json.loads(target.read_text())
    validate_dashboard_payload(payload)
    assert {row["code"] for row in payload["warnings"]} >= {
        "missing_price", "automatic_rule_violation", "missing_prediction_metadata",
    }
    assert {
        (row["m1"], row["m2"]) for row in payload["manual"]
    } == {("a", "a"), ("a", "missing")}
    assert all(
        row["warning_state"] == "unavailable"
        for row in payload["manual"]
        if row["metric"] in {"choice_accuracy", "p_choose_m1"}
    )
    assert any(
        row["metric"] == "cost" and row["p50"] is None
        and row["warning_state"] == "missing_price"
        for row in payload["automatic"]
    )


def test_dashboard_records_restore_manual_identity_after_real_summary_nulls():
    identity = {
        "judge": "j", "m1": "a", "m2": "b",
        "judge_reasoning_class": "non_reasoning",
        "m1_reasoning_class": "reasoning",
        "m2_reasoning_class": "non_reasoning",
    }
    # Task 7's tidy summary retains every grouping column. Per-model rows
    # therefore carry null comparison identities, which must not erase the
    # exact manual scenario during dashboard serialization.
    summary = pd.DataFrame([{
        "configuration": "manual", "mode": "manual", "comparison_id": None,
        "judge": None, "m1": None, "m2": None,
        "model": "a", "reasoning_class": "reasoning", "role": "candidate",
        "metric": "cost", "warning_state": "ok", "p5": 1.0, "p50": 2.0,
        "p95": 3.0, "n_trials": 3, "exact": False, "value": None,
    }])
    catalogue = pd.DataFrame([
        {"model_id": "j", "theta": 0.0, "reasoning_class": "non_reasoning",
         "observed_judge": True, "observed_candidate": False},
        {"model_id": "a", "theta": 1.0, "reasoning_class": "reasoning",
         "observed_judge": False, "observed_candidate": True},
        {"model_id": "b", "theta": -1.0, "reasoning_class": "non_reasoning",
         "observed_judge": False, "observed_candidate": False},
    ])
    result = SimulationResult(pd.DataFrame(), summary, pd.DataFrame(columns=["code", "message"]))

    rows, _ = cli._dashboard_records(result, identity, catalogue)

    assert all({key: row[key] for key in cli._MANUAL_COLUMNS} == identity for row in rows)
    assert rows[0]["row_prediction_status"] == "observed"
    assert rows[0]["scenario_prediction_status"] == "predicted"
    assert rows[0]["scenario_extrapolation"] is True


def test_export_dashboard_applies_configured_pool_columns_and_keeps_resolved_metadata(
    tmp_path, monkeypatch,
):
    dashboard = {
        "judge_counts": [1], "anchor_counts": [1], "comparator_counts": [1],
        "attempted_item_counts": [1], "presentation_counts": [1],
        "strength_filters": [True], "manual_triple_limit": 0, "manual_triples": [],
        "judge_pool_column": "custom_judges", "candidate_pool_column": "custom_candidates",
    }
    config, output, _ = pipeline_artifacts(tmp_path, dashboard=dashboard)
    catalogue = pd.read_csv(output / "catalogue.csv")
    catalogue["custom_judges"] = catalogue["model_id"].eq("b")
    catalogue["custom_candidates"] = catalogue["model_id"].eq("a")
    catalogue.to_csv(output / "catalogue.csv", index=False)
    write_generation(output, "catalogue.manifest.json", CATALOGUE_FILES)
    hashes = refresh_artifact_provenance(output)
    write_valid_decision(output, hashes)
    selected = []

    def fake_simulate(comparisons, resolved, **kwargs):
        selected.extend(comparisons(np.random.default_rng(0))["judge"].tolist())
        return fake_result(resolved.configuration)

    monkeypatch.setattr(cli, "simulate", fake_simulate)
    target = tmp_path / "dashboard.json"

    assert main([
        "export-dashboard", "--config", str(config), "--out", str(target),
        "--trials", "3", "--seed", "19",
    ]) == 0

    row = json.loads(target.read_text())["automatic"][0]
    assert selected == ["b"]
    assert row["judge_pool"] == "custom_judges"
    assert row["anchor_pool"] == "custom_candidates"
    assert row["resolved_configuration"] == {
        "accuracy_target": "higher_theta",
        "grid": {"J": 1, "M": 1, "N": 1, "attempted_items": 1},
        "manual": {"identity": None, "limit": 0, "source": None},
        "pools": {
            "candidate_column": "custom_candidates",
            "comparator": "all_released",
            "judge_column": "custom_judges",
        },
        "presentations": 1,
        "response_policy": "reuse_responses",
        "seed": 19,
        "selected_judges": [],
        "strength_filter": True,
        "trials": 3,
    }


def test_export_dashboard_samples_default_manual_bank_with_truncation_warning_deterministically(
    tmp_path, monkeypatch,
):
    dashboard = {
        "judge_counts": [1], "anchor_counts": [1], "comparator_counts": [1],
        "attempted_item_counts": [1], "presentation_counts": [1],
        "manual_triples": [{
            "judge": "j", "m1": "a", "m2": "b", "m1_reasoning_class": "reasoning",
        }],
        "manual_triple_limit": 1,
    }
    config, output, hashes = pipeline_artifacts(tmp_path, dashboard=dashboard)
    write_valid_decision(output, hashes)
    calls = []

    def fake_simulate(comparisons, resolved, **kwargs):
        calls.append(resolved.mode)
        result = fake_result(resolved.configuration)
        return SimulationResult(
            result.trials.assign(mode=resolved.mode), result.summary.assign(mode=resolved.mode), result.warnings,
        )

    monkeypatch.setattr(cli, "simulate", fake_simulate)
    target = tmp_path / "dashboard.json"
    command = [
        "export-dashboard", "--config", str(config), "--out", str(target),
        "--trials", "3", "--seed", "19",
    ]

    assert main(command) == 0
    first = target.read_bytes()
    assert main(command) == 0
    assert target.read_bytes() == first

    payload = json.loads(first)
    identities = {
        tuple(row[key] for key in cli._MANUAL_COLUMNS)
        for row in payload["manual"]
    }
    assert len(identities) == 2
    assert (
        "j", "a", "b", "non_reasoning", "reasoning", "non_reasoning",
    ) in identities
    warning = next(row for row in payload["warnings"] if row["code"] == "manual_bank_truncated")
    assert warning["total_count"] == 19
    assert warning["exported_count"] == 2
    assert warning["omitted_count"] == len(warning["omitted_identities"]) == 17
    assert calls == ["automatic", "manual", "manual"] * 2


def test_default_manual_bank_excludes_self_and_prediction_ineligible_models():
    catalogue = pd.DataFrame([
        {"model_id": "j", "theta": 0.0, "release": "2024-01-01", "family": "j", "reasoning_class": "non_reasoning"},
        {"model_id": "a", "theta": 1.0, "release": "2025-01-01", "family": "a", "reasoning_class": "reasoning"},
        {"model_id": "b", "theta": -1.0, "release": "2023-01-01", "family": "b", "reasoning_class": "non_reasoning"},
        {"model_id": "missing", "theta": None, "release": None, "family": "missing", "reasoning_class": "reasoning"},
    ])

    rows, _ = cli._dashboard_manual_bank(catalogue, [], seed=0, limit=1000)

    assert rows
    assert all(row["m1"] != row["m2"] for row in rows)
    assert all("missing" not in (row["judge"], row["m1"], row["m2"]) for row in rows)


def test_project_registers_mc_cost_entry_point():
    project = tomllib.loads(Path("pyproject.toml").read_text())
    assert project["project"]["scripts"]["mc-cost"] == "mc_cost.cli:main"


def test_built_wheel_is_self_contained_outside_checkout(tmp_path):
    wheel_directory = tmp_path / "dist"
    built = subprocess.run(
        [
            "uv", "build", "--wheel", "--no-build-isolation",
            "--out-dir", str(wheel_directory),
        ],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
    )
    assert built.returncode == 0, built.stderr
    wheel = next(wheel_directory.glob("*.whl"))
    installed_tree = tmp_path / "installed"
    installed = subprocess.run(
        ["uv", "pip", "install", "--target", str(installed_tree), "--no-deps", str(wheel)],
        capture_output=True, text=True,
    )
    assert installed.returncode == 0, installed.stderr
    outside = tmp_path / "outside"
    outside.mkdir()
    clean_env = {**os.environ, "PYTHONPATH": str(installed_tree)}
    imported = subprocess.run(
        [
            sys.executable, "-c",
            "from mc_cost.cli import main; from mc_cost.gpqa import prepare; "
            "from mc_cost.ranking import collect_pair_wins, components, fit_bt, lomo_predictions",
        ],
        cwd=outside, env=clean_env, capture_output=True, text=True,
    )
    assert imported.returncode == 0, imported.stderr
    help_result = subprocess.run(
        [str(installed_tree / "bin/mc-cost"), "--help"],
        cwd=outside, env=clean_env, capture_output=True, text=True,
    )
    assert help_result.returncode == 0, help_result.stderr
    assert all(command in help_result.stdout for command in cli._COMMANDS)
