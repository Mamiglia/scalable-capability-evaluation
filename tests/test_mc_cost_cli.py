import json
import hashlib
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import mc_cost.cli as cli
from mc_cost.cli import main, write_run_config
from mc_cost.data import JUDGMENT_COLUMNS
from mc_cost.model import PosteriorArtifact, load_artifact, save_artifact
from mc_cost.simulation import SimulationResult


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
    config = config_file(tmp_path, paths={
        "output": str(tmp_path / "artifacts"),
        "canonical_models": str(canonical), "model_pool": str(pool),
        "irt_capabilities": str(capabilities), "irt_posterior": str(posterior),
    })

    assert main(["catalogue", "--config", str(config), "--seed", "7"]) == 0

    stored = pd.read_csv(tmp_path / "artifacts/catalogue.csv")
    assert stored["model_id"].tolist() == ["fitted"]
    with np.load(tmp_path / "artifacts/capability_draws.npz", allow_pickle=False) as draws:
        assert draws["models"].tolist() == ["fitted"]
        assert draws["draws"].tolist() == [[0.4], [0.6]]


def test_prepare_calls_raw_boundaries_and_writes_declared_tables(tmp_path, monkeypatch):
    output = tmp_path / "artifacts"
    output.mkdir()
    catalogue = pd.DataFrame([
        {"model_id": "j", "theta": 0.0, "release": "2024-01-01"},
        {"model_id": "a", "theta": 1.0, "release": "2025-01-01"},
        {"model_id": "b", "theta": -1.0, "release": "2023-01-01"},
    ])
    catalogue.to_csv(output / "catalogue.csv", index=False)
    items = tmp_path / "items.csv"
    pd.DataFrame([{
        "irt_model": "fixed_3pl", "fit": "gpqa", "benchmark": "gpqa",
        "item": "i", "difficulty": 0.2,
    }]).to_csv(items, index=False)
    logs = tmp_path / "logs"
    logs.mkdir()
    models = tmp_path / "models.csv"
    models.touch()
    responses = tmp_path / "responses.parquet"
    responses.touch()
    attempts = pd.DataFrame([{
        "chosen_left": 1, "judge": "j", "item": "i", "left_model": "a", "right_model": "b",
        "length_difference": 0.2, "same_model_family": 0, "self_comparison": 0,
        "reasoning_difference": 0, "left_chars": 10, "right_chars": 8,
        "judge_input_tokens": 20, "judge_output_tokens": 4,
        "judge_reasoning_tokens": 0, "judge_total_tokens": 24,
    }])
    calls = []

    def fake_prepare(*args, **kwargs):
        calls.append((args, kwargs))
        return attempts, pd.DataFrame(), pd.DataFrame()

    monkeypatch.setattr(cli, "prepare_gpqa", fake_prepare)
    monkeypatch.setattr(cli, "candidate_token_rows", lambda *_: pd.DataFrame())
    config = config_file(tmp_path, paths={
        "output": str(output), "irt_items": str(items), "judge_logs": str(logs),
        "judge_models": str(models), "epoch_responses": str(responses),
        "candidate_log_roots": [],
    })

    assert main(["prepare", "--config", str(config), "--seed", "9"]) == 0

    assert calls and calls[0][0] == (logs, models, responses)
    assert pd.read_parquet(output / "judgments.parquet").shape[0] == 1
    assert (output / "token_candidate.parquet").is_file()
    assert (output / "token_judge.parquet").is_file()
    assert (output / "scales.json").is_file()
    assert (output / "judgment_standardization.json").is_file()


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
    (output / "scales.json").write_text(json.dumps({
        "mean": {"theta_gap": 0}, "scale": {"theta_gap": 1},
    }))
    (output / "capability_draws.npz").write_bytes(b"draws")
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
    (output / "scales.json").write_text(json.dumps({
        column: {"mean": 0.0, "scale": 1.0}
        for column in (
            "theta_gap", "oversight", "difficulty", "theta_gap_x_oversight",
            "theta_gap_x_difficulty", "same_family_x_theta_gap", "reasoning_difference",
            "release_gap_days", "relative_release_days", "length_difference",
        )
    }))
    np.savez_compressed(
        output / "capability_draws.npz", models=np.array(["j", "a", "b"]),
        draws=np.array([[0.0, 1.0, -1.0], [0.1, 1.1, -0.9]]),
    )
    irt = tmp_path / "irt.npz"
    np.savez_compressed(
        irt, fixed_3pl__gpqa__a=np.ones((2, 1)), fixed_3pl__gpqa__b=np.zeros((2, 1)),
        fixed_3pl__gpqa__floor=np.array([0.25]), fixed_3pl__gpqa__items=np.array(["gpqa:i"]),
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
        save_artifact(
            PosteriorArtifact(
                {"draw": np.array([1.0])},
                {"source_hashes": hashes, "conditional_judges": False}, {},
            ),
            output / f"model/{name}",
        )
    config = config_file(
        tmp_path,
        paths={"output": str(output), "irt_posterior": str(irt)},
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
                "observed": 0, "probability": 0.5,
            }
            for spec in ("baseline", "core", "core_release")
            for population in ("conditional", "pooled")
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
    assert load_artifact(output / "model/core").metadata["conditional_judges"] is True
    assert calls == ["judgments", "candidate", "judge"]
    for name in (
        "judgment_predictions.csv", "judgment_folds.csv", "token_candidate_predictions.csv",
        "token_candidate_summary.csv", "token_judge_predictions.csv", "token_judge_summary.csv",
    ):
        assert (output / f"validation/{name}").is_file()
    replay = pd.read_csv(output / "validation/posterior_predictive_replay.csv")
    assert replay.loc[replay["check"].eq("position_effect"), "valid"].all()


def fake_result(configuration="test"):
    trials = pd.DataFrame([{
        "trial": 0, "configuration": configuration, "mode": "automatic",
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
    validation.mkdir()
    (validation / "decision.json").write_text(json.dumps({
        "selected_judgment_spec": "baseline", "data_hashes": hashes,
        "judge_covariate_gate": {"enabled": False},
        "validation_timestamp": "2026-08-15T00:00:00Z",
    }))
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
    assert len(seen) == 2


def test_simulate_refuses_stale_decision_before_calling_engine(tmp_path, monkeypatch, capsys):
    config, output, hashes = pipeline_artifacts(tmp_path)
    validation = output / "validation"
    validation.mkdir()
    (validation / "decision.json").write_text(json.dumps({
        "selected_judgment_spec": "baseline", "data_hashes": hashes,
        "judge_covariate_gate": {"enabled": False},
    }))
    with (output / "judgments.parquet").open("ab") as handle:
        handle.write(b"stale")
    monkeypatch.setattr(cli, "simulate", lambda *args, **kwargs: pytest.fail("simulation ran"))

    assert main(["simulate", "--config", str(config)]) == 2
    assert "hash mismatch" in capsys.readouterr().err


def test_manual_run_config_keeps_exact_rows(tmp_path, monkeypatch):
    config, output, hashes = pipeline_artifacts(tmp_path)
    validation = output / "validation"
    validation.mkdir()
    (validation / "decision.json").write_text(json.dumps({
        "selected_judgment_spec": "baseline", "data_hashes": hashes,
        "judge_covariate_gate": {"enabled": False},
    }))
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
    validation = output / "validation"
    validation.mkdir()
    (validation / "decision.json").write_text(json.dumps({
        "selected_judgment_spec": "baseline", "data_hashes": hashes,
        "judge_covariate_gate": {"enabled": True},
    }))
    monkeypatch.setattr(cli, "simulate", lambda *args, **kwargs: pytest.fail("simulation ran"))

    assert main(["simulate", "--config", str(config)]) == 2
    assert "judge-covariate gate" in capsys.readouterr().err


def test_export_dashboard_runs_grid_and_manual_bank_then_replaces_json(tmp_path, monkeypatch):
    dashboard = {
        "judge_counts": [1], "anchor_counts": [1], "comparator_counts": [1],
        "attempted_item_counts": [1], "presentation_counts": [1],
        "manual_triples": [{"judge": "j", "m1": "a", "m2": "b"}],
    }
    config, output, hashes = pipeline_artifacts(tmp_path, dashboard=dashboard)
    validation = output / "validation"
    validation.mkdir()
    (validation / "decision.json").write_text(json.dumps({
        "selected_judgment_spec": "baseline", "data_hashes": hashes,
        "judge_covariate_gate": {"enabled": False},
        "validation_timestamp": "2026-08-15T00:00:00Z",
    }))
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
    assert calls == ["automatic", "manual"]
    assert {row["mode"] for row in payload["scenarios"]} == {"automatic", "manual"}
    assert payload["source_hashes"]["decision"]


def test_project_registers_mc_cost_entry_point():
    project = tomllib.loads(Path("pyproject.toml").read_text())
    assert project["project"]["scripts"]["mc-cost"] == "mc_cost.cli:main"
