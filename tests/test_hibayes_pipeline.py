from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

try:
    import zipfile_zstd  # noqa: F401
except ModuleNotFoundError:
    # The system test environment lacks this optional ZIP codec registration.
    # These unit tests do not open compressed eval logs.
    sys.modules["zipfile_zstd"] = SimpleNamespace()


HIBAYES = Path(__file__).resolve().parents[1] / "scripts" / "hibayes"
sys.path.insert(0, str(HIBAYES))

import evaluation_power
import fit_features
import pipeline
import plotting
import prepare_gpqa
import sae_judge
import transcript_features


def test_prompt_parser_supports_numbered_transcripts():
    prompt = """Input question:
---
Why?
---
Transcript 1:
---
first response
---
still first
---
Multiple choice options as shown to Model 2:
A. one
B. two

Transcript 2:
---
second response
---
Which of the two models is more capable?
"""
    assert prepare_gpqa.transcript_text(prompt) == (
        "first response\n---\nstill first", "second response"
    )
    assert prepare_gpqa.transcript_features.question_of(prompt) == "Why?"


def test_local_feature_contract_and_capability_evidence():
    names = transcript_features.feature_names()
    assert len(names) == 138
    assert sum(map(len, transcript_features.families().values())) == 138

    answer_only = transcript_features.features("**ANSWER: C**")
    assert answer_only["n_answer_mask_tokens"] == 1
    assert np.isnan(answer_only["proc_n_steps"])

    full = transcript_features.features(
        "First, calculate the value. Check the units. Therefore the answer is 2.\nANSWER: B"
    )
    assert full["lex_verification"] > 0
    assert np.isfinite(full["proc_n_steps"])

    capability = transcript_features.capability_betas()
    assert set(capability["feature"]) == set(names)
    assert int(capability["discovery"].sum()) == 35
    assert len(transcript_features.evidence_betas("SAE")) == 32
    assert len(transcript_features.evidence_betas("TF-IDF")) == 240


def test_sae_config_does_not_need_an_existing_hand_built_fit():
    config = sae_judge.marginal_config("d_sae_s0_000_within")
    model = config["model"]["models"][0]["config"]
    assert model["predictors"] == ["d_sae_s0_000_within", "major_is_right"]
    assert model["pool_judge_effects"] is True
    report = config["communicate"]["communicators"][0]["judge_feature_report"]
    assert "make_plots" not in report


def test_sae_embedding_store_detects_mixed_files(tmp_path, monkeypatch):
    store = tmp_path / "embeddings"
    monkeypatch.setattr(sae_judge, "STORE", store)
    monkeypatch.setattr(sae_judge, "EMBEDDINGS", store / "unique_embeddings.npy")
    monkeypatch.setattr(sae_judge, "EMBEDDING_ROWS", store / "rows.parquet")
    monkeypatch.setattr(sae_judge, "EMBEDDING_MANIFEST", store / "manifest.json")
    embeddings = np.arange(12, dtype=np.float16).reshape(3, 4)
    rows = pd.DataFrame({
        "sample_uuid": ["a", "b"], "major_idx": [0, 1], "minor_idx": [1, 2]
    })
    sae_judge.write_embedding_store(embeddings, rows, n_texts=3, n_chunks=3)
    loaded, loaded_rows = sae_judge.load_embedding_store()
    np.testing.assert_array_equal(loaded, embeddings)
    assert loaded_rows.equals(rows)

    path = store / "rows.parquet"
    path.write_bytes(path.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="does not match"):
        sae_judge.load_embedding_store()


def test_local_feature_contract_preserves_view_and_visibility_edges():
    contract = transcript_features.gpqa_features
    question = "A prompt line that is deliberately longer than forty characters."
    masked = contract.answer_masked(
        f"Restating: {question}\r\nFirst option A).\r\nANSWER： `C`", question
    )
    assert question not in masked
    assert "<OPTION>" in masked
    assert "<ANSWERMASK>" in masked
    assert "\r" not in masked

    assert contract.classify_visibility("x" * 19) == "answer_only"
    assert contract.classify_visibility("x" * 20) == "full_response"
    assert contract.classify_visibility("ANSWER: A", "reasoning") == "reasoning_summary"

    # The frozen upstream API ignores this argument and retokenizes the text.
    assert contract.process_features("Check the result.", 1) == contract.process_features(
        "Check the result.", 10_000
    )


def test_judge_answer_prefers_named_scorer():
    sample = {
        "id": "sample",
        "scores": {
            "other": {"answer": "A"},
            "judge_scorer": {"answer": "2", "metadata": {"confidence": 0.8}},
        },
    }
    assert prepare_gpqa.judge_answer(sample) == ("2", 0.8)


def test_resolved_missing_source_does_not_use_fallback(tmp_path, monkeypatch):
    models = tmp_path / "models.csv"
    pd.DataFrame({"model": ["short"], "accuracy": [0.7]}).to_csv(models, index=False)
    monkeypatch.setattr(
        prepare_gpqa.pd,
        "read_parquet",
        lambda *args, **kwargs: pd.DataFrame({"model": ["epoch"], "correct": [1]}),
    )
    monkeypatch.setattr(prepare_gpqa, "resolve_sources", lambda root: {"short": "missing"})
    capabilities = prepare_gpqa.load_capabilities(
        models, tmp_path / "responses.parquet", tmp_path / "logs"
    )
    assert "short" not in capabilities


def test_item_votes_pool_repeated_rows():
    frame = pd.DataFrame({
        "item": ["a", "a", "b", "b"],
        "is_correct": [0, 1, 1, 1],
    })
    np.testing.assert_array_equal(evaluation_power.item_votes(frame), [0.5, 1.0])


def test_power_curve_is_invariant_to_duplicate_rows():
    frame = pd.DataFrame({
        "model_pair": ["p"] * 8,
        "item": [f"i{i}" for i in range(8)],
        "is_correct": [1, 1, 1, 1, 1, 0, 0, 0],
        "gap": [0.12] * 8,
    })
    first = evaluation_power.power_curve(frame, sizes=(3,), B=40)
    second = evaluation_power.power_curve(
        pd.concat([frame, frame], ignore_index=True), sizes=(3,), B=40
    )
    assert first == second


def test_length_control_uses_matched_cells_only():
    rows = [
        ("p", "j", "i", 0, "raw", 1),
        ("p", "j", "i", 0, "truncated_10", 0),
        ("p", "j", "unmatched", 0, "raw", 1),
    ]
    frame = pd.DataFrame(rows, columns=[
        "model_pair", "judge", "item", "major_is_left",
        "transcript_condition", "is_correct",
    ])
    result = evaluation_power.length_control(frame)
    assert result["n_matched_cells"] == 1
    assert {row["condition"]: row["accuracy"] for row in result["matched_cells"]} == {
        "raw": 1.0,
        "truncated_10": 0.0,
    }


def test_judge_similarity_excludes_repeated_controls(tmp_path, monkeypatch):
    base = tmp_path / "features_within"
    for index, values in enumerate(([0.1, 0.2], [-0.3, -0.4])):
        report = base / f"marginal__family__d_f{index}" / "communicate"
        report.mkdir(parents=True)
        pd.DataFrame({
            "judge": ["a", "b", "a", "b"],
            "feature": [f"d_f{index}", f"d_f{index}",
                        "baseline_accuracy", "baseline_accuracy"],
            "mean_log_odds": [*values, 10.0, -10.0],
        }).to_csv(report / "judge_feature_effects.csv", index=False)
    monkeypatch.setattr(evaluation_power, "HIB", tmp_path)
    evaluation_power.out.clear()
    _, result, matrix = evaluation_power.judge_bias_similarity()
    assert result["n_features"] == 2
    assert len(matrix) == 2


def test_complete_case_csv_honors_custom_outcome_and_subset(tmp_path):
    frame = pd.DataFrame({
        "custom_outcome": [1, 0], "judge": ["j", "j"], "item": ["a", "b"],
        "transcript_condition": ["raw", "raw"], "model_pair": ["p", "p"],
        "self_comparison": [0, 0], "same_model_family": [1, 0],
        "major_is_right": [0, 1], "d_x_within": [0.2, np.nan], "gap": [0.1, 0.2],
    })
    source = tmp_path / "source.csv"
    frame.to_csv(source, index=False)
    result = fit_features.complete_case_csv(
        source, tmp_path, ["d_x_within", "major_is_right"], "gap",
        "custom_outcome", "same_model_family=1",
    )
    loaded = pd.read_csv(result)
    assert len(loaded) == 1
    assert "custom_outcome" in loaded
    with pytest.raises(ValueError, match="no complete rows"):
        fit_features.complete_case_csv(
            source, tmp_path / "empty", ["d_x_within", "major_is_right"], "gap",
            "custom_outcome", "same_model_family=0",
        )


def test_factor_components_are_standardized_and_oriented():
    data = pd.DataFrame({
        "d_a_within": [-2.0, -1.0, 1.0, 2.0],
        "d_b_within": [-1.8, -0.9, 0.8, 1.9],
    })
    scores, loadings = fit_features.factorize(
        data, list(data.columns), "family"
    )
    np.testing.assert_allclose(scores.std(ddof=0), 1.0)
    feature_loadings = loadings.drop(index="__variance_explained__")
    for column in feature_loadings:
        values = feature_loadings[column].to_numpy()
        assert values[np.abs(values).argmax()] > 0


def test_collect_rejects_stale_result(tmp_path):
    input_path = tmp_path / "input.csv"
    input_path.write_text("fixture\n")
    expected = fit_features.build_config(input_path, ["d_x_within"], "is_correct")
    (tmp_path / "marginal__family__d_x.yaml").write_text(
        yaml.safe_dump(expected, sort_keys=False)
    )
    run = tmp_path / "marginal__family__d_x"
    (run / "communicate").mkdir(parents=True)
    model_dir = run / "models" / "judge_feature_logistic_model"
    model_dir.mkdir(parents=True)
    (run / "source_config.yaml").write_text(yaml.safe_dump(expected, sort_keys=False))
    source_files = [
        fit_features.ROOT / "scripts/hibayes/model.py",
        fit_features.ROOT / "scripts/hibayes/communicators.py",
        input_path,
    ]
    (run / "source_hashes.sha256").write_text("".join(
        f"{fit_features.sha256(path)}  {path}\n" for path in source_files
    ))
    incomplete_hashes = (
        f"{fit_features.sha256(source_files[0])}  {source_files[0]}\n"
    )
    (run / "source_hashes.sha256").write_text(incomplete_hashes)
    assert not fit_features.result_matches_config(expected, run)
    (run / "source_hashes.sha256").write_text("".join(
        f"{fit_features.sha256(path)}  {path}\n" for path in source_files
    ))
    pd.DataFrame({
        "effect": ["d_x_within"], "mean_log_odds": [0.1],
        "hdi_2.5%": [-0.1], "hdi_97.5%": [0.3],
    }).to_csv(run / "communicate" / "population_effects.csv", index=False)
    model_config = {
        "extra_kwargs": {"predictors": ["d_x_within"]},
        "fit": expected["model"]["models"][0]["config"]["fit"],
    }
    config_path = model_dir / "model_config.json"
    config_path.write_text(json.dumps(model_config))
    assert fit_features.collect(tmp_path)["feature"].tolist() == ["x"]
    snapshot = run / "source_config.yaml"
    wrong_source = yaml.safe_load(snapshot.read_text())
    wrong_source["data_process"]["processors"][0]["extract_observed_feature"][
        "feature_name"
    ] = "chosen_left"
    snapshot.write_text(yaml.safe_dump(wrong_source, sort_keys=False))
    with pytest.raises(SystemExit, match="stale"):
        fit_features.collect(tmp_path)
    snapshot.write_text(yaml.safe_dump(expected, sort_keys=False))
    model_config["extra_kwargs"]["predictors"] = ["wrong"]
    config_path.write_text(json.dumps(model_config))
    with pytest.raises(SystemExit, match="stale"):
        fit_features.collect(tmp_path)


def test_legacy_provenance_uses_recorded_input_outcome_and_order(tmp_path):
    config = fit_features.build_config(Path("sample.csv"), ["d_x_within"], "is_correct")
    logs = tmp_path / "logs"
    logs.mkdir()
    logs.joinpath("logs_process.txt").write_text(
        "Loading extracted data from: ['sample.csv']\n"
        "Extracted 'is_correct' -> 'obs'\n"
        "Category order: ['raw', 'truncated_10', 'truncated_50', 'truncated_90']\n"
    )
    assert fit_features.result_matches_config(config, tmp_path)
    config["data_process"]["processors"][0]["extract_observed_feature"][
        "feature_name"
    ] = "chosen_left"
    assert not fit_features.result_matches_config(config, tmp_path)


def test_strict_json_replaces_nonfinite_numbers(tmp_path):
    path = tmp_path / "result.json"
    plotting.write_json(path, {
        "nan": np.nan, "inf": np.inf, "ok": 1.5,
        "count": np.int64(2), "flag": np.bool_(True),
    })
    assert json.loads(path.read_text()) == {
        "nan": None, "inf": None, "ok": 1.5, "count": 2, "flag": True,
    }
    assert "NaN" not in path.read_text()


def test_pipeline_dispatches_without_reimplementing_stage(monkeypatch):
    calls = []
    fake = SimpleNamespace(main=lambda args: calls.append(args) or 0)
    monkeypatch.setattr(pipeline.importlib, "import_module", lambda name: fake)
    args = ["--logs", "logs", "--out", "all.csv", "--out-correctness", "correct.csv"]
    assert pipeline.main(["prepare", *args]) == 0
    assert calls == [args]
    with pytest.raises(SystemExit):
        pipeline.main(["prepare", "--logs", "logs"])
