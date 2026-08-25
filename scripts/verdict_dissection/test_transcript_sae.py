import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
from transcript_sae import (TopKSAE, bootstrap_train_indices, decoder_matches,
                            dense_identity, development_report, fit_sae,
                            load_dense_artifacts, load_development_inputs,
                            load_sae, random_dictionary,
                            reconstruct_retained, reconstruction_metrics,
                            retained_features, run_grouped_sae_suite, run_sae_suite,
                            sha256_file, verify_artifact_files,
                            write_development_report, write_sae_artifacts)


SAE_CONFIG = {"width": 32, "sensitivity_widths": [16, 64], "top_k": 4,
              "learning_rate": 1e-3, "batch_size": 512, "max_epochs": 1, "patience": 1,
              "min_decoder_cosine": 0.8, "min_noncanonical_matches": 3,
              "min_training_support": 50, "bootstrap_replicates": 20,
              "min_explained_variance_gain": 0.1, "min_centered_cosine_gain": 0.1,
              "min_validation_active_fraction": 0.9}


def test_topk_sae_keeps_exactly_two_positive_features():
    sae = TopKSAE(4, 4, 2, torch.zeros(4))
    with torch.no_grad():
        sae.encoder.weight.copy_(torch.eye(4))
        sae.encoder.bias.zero_()
        sae.decoder.weight.copy_(torch.eye(4))
    code = sae.encode(torch.tensor([[1.0, 3.0, 2.0, -1.0]]))
    assert torch.equal(code, torch.tensor([[0.0, 3.0, 2.0, 0.0]]))
    reconstruction, residual = sae.reconstruct_with_residual(
        torch.tensor([[1.0, 3.0, 2.0, -1.0]])
    )
    assert torch.allclose(reconstruction + residual, torch.tensor([[1.0, 3.0, 2.0, -1.0]]))


def test_decoder_matching_is_positive_and_one_to_one():
    canonical = np.eye(3)
    alternatives = [canonical[:, [2, 0, 1]], -canonical]
    matches = decoder_matches(canonical, alternatives)
    assert np.allclose(matches[:, 0], 1.0)
    assert np.all(matches[:, 1] == 0.0)


def test_retention_requires_three_noncanonical_matches_and_fifty_training_rows():
    matches = np.array([[0.9, 0.85, 0.81, 0.1], [0.9, 0.9, 0.9, 0.9]])
    activations = np.zeros((60, 2))
    activations[:50, 0] = 1
    activations[:49, 1] = 1
    assert np.array_equal(retained_features(matches, activations), np.array([0]))


def test_reconstruction_metrics_are_centred_on_training_mean():
    mean = np.array([1.0, 1.0])
    inputs = np.array([[2.0, 1.0], [1.0, 2.0]])
    reconstructions = np.array([[1.5, 1.0], [1.0, 1.5]])
    metrics = reconstruction_metrics(inputs, reconstructions, mean)
    assert metrics["explained_variance"] == 0.75
    assert metrics["median_centered_cosine"] == 1.0

    with pytest.raises(ValueError, match="same 2D shape"):
        reconstruction_metrics(inputs, reconstructions[:, :1], mean)


def test_random_dictionary_is_seeded_and_unit_normalised():
    first = random_dictionary(8, 4, seed=0)
    second = random_dictionary(8, 4, seed=0)
    assert np.array_equal(first, second)
    assert np.allclose(np.linalg.norm(first, axis=0), 1.0)


def test_fit_sae_is_deterministic_and_keeps_best_validation_epoch():
    train = np.tile(np.eye(4, dtype=np.float32), (8, 1))
    validation = np.eye(4, dtype=np.float32)
    first, first_state = fit_sae(train, validation, width=4, k=2, seed=7,
                                 batch_size=8, max_epochs=20, patience=4)
    second, second_state = fit_sae(train, validation, width=4, k=2, seed=7,
                                   batch_size=8, max_epochs=20, patience=4)
    assert first_state["best_epoch"] < 20
    assert first_state["validation_mse"] == second_state["validation_mse"]
    assert torch.equal(first.decoder.weight, second.decoder.weight)


def test_fit_sae_can_use_the_frozen_full_training_mean():
    train = np.eye(4, dtype=np.float32)
    validation = np.eye(4, dtype=np.float32)
    frozen_mean = np.full(4, 0.125, dtype=np.float32)
    model, _ = fit_sae(train, validation, width=4, k=2, seed=0, mean=frozen_mean,
                       batch_size=4, max_epochs=1, patience=1)
    assert np.array_equal(model.mean.numpy(), frozen_mean)


def test_sae_artifacts_are_hashed_and_never_overwritten(tmp_path):
    sae = TopKSAE(4, 4, 2, torch.zeros(4))
    arrays = {"activations": np.ones((2, 4)), "residuals": np.zeros((2, 4))}
    manifest = write_sae_artifacts(tmp_path / "sae", {"gpqa": sae}, arrays, {"seed": 0})
    assert manifest["outputs"]["activations.npy"] == sha256_file(
        tmp_path / "sae" / "activations.npy")
    assert (tmp_path / "sae" / "manifest.sha256").read_text().strip() == sha256_file(
        tmp_path / "sae" / "manifest.json")
    with pytest.raises(FileExistsError):
        write_sae_artifacts(tmp_path / "sae", {"gpqa": sae}, arrays, {})

    loaded = load_sae(tmp_path / "sae" / "sae_weights__gpqa.npz")
    inputs = torch.tensor([[2.0, 1.0, 0.0, 0.0]])
    assert np.allclose(loaded(inputs).detach().numpy(), sae(inputs).detach().numpy())


def test_sparse_activations_use_canonical_retained_feature_ids(tmp_path):
    sae = TopKSAE(4, 4, 2, torch.zeros(4))
    arrays = {
        "activations": np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32),
        "retained_feature_ids": np.array([5, 9]),
    }
    rows = pd.DataFrame({"transcript_key": ["a" * 64, "b" * 64]})
    write_sae_artifacts(tmp_path / "sae", {"gpqa": sae}, arrays, {}, rows=rows)
    sparse = pd.read_parquet(tmp_path / "sae" / "activations_sparse.parquet")
    assert sparse["feature"].tolist() == [5, 9]


def test_dense_inputs_are_hash_checked_before_sae_fitting(tmp_path):
    dense = tmp_path / "dense"
    dense.mkdir()
    rows = pd.DataFrame({"transcript_key": ["a" * 64],
                         "split_role": ["representation_train"]})
    rows.to_parquet(dense / "rows.parquet", index=False)
    np.save(dense / "transcript_only__final.npy", np.ones((1, 4), dtype=np.float32))
    outputs = {name: sha256_file(dense / name) for name in
               ("rows.parquet", "transcript_only__final.npy")}
    manifest_path = dense / "manifest.json"
    manifest_path.write_text(json.dumps({
        "family": "qwen", "runtime": {}, "outputs": outputs,
    }))
    (dense / "manifest.sha256").write_text(sha256_file(manifest_path) + "\n")
    loaded_rows, loaded_inputs, _ = load_dense_artifacts(dense)
    assert loaded_rows.equals(rows)
    assert np.array_equal(loaded_inputs, np.ones((1, 4), dtype=np.float32))

    (dense / "transcript_only__final.npy").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="artifact SHA-256"):
        load_dense_artifacts(dense)


def test_dense_identity_covers_model_config_freeze_and_runtime():
    manifest = {
        "family": "qwen", "model": {"revision": "r"}, "model_files": {"w": "h"},
        "config_sha256": "a", "phase1_freeze_sha256": "b",
        "embedding_code_sha256": "c", "git_commit": "d",
        "extraction": {"dtype": "bfloat16"}, "runtime": {"gpu": "A100"},
    }
    identity = dense_identity(manifest)
    for field in manifest:
        changed = manifest | {field: "different"}
        assert dense_identity(changed) != identity


def test_bootstrap_samples_whole_items_within_benchmark():
    rows = pd.DataFrame({"benchmark": ["a", "a", "a", "b"],
                         "item_group_key": ["1", "1", "2", "3"]})
    first = bootstrap_train_indices(rows, seed=4)
    second = bootstrap_train_indices(rows, seed=4)
    assert np.array_equal(first, second)
    assert list(first).count(0) == list(first).count(1)


def test_reconstruct_retained_keeps_only_stable_codes_and_exact_residual():
    sae = TopKSAE(3, 3, 2, torch.zeros(3))
    with torch.no_grad():
        sae.encoder.weight.copy_(torch.eye(3))
        sae.encoder.bias.zero_()
        sae.decoder.weight.copy_(torch.eye(3))
    inputs = np.array([[3.0, 2.0, 1.0]], dtype=np.float32)
    codes, reconstructions, residuals = reconstruct_retained(sae, inputs, np.array([0]))
    assert np.array_equal(codes, np.array([[3.0]], dtype=np.float32))
    assert np.allclose(reconstructions + residuals, inputs)
    assert np.array_equal(reconstructions, np.array([[3.0, 0.0, 0.0]], dtype=np.float32))


def suite_rows(benchmark, n_train, n_validation, n_confirmation=0):
    roles = (["representation_train"] * n_train + ["representation_validation"] * n_validation
             + ["confirmation"] * n_confirmation)
    return pd.DataFrame({"benchmark": [benchmark] * len(roles),
                         "item_group_key": [f"{benchmark}-{i // 8}" for i in range(len(roles))],
                         "split_role": roles})


def test_run_sae_suite_fits_train_rows_and_encodes_confirmation_rows():
    rng = np.random.default_rng(0)
    inputs = rng.normal(size=(3_200 + 64 + 32, 64)).astype(np.float32)
    rows = suite_rows("gpqa", 3_200, 64, 32)
    _, arrays, report, _ = run_sae_suite(inputs, rows, SAE_CONFIG, device="cpu")
    assert report["widths"] == {"narrow": 16, "primary": 32, "wide": 64}
    assert report["n_train"] == 3_200 and report["n_validation"] == 64
    assert arrays["match_cosines"].shape == (32, 4)
    assert arrays["narrow_encoder_weight"].shape == (16, 64)
    assert arrays["wide_encoder_weight"].shape == (64, 64)
    assert arrays["residuals"].shape == inputs.shape
    assert np.allclose(arrays["reconstructions"] + arrays["residuals"], inputs)
    assert set(("full_validation", "retained_validation", "random_validation")) <= set(report)
    assert 0 <= report["retained_validation_active_fraction"] <= 1
    assert report["status"] in {"passed", "null"}
    assert set(("narrow_validation", "wide_validation", "bootstrap")) <= set(report)


def test_too_few_rows_is_a_dense_only_null_not_an_exception(tmp_path):
    inputs = np.ones((20, 8), dtype=np.float32)
    model, arrays, report, state = run_sae_suite(
        inputs, suite_rows("gpqa", 16, 4), SAE_CONFIG | {"top_k": 2, "batch_size": 8},
        device="cpu")
    assert model is None and arrays == {} and state is None
    assert report == {"status": "null", "reason": "too_few_rows",
                      "n_train": 16, "n_validation": 4}
    manifest = write_sae_artifacts(tmp_path / "null", {}, arrays, {"report": report})
    assert manifest["report"]["status"] == "null"
    assert not any(name.startswith("sae_weights") for name in manifest["outputs"])
    verify_artifact_files(tmp_path / "null", manifest)


def test_grouped_suite_fits_one_sae_per_group_and_stacks_activations(tmp_path):
    rng = np.random.default_rng(0)
    rows = pd.concat([suite_rows("gpqa", 800, 64, 16), suite_rows("aime2024", 400, 32, 8),
                      suite_rows("aime2025", 400, 32, 8), suite_rows("swe_bench_mini", 4, 4)],
                     ignore_index=True).sample(frac=1, random_state=0).reset_index(drop=True)
    inputs = rng.normal(size=(len(rows), 16)).astype(np.float32)
    config = SAE_CONFIG | {"width": 8, "sensitivity_widths": [4, 16], "top_k": 2,
                           "min_training_support": 5, "min_noncanonical_matches": 0,
                           "min_decoder_cosine": 0.0,
                           "fit_groups": {"gpqa": ["gpqa"], "aime": ["aime2024", "aime2025"],
                                          "swe_bench_mini": ["swe_bench_mini"]}}
    saes, arrays, reports = run_grouped_sae_suite(inputs, rows, config, device="cpu")
    assert set(saes) == {"gpqa", "aime"}
    assert reports["swe_bench_mini"]["reason"] == "too_few_rows"
    labels = arrays["feature_labels"]
    assert len(labels) == 16 and arrays["activations"].shape == (len(rows), 16)
    groups = np.array([label.split(":")[0] for label in labels])
    assert list(dict.fromkeys(groups)) == ["gpqa", "aime"]
    gpqa_rows = rows["benchmark"].eq("gpqa").to_numpy()
    aime_rows = rows["benchmark"].isin(["aime2024", "aime2025"]).to_numpy()
    assert not arrays["activations"][np.ix_(gpqa_rows, groups == "aime")].any()
    assert not arrays["activations"][np.ix_(aime_rows, groups == "gpqa")].any()
    assert (arrays["activations"][gpqa_rows] > 0).sum(axis=1).max() <= 2
    assert np.allclose(arrays["reconstructions"] + arrays["residuals"], inputs, atol=1e-6)
    assert not arrays["reconstructions"][rows["benchmark"].eq("swe_bench_mini")].any()
    assert arrays["gpqa__match_cosines"].shape == (8, 4)
    with pytest.raises(ValueError, match="without a fit group"):
        run_grouped_sae_suite(inputs, rows, config | {"fit_groups": {"gpqa": ["gpqa"]}},
                              device="cpu")
    manifest = write_sae_artifacts(tmp_path / "sae", saes, arrays, {"report": reports},
                                   rows=rows.assign(transcript_key=[f"{i:064d}" for i in range(len(rows))]))
    assert {"sae_weights__gpqa.npz", "sae_weights__aime.npz", "feature_labels.npy"} <= set(manifest["outputs"])
    assert np.load(tmp_path / "sae" / "feature_labels.npy").tolist() == labels.tolist()


def test_development_report_builds_atlas_shortcut_audits_and_cross_family_matches(tmp_path):
    rows = pd.DataFrame({
        "transcript_key": [f"k{index}" for index in range(8)],
        "item_group_key": [f"i{index // 2}" for index in range(8)],
        "benchmark": ["gpqa", "aime2024", "aime2025", "aime2026",
                      "intercode", "intercode", "swe_bench_mini", "swe_bench_mini"],
        "transcript": [f"text {index}" for index in range(8)],
        "chunk_count": [1] * 4 + [2] * 4,
    })
    activations = np.column_stack([np.arange(8), np.arange(8)[::-1]]).astype(float)
    dense = {"transcript_only__final": np.column_stack([np.arange(8), np.ones(8)])}
    a1 = pd.DataFrame({"transcript_key": rows.transcript_key,
                       "length": [np.nan, *np.arange(1, 8)],
                       "formatting": np.arange(8) % 2,
                       "a_judge_truncated": [False, False, False, False,
                                             True, True, True, True],
                       "a1_missing": [True, False, False, False, False, False, False, False]})
    scorer = pd.DataFrame({
        "transcript_key": rows.transcript_key, "full_bits_per_char": np.arange(8),
        "coverage_complete": [True] * 4 + [False] * 4,
    })
    a2 = {name: scorer.copy() for name in ("llama", "qwen", "gemma")}
    audit = pd.DataFrame({"transcript_key": rows.transcript_key,
                          "model": ["gpt-5"] * 4 + ["gemma-3-12b-it"] * 4})
    report = development_report(rows, activations, dense, a1, a2,
                                bge_activations=activations.copy(),
                                feature_ids=np.array([5, 9]),
                                bge_feature_ids=np.array([7, 11]),
                                cross_family_mask=rows.chunk_count.eq(1).to_numpy(),
                                audit_map=audit)
    assert set(report) == {"atlas", "support", "prevalence", "coactivation",
                           "shortcut_audit", "cross_family"}
    assert set(report["atlas"].feature) == {5, 9}
    assert "llama" in report["shortcut_audit"]["correlations"]
    assert "a1" in report["shortcut_audit"]["correlations"]
    assert "b" in report["shortcut_audit"]["correlations"]
    assert "a_judge_truncated" in report["shortcut_audit"]["correlations"]["a1"]["all"]["5"]
    assert "chunk_count" in report["shortcut_audit"]["correlations"]["b"]["all"]["5"]
    assert report["shortcut_audit"]["grouped_oof_r2"]["a1_complete"]["n_rows"] == 7
    tags = report["shortcut_audit"]["tags"]
    assert any(tag["criterion"] == "abs_rho>=0.5" for tag in tags)
    assert any(tag["criterion"] == "oof_r2>=0.5" for tag in tags)
    assert all(abs(tag.get("rho", 0.5)) >= 0.5 if "rho" in tag else tag["r2"] >= 0.5
               for tag in tags)
    assert report["cross_family"]["matched_fraction"] == 1.0
    assert report["cross_family"]["single_chunk_both"]["matched_fraction"] == 1.0
    assert "provider" in report["prevalence"]
    assert set(report["prevalence"]["task_family"]) == {"GPQA", "AIME", "InterCode", "SWE"}
    assert set(report["prevalence"]["benchmark"]) >= {"aime2024", "aime2025", "aime2026"}
    assert "provider" in report["support"] and "task_family" in report["support"]
    assert report["coactivation"]["feature_ids"] == [5, 9]
    assert report["coactivation"]["active_together"][0][1] == 6
    assert set(report["cross_family"]["matched_support"]) == {
        "global", "single_chunk_both", "benchmark", "provider"}
    manifest = write_development_report(
        tmp_path / "report", report, {"qwen_sae": "a" * 64})
    assert manifest["kind"] == "development_report"
    assert (tmp_path / "report" / "atlas.parquet").is_file()


def test_development_report_rejects_outcome_or_identity_selectors():
    rows = pd.DataFrame({"transcript_key": ["k"], "item_group_key": ["i"],
                         "benchmark": ["gpqa"], "transcript": ["text"]})
    with pytest.raises(ValueError, match="forbidden learner column"):
        development_report(rows, np.ones((1, 1)), {},
                           pd.DataFrame({"transcript_key": ["k"], "verdict": [1]}), {})

    safe_a1 = pd.DataFrame({"transcript_key": ["k"], "a_judge_truncated": [False],
                            "a1_missing": [False]})
    scorer = pd.DataFrame({"transcript_key": ["k"], "metric": [1.0],
                           "coverage_complete": [True]})
    with pytest.raises(ValueError, match="forbidden learner column"):
        development_report(
            rows, np.ones((1, 1)), {}, safe_a1,
            {name: scorer.assign(judge_identity="x")
             for name in ("llama", "qwen", "gemma")},
            audit_map=pd.DataFrame({"transcript_key": ["k"], "model": ["gpt-5"]}))


def test_cli_shaped_development_inputs_verify_source_and_row_order(tmp_path):
    from transcript_embeddings import write_dense_artifacts

    keys = ["a" * 64, "b" * 64]
    dense_rows = pd.DataFrame({"transcript_key": keys,
                               "item_group_key": ["i1", "i2"],
                               "split_role": ["representation_train",
                                              "representation_validation"]})
    report_rows = dense_rows.assign(
        benchmark="gpqa", question=["q1", "q2"], transcript=["t1", "t2"],
        chunk_count=[1, 1])
    audit_rows = pd.DataFrame({"transcript_key": keys, "model": ["gpt-5", "gpt-5"]})
    dense_manifest = write_dense_artifacts(
        tmp_path / "dense", dense_rows,
        {"transcript_only__final": np.eye(2, dtype=np.float32)},
        {"family": "qwen"}, report_rows=report_rows, audit_rows=audit_rows)
    sae = TopKSAE(2, 2, 1, torch.zeros(2))
    write_sae_artifacts(
        tmp_path / "sae", {"gpqa": sae},
        {"activations": np.eye(2, dtype=np.float32),
         "retained_feature_ids": np.array([4, 9])},
        {"dense_manifest_sha256": sha256_file(tmp_path / "dense" / "manifest.json"),
         "report": {"status": "passed"}}, rows=dense_rows)
    loaded_rows, activations, dense_views, feature_ids, _, _ = load_development_inputs(
        tmp_path / "dense", tmp_path / "sae")
    assert loaded_rows.transcript.tolist() == ["t1", "t2"]
    assert np.array_equal(activations, np.eye(2))
    assert np.array_equal(feature_ids, [4, 9])
    assert dense_manifest["row_order_hash"] == dense_manifest["report_row_order_hash"]
    a1 = pd.DataFrame({"transcript_key": keys, "length": [1.0, 2.0],
                       "a_judge_truncated": [False, False],
                       "a1_missing": [False, False]})
    scorer = pd.DataFrame({"transcript_key": keys, "metric": [1.0, 2.0],
                           "coverage_complete": [True, True]})
    report = development_report(
        loaded_rows, activations, dense_views, a1,
        {name: scorer.copy() for name in ("llama", "qwen", "gemma")},
        feature_ids=feature_ids, audit_map=audit_rows)
    assert set(report["atlas"].feature) == {4, 9}
