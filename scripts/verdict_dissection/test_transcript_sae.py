import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from transcript_sae import (centre_by_item, dense_identity, explained_variance, load_dense_artifacts, load_sae,
                            run_grouped_sae_suite, sha256_file, verify_artifact_files,
                            write_sae_artifacts)


SAE_CONFIG = {"width": 8, "top_k": 2, "matryoshka_prefixes": [4, 8],
              "dead_neuron_threshold_steps": 64, "batch_size": 32, "learning_rate": 5e-4,
              "max_epochs": 3, "min_epochs": 1, "patience": 3, "seed": 0}


def suite_rows(benchmark, n_train, n_validation, n_confirmation=0):
    roles = (["representation_train"] * n_train + ["representation_validation"] * n_validation
             + ["confirmation"] * n_confirmation)
    return pd.DataFrame({"benchmark": [benchmark] * len(roles),
                         "item_group_key": [f"{benchmark}-{i // 8}" for i in range(len(roles))],
                         "split_role": roles})


def test_explained_variance_is_one_for_perfect_reconstruction():
    inputs = np.array([[1.0, 2.0], [3.0, 4.0], [0.0, 1.0]])
    assert explained_variance(inputs, inputs) == 1.0
    assert explained_variance(inputs, np.broadcast_to(inputs.mean(0), inputs.shape)) == 0.0


def test_grouped_suite_fits_one_sae_per_group_and_stacks_signed_activations(tmp_path):
    rng = np.random.default_rng(0)
    rows = pd.concat([suite_rows("gpqa", 200, 32, 16), suite_rows("aime2024", 100, 16, 8),
                      suite_rows("aime2025", 100, 16, 8), suite_rows("swe_bench_mini", 4, 4)],
                     ignore_index=True).sample(frac=1, random_state=0).reset_index(drop=True)
    inputs = rng.normal(size=(len(rows), 16)).astype(np.float32)
    config = SAE_CONFIG | {"fit_groups": {"gpqa": ["gpqa"], "aime": ["aime2024", "aime2025"],
                                          "swe_bench_mini": ["swe_bench_mini"]}}
    saes, arrays, reports = run_grouped_sae_suite(inputs, rows, config)
    assert set(saes) == {"gpqa", "aime"}
    assert reports["swe_bench_mini"] == {"n_rows": 8, "n_train": 4, "n_validation": 4,
                                         "status": "null", "reason": "too_few_rows"}
    labels = arrays["feature_labels"]
    assert labels.tolist() == [f"gpqa:{i}" for i in range(8)] + [f"aime:{i}" for i in range(8)]
    assert arrays["activations"].shape == (len(rows), 16)
    groups = np.array([label.split(":")[0] for label in labels])
    gpqa_rows = rows["benchmark"].eq("gpqa").to_numpy()
    aime_rows = rows["benchmark"].isin(["aime2024", "aime2025"]).to_numpy()
    assert not arrays["activations"][np.ix_(gpqa_rows, groups == "aime")].any()
    assert not arrays["activations"][np.ix_(aime_rows, groups == "gpqa")].any()
    assert arrays["activations"][gpqa_rows].any()          # confirmation rows are encoded too
    assert np.allclose(arrays["reconstructions"] + arrays["residuals"], inputs, atol=1e-6)
    assert not arrays["reconstructions"][rows["benchmark"].eq("swe_bench_mini")].any()
    gpqa = reports["gpqa"]
    assert gpqa["status"] == "fitted" and len(gpqa["prevalence"]) == 8 and gpqa["epochs_run"] <= 3
    gpqa_activations = arrays["activations"][np.ix_(gpqa_rows, groups == "gpqa")]
    assert all((gpqa_activations > 0).sum(0) >= (gpqa_activations < 0).sum(0)) and len(gpqa["sign_flipped"]) == 8
    assert -1 <= gpqa["validation_explained_variance"] <= 1
    with pytest.raises(ValueError, match="without a fit group"):
        run_grouped_sae_suite(inputs, rows, config | {"fit_groups": {"gpqa": ["gpqa"]}})

    manifest = write_sae_artifacts(tmp_path / "sae", saes, arrays, {"report": reports},
                                   rows=rows.assign(transcript_key=[f"{i:064d}" for i in range(len(rows))]))
    assert {"sae__gpqa.pt", "sae__aime.pt", "feature_labels.npy", "rows.parquet"} <= set(manifest["outputs"])
    verify_artifact_files(tmp_path / "sae", manifest)
    assert np.load(tmp_path / "sae" / "feature_labels.npy").tolist() == labels.tolist()
    with pytest.raises(FileExistsError):
        write_sae_artifacts(tmp_path / "sae", saes, arrays, {})
    loaded = load_sae(tmp_path / "sae" / "sae__gpqa.pt")
    assert loaded.m_total_neurons == 8 and loaded.k_active_neurons == 2


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


def test_centre_by_item_removes_the_item_mean():
    inputs = np.array([[1.0, 2.0], [3.0, 4.0], [10.0, 10.0]], dtype=np.float32)
    centred = centre_by_item(inputs, np.array(["a", "a", "b"]))
    assert np.allclose(centred, [[-1.0, -1.0], [1.0, 1.0], [0.0, 0.0]])
    assert inputs[0, 0] == 1.0     # input untouched
