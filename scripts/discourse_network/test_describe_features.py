import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from discourse_network.describe_features import (build_requests, cache_path, choose_fidelity_examples,
                                                  conservative_full_preflight, description_result,
                                                  reconstruct_edge_space, select_top_examples)
from discourse_network.features import EdgeSpace


def test_reconstruct_edge_space_uses_development_edges_only():
    embeddings = np.arange(48, dtype=np.float32).reshape(6, 8) / 10
    nodes = pd.DataFrame({"transcript_key": ["a"] * 3 + ["b"] * 3,
                          "node_id": [0, 1, 2] * 2, "bench": ["gpqa"] * 6,
                          "item": ["dev"] * 3 + ["confirmation"] * 3})
    edges = pd.DataFrame({"transcript_key": ["a", "a", "b", "b"],
                          "source_node_id": [0, 1, 0, 1], "target_node_id": [1, 2, 1, 2]})
    split = pd.DataFrame({"bench": ["gpqa", "gpqa"], "item": ["dev", "confirmation"],
                          "split": ["representation_train", "confirmation"]})
    got, pairs, development = reconstruct_edge_space(
        nodes, edges, embeddings, split, ["gpqa"], components=2, seed=7)
    expected = EdgeSpace.fit(embeddings, [pairs[i] for i in np.flatnonzero(development)], 2, 7)
    np.testing.assert_allclose(got.scaler.mean_, expected.scaler.mean_)
    np.testing.assert_allclose(got.pca.components_, expected.pca.components_)
    assert development.tolist() == [True, True, False, False]


def test_top_examples_are_unique_by_transcript_and_include_context():
    units = pd.DataFrame({"transcript_key": ["a", "a", "b", "c"], "node_id": [0, 1, 0, 0],
                          "text": ["A0", "A1", "B0", "C0"]})
    examples = select_top_examples(units, np.array([9, 10, 8, 7]), 3, kind="node")
    assert [x["transcript_key"] for x in examples] == ["a", "b", "c"]
    assert examples[0]["text"] == "A1" and examples[0]["previous"] == "A0"


def test_fidelity_sample_is_deterministic_and_balanced():
    activations = np.r_[np.ones(150), np.zeros(150)]
    first = choose_fidelity_examples(activations, "node_sae/gpqa/f00", n_each=100)
    second = choose_fidelity_examples(activations, "node_sae/gpqa/f00", n_each=100)
    assert np.array_equal(first, second)
    assert (activations[first] > 0).sum() == 100 and (activations[first] == 0).sum() == 100


def test_requests_use_prompt_hash_cache_and_preflight_ceiling(tmp_path):
    prompts = [("description/node/gpqa/f00", "describe me", "claude-sonnet-5", 100),
               ("fidelity/node/gpqa/f00/000", "yes or no", "claude-haiku-4-5", 4)]
    cached = cache_path(tmp_path, "describe me")
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_text(json.dumps({"text": "cached"}))
    requests, estimate = build_requests(prompts, tmp_path, ceiling=16)
    assert len(requests) == 1 and requests[0]["custom_id"].startswith("fidelity/")
    assert set(requests[0]) == {"custom_id", "params"}
    assert requests[0]["params"]["model"] == "claude-haiku-4-5"
    assert estimate["cached"] == 1 and estimate["projected_usd"] < 16
    with pytest.raises(RuntimeError, match="ceiling"):
        build_requests(prompts * 100000, tmp_path, ceiling=0.0001)


def test_description_result_has_the_accepted_schema():
    result = description_result("verification", ["yes", "yes", "no", "no"], [3, 2, 0, 0])
    assert result["description"] == "verification" and result["fidelity"] > 0.3
    assert result["n_examples"] == 4 and set(result) == {"description", "fidelity", "n_examples"}


def test_accepted_run_preflight_covers_all_228_concepts():
    estimate = conservative_full_preflight(100 + 128)
    assert estimate["description_requests"] == 228
    assert estimate["fidelity_requests"] == 45_600
    assert estimate["projected_usd"] == 13.338 and estimate["ceiling_usd"] == 16
