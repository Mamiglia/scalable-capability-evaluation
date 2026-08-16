import json

import numpy as np
import pandas as pd
import pytest


def _verdict(picked_major=1):
    return {
        "orders": [{
            "slot_of_major": "1",
            "pick": "1",
            "picked_major": picked_major,
        }]
    }


def test_epoch_model_aliases_match_viewer_names():
    from inputs import canonical_model

    aliases = {
        "gpt-5.5_low": "gpt-5.5-low",
        "gemini-3.5-flash_high": "gemini-3.5-flash",
        "gpt-4.1-nano-2025-04-14": "gpt-4.1-nano",
        "Llama-3.1-8B-Instruct": "llama-3.1-8b",
        "gpt-5.4-mini-2026-03-17_high": "gpt-5.4-mini-high",
    }

    assert {raw: canonical_model(raw) for raw in aliases} == aliases


def test_standard_candidates_exclude_nonstandard_effort_variants():
    from inputs import is_standard_candidate

    expected = {
        "gpt-5-mini-medium": True,
        "o3-mini-medium": True,
        "qwen-2.5-72b-instruct": True,
        "gpt-4.5-preview": False,
        "gpt-5.5_low": False,
        "gpt-5.4-mini-high": False,
        "gemini-3.1-flash-lite": False,
        "gemini-3.5-flash_high": False,
        "ling-2.6-flash": False,
    }

    assert {name: is_standard_candidate(name) for name in expected} == expected


def test_load_votes_is_full_only_standard_candidates_and_allows_deepseek_judge(
    tmp_path,
):
    from inputs import load_votes

    dataset = tmp_path / "data-expansion-gpqa"
    pairs = dataset / "pairs"
    pairs.mkdir(parents=True)

    (pairs / "m1__m2__full.json").write_text(json.dumps({
        "i-full": {
            "judge-ok": _verdict(),
            "deepseek-v4-flash-0731": _verdict(),
        }
    }))
    (pairs / "m3__m4__10.json").write_text(json.dumps({
        "i-truncated": {"judge-ok": _verdict()}
    }))
    (pairs / "deepseek-v4-flash-0731__m2__full.json").write_text(json.dumps({
        "i-deepseek": {"judge-ok": _verdict()}
    }))
    (pairs / "gpt-4.5-preview__m2__full.json").write_text(json.dumps({
        "i-preview": {"judge-ok": _verdict()}
    }))
    (pairs / "gpt-5.5_low__m2__full.json").write_text(json.dumps({
        "i-low": {"judge-ok": _verdict()}
    }))
    (pairs / "gemini-3.5-flash_high__m2__full.json").write_text(json.dumps({
        "i-flash": {"judge-ok": _verdict()}
    }))

    got = load_votes(tmp_path, {"data-expansion-gpqa": "gpqa"})

    assert got[["major", "minor", "judge", "item_id"]].to_dict("records") == [
        {
            "major": "m1",
            "minor": "m2",
            "judge": "judge-ok",
            "item_id": "i-full",
        },
        {
            "major": "m1",
            "minor": "m2",
            "judge": "deepseek-v4-flash-0731",
            "item_id": "i-full",
        },
    ]
    assert got["condition"].eq("full").all()


def test_gpqa_gemini_uses_rescored_log_but_aime_uses_standard(tmp_path):
    from inputs import select_local_eval_paths

    roots = {}
    created = {}
    for benchmark, models in {
        "gpqa": ["gemini-3.1-flash-lite", "gemini-3.1-flash-lite-rescored"],
        "aime2025": ["gemini-3.1-flash-lite"],
    }.items():
        roots[benchmark] = tmp_path / benchmark
        for model in models:
            path = roots[benchmark] / model / "0" / "log.eval"
            path.parent.mkdir(parents=True)
            path.touch()
            created[(benchmark, model)] = path

    got = select_local_eval_paths(roots)

    assert got[("gpqa", "gemini-3.1-flash-lite")] == created[
        ("gpqa", "gemini-3.1-flash-lite-rescored")
    ]
    assert got[("aime2025", "gemini-3.1-flash-lite")] == created[
        ("aime2025", "gemini-3.1-flash-lite")
    ]
    assert ("gpqa", "gemini-3.1-flash-lite-rescored") not in got


def test_read_named_evals_uses_declared_identity(tmp_path):
    from inputs import read_named_evals

    path = tmp_path / "timestamp.eval"
    path.touch()

    def reader(_path):
        yield {
            "benchmark": "wrong-label",
            "model": "timestamp",
            "item": "i",
            "question": "q",
            "correct": True,
            "answer": "C",
            "target": "C",
            "epoch": 1,
        }

    got = read_named_evals(
        {("gpqa", "gemini-3.1-flash-lite"): path},
        source="expansion",
        source_priority=2,
        reader=reader,
    )
    assert got.loc[0, ["benchmark", "model", "source"]].tolist() == [
        "gpqa",
        "gemini-3.1-flash-lite",
        "expansion",
    ]


def test_validation_reports_expanding_graphs_without_hard_coding_their_size(
    monkeypatch,
):
    from inputs import validate_inputs

    counts = pd.DataFrame([
        {"scope": f"source:{source}", "n_evaluated_models": evaluated,
         "n_graph_models": graph, "n_components": 1,
         "component_sizes": str(graph)}
        for source, evaluated, graph in (
            ("gpqa-pilot-rd2", 10, 5),
            ("data-expansion-gpqa", 44, 32),
            ("data-expansion-aime2025", 8, 4),
            ("data-expansion-aime2026", 8, 4),
        )
    ] + [{
        "scope": "gpqa", "n_evaluated_models": np.nan,
        "n_graph_models": 34, "n_components": 1,
        "component_sizes": "34",
    }])
    monkeypatch.setattr("inputs.preflight_counts", lambda *_: counts)
    responses = pd.DataFrame([
        {
            "benchmark": "gpqa", "model": "gemini-3.1-flash-lite",
            "source": "expansion", "correct": index < 579,
        }
        for index in range(792)
    ])
    votes = pd.DataFrame(columns=["benchmark", "major", "minor"])
    metadata = pd.DataFrame([{"benchmark": "gpqa", "model": "gpt-4.1"}])

    assert validate_inputs(
        responses,
        votes,
        metadata,
        require_four_opponents=False,
    ).equals(counts)


def test_preflight_separates_response_and_graph_counts():
    from inputs import preflight_counts

    responses = pd.DataFrame({
        "benchmark": ["gpqa"] * 3,
        "model": ["judged-a", "judged-b", "response-only"],
    })
    votes = pd.DataFrame([{
        "source": "s",
        "benchmark": "gpqa",
        "condition": "full",
        "major": "judged-a",
        "minor": "judged-b",
        "judge": "deepseek-v4-flash-0731",
    }])

    gpqa = preflight_counts(responses, votes).query("scope == 'gpqa'").iloc[0]

    assert gpqa["n_response_models"] == 3
    assert gpqa["n_graph_models"] == 2
    assert gpqa["component_sizes"] == "2"


def test_preflight_rejects_nonstandard_pair_members():
    from inputs import preflight_counts

    responses = pd.DataFrame({"benchmark": ["gpqa"], "model": ["m1"]})
    votes = pd.DataFrame([{
        "source": "s",
        "benchmark": "gpqa",
        "condition": "full",
        "major": "gpt-5-high",
        "minor": "m1",
    }])

    with pytest.raises(ValueError, match="nonstandard evaluated model"):
        preflight_counts(responses, votes)


def test_underconnected_models_counts_distinct_opponents():
    from inputs import underconnected_models

    votes = pd.DataFrame([
        {"major": "a", "minor": opponent, "parse_ok": True}
        for opponent in ("b", "c", "d", "e")
    ] + [
        # A repeated judgment does not create another opponent.
        {"major": "a", "minor": "b", "parse_ok": True},
        {"major": "b", "minor": "c", "parse_ok": False},
    ])

    assert underconnected_models(votes, {"a", "b", "c", "d", "e"}) == {
        "b": 1,
        "c": 1,
        "d": 1,
        "e": 1,
    }
