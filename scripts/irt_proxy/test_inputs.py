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


def test_correction_votes_keep_deepseek_and_effort_candidates(tmp_path):
    from inputs import load_votes

    pairs = tmp_path / "gpqa" / "pairs"
    pairs.mkdir(parents=True)
    (pairs / "deepseek-v4-flash-0731__gpt-5-high__full.json").write_text(
        json.dumps({"i": {"j": _verdict()}})
    )

    got = load_votes(tmp_path, {"gpqa": "gpqa"}, all_candidates=True)

    assert got[["major", "minor"]].iloc[0].tolist() == [
        "deepseek-v4-flash-0731", "gpt-5-high",
    ]


def test_hf_vote_loader_uses_raw_item_samples(tmp_path):
    from types import SimpleNamespace as NS
    from inputs import load_hf_votes

    pair = tmp_path / "raw" / "pilot-j-gpqa-a-vs-b"
    pair.mkdir(parents=True)
    (pair / "logs.json").write_text(json.dumps({"run.eval": {"status": "success"}}))
    eval_path = tmp_path / "evals" / "run.eval"
    eval_path.parent.mkdir()
    eval_path.touch()
    log = NS(
        status="success",
        eval=NS(metadata={
            "major_path": "data/gpqa/hash/a/0/a.eval",
            "minor_path": "data/gpqa/hash/b/0/b.eval",
        }),
        samples=[NS(id="item_1", scores={"judge": NS(answer="1")})],
    )

    got = load_hf_votes(tmp_path / "raw", tmp_path / "evals", reader=lambda _: log)

    assert got.loc[0, ["source", "item_id", "left_model", "right_model", "chosen_left"]].tolist() == [
        "hf-judging-2026-08-17", "item", "a", "b", 1.0,
    ]


def test_hf_vote_loader_filters_mixed_archive_before_reading_evals(tmp_path):
    from types import SimpleNamespace as NS
    from inputs import load_hf_votes

    raw = tmp_path / "raw"
    gpqa = raw / "pilot-j-gpqa-a-vs-b"
    other = raw / "pilot-j-mbpp-a-vs-b"
    gpqa.mkdir(parents=True)
    other.mkdir()
    (gpqa / "logs.json").write_text(json.dumps({
        "gpqa.eval": {"status": "success"},
    }))
    (other / "logs.json").write_text(json.dumps({
        "other.eval": {"status": "success"},
    }))
    (gpqa / "gpqa.eval").touch()
    (other / "other.eval").touch()

    def reader(path):
        assert path.name == "gpqa.eval"
        return NS(
            status="success",
            eval=NS(metadata={
                "major_path": "data/gpqa/hash/a/0/a.eval",
                "minor_path": "data/gpqa/hash/b/0/b.eval",
            }),
            samples=[NS(id="item_1", scores={"judge": NS(answer="1")})],
        )

    got = load_hf_votes(raw, raw, benchmark="gpqa", reader=reader)

    assert got["benchmark"].tolist() == ["gpqa"]
    assert got["item_id"].tolist() == ["item"]


def test_pair_feature_rows_are_left_minus_right():
    from inputs import pair_feature_rows

    got = pair_feature_rows(pd.DataFrame([{
        "left_text": "I perhaps?", "right_text": "plain",
    }]), {
        "personal_pronouns": ["I"], "hedging": ["perhaps"],
        "backtracking": [], "uncertainty": [], "confidence": [], "questions": ["?"],
    })

    assert got.loc[0, ["delta_log_chars", "delta_personal_pronouns", "delta_hedging", "delta_questions"]].gt(0).all()


def test_hf_response_resolution_uses_minor_path_on_second_slot():
    from inputs import resolve_hf_responses

    votes = pd.DataFrame([{
        "benchmark": "gpqa", "item_id": "i", "order": 2,
        "major_path": "data/gpqa/h/a/0/a.eval", "minor_path": "data/gpqa/h/b/1/b.eval",
        "left_model": "b", "right_model": "a",
    }])
    recorded = pd.DataFrame([
        {"benchmark": "gpqa", "model": "a", "item": "i", "epoch": 0,
         "transcript": "major", "source_path": "data/gpqa/h/a/0/a.eval"},
        {"benchmark": "gpqa", "model": "b", "item": "i", "epoch": 1,
         "transcript": "minor", "source_path": "data/gpqa/h/b/1/b.eval"},
    ])

    got = resolve_hf_responses(votes, pd.DataFrame(), pd.DataFrame(), recorded)

    assert got.loc[0, ["left_text", "right_text", "left_epoch", "right_epoch"]].tolist() == [
        "minor", "major", 1, 0,
    ]


def test_hf_response_resolution_falls_back_to_canonical_epoch_transcript():
    from inputs import resolve_hf_responses

    votes = pd.DataFrame([{
        "benchmark": "gpqa", "item_id": "i",
        "major_path": "data/gpqa/h/a/0/a.eval", "minor_path": "data/gpqa/h/b/0/b.eval",
        "left_model": "a", "right_model": "b",
    }])
    epoch = pd.DataFrame([
        {"benchmark": "gpqa", "model": "a", "item": "i", "epoch": 3, "transcript": "a epoch"},
        {"benchmark": "gpqa", "model": "b", "item": "i", "epoch": 4, "transcript": "b epoch"},
    ])

    got = resolve_hf_responses(votes, pd.DataFrame(), epoch)

    assert got.loc[0, ["left_text", "right_text", "left_epoch", "right_epoch"]].tolist() == [
        "a epoch", "b epoch", 3, 4,
    ]


def test_viewer_correction_votes_reject_response_log_length_mismatch(tmp_path):
    from inputs import _viewer_correction_votes

    pairs = tmp_path / "gpqa" / "pairs"
    transcripts = tmp_path / "gpqa" / "transcripts"
    pairs.mkdir(parents=True)
    transcripts.mkdir()
    (pairs / "a__b__full.json").write_text(json.dumps({"i": {"j": _verdict()}}))
    (transcripts / "a.json").write_text(json.dumps({"i": "short"}))
    (transcripts / "b.json").write_text(json.dumps({"i": "right"}))
    response_rows = pd.DataFrame([
        {"benchmark": "gpqa", "model": "a", "item": "i", "epoch": 0, "transcript": "too long"},
        {"benchmark": "gpqa", "model": "b", "item": "i", "epoch": 0, "transcript": "right"},
    ])

    with pytest.raises(ValueError, match="length"):
        _viewer_correction_votes(tmp_path, {"gpqa": "gpqa"}, response_rows)


def test_correction_data_preserves_transcript_epochs(tmp_path, monkeypatch):
    import inputs

    votes = pd.DataFrame([{
        "source": "viewer", "benchmark": "gpqa", "condition": "full", "judge": "j", "item_id": "i",
        "left_model": "a", "right_model": "b", "left_text": "a", "right_text": "b",
        "left_epoch": 1, "right_epoch": 2, "chosen_left": 1.0,
    }])
    scored = pd.DataFrame([
        {"source": "epoch", "source_path": f"{model}.eval", "source_priority": 0,
         "benchmark": "gpqa", "model": model, "item": "i", "epoch": epoch,
         "answer": "A", "correct": True}
        for model, epoch in (("a", 1), ("b", 2))
    ])
    monkeypatch.setattr(
        inputs, "_viewer_correction_votes",
        lambda *_args, **_kwargs: (votes, pd.DataFrame(columns=[
            "source", "judge", "artifact", "reason", "count",
        ])),
    )
    monkeypatch.setattr(inputs, "_canonical_response_rows", lambda *_: pd.DataFrame())
    monkeypatch.setattr(inputs, "load_responses", lambda *_args, **_kwargs: scored)
    monkeypatch.setattr(inputs, "pair_feature_rows", lambda votes, _: votes)
    lexicon = tmp_path / "lexicon.yaml"
    lexicon.write_text("{}")

    _, responses = inputs.load_correction_data(tmp_path, {
        "paths": {"viewer_data": "viewer"},
        "correction": {"viewer_datasets": {}, "style_lexicon": "lexicon.yaml"},
    })

    assert responses.set_index("model")["epoch"].to_dict() == {"a": 1, "b": 2}


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


def test_correction_transcripts_are_source_qualified(tmp_path):
    from inputs import _canonical_response_rows, _viewer_correction_votes

    viewer = tmp_path / "viewer"
    for source, text in (("round-a", "answer from A"), ("round-b", "answer from B")):
        pairs = viewer / source / "pairs"
        transcripts = viewer / source / "transcripts"
        pairs.mkdir(parents=True)
        transcripts.mkdir()
        (pairs / "shared__other__full.json").write_text(
            json.dumps({"i": {"judge": _verdict()}})
        )
        (transcripts / "shared.json").write_text(json.dumps({"i": text}))
        (transcripts / "other.json").write_text(json.dumps({"i": f"other {source}"}))

    config = {
        "paths": {"viewer_data": "viewer"},
        "correction": {"viewer_datasets": {"round-a": "gpqa", "round-b": "gpqa"}},
    }
    responses = _canonical_response_rows(tmp_path, config)
    votes = _viewer_correction_votes(
        viewer, config["correction"]["viewer_datasets"], responses,
    )

    assert responses.set_index(["source", "model", "item"])["transcript"].to_dict() == {
        ("round-a", "other", "i"): "other round-a",
        ("round-a", "shared", "i"): "answer from A",
        ("round-b", "other", "i"): "other round-b",
        ("round-b", "shared", "i"): "answer from B",
    }
    assert votes.set_index("source")["left_text"].to_dict() == {
        "round-a": "answer from A",
        "round-b": "answer from B",
    }


def test_invalid_viewer_verdicts_are_excluded_with_auditable_counts(tmp_path):
    from inputs import _canonical_response_rows, _viewer_correction_votes

    viewer = tmp_path / "viewer"
    pairs = viewer / "round" / "pairs"
    transcripts = viewer / "round" / "transcripts"
    pairs.mkdir(parents=True)
    transcripts.mkdir()
    (pairs / "a__b__full.json").write_text(json.dumps({
        "i": {
            "valid": _verdict(1),
            "invalid": _verdict(None),
            "missing": {"orders": [None]},
        },
    }))
    (transcripts / "a.json").write_text(json.dumps({"i": "a"}))
    (transcripts / "b.json").write_text(json.dumps({"i": "b"}))
    config = {
        "paths": {"viewer_data": "viewer"},
        "correction": {"viewer_datasets": {"round": "gpqa"}},
    }

    votes, exclusions = _viewer_correction_votes(
        viewer,
        config["correction"]["viewer_datasets"],
        _canonical_response_rows(tmp_path, config),
        return_exclusions=True,
    )

    assert votes["judge"].tolist() == ["valid"]
    assert exclusions.to_dict("records") == [
        {
            "source": "round",
            "judge": "invalid",
            "artifact": "round/pairs/a__b__full.json",
            "reason": "invalid viewer verdict",
            "count": 1,
        },
        {
            "source": "round",
            "judge": "missing",
            "artifact": "round/pairs/a__b__full.json",
            "reason": "missing viewer verdict",
            "count": 1,
        },
    ]


def test_correction_loader_keeps_self_pair_with_distinct_epoch_transcripts(tmp_path):
    from inputs import _canonical_response_rows, _viewer_correction_votes, load_votes

    viewer = tmp_path / "viewer"
    pairs = viewer / "round" / "pairs"
    transcripts = viewer / "round" / "transcripts"
    pairs.mkdir(parents=True)
    transcripts.mkdir()
    (pairs / "same__same__full.json").write_text(
        json.dumps({"i": {"judge": _verdict(None)}})
    )
    (transcripts / "same.json").write_text(json.dumps({"i": "first"}))
    (transcripts / "same__ep1.json").write_text(json.dumps({"i": "second"}))
    config = {
        "paths": {"viewer_data": "viewer"},
        "correction": {"viewer_datasets": {"round": "gpqa"}},
    }

    raw = load_votes(viewer, {"round": "gpqa"}, all_candidates=True)
    votes = _viewer_correction_votes(
        viewer,
        {"round": "gpqa"},
        _canonical_response_rows(tmp_path, config),
    )

    assert raw[["major", "minor"]].iloc[0].tolist() == ["same", "same"]
    assert votes.loc[0, ["left_text", "right_text"]].tolist() == ["first", "second"]


def test_hf_loader_deduplicates_only_repeated_eval_artifacts(tmp_path):
    from types import SimpleNamespace as NS
    from inputs import load_hf_votes

    raw = tmp_path / "raw"
    for name in ("manifest-a", "manifest-b"):
        directory = raw / f"{name}-gpqa-a-vs-b"
        directory.mkdir(parents=True)
        (directory / "logs.json").write_text(json.dumps({
            "run.eval": {"status": "success"},
        }))
    eval_path = tmp_path / "evals" / "run.eval"
    eval_path.parent.mkdir()
    eval_path.touch()
    log = NS(
        status="success",
        eval=NS(
            metadata={
                "major_path": "data/gpqa/hash/a/0/a.eval",
                "minor_path": "data/gpqa/hash/b/0/b.eval",
            },
            model="judge",
        ),
        samples=[NS(id="item_1", scores={"judge": NS(answer="1", value="I")})],
    )

    got = load_hf_votes(raw, tmp_path / "evals", reader=lambda _: log)

    assert len(got) == 1
    assert got.loc[0, "chosen_left"] == 1.0
    assert got.loc[0, "round"] == "run"
    assert "value" not in got.columns


def test_canonical_scored_pool_joins_viewer_restored_hf_and_epoch_aliases():
    from inputs import canonical_scored_pool

    columns = {
        "benchmark": "gpqa", "item": "i", "epoch": 1,
        "answer": "A", "target": "A", "question": "q",
    }
    viewer = pd.DataFrame([{**columns, "model": "gpt5", "correct": False,
                            "source": "data-expansion-gpqa", "source_priority": 1,
                            "source_path": "viewer.eval"}])
    restored = pd.DataFrame([{**columns, "model": "gpt-5", "correct": True,
                              "source": "hf-restored", "source_priority": 3,
                              "source_path": "hf.eval"}])
    epoch = pd.DataFrame([{**columns, "model": "other", "correct": True,
                           "source": "epoch", "source_priority": 0,
                           "source_path": "epoch.eval"}])

    got = canonical_scored_pool([viewer, restored, epoch])

    assert set(got["source"]) == {"data-expansion-gpqa", "hf-restored", "epoch"}
    assert got.loc[got["source"].eq("data-expansion-gpqa"), "model"].item() == "gpt-5"


def test_correction_data_returns_scored_pool_and_exclusions(tmp_path, monkeypatch):
    import inputs

    vote = pd.DataFrame([{
        "source": "gpqa-pilot-rd2", "artifact": "pilot/pair.json",
        "round": "pilot:pair", "benchmark": "gpqa", "condition": "full",
        "judge": "j", "item_id": "i", "order": 1,
        "left_model": "gpt-5", "right_model": "other",
        "left_text": "left", "right_text": "right",
        "left_epoch": 0, "right_epoch": 0, "chosen_left": 1.0,
    }])
    exclusions = pd.DataFrame([{
        "source": "gpqa-pilot-rd2", "judge": "j",
        "artifact": "pilot/bad.json", "reason": "invalid viewer verdict",
        "count": 1,
    }])
    scored = pd.DataFrame([{
        "benchmark": "gpqa", "model": "gpt5", "item": "i", "epoch": 1,
        "answer": "A", "target": "A", "question": "q", "correct": True,
        "source": "epoch", "source_priority": 0, "source_path": "epoch.eval",
    }])
    monkeypatch.setattr(inputs, "_canonical_response_rows", lambda *_: pd.DataFrame())
    monkeypatch.setattr(
        inputs, "_viewer_correction_votes",
        lambda *_args, **_kwargs: (vote, exclusions),
    )
    monkeypatch.setattr(inputs, "load_responses", lambda *_args, **_kwargs: scored)
    monkeypatch.setattr(inputs, "pair_feature_rows", lambda rows, _lexicon: rows)
    lexicon = tmp_path / "lexicon.yaml"
    lexicon.write_text("{}")
    config = {
        "paths": {"viewer_data": "viewer"},
        "correction": {
            "viewer_datasets": {"gpqa-pilot-rd2": "gpqa"},
            "style_lexicon": "lexicon.yaml",
        },
    }

    votes, responses, excluded = inputs.load_correction_data(
        tmp_path, config, return_exclusions=True,
    )

    assert votes["left_model"].tolist() == ["gpt-5"]
    assert responses["model"].tolist() == ["gpt-5"]
    pd.testing.assert_frame_equal(excluded, exclusions)


def test_hf_loader_excludes_missing_phi_judge_before_eval_validation(tmp_path):
    from inputs import load_hf_votes

    manifest = tmp_path / "raw" / "pilot-Phi3-medium-gpqa-a-vs-b"
    manifest.mkdir(parents=True)
    (manifest / "logs.json").write_text(json.dumps({
        "missing.eval": {"status": "success"},
    }))

    votes, exclusions = load_hf_votes(
        tmp_path / "raw", tmp_path / "evals", return_exclusions=True,
    )

    assert votes.empty
    assert exclusions.to_dict("records") == [{
        "source": "hf-judging-2026-08-17",
        "judge": "Phi3-medium",
        "artifact": "hf-judging-2026-08-17/missing.eval",
        "reason": "excluded_phi_judge",
        "count": 1,
    }]


def test_hf_loader_still_fails_for_missing_non_phi_success(tmp_path):
    from inputs import load_hf_votes

    manifest = tmp_path / "raw" / "pilot-ling-gpqa-a-vs-b"
    manifest.mkdir(parents=True)
    (manifest / "logs.json").write_text(json.dumps({
        "missing.eval": {"status": "success"},
    }))

    with pytest.raises(ValueError, match="missing raw eval.*missing.eval"):
        load_hf_votes(tmp_path / "raw", tmp_path / "evals")


def test_viewer_missing_transcript_is_audited_without_dropping_valid_item(tmp_path):
    from inputs import _canonical_response_rows, _viewer_correction_votes

    viewer = tmp_path / "viewer"
    source = viewer / "data-expansion-gpqa"
    pairs = source / "pairs"
    transcripts = source / "transcripts"
    pairs.mkdir(parents=True)
    transcripts.mkdir()
    (pairs / "a__b__full.json").write_text(json.dumps({
        "valid": {"judge": _verdict()},
        "missing": {"judge": _verdict()},
    }))
    (transcripts / "a.json").write_text(json.dumps({
        "valid": "a valid", "missing": "a missing",
    }))
    (transcripts / "b.json").write_text(json.dumps({"valid": "b valid"}))
    config = {
        "paths": {"viewer_data": "viewer"},
        "correction": {"viewer_datasets": {"data-expansion-gpqa": "gpqa"}},
    }

    votes, exclusions = _viewer_correction_votes(
        viewer, config["correction"]["viewer_datasets"],
        _canonical_response_rows(tmp_path, config), return_exclusions=True,
    )

    assert votes["item_id"].tolist() == ["valid"]
    assert exclusions.to_dict("records") == [{
        "source": "data-expansion-gpqa",
        "judge": "judge",
        "artifact": "data-expansion-gpqa/pairs/a__b__full.json",
        "reason": "missing_viewer_transcript",
        "count": 1,
    }]


def test_correction_data_uses_hf_viewer_and_benchmark_score_reductions(
    tmp_path, monkeypatch,
):
    import inputs

    viewer = tmp_path / "viewer" / "hf-judging-2026-08-17-gpqa"
    (viewer / "pairs").mkdir(parents=True)
    (viewer / "transcripts").mkdir()
    (viewer / "pairs" / "gpt-5__gpt-4.1__full.json").write_text(
        json.dumps({"item__ep0-0": {"judge": _verdict()}})
    )
    (viewer / "transcripts" / "gpt-5.json").write_text(
        json.dumps({"item__ep0-0": "judged gpt-5"})
    )
    (viewer / "transcripts" / "gpt-4.1.json").write_text(
        json.dumps({"item__ep0-0": "judged gpt-4.1"})
    )

    score_root = tmp_path / "scores"

    def write_scores(raw_model, *, value, answer, explanation):
        path = score_root / "hash" / raw_model / "0" / "logs.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({
            "invalid.eval": {
                "status": "success", "invalidated": True,
                "reductions": [{"scorer": "choice", "samples": [{
                    "sample_id": "item", "value": 1 - value,
                    "answer": "Z", "explanation": "invalidated",
                }]}],
            },
            "valid.eval": {
                "status": "success", "invalidated": False,
                "reductions": [{"scorer": "choice", "samples": [{
                    "sample_id": "item", "value": value,
                    "answer": answer, "explanation": explanation,
                }]}],
            },
        }))

    write_scores("gpt5", value=1.0, answer="A", explanation="scored gpt-5")
    write_scores("gpt4.1", value=0.0, answer="B", explanation="scored gpt-4.1")
    write_scores("unused", value=1.0, answer="C", explanation="not in viewer")

    monkeypatch.setattr(inputs, "load_responses", lambda *_args, **_kwargs: pd.DataFrame())
    monkeypatch.setattr(inputs, "pair_feature_rows", lambda rows, _lexicon: rows)
    monkeypatch.setattr(
        inputs, "load_hf_votes",
        lambda *_args, **_kwargs: pytest.fail("raw HF loader is audit-only"),
    )
    lexicon = tmp_path / "lexicon.yaml"
    lexicon.write_text("{}")
    config = {
        "paths": {"viewer_data": "viewer"},
        "correction": {
            "benchmark": "gpqa",
            "viewer_datasets": {"hf-judging-2026-08-17-gpqa": "gpqa"},
            "hf_benchmark_scores": "scores",
            "style_lexicon": "lexicon.yaml",
        },
    }

    votes, responses, exclusions = inputs.load_correction_data(
        tmp_path, config, return_exclusions=True,
    )

    assert votes["source"].tolist() == ["hf-judging-2026-08-17"]
    assert votes["item_id"].tolist() == ["item"]
    assert set(votes["left_model"]) | set(votes["right_model"]) == {
        "gpt-5", "gpt-4.1",
    }
    assert responses.set_index("model")["correct"].to_dict() == {
        "gpt-4.1": False, "gpt-5": True,
    }
    assert responses.set_index("model")["transcript"].to_dict() == {
        "gpt-4.1": "scored gpt-4.1", "gpt-5": "scored gpt-5",
    }
    assert exclusions.empty


def test_hf_benchmark_scores_keep_numeric_incorrect_blank_answer(tmp_path):
    from inputs import load_hf_benchmark_scores

    path = tmp_path / "hash" / "gpt5" / "0" / "logs.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({
        "run.eval": {
            "status": "success", "invalidated": False,
            "reductions": [{"scorer": "choice", "samples": [{
                "sample_id": "item", "value": 0,
                "answer": "", "explanation": "",
            }]}],
        },
    }))

    got = load_hf_benchmark_scores(tmp_path).iloc[0]

    assert got["model"] == "gpt-5"
    assert got["answer"] == "<no_answer>"
    assert got["transcript"] == ""
    assert not bool(got["correct"])
