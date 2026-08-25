"""main.py checks: feature calculation on a toy store, z-scoring."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import main
from main import KEYS, build_transcript_store, calculate_raw_features, normalise
from judge.dataset import Transcript, responses_to_sample

STORE = pd.DataFrame([
    {"bench": "gpqa", "model": "x", "item": "q1", "epoch": 0, "log": "run_a", "sample_epoch": 1, "content_hash": "hash_a", "question": "Why?",
     "transcript": "Because $n = 2$. Therefore it holds, I think.", "correct": True, "is_canonical": True},
    {"bench": "gpqa", "model": "y", "item": "q1", "epoch": 0, "log": "run_b", "sample_epoch": 1, "content_hash": "hash_b", "question": "Why?",
     "transcript": "It just does. No doubt about it.", "correct": True, "is_canonical": True},
])


def test_raw_features_keep_keys_and_have_no_nans(tmp_path):
    raw = calculate_raw_features(STORE, surprisal_cache=tmp_path)   # empty cache -> basic features only
    assert list(raw.columns[:7]) == KEYS
    assert raw.notna().all().all()
    assert raw.loc[0, "personal_pronouns_count"] > raw.loc[1, "personal_pronouns_count"]


def test_normalise_gives_zero_mean_unit_sd(tmp_path):
    raw = calculate_raw_features(STORE, surprisal_cache=tmp_path)
    dm, scaling = normalise(raw)
    feats = [c for c in dm.columns if c not in KEYS]
    assert np.allclose(dm[feats].mean(), 0, atol=1e-12)
    assert set(scaling.index) == set(feats)
    constant = [c for c in feats if raw[c].nunique() == 1]
    assert all((dm[c] == 0).all() for c in constant)     # sd 0 columns stay 0, not NaN


def test_store_deduplicates_identical_hf_alias_copies(tmp_path, monkeypatch):
    duplicate = STORE.iloc[[0, 0]].reset_index(drop=True)
    duplicate.loc[0, "is_canonical"] = False
    monkeypatch.setitem(main.CFG["data"], "collections", {"gpqa": "collection"})
    monkeypatch.setattr(main, "OUT", tmp_path)
    monkeypatch.setattr(main, "STORE", tmp_path / "store.parquet")
    monkeypatch.setattr(main, "read_rollouts", lambda bench: duplicate)

    built = build_transcript_store()
    assert len(built) == 1 and bool(built.iloc[0].is_canonical)


def test_verdict_draw_key_keeps_distinct_rollout_draws():
    first = {
        "bench": "gpqa", "judge": "judge", "item": "q1",
        "model_left": "x", "model_right": "y", "epoch_left": 0,
        "epoch_right": 0, "log_left": "run_a", "log_right": "run_b",
        "judge_log": "batch_a/judge.eval", "judge_sample": "q1_1", "judge_epoch": 1,
    }
    rerun = first | {"epoch_left": 1, "log_left": "run_c"}
    repeated_judgment = first | {"judge_log": "batch_b/judge.eval"}
    repeated_epoch = first | {"judge_epoch": 2}

    assert main.verdict_draw_key(first) != main.verdict_draw_key(rerun)
    assert main.verdict_draw_key(first) != main.verdict_draw_key(repeated_judgment)
    assert main.verdict_draw_key(first) != main.verdict_draw_key(repeated_epoch)


def test_canonical_protocol_keeps_legacy_max_length_unknown():
    meta = {"include_reasoning": False, "truncate_frac": None,
            "require_both_correct": True, "include_reverse": True}
    protocol = main.canonical_protocol(meta)
    assert protocol["max_length"] is None and not protocol["max_length_recorded"]

    with pytest.raises(ValueError, match="truncate_frac"):
        main.canonical_protocol(meta | {"truncate_frac": 0.5})


def test_canonical_log_sets_replace_identical_and_combine_disjoint():
    selected = main.select_canonical_logs([
        ("old", {"a", "b"}), ("new", {"a", "b"}), ("shard", {"c"}),
    ])
    assert [name for name, _ in selected] == ["new", "shard"]

    with pytest.raises(ValueError, match="partially overlapping"):
        main.select_canonical_logs([("a", {"x", "y"}), ("b", {"y", "z"})])


def test_pair_rows_uses_first_correct_sample_epoch(tmp_path, monkeypatch):
    verdicts = pd.DataFrame([{
        "bench": "gpqa", "judge": "judge", "item": "q1", "model_left": "x",
        "model_right": "y", "epoch_left": 0, "epoch_right": 0,
        "log_left": "run_x", "log_right": "run_y", "pick_left": 1,
    }])
    raw = pd.DataFrame([
        {"bench": "gpqa", "model": "x", "item": "q1", "epoch": 0, "log": "run_x",
         "sample_epoch": 1, "content_hash": "x1", "feature": 3.0},
        {"bench": "gpqa", "model": "x", "item": "q1", "epoch": 0, "log": "run_x",
         "sample_epoch": 2, "content_hash": "x2", "feature": 9.0},
        {"bench": "gpqa", "model": "y", "item": "q1", "epoch": 0, "log": "run_y",
         "sample_epoch": 1, "content_hash": "y1", "feature": 1.0},
    ])
    monkeypatch.setattr(main, "PAIRS", tmp_path / "pairs.parquet")

    pairs = main.pair_rows(verdicts, raw)
    assert len(pairs) == 1 and pairs.iloc[0].d_feature == 2.0


def test_reconcile_verdicts_requires_exact_judge_input():
    left = Transcript("q1", "Question?", "Left answer", ("A", "B"), 1)
    right = Transcript("q1", "Question?", "Right answer", ("B", "A"), 1)
    judged = responses_to_sample(left, right, max_len=262144)
    base = {"bench": "gpqa", "item": "q1", "epoch": 0, "correct": True, "is_canonical": True,
            "transcript_parts_json": "[]"}
    store = pd.DataFrame([
        base | {"model": "x", "log": "run_x", "sample_epoch": 1, "content_hash": "x",
                "question": left.question, "transcript": left.text,
                "options_json": '["A", "B"]', "is_canonical": False},
        base | {"model": "y", "log": "run_y", "sample_epoch": 1, "content_hash": "y",
                "question": right.question, "transcript": right.text,
                "options_json": '["B", "A"]'},
    ])
    verdict = {"bench": "gpqa", "judge": "judge", "judge_log": "judge.eval", "item": "q1",
               "model_left": "x", "model_right": "y", "epoch_left": 0, "epoch_right": 0,
               "log_left": "run_x", "log_right": "run_y", "max_length": 262144,
               "judge_input": str(judged.input),
               "judge_sample_metadata_json": json.dumps(judged.metadata)}

    accepted, report = main.reconcile_verdicts(pd.DataFrame([verdict]), store)
    assert len(accepted) == 1 and report["accepted_rows"] == 1

    legacy_input = responses_to_sample(left, right, max_len=262144,
                                       format_str=main.LEGACY_USER_PROMPT).input
    legacy, report = main.reconcile_verdicts(pd.DataFrame([
        verdict | {"max_length": None, "judge_input": str(legacy_input)}]), store)
    assert len(legacy) == 1 and legacy.iloc[0].max_length_resolution == "non_binding"

    rejected, report = main.reconcile_verdicts(
        pd.DataFrame([verdict | {"judge_input": str(judged.input) + " altered"}]), store)
    assert rejected.empty and report["input_mismatch_rows"] == 1


def test_reconcile_identifies_missing_binding_max_length():
    text = "x" * 70000
    transcript = Transcript("q1", "Question?", text, (), 1)
    judged = responses_to_sample(transcript, transcript, max_len=65536,
                                 format_str=main.LEGACY_USER_PROMPT)
    base = {"bench": "gpqa", "item": "q1", "epoch": 0, "correct": True,
            "is_canonical": False, "sample_epoch": 1, "question": transcript.question,
            "transcript": text, "options_json": "[]", "transcript_parts_json": "[]"}
    store = pd.DataFrame([
        base | {"model": "x", "log": "run_x", "content_hash": "x"},
        base | {"model": "y", "log": "run_y", "content_hash": "y"},
    ])
    verdict = pd.DataFrame([{
        "bench": "gpqa", "judge": "judge", "judge_log": "judge.eval", "item": "q1",
        "model_left": "x", "model_right": "y", "epoch_left": 0, "epoch_right": 0,
        "log_left": "run_x", "log_right": "run_y", "max_length": None,
        "judge_input": str(judged.input), "judge_sample_metadata_json": json.dumps(judged.metadata),
    }])

    accepted, _ = main.reconcile_verdicts(verdict, store)
    assert len(accepted) == 1 and accepted.iloc[0].max_length_resolution == "identified_65536"


def test_scale_fold_uses_training_rows_only():
    train = pd.DataFrame({"d_feature": [1.0, 3.0]})
    test = pd.DataFrame({"d_feature": [100.0]})
    scaled_train, scaled_test, scaling = main.scale_fold(train, test, ["d_feature"])

    assert np.allclose(scaled_train.d_feature, [-2 ** -0.5, 2 ** -0.5])
    assert scaled_test.d_feature.iloc[0] > 60
    assert scaling.loc["d_feature", "mean"] == 2.0


def test_item_split_is_seeded_subset_stratified_64_16_20():
    items = pd.DataFrame([
        {"bench": bench, "item": str(item)}
        for bench in ("gpqa", "aime2024") for item in range(100)
    ])
    first = main.make_item_split(items, seed=0)
    second = main.make_item_split(items.sample(frac=1, random_state=9), seed=0)

    assert first.equals(second)
    assert first.groupby(["bench", "split"]).size().unstack().to_dict("index") == {
        "aime2024": {"confirmation": 20, "representation_train": 64, "representation_validation": 16},
        "gpqa": {"confirmation": 20, "representation_train": 64, "representation_validation": 16},
    }


def test_hash_files_records_sha256_and_size(tmp_path):
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("abc")
    assert main.hash_files({"artifact": artifact}) == {
        "artifact": {"path": str(artifact), "size": 3,
                     "sha256": "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"}
    }


def test_canonical_content_hash_is_order_stable_and_population_scoped():
    original = main.canonical_content_hash(STORE)
    assert original == main.canonical_content_hash(STORE.iloc[::-1])
    noncanonical = STORE.iloc[[0]].copy()
    noncanonical.loc[:, ["is_canonical", "content_hash"]] = [False, "changed"]
    assert original == main.canonical_content_hash(pd.concat([STORE, noncanonical]))
    changed_canonical = STORE.copy()
    changed_canonical.loc[0, "content_hash"] = "changed"
    assert original != main.canonical_content_hash(changed_canonical)
