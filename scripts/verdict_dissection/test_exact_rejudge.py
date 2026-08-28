"""Regression checks for deterministic exact-input manifest selection."""

import sys
from pathlib import Path

import pandas as pd


sys.path.insert(0, str(Path(__file__).parent))


def test_manifest_pools_aime_excludes_answer_only_and_keeps_one_order():
    """Dropping reasoning filtering or order sharing makes this selected manifest wrong."""
    from exact_rejudge import build_manifest, sha256_text, validate_manifest

    answer_only_hash = "answer-only"
    store = pd.DataFrame([
        {"content_hash": "a", "transcript": "reasoning " * 20},
        {"content_hash": "b", "transcript": "reasoning " * 20},
        {"content_hash": "c", "transcript": "reasoning " * 20},
        {"content_hash": "d", "transcript": "reasoning " * 20},
        {"content_hash": "e", "transcript": "reasoning " * 20},
        {"content_hash": "f", "transcript": "reasoning " * 20},
        {"content_hash": answer_only_hash, "transcript": "ANSWER: A"},
    ])

    rows = []
    for bench, pair, left, right in [
        ("aime2024", "1", "a", "b"), ("aime2025", "2", "c", "d"),
        ("gpqa", "3", "a", "c"), ("gpqa", "4", "b", "d"),
    ]:
        for judge in ("judge-a", "judge-b"):
            for shown_left, shown_right in ((left, right), (right, left)):
                rows.append({
                    "bench": bench, "judge": judge, "item": pair,
                    "model_left": "left", "model_right": "right",
                    "epoch_left": 0, "epoch_right": 0,
                    "log_left": "left-log", "log_right": "right-log",
                    "judge_log": f"{judge}/{bench}/{pair}.eval",
                    "judge_sample": f"{pair}_{shown_left}", "judge_epoch": 0,
                    "content_hash_l": shown_left, "content_hash_r": shown_right,
                    "judge_config_hash": "dominant", "judge_input": f"input:{shown_left}:{shown_right}",
                    "pick_left": int(shown_left == left),
                })
            # A repeated historical draw must never become a second manifest row.
            rows.append(rows[-1] | {"judge_epoch": 1})
            # A non-dominant configuration is not eligible.
            rows.append(rows[-1] | {"judge_config_hash": "minority", "judge_epoch": 2})

    rows.append({
        "bench": "gpqa", "judge": "judge-a", "item": "bad", "model_left": "left",
        "model_right": "right", "epoch_left": 0, "epoch_right": 0,
        "log_left": "left-log", "log_right": "right-log", "judge_log": "bad.eval",
        "judge_sample": "bad", "judge_epoch": 0, "content_hash_l": answer_only_hash,
        "content_hash_r": "f", "judge_config_hash": "dominant", "judge_input": "bad input",
        "pick_left": 1,
    })
    pairs = pd.DataFrame(rows)

    def fake_prompt_loader(row):
        return {"system_prompt": f"system:{row['judge']}", "usage": {"input_tokens": 1},
                "model_call": {"model": row["judge"]}}

    manifest = build_manifest(pairs, store, fake_prompt_loader, seed=20260828, per_cell=2)
    assert set(manifest.bench) == {"aime", "gpqa"}
    assert manifest.groupby(["judge", "bench"]).size().eq(2).all()
    assert not manifest.duplicated("source_draw_id").any()
    assert not manifest.duplicated(["judge", "bench", "underlying_pair_hash"]).any()
    assert manifest.groupby(["bench", "underlying_pair_hash"]).input_hash.nunique().max() == 1
    assert manifest.input_hash.eq(manifest.judge_input.map(sha256_text)).all()
    assert manifest.system_prompt_hash.eq(manifest.system_prompt.map(sha256_text)).all()
    assert not manifest.content_hash_l.isin({answer_only_hash}).any()
    validate_manifest(manifest, store, per_cell=2)


def test_fresh_kwargs_remove_generation_seed_and_disable_response_cache():
    """A model seed or replay cache would turn a fresh draw into the wrong experiment."""
    from exact_rejudge import fresh_eval_kwargs

    got = fresh_eval_kwargs({"model": "openrouter/x/y", "seed": 7,
                             "extra_headers": {"Existing": "yes"}})
    assert "seed" not in got
    assert got["extra_headers"] == {
        "Existing": "yes",
        "X-OpenRouter-Cache": "false",
        "X-OpenRouter-Metadata": "enabled",
    }


def test_summaries_count_missing_as_disagreement_and_transitions():
    """Dropping missing answers or counting them answered inflates reliability."""
    from exact_rejudge import _bayes_upper, _wilson, summaries

    rows = pd.DataFrame([
        {"judge": "j", "bench": "a", "target": "1", "repeated_verdict": "1"},
        {"judge": "j", "bench": "a", "target": "1", "repeated_verdict": "2"},
        {"judge": "j", "bench": "a", "target": "2", "repeated_verdict": None},
        {"judge": "j", "bench": "b", "target": "2", "repeated_verdict": "1"},
        {"judge": "j", "bench": "b", "target": "2", "repeated_verdict": "2"},
        {"judge": "j", "bench": "b", "target": "1", "repeated_verdict": None},
    ])
    summary, by_judge = summaries(rows, seed=20260828, bootstrap_replicates=200)
    a = summary.set_index("benchmark").loc["a"]
    b = summary.set_index("benchmark").loc["b"]
    assert a.n == b.n == 3
    assert a.answer_rate == b.answer_rate == 2 / 3
    assert a.raw_agreement == b.raw_agreement == 1 / 3
    assert a.answered_only_agreement == b.answered_only_agreement == 1 / 2
    assert a.transition_1_to_2 == 1 and a.transition_2_to_1 == 0
    assert b.transition_1_to_2 == 0 and b.transition_2_to_1 == 1
    assert by_judge.loc[0, "macro_agreement"] == 1 / 3
    assert by_judge.loc[0, "bayes_accuracy_lower"] == 1 / 3
    assert pd.isna(by_judge.loc[0, "bayes_accuracy_upper"])
    _, repeated = summaries(rows, seed=20260828, bootstrap_replicates=200)
    pd.testing.assert_frame_equal(by_judge, repeated)
    low, high = _wilson(1, 3)
    assert 0 < low < 1 / 3 < high < 1
    assert _bayes_upper(0.9) == 0.5 + (0.2 ** 0.5)
