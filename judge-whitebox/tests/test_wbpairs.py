import pathlib
import pandas as pd
import pytest
import wbpairs


def make_synthetic(tmp_path):
    """Create a synthetic pairs.parquet with 4 rows (2 pair_uids × 2 orders)."""
    data = [
        {
            "pair_uid": "p1",
            "order": 0,
            "item": "i1",
            "model_1": "gpt-4",
            "model_2": "claude-3",
            "cap_1": 0.8,
            "cap_2": 0.7,
            "label": 1,
            "question": "Q1?",
            "options_1": "a",
            "options_2": "b",
            "text_1": "answer1",
            "text_2": "answer2",
            "split": "eval_mm",
        },
        {
            "pair_uid": "p1",
            "order": 1,
            "item": "i1",
            "model_1": "claude-3",
            "model_2": "gpt-4",
            "cap_1": 0.7,
            "cap_2": 0.8,
            "label": 2,
            "question": "Q1?",
            "options_1": "b",
            "options_2": "a",
            "text_1": "answer2",
            "text_2": "answer1",
            "split": "eval_mm",
        },
        {
            "pair_uid": "p2",
            "order": 0,
            "item": "i2",
            "model_1": "gpt-4",
            "model_2": "claude-3",
            "cap_1": 0.75,
            "cap_2": 0.65,
            "label": 1,
            "question": "Q2?",
            "options_1": "c",
            "options_2": "d",
            "text_1": "answer3",
            "text_2": "answer4",
            "split": "train",
        },
        {
            "pair_uid": "p2",
            "order": 1,
            "item": "i2",
            "model_1": "claude-3",
            "model_2": "gpt-4",
            "cap_1": 0.65,
            "cap_2": 0.75,
            "label": 2,
            "question": "Q2?",
            "options_1": "d",
            "options_2": "c",
            "text_1": "answer4",
            "text_2": "answer3",
            "split": "train",
        },
    ]
    df = pd.DataFrame(data)
    path = tmp_path / "pairs.parquet"
    df.to_parquet(path)
    return path


def test_load_pairs_filters_and_asserts(tmp_path):
    make_synthetic(tmp_path)
    c = {"paths": {"pairs": str(tmp_path / "pairs.parquet")}}
    out = wbpairs.load_pairs(c, split="eval_mm", limit=1)
    assert set(out.pair_uid) == {"p1"} and set(out.order) == {0, 1}
    assert len(out) == 2


def test_load_pairs_respects_limit_by_pair_uid(tmp_path):
    make_synthetic(tmp_path)
    c = {"paths": {"pairs": str(tmp_path / "pairs.parquet")}}
    # limit=2 means 2 pair_uids, so should get both p1 and p2
    out = wbpairs.load_pairs(c, limit=2)
    assert len(out) == 4
    assert set(out.pair_uid) == {"p1", "p2"}


def test_load_pairs_no_think_in_texts(tmp_path):
    data = [
        {
            "pair_uid": "p1",
            "order": 0,
            "item": "i1",
            "model_1": "gpt-4",
            "model_2": "claude-3",
            "cap_1": 0.8,
            "cap_2": 0.7,
            "label": 1,
            "question": "Q1?",
            "options_1": "a",
            "options_2": "b",
            "text_1": "answer with <think>tag</think>",
            "text_2": "clean",
            "split": "eval_mm",
        },
        {
            "pair_uid": "p1",
            "order": 1,
            "item": "i1",
            "model_1": "claude-3",
            "model_2": "gpt-4",
            "cap_1": 0.7,
            "cap_2": 0.8,
            "label": 2,
            "question": "Q1?",
            "options_1": "b",
            "options_2": "a",
            "text_1": "clean",
            "text_2": "answer with <think>tag</think>",
            "split": "eval_mm",
        },
    ]
    df = pd.DataFrame(data)
    path = tmp_path / "pairs.parquet"
    df.to_parquet(path)

    c = {"paths": {"pairs": str(path)}}
    with pytest.raises(ValueError, match="<think>"):
        wbpairs.load_pairs(c)


def test_load_pairs_asserts_both_orders(tmp_path):
    # only order=0, missing order=1
    data = [
        {
            "pair_uid": "p1",
            "order": 0,
            "item": "i1",
            "model_1": "gpt-4",
            "model_2": "claude-3",
            "cap_1": 0.8,
            "cap_2": 0.7,
            "label": 1,
            "question": "Q1?",
            "options_1": "a",
            "options_2": "b",
            "text_1": "answer1",
            "text_2": "answer2",
            "split": "eval_mm",
        },
    ]
    df = pd.DataFrame(data)
    path = tmp_path / "pairs.parquet"
    df.to_parquet(path)

    c = {"paths": {"pairs": str(path)}}
    with pytest.raises(AssertionError):
        wbpairs.load_pairs(c)
