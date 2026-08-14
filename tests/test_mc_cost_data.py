from types import SimpleNamespace

import pandas as pd
import pytest

from mc_cost.data import (
    JUDGMENT_COLUMNS,
    candidate_token_rows,
    judge_token_rows,
    prepare_judgments,
    write_prepared_tables,
)


def test_prepare_judgments_builds_signed_and_symmetric_features():
    attempts = pd.DataFrame([{
        "chosen_left": 1, "judge": "j", "item": "i", "left_model": "a", "right_model": "b",
        "length_difference": 0.4, "same_model_family": 0, "self_comparison": 0,
        "reasoning_difference": 1, "left_chars": 120, "right_chars": 80,
        "judge_input_tokens": 300, "judge_output_tokens": 40,
        "judge_reasoning_tokens": 10, "judge_total_tokens": 340,
    }])
    catalogue = pd.DataFrame([
        {"model_id": "j", "theta": -0.5, "release": "2024-01-01", "family": "judge"},
        {"model_id": "a", "theta": 1.0, "release": "2025-01-01", "family": "a"},
        {"model_id": "b", "theta": 0.0, "release": "2023-01-01", "family": "b"},
    ])
    items = pd.DataFrame([{"irt_model": "fixed_3pl", "fit": "gpqa", "benchmark": "gpqa",
                           "item": "i", "difficulty": 0.7}])

    got = prepare_judgments(attempts, catalogue, items)

    row = got.iloc[0]
    assert list(got.columns[:len(JUDGMENT_COLUMNS)]) == JUDGMENT_COLUMNS
    assert row.theta_gap == 1.0
    assert row.oversight == 1.0
    assert row.theta_gap_x_oversight == 1.0
    assert row.theta_gap_x_difficulty == 0.7
    assert row.release_gap_days == 731
    assert row.relative_release_days == 0.5
    token_row = judge_token_rows(got).iloc[0]
    assert token_row.input_tokens == 300
    assert token_row.ordinary_output_tokens == 30


def test_prepare_judgments_names_missing_metadata():
    attempts = pd.DataFrame([{
        "chosen_left": 1, "judge": "judge", "item": "missing", "left_model": "left", "right_model": "right",
        "length_difference": 0.0, "same_model_family": 0, "self_comparison": 0,
        "reasoning_difference": 0, "left_chars": 1, "right_chars": 1,
    }])
    catalogue = pd.DataFrame([
        {"model_id": "judge", "theta": 0.0, "release": "2024-01-01"},
        {"model_id": "left", "theta": 1.0, "release": "2024-01-01"},
        {"model_id": "right", "theta": 0.0, "release": None},
    ])

    with pytest.raises(ValueError, match="right.*missing"):
        prepare_judgments(attempts, catalogue, pd.DataFrame())


def test_judge_tokens_define_reasoning_as_output_subset():
    rows = pd.DataFrame([{
        "judge": "j", "item": "i", "chosen_left": 1, "left_chars": 100, "right_chars": 200,
        "judge_input_tokens": 500, "judge_output_tokens": 80,
        "judge_reasoning_tokens": 30, "judge_total_tokens": 580,
    }])
    got = judge_token_rows(rows)
    assert got.iloc[0].ordinary_output_tokens == 50
    assert got.iloc[0].output_tokens == 80
    assert got.iloc[0].transcript_chars == 300


def test_candidate_rows_deduplicate_and_exclude_zero_usage(monkeypatch):
    import mc_cost.data as data

    usage = lambda total, output=20, reasoning=5: SimpleNamespace(
        input_tokens=total - output, output_tokens=output, reasoning_tokens=reasoning, total_tokens=total,
    )
    kept = SimpleNamespace(
        id="i_1", epoch=1, error=None, model_usage={"m": usage(100)},
        scores={"s": SimpleNamespace(value="C")},
        output=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))]),
    )
    duplicate = SimpleNamespace(
        id="i_1", epoch=1, error=None, model_usage={"m": usage(100)}, scores={}, output=kept.output,
    )
    zero = SimpleNamespace(
        id="i_2", epoch=1, error=None, model_usage={"m": usage(0, 0, 0)}, scores={}, output=kept.output,
    )
    log = SimpleNamespace(
        eval=SimpleNamespace(task="gpqa", model="openrouter/candidate", task_args={}),
        samples=[kept, duplicate, zero],
    )
    monkeypatch.setattr(data, "list_eval_logs", lambda _: [SimpleNamespace(name="one")])
    monkeypatch.setattr(data, "read_eval_log", lambda _: log)
    catalogue = pd.DataFrame([{
        "model_id": "candidate", "theta": 0.5, "release": "2025-01-01", "reasoning_class": "reasoning",
    }])
    items = pd.DataFrame([{
        "irt_model": "fixed_3pl", "fit": "gpqa", "benchmark": "gpqa", "item": "i", "difficulty": 0.4,
    }])

    got = candidate_token_rows(["unused"], catalogue, items)

    assert len(got) == 1
    row = got.iloc[0]
    assert row.role == "candidate"
    assert row.correct == 1
    assert row.theta == 0.5
    assert row.difficulty == 0.4
    assert row.ordinary_output_tokens == 15
    assert got.attrs["accounting"]["zero_usage_or_error"] == 1


def test_candidate_rows_reject_missing_fitting_metadata(monkeypatch):
    import mc_cost.data as data

    sample = SimpleNamespace(
        id="missing_1", epoch=1, error=None,
        model_usage={"m": SimpleNamespace(input_tokens=1, output_tokens=1, reasoning_tokens=0, total_tokens=2)},
        scores={}, output=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))]),
    )
    log = SimpleNamespace(
        eval=SimpleNamespace(task="gpqa", model="openrouter/candidate", task_args={}), samples=[sample],
    )
    monkeypatch.setattr(data, "list_eval_logs", lambda _: [SimpleNamespace(name="one")])
    monkeypatch.setattr(data, "read_eval_log", lambda _: log)
    catalogue = pd.DataFrame([{"model_id": "candidate", "theta": None, "release": None}])

    with pytest.raises(ValueError, match="candidate.*missing.*theta.*release.*item.*missing"):
        candidate_token_rows(["unused"], catalogue, pd.DataFrame())


def test_write_prepared_tables_stores_training_standardization(tmp_path):
    judgments = pd.DataFrame([{column: 0.0 for column in JUDGMENT_COLUMNS}, {column: 2.0 for column in JUDGMENT_COLUMNS}])
    judgments[["judge", "item", "left_model", "right_model"]] = "identity"
    paths = write_prepared_tables(tmp_path, judgments, pd.DataFrame(), pd.DataFrame(), training_rows=[True, True])

    stored = pd.read_parquet(paths["judgments"])
    stats = pd.read_json(paths["judgment_standardization"])
    assert stored.theta_gap.tolist() == [-1.0, 1.0]
    assert stored.reasoning_difference.tolist() == [-1.0, 1.0]
    assert stats.loc["theta_gap", "mean"] == 1.0
    assert stats.loc["reasoning_difference", "mean"] == 1.0
    assert paths["candidate_tokens"].exists()
    assert paths["judge_tokens"].exists()


def test_write_prepared_tables_uses_only_explicit_training_rows(tmp_path):
    judgments = pd.DataFrame([{column: 0.0 for column in JUDGMENT_COLUMNS} for _ in range(3)])
    judgments["theta_gap"] = [0.0, 2.0, 100.0]
    judgments["reasoning_difference"] = [0.0, 2.0, 100.0]
    judgments[["judge", "item", "left_model", "right_model"]] = "identity"

    paths = write_prepared_tables(
        tmp_path, judgments, pd.DataFrame(), pd.DataFrame(), training_rows=[True, True, False],
    )

    stored = pd.read_parquet(paths["judgments"])
    stats = pd.read_json(paths["judgment_standardization"])
    assert stats.loc["theta_gap", "mean"] == 1.0
    assert stored.theta_gap.tolist() == [-1.0, 1.0, 99.0]
    assert stats.loc["reasoning_difference", "mean"] == 1.0
    assert stored.reasoning_difference.tolist() == [-1.0, 1.0, 99.0]
