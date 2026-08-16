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


@pytest.mark.parametrize("task", ["inspect_evals/gpqa_diamond", "GPQA Diamond", "gpqa-diamond"])
def test_gpqa_task_names_share_the_irt_benchmark_key(task):
    from mc_cost.data import _benchmark

    assert _benchmark(task) == "gpqa"


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
    assert row.same_model_family == 0
    assert row.release_gap_days == 731
    assert row.relative_release_days == 0.5
    token_row = judge_token_rows(got).iloc[0]
    assert token_row.input_tokens == 300
    assert token_row.ordinary_output_tokens == 30
    assert token_row.difficulty == 0.7


def test_prepare_judgments_names_missing_metadata():
    attempts = pd.DataFrame([{
        "chosen_left": 1, "judge": "judge", "item": "missing", "left_model": "left", "right_model": "right",
        "length_difference": 0.0, "same_model_family": 0, "self_comparison": 0,
        "reasoning_difference": 0, "left_chars": 1, "right_chars": 1,
    }])
    catalogue = pd.DataFrame([
        {"model_id": "judge", "theta": 0.0, "release": "2024-01-01", "family": "judge"},
        {"model_id": "left", "theta": 1.0, "release": "2024-01-01", "family": "left"},
        {"model_id": "right", "theta": 0.0, "release": None, "family": "right"},
    ])

    with pytest.raises(ValueError, match="right.*missing"):
        prepare_judgments(attempts, catalogue, pd.DataFrame())


@pytest.mark.parametrize("left,right", [
    ("gemma3-12b-it", "gemini-3.5-flash"),
    ("gpt-5", "o3"),
])
def test_prepare_judgments_recomputes_family_from_canonical_catalogue(left, right):
    attempts = pd.DataFrame([{
        "chosen_left": 1, "judge": "judge", "item": "i",
        "left_model": left, "right_model": right,
        "length_difference": 0.0, "same_model_family": 0, "self_comparison": 0,
        "reasoning_difference": 0, "left_chars": 1, "right_chars": 1,
        "include_reasoning": False,
    }])
    catalogue = pd.DataFrame([
        {"model_id": "judge", "theta": 0.0, "release": "2024-01-01", "family": "openai"},
        {"model_id": left, "theta": 1.0, "release": "2025-01-01", "family": "google" if left.startswith("gemma") else "openai"},
        {"model_id": right, "theta": 0.0, "release": "2023-01-01", "family": "google" if right.startswith("gemini") else "openai"},
    ])
    items = pd.DataFrame([{
        "irt_model": "fixed_3pl", "fit": "gpqa", "benchmark": "gpqa",
        "item": "i", "difficulty": 0.0,
    }])

    got = prepare_judgments(attempts, catalogue, items)

    assert got.loc[0, "same_model_family"] == 1
    assert got.loc[0, "same_family_x_theta_gap"] == 1.0


def test_prepare_judgments_names_missing_canonical_family():
    attempts = pd.DataFrame([{
        "chosen_left": 1, "judge": "judge", "item": "i", "left_model": "left", "right_model": "right",
        "length_difference": 0.0, "same_model_family": 0, "self_comparison": 0,
        "reasoning_difference": 0, "left_chars": 1, "right_chars": 1,
    }])
    catalogue = pd.DataFrame([
        {"model_id": "judge", "theta": 0.0, "release": "2024-01-01", "family": "judge"},
        {"model_id": "left", "theta": 1.0, "release": "2025-01-01", "family": None},
        {"model_id": "right", "theta": 0.0, "release": "2023-01-01", "family": "right"},
    ])
    items = pd.DataFrame([{
        "irt_model": "fixed_3pl", "fit": "gpqa", "benchmark": "gpqa",
        "item": "i", "difficulty": 0.0,
    }])

    with pytest.raises(ValueError, match="left left missing family"):
        prepare_judgments(attempts, catalogue, items)


def test_judge_tokens_define_reasoning_as_output_subset():
    rows = pd.DataFrame([{
        "judge": "j", "item": "i", "chosen_left": 1, "left_chars": 100, "right_chars": 200,
        "difficulty": 0.4,
        "judge_input_tokens": 500, "judge_output_tokens": 80,
        "judge_reasoning_tokens": 30, "judge_total_tokens": 580,
    }])
    got = judge_token_rows(rows)
    assert got.iloc[0].ordinary_output_tokens == 50
    assert got.iloc[0].output_tokens == 80
    assert got.iloc[0].transcript_chars == 300
    assert got.iloc[0].difficulty == 0.4


def test_judge_tokens_exclude_reasoning_above_billed_output():
    rows = pd.DataFrame([
        {
            "judge": "j", "item": item, "chosen_left": 1,
            "left_chars": 100, "right_chars": 200, "log_file": "judge.eval",
            "judge_input_tokens": 500, "judge_output_tokens": 80,
            "judge_reasoning_tokens": reasoning, "judge_total_tokens": 580,
        }
        for item, reasoning in (("valid", 30), ("invalid", 90))
    ])

    got = judge_token_rows(rows)

    assert got["item"].tolist() == ["valid"]
    assert got.attrs["accounting"] == {"invalid_token_accounting": 1}
    assert got.attrs["exclusions"] == [{
        "source": "judge_token", "source_log": "judge.eval", "judge": "j",
        "model": "j", "item": "invalid", "reason": "invalid_token_accounting",
        "missing_fields": ["output_tokens", "reasoning_tokens"], "count": 1,
    }]


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
        eval=SimpleNamespace(task="inspect_evals/gpqa_diamond", model="openrouter/candidate", task_args={}),
        samples=[kept, duplicate, zero],
    )
    monkeypatch.setattr(data, "list_eval_logs", lambda _: [SimpleNamespace(name="one")])
    monkeypatch.setattr(data, "read_eval_log", lambda _: log)
    catalogue = pd.DataFrame([{
        "model_id": "candidate", "theta": 0.5, "release": "2025-01-01", "reasoning_class": "reasoning",
        "family": "candidate",
    }])
    items = pd.DataFrame([{
        "irt_model": "fixed_3pl", "fit": "gpqa", "benchmark": "gpqa", "item": "i", "difficulty": 0.4,
    }])

    got = candidate_token_rows(["unused"], catalogue, items)

    assert len(got) == 1
    row = got.iloc[0]
    assert row.role == "candidate"
    assert row.benchmark == "gpqa"
    assert row.correct == 1
    assert row.theta == 0.5
    assert row.difficulty == 0.4
    assert row.ordinary_output_tokens == 15
    assert got.attrs["accounting"]["zero_usage_or_error"] == 1
    assert got.attrs["accounting"]["duplicate_samples"] == 1
    assert {
        entry["reason"]: entry for entry in got.attrs["exclusions"]
    }["duplicate_sample"] == {
        "source": "candidate", "source_log": "one", "judge": None,
        "model": "candidate", "item": "i", "reason": "duplicate_sample",
        "missing_fields": [], "count": 1,
    }


def test_candidate_rows_normalize_separate_reasoning_and_exclude_inconsistent_usage(monkeypatch):
    import mc_cost.data as data

    def sample(item, usage):
        return SimpleNamespace(
            id=f"{item}_1", epoch=1, error=None, model_usage={"m": usage}, scores={},
            output=SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content="answer"),
            )]),
        )

    log = SimpleNamespace(
        eval=SimpleNamespace(task="gpqa", model="openrouter/candidate"),
        samples=[
            sample("valid", SimpleNamespace(
                input_tokens=10, output_tokens=20, reasoning_tokens=30, total_tokens=60,
            )),
            sample("invalid", SimpleNamespace(
                input_tokens=10, output_tokens=20, reasoning_tokens=30, total_tokens=30,
            )),
        ],
    )
    monkeypatch.setattr(data, "list_eval_logs", lambda _: [SimpleNamespace(name="one")])
    monkeypatch.setattr(data, "read_eval_log", lambda _: log)
    catalogue = pd.DataFrame([{
        "model_id": "candidate", "theta": 0.5, "release": "2025-01-01",
        "reasoning_class": "reasoning", "family": "candidate",
    }])
    items = pd.DataFrame([
        {"irt_model": "fixed_3pl", "fit": "gpqa", "benchmark": "gpqa", "item": item,
         "difficulty": 0.4}
        for item in ("valid", "invalid")
    ])

    got = candidate_token_rows(["unused"], catalogue, items)

    assert got[["output_tokens", "reasoning_tokens", "ordinary_output_tokens"]].values.tolist() == [
        [50, 30, 20],
    ]
    assert got.attrs["accounting"]["invalid_token_accounting"] == 1
    assert got.attrs["exclusions"] == [{
        "source": "candidate", "source_log": "one", "judge": None,
        "model": "candidate", "item": "invalid", "reason": "invalid_token_accounting",
        "missing_fields": ["output_tokens", "reasoning_tokens"], "count": 1,
    }]


def test_candidate_rows_keep_final_and_full_transcript_lengths_separate(monkeypatch):
    import mc_cost.data as data

    message = SimpleNamespace(content=[
        SimpleNamespace(type="reasoning", text="private chain"),
        SimpleNamespace(type="text", text="final answer"),
    ])
    sample = SimpleNamespace(
        id="i_1", epoch=1, error=None,
        model_usage={"m": SimpleNamespace(
            input_tokens=10, output_tokens=20, reasoning_tokens=10, total_tokens=30,
        )},
        scores={"s": SimpleNamespace(value="C")},
        output=SimpleNamespace(choices=[SimpleNamespace(message=message)]),
    )
    log = SimpleNamespace(
        eval=SimpleNamespace(task="gpqa", model="openrouter/candidate"), samples=[sample],
    )
    monkeypatch.setattr(data, "list_eval_logs", lambda _: [SimpleNamespace(name="one")])
    monkeypatch.setattr(data, "read_eval_log", lambda _: log)
    catalogue = pd.DataFrame([{
        "model_id": "candidate", "theta": 0.5, "release": "2025-01-01",
        "family": "candidate", "reasoning_class": "reasoning",
    }])
    items = pd.DataFrame([{
        "irt_model": "fixed_3pl", "fit": "gpqa", "benchmark": "gpqa",
        "item": "i", "difficulty": 0.4,
    }])

    row = candidate_token_rows(
        ["unused"], catalogue, items, presentation_max_chars=5,
    ).iloc[0]

    assert row.response_chars == 5
    assert row.uncapped_response_chars == len("final answer")
    assert row.full_response_chars > row.response_chars
    assert row.generated_reasoning == 1
    assert row.visible_reasoning == 0


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
    catalogue = pd.DataFrame([{
        "model_id": "candidate", "theta": None, "release": None, "family": "candidate",
    }])

    with pytest.raises(ValueError, match="candidate.*missing.*theta.*release.*item.*missing"):
        candidate_token_rows(["unused"], catalogue, pd.DataFrame())


def test_mc_candidate_rows_account_for_incomplete_logs(monkeypatch):
    import mc_cost.data as data

    sample = SimpleNamespace(id="i_1", epoch=1)
    log = SimpleNamespace(
        eval=SimpleNamespace(task="gpqa", model="openrouter/not-fitted"), samples=[sample, sample],
    )
    monkeypatch.setattr(data, "list_eval_logs", lambda _: [SimpleNamespace(name="one")])
    monkeypatch.setattr(data, "read_eval_log", lambda _: log)

    got = candidate_token_rows(
        ["unused"], pd.DataFrame([{
            "model_id": "fitted", "theta": 0.0, "release": "2025-01-01", "family": "fitted",
        }]),
        pd.DataFrame(), skip_incomplete=True,
    )

    assert got.empty
    assert got.attrs["accounting"]["missing_catalogue_metadata"] == 2
    assert got.attrs["exclusions"] == [{
        "source": "candidate", "source_log": "one", "judge": None,
        "model": "not-fitted", "item": None, "reason": "unknown_model",
        "missing_fields": ["catalogue"], "count": 2,
    }]


def test_mc_candidate_rows_name_error_zero_and_missing_metadata_exclusions(monkeypatch):
    import mc_cost.data as data

    usage = SimpleNamespace(input_tokens=1, output_tokens=1, reasoning_tokens=0, total_tokens=2)
    samples = [
        SimpleNamespace(id="error_1", epoch=1, error="boom", model_usage={"m": usage}),
        SimpleNamespace(
            id="zero_1", epoch=1, error=None,
            model_usage={"m": SimpleNamespace(
                input_tokens=0, output_tokens=0, reasoning_tokens=0, total_tokens=0,
            )},
        ),
        SimpleNamespace(id="missing_1", epoch=1, error=None, model_usage={"m": usage}),
    ]
    log = SimpleNamespace(
        eval=SimpleNamespace(task="gpqa", model="openrouter/candidate"), samples=samples,
    )
    monkeypatch.setattr(data, "list_eval_logs", lambda _: [SimpleNamespace(name="one")])
    monkeypatch.setattr(data, "read_eval_log", lambda _: log)
    catalogue = pd.DataFrame([{
        "model_id": "candidate", "theta": None, "release": None, "family": "candidate",
    }])

    got = candidate_token_rows(["unused"], catalogue, pd.DataFrame(), skip_incomplete=True)

    exclusions = {row["reason"]: row for row in got.attrs["exclusions"]}
    assert exclusions == {
        "sample_error": {
            "source": "candidate", "source_log": "one", "judge": None,
            "model": "candidate", "item": "error", "reason": "sample_error",
            "missing_fields": [], "count": 1,
        },
        "zero_usage": {
            "source": "candidate", "source_log": "one", "judge": None,
            "model": "candidate", "item": "zero", "reason": "zero_usage",
            "missing_fields": [], "count": 1,
        },
        "missing_metadata": {
            "source": "candidate", "source_log": "one", "judge": None,
            "model": "candidate", "item": "missing", "reason": "missing_metadata",
            "missing_fields": ["difficulty", "release", "theta"], "count": 1,
        },
    }


def test_write_prepared_tables_stores_training_standardization(tmp_path):
    judgments = pd.DataFrame([{column: 0.0 for column in JUDGMENT_COLUMNS}, {column: 2.0 for column in JUDGMENT_COLUMNS}])
    judgments[["judge", "item", "left_model", "right_model"]] = "identity"
    paths = write_prepared_tables(tmp_path, judgments, pd.DataFrame(), pd.DataFrame(), training_rows=[True, True])

    stored = pd.read_parquet(paths["judgments"])
    stats = pd.read_json(paths["judgment_standardization"])
    assert stored.theta_gap.tolist() == [-1.0, 1.0]
    assert stored.reasoning_difference.tolist() == [0.0, 2.0]
    assert stats.loc["theta_gap", "mean"] == 1.0
    assert "reasoning_difference" not in stats.index
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
    assert "reasoning_difference" not in stats.index
    assert stored.reasoning_difference.tolist() == [0.0, 2.0, 100.0]
