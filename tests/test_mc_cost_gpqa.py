import json
import zipfile

import pandas as pd

import mc_cost.gpqa as gpqa
from mc_cost.gpqa import extract_attempts, judge_answer, major_is_left, transcript_text


def test_current_baseline_numeric_schema_maps_to_left_right():
    sample = {
        "input": (
            "Transcript 1:\n---\nfirst\n---\n\n"
            "Multiple choice options as shown to Model 2:\nA) option\n\n"
            "Transcript 2:\n---\nsecond\n---\n\nWhich of"
        ),
        "scores": {"judge_scorer": {"answer": "2", "metadata": {"confidence": 5}}},
        "target": 1,
    }

    assert transcript_text(sample) == ("first", "second")
    assert judge_answer(sample) == ("B", 5)
    assert major_is_left(sample)


def test_extractor_accounts_for_truncated_prompt_instead_of_aborting_file(tmp_path):
    path = tmp_path / "judge" / "a-vs-b" / "truncated.eval"
    path.parent.mkdir(parents=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("header.json", json.dumps({
            "status": "success",
            "eval": {"eval_id": "eval", "metadata": {}, "model": "judge", "task": "gpqa"},
        }))
        archive.writestr("samples/one.json", json.dumps({
            "id": "item_1", "uuid": "one", "input": "Transcript 1:\n---\ntruncated",
            "target": 1,
            "scores": {"judge_scorer": {"answer": "1", "metadata": {"confidence": 4}}},
        }))
        archive.writestr("samples/two.json", json.dumps({
            "id": "item_2", "uuid": "two", "input": "unused", "target": 1,
            "scores": {"judge_scorer": {"answer": "invalid", "metadata": {}}},
        }))

    attempts, accounting = extract_attempts(tmp_path)

    assert attempts.empty
    assert accounting.loc["judge", "attempts_skipped_unparseable_transcript"] == 1
    assert accounting.loc["judge", "attempts_skipped_invalid_answer"] == 1
    assert attempts.attrs["exclusions"] == [
        {
            "source": "judgment", "source_log": str(path), "judge": "judge",
            "model": "a|b", "item": "item", "reason": "invalid_answer",
            "missing_fields": [], "count": 1,
        },
        {
            "source": "judgment", "source_log": str(path), "judge": "judge",
            "model": "a|b", "item": "item", "reason": "unparseable_transcript",
            "missing_fields": [], "count": 1,
        },
    ]


def test_extractor_names_file_and_duplicate_exclusions(tmp_path):
    directory = tmp_path / "judge" / "a-vs-b"
    directory.mkdir(parents=True)
    headerless = directory / "01_headerless.eval"
    with zipfile.ZipFile(headerless, "w") as archive:
        archive.writestr("samples/ignored.json", "{}")
    failed = directory / "02_failed.eval"
    with zipfile.ZipFile(failed, "w") as archive:
        archive.writestr("header.json", json.dumps({"status": "error"}))
    duplicate = directory / "03_duplicate.eval"
    sample = {
        "id": "item_1", "uuid": "same", "target": 1,
        "input": "Transcript 1:\n---\nleft\n---\nTranscript 2:\n---\nright\n---\nWhich of",
        "scores": {"judge_scorer": {"answer": "1", "metadata": {}}},
    }
    with zipfile.ZipFile(duplicate, "w") as archive:
        archive.writestr("header.json", json.dumps({
            "status": "success",
            "eval": {"eval_id": "eval", "metadata": {}, "model": "judge", "task": "gpqa"},
        }))
        archive.writestr("samples/one.json", json.dumps(sample))
        archive.writestr("samples/two.json", json.dumps(sample))

    attempts, accounting = extract_attempts(tmp_path)

    assert len(attempts) == 1
    assert accounting.loc["judge", [
        "files_skipped_no_header", "files_skipped_not_success", "duplicate_attempts_skipped",
    ]].tolist() == [1, 1, 1]
    exclusions = {row["reason"]: row for row in attempts.attrs["exclusions"]}
    assert exclusions == {
        "missing_header": {
            "source": "judgment", "source_log": str(headerless), "judge": "judge",
            "model": "a|b", "item": None, "reason": "missing_header",
            "missing_fields": [], "count": 1,
        },
        "non_success": {
            "source": "judgment", "source_log": str(failed), "judge": "judge",
            "model": "a|b", "item": None, "reason": "non_success",
            "missing_fields": [], "count": 1,
        },
        "duplicate_attempt": {
            "source": "judgment", "source_log": str(duplicate), "judge": "judge",
            "model": "a|b", "item": "item", "reason": "duplicate_attempt",
            "missing_fields": [], "count": 1,
        },
    }


def test_mc_prepare_names_dropped_catalogue_metadata_identity(monkeypatch):
    rows = pd.DataFrame([
        {
            "judge": "j", "left_model": "a", "right_model": "b", "model_pair": "a|b",
            "item": "i1", "length_difference": 0.0, "transcript_condition": "raw",
            "log_file": "kept.eval",
        },
        {
            "judge": "j", "left_model": "b", "right_model": "a", "model_pair": "a|b",
            "item": "i2", "length_difference": 1.0, "transcript_condition": "raw",
            "log_file": "kept.eval",
        },
        {
            "judge": "j", "left_model": "a", "right_model": "c", "model_pair": "a|c",
            "item": "i3", "length_difference": 2.0, "transcript_condition": "raw",
            "log_file": "missing.eval",
        },
    ])
    accounting = pd.DataFrame({"attempts_kept": [3, 1]}, index=["j", "dropped"])
    monkeypatch.setattr(gpqa, "extract_attempts", lambda _: (rows, accounting))
    catalogue = pd.DataFrame([
        {"model_id": "j", "theta": 0.0, "release": "2024-01-01"},
        {"model_id": "a", "theta": 1.0, "release": "2025-01-01"},
        {"model_id": "b", "theta": -1.0, "release": "2023-01-01"},
        {"model_id": "c", "theta": 0.5, "release": None},
    ])

    attempts, got_accounting, eligibility = gpqa.prepare_for_mc(
        None, catalogue, min_pairs=0, min_samples=1,
    )

    assert len(attempts) == 2
    assert "length_difference_sd" not in attempts
    assert got_accounting.loc["j", "attempts_skipped_missing_catalogue_metadata"] == 1
    assert eligibility.loc["j", "eligible"]
    assert eligibility.loc["dropped"].to_dict() == {"pairs": 0, "samples": 0, "eligible": False}
    assert attempts.attrs["exclusions"] == [{
        "source": "judgment", "source_log": "missing.eval", "judge": "j",
        "model": "c", "item": "i3", "reason": "missing_catalogue_metadata",
        "missing_fields": ["right_model.release"], "count": 1,
    }]
