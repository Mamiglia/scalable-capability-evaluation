import json
import zipfile

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

    attempts, accounting = extract_attempts(tmp_path)

    assert attempts.empty
    assert accounting.loc["judge", "attempts_skipped_unparseable_transcript"] == 1
