import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from build_manipulation_pairs import build_truncation_frame
from manipulations_verdicts import CONDITIONS, conditions_for_shard, write_records


def test_truncation_notes_are_self_contained_and_match_judging_source():
    import prompts
    from configs.experiment import TRUNCATION_NOTE, TRUNCATION_NOTE_END

    assert prompts.TRUNCATION_NOTE == TRUNCATION_NOTE
    assert prompts.TRUNCATION_NOTE_END == TRUNCATION_NOTE_END


def test_truncation_frame_matches_pairwise_judging_semantics():
    cfg = {
        "comparisons": [["strong", "weak"]],
        "sample_ids": {"strong-vs-weak": ["item"]},
    }
    metadata = {
        ("strong", "item"): {
            "text": "abcdefghij\nANSWER: A",
            "options": "A) one\nB) two",
            "question": "question",
        },
        ("weak", "item"): {
            "text": "123456\nANSWER: B",
            "options": "A) two\nB) one",
            "question": "question",
        },
    }

    frame = build_truncation_frame(
        cfg, metadata, {"strong": 2.0, "weak": 1.0},
        condition="trunc_first_50", frac=0.5, keep="first",
    )

    assert len(frame) == 2
    assert frame.label.tolist() == [1, 2]
    assert frame.text_1.tolist() == ["abc", "123"]
    assert frame.text_2.tolist() == ["123", "abc"]
    assert frame.split.unique().tolist() == ["trunc_first_50"]


def test_four_shards_cover_each_condition_once():
    assigned = [condition for shard in range(4) for condition in conditions_for_shard(shard)]
    assert len(CONDITIONS) == 10
    assert sorted(assigned) == sorted(CONDITIONS)
    assert len(assigned) == len(set(assigned))
    assert [len(conditions_for_shard(i)) for i in range(4)] == [3, 3, 2, 2]


def test_result_writer_resumes_one_deterministic_partial_file(tmp_path):
    out = tmp_path / "condition.jsonl"
    partial = out.with_suffix(".jsonl.partial")
    partial.write_text(json.dumps({"row": 0}) + "\n")

    write_records(out, [{"row": 0}, {"row": 1}, {"row": 2}])

    assert not partial.exists()
    assert [json.loads(line) for line in out.read_text().splitlines()] == [
        {"row": 0}, {"row": 1}, {"row": 2}
    ]
