import sys
from pathlib import Path
from types import SimpleNamespace

from inspect_ai.model import ModelOutput
from inspect_ai.scorer import Score

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts" / "judge"))

import rejudge_unanswered


def test_unanswered_detection_reduces_all_logs_and_filters_stale_ids(tmp_path, monkeypatch):
    empty = SimpleNamespace(
        id="empty_1",
        output=ModelOutput.from_content(model="m", content=""),
        scores={"judge_scorer": Score(value="N", answer=None)},
    )
    nonempty = SimpleNamespace(
        id="nonempty_2",
        output=ModelOutput.from_content(model="m", content="I cannot choose."),
        scores={"judge_scorer": Score(value="N", answer=None)},
    )
    answered = SimpleNamespace(
        id="empty_1",
        output=ModelOutput.from_content(model="m", content="ANSWER: 1"),
        scores={"judge_scorer": Score(value="C", answer="1")},
    )
    scoreless = SimpleNamespace(id="scoreless_1", output=None, scores=None)
    old, latest = tmp_path / "old.eval", tmp_path / "latest.eval"
    monkeypatch.setattr(rejudge_unanswered, "success_logs", lambda _directory: [old, latest])
    monkeypatch.setattr(
        rejudge_unanswered,
        "read_eval_log",
        lambda path: SimpleNamespace(
            samples=[empty, scoreless] if Path(path) == old else [answered, nonempty]
        ),
    )

    assert rejudge_unanswered.unanswered_sample_ids(tmp_path) == {"scoreless_1"}
    assert rejudge_unanswered.unanswered_sample_ids(tmp_path, all_unparsed=True) == {
        "scoreless_1",
        "nonempty_2",
    }

    task = SimpleNamespace(dataset=[SimpleNamespace(id="nonempty_2")])
    assert rejudge_unanswered.current_retry_ids(task, {"nonempty_2", "stale_1"}) == [
        "nonempty_2"
    ]
    assert rejudge_unanswered.current_retry_ids(None, {"nonempty_2"}) == []
