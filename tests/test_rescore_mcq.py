import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts" / "judge"))

import rescore_mcq


def test_mcq_answer_policy(monkeypatch, tmp_path):
    accepted = {
        "ANSWER: $C$": "C",
        "**ANSWER**: A": "A",
        r"ANSWER: \boxed{C}": "C",
        "ANSWER: (C)": "C",
        "ANSWER: C because the others fail": "C",
        "ANSWER: C\n...\nANSWER: E": "C",
        "**Answer:** Apple pie": "",
    }
    for text, expected in accepted.items():
        assert rescore_mcq.parse_answer(text, 4) == expected
    assert rescore_mcq.parse_answer("ANSWER: $C$", 4, "reject") == ""

    rules = []
    with monkeypatch.context() as patch:
        patch.setattr(
            rescore_mcq, "rescore", lambda _src, _dst, rule: rules.append(rule) or "ok"
        )
        patch.setattr(
            sys,
            "argv",
            ["rescore_mcq", str(tmp_path / "in.eval"), "--out", str(tmp_path / "out.eval")],
        )
        assert rescore_mcq.main() == 0
    assert rules == ["accept"]


def test_mcq_rescore_safety_and_metrics(monkeypatch, tmp_path):
    def sample(item, value, completion, stop):
        return SimpleNamespace(
            id=item,
            choices=["a", "b", "c", "d"],
            target="C",
            output=SimpleNamespace(completion=completion, choices=[1], stop_reason=stop),
            scores={"choice": SimpleNamespace(value=value, answer="", metadata=None)},
        )

    log = SimpleNamespace(
        status="success",
        samples=[
            sample("item-1", "I", "ANSWER: $C$", "max_tokens"),
            sample("item-1", "C", "ANSWER: C", "stop"),
            sample("item-2", "I", "ANSWER: $C$", "stop"),
        ],
        results=SimpleNamespace(
            scores=[
                SimpleNamespace(
                    metrics={
                        "accuracy": SimpleNamespace(value=1 / 3),
                        "stderr": SimpleNamespace(value=0.0),
                    }
                )
            ]
        ),
        eval=SimpleNamespace(metadata={}),
    )
    written = []
    monkeypatch.setattr(rescore_mcq, "read_eval_log", lambda _path: log)
    monkeypatch.setattr(
        rescore_mcq, "write_eval_log", lambda value, path: written.append((value, path))
    )
    rescore_mcq.rescore(tmp_path / "in.eval", tmp_path / "out.eval")
    assert [s.scores["choice"].value for s in log.samples] == ["I", "C", "C"]
    assert log.results.scores[0].metrics["accuracy"].value == pytest.approx(0.75)
    assert log.results.scores[0].metrics["stderr"].value == pytest.approx(0.25)
    assert written[0][1] == tmp_path / "out.eval"

    for invalid in (
        SimpleNamespace(status="error", samples=None),
        SimpleNamespace(status="success", samples=[]),
        SimpleNamespace(status="success", samples=[SimpleNamespace(scores=None)]),
    ):
        monkeypatch.setattr(rescore_mcq, "read_eval_log", lambda _path, value=invalid: value)
        assert rescore_mcq.rescore(tmp_path / "in.eval", tmp_path / "out.eval").startswith(
            "skip"
        )

    root = tmp_path / "logs"
    root.mkdir()
    monkeypatch.setattr(
        sys, "argv", ["rescore_mcq", str(root), "--out", str(root / "rescored")]
    )
    with pytest.raises(SystemExit):
        rescore_mcq.main()
