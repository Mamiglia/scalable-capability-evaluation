import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts" / "judge"))

import rescore_aime


def test_aime_answer_policy(monkeypatch, tmp_path):
    accepted = {
        "**ANSWER:** \\(259\\)": "259",
        "\\[\n\\text{ANSWER: }73\n\\]": "73",
        "ANSWER: 5\nno wait\nANSWER: 7": "7",
        "working\n\n204": "204",
        "**Final Answer:**\n\n$$\n\\boxed{70}": "70",
    }
    for text, target in accepted.items():
        assert rescore_aime.parse_answer(text, target) == (target, True)
    assert not rescore_aime.parse_answer("**ANSWER:** \\(259\\)", "259", "reject")[1]

    rules = []
    with monkeypatch.context() as patch:
        patch.setattr(
            rescore_aime, "rescore", lambda _src, _dst, rule: rules.append(rule) or "ok"
        )
        patch.setattr(
            sys,
            "argv",
            ["rescore_aime", str(tmp_path / "in.eval"), "--out", str(tmp_path / "out.eval")],
        )
        assert rescore_aime.main() == 0
    assert rules == ["accept"]


def test_aime_rescore_safety_and_metrics(monkeypatch, tmp_path):
    def sample(item, value, completion, stop):
        return SimpleNamespace(
            id=item,
            target="259",
            messages=[SimpleNamespace(text=completion)],
            output=SimpleNamespace(completion=completion, choices=[1], stop_reason=stop),
            scores={"aime_scorer": SimpleNamespace(value=value, answer="", metadata={})},
        )

    log = SimpleNamespace(
        status="success",
        samples=[
            sample("item-1", "I", "**ANSWER:** \\(259\\)", "max_tokens"),
            sample("item-1", "C", "ANSWER: 259", "stop"),
            sample("item-2", "I", "**ANSWER:** \\(259\\)", "stop"),
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
    monkeypatch.setattr(rescore_aime, "read_eval_log", lambda _path: log)
    monkeypatch.setattr(
        rescore_aime, "write_eval_log", lambda value, path: written.append((value, path))
    )
    rescore_aime.rescore(tmp_path / "in.eval", tmp_path / "out.eval")
    assert [s.scores["aime_scorer"].value for s in log.samples] == ["I", "C", "C"]
    assert log.results.scores[0].metrics["accuracy"].value == pytest.approx(0.75)
    assert log.results.scores[0].metrics["stderr"].value == pytest.approx(0.25)
    assert written[0][1] == tmp_path / "out.eval"

    for invalid in (
        SimpleNamespace(status="error", samples=None),
        SimpleNamespace(status="success", samples=[]),
        SimpleNamespace(status="success", samples=[SimpleNamespace(scores=None)]),
    ):
        monkeypatch.setattr(rescore_aime, "read_eval_log", lambda _path, value=invalid: value)
        assert rescore_aime.rescore(tmp_path / "in.eval", tmp_path / "out.eval").startswith(
            "skip"
        )

    root = tmp_path / "logs"
    root.mkdir()
    monkeypatch.setattr(
        sys, "argv", ["rescore_aime", str(root), "--out", str(root / "rescored")]
    )
    with pytest.raises(SystemExit):
        rescore_aime.main()
