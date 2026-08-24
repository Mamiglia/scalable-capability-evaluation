"""features_basic checks: span exclusion, position split, connective chains."""
import math
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))
from features_basic import basic_features

CFG = yaml.safe_load((Path(__file__).resolve().parents[2] / "configs" / "verdict_dissection.yaml").read_text())

AGENTIC = "[Assistant]: short answer\n\n[Tool]: " + "x" * 10000


def test_tool_text_excluded_from_length():
    assert basic_features(AGENTIC, CFG)["log_chars"] < math.log(1000)


def test_conclusion_split():
    body = "The derivation proceeds cleanly. " * 40
    tail = "Perhaps this might possibly be wrong. "
    f = basic_features(body + tail, CFG)
    assert f["hedging_rate_conclusion"] > f["hedging_rate"]


def test_connective_chain_and_verification():
    text = ("Suppose x is even. Since x is even, x squared is even. "
            "Therefore four divides x squared. Let me check: 4 divides 16.")
    f = basic_features(text, CFG)
    assert f["inference_chain_max"] >= 3
    assert f["verification_present"] == 1.0
    assert 0.5 < f["verification_mean_position"] < 1.0


def test_formatting_and_answer_marker():
    text = "1. First\n2. Second\n```py\ncode\n```\nFinal answer: 42"
    f = basic_features(text, CFG)
    assert f["numbered_steps_per_kchar"] > 0 and f["code_density"] > 0
    assert f["answer_marker_present"] == 1.0
    assert 0.5 < f["answer_marker_position"] <= 1.0


def test_no_answer_marker():
    f = basic_features("Just some text with no marker.", CFG)
    assert f["answer_marker_present"] == 0.0 and f["answer_marker_position"] == 1.0


def test_sentence_split_guards_decimals_and_latex():
    from features_basic import _sentences
    text = "The value is 3.14 exactly. Next $x = 2. 5$ holds. Done."
    parts = [text[s:e].strip() for s, e in _sentences(text)]
    assert parts[0] == "The value is 3.14 exactly."
    assert any("$x = 2. 5$" in p for p in parts)
    assert len(_sentences("Is it prime?\nYes it is")) == 2
