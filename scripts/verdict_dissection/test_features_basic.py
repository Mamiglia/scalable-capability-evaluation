"""features_basic checks: span exclusion, term counts, conclusion split, formatting."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from features_basic import basic_features

AGENTIC = "[Assistant]: short answer\n\n[Tool]: " + "x" * 10000


def test_tool_text_excluded_from_length():
    assert basic_features(AGENTIC)["log_chars"] < math.log(1000)


def test_term_counts_are_whole_words():
    f = basic_features("I think we might. Imagine wet mice.")
    assert f["personal_pronouns_per_kchar"] > 0      # I, we
    assert f["hedging_per_kchar"] > 0                # might
    assert basic_features("Imagine wet mice.")["personal_pronouns_per_kchar"] == 0


def test_conclusion_split():
    body = "The derivation proceeds cleanly. " * 40
    tail = "Perhaps this might possibly be wrong. "
    f = basic_features(body + tail)
    assert f["hedging_per_kchar_conclusion"] > f["hedging_per_kchar"]


def test_formatting_and_answer_marker():
    text = "1. First\n2. Second\n```py\ncode\n```\nFinal answer: 42"
    f = basic_features(text)
    assert f["numbered_steps_per_kchar"] > 0 and f["code_density"] > 0
    assert f["answer_marker_present"] == 1.0
    assert 0.5 < f["answer_marker_position"] <= 1.0


def test_no_answer_marker():
    f = basic_features("Just some text with no marker.")
    assert f["answer_marker_present"] == 0.0 and f["answer_marker_position"] == 1.0
