"""features_basic checks: span exclusion, term rules, formatting."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from features_basic import basic_features

AGENTIC = "[Assistant]: short answer\n\n[Tool]: " + "x" * 10000


def test_tool_text_excluded_from_length():
    assert basic_features(AGENTIC)["log_chars"] < math.log(1000)


def test_pronoun_I_is_the_pronoun_not_the_symbol():
    assert basic_features("I need to check this.")["personal_pronouns_per_kword"] > 0
    assert basic_features("where I is the moment of inertia, I = mu R^2")["personal_pronouns_per_kword"] == 0
    assert basic_features("index i runs over items")["personal_pronouns_per_kword"] == 0


def test_overlapping_terms_count_once():
    one = basic_features("Oh wait, no.")["backtracking_per_kword"]
    assert one == 1000.0 / 3          # 3 words, one hit


def test_formatting():
    f = basic_features("1. First\n2. Second\n```py\ncode\n```\n- bullet")
    assert f["numbered_steps_per_kword"] > 0 and f["bullets_per_kword"] > 0
    assert f["code_density"] > 0 and f["latex_density"] == 0


def test_verification_inflections():
    assert basic_features("I verified it and am checking again.")["verification_per_kword"] == 1000.0 * 2 / 7
