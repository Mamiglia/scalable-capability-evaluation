"""Tests for feature_rates in style_features.py."""
from style_features import feature_rates


def test_multiword_phrase_counted_once():
    # "not sure" is one uncertainty hit, not counted per component word.
    text = "I am not sure about this."
    rates = feature_rates(text, {"uncertainty": ["not sure", "unsure"]})
    assert rates["uncertainty"] == 1000 * 1 / 6


def test_normalises_per_1000_words():
    text = " ".join(["perhaps"] * 10 + ["x"] * 90)  # 100 words, 10 hits
    rates = feature_rates(text, {"hedging": ["perhaps"]})
    assert rates["hedging"] == 100.0


def test_capital_i_matches_only_standalone_capitalised_i():
    # capital standalone "I" counts twice; lowercase "i" and "Illinois" don't.
    text = "I know that i is not the same as I, nor as Illinois."
    rates = feature_rates(text, {"personal_pronouns": ["I", "we"]})
    n_words = len(text.split())
    assert rates["personal_pronouns"] == 1000 * 2 / n_words


def test_question_mark_counted_per_character():
    text = "what???"  # one word, three "?" characters
    rates = feature_rates(text, {"questions": ["?"]})
    assert rates["questions"] == 1000 * 3 / 1


def test_overlapping_terms_in_same_category_not_double_counted():
    # "oh wait" should fire once for backtracking, not once for "oh wait"
    # and again for the "wait" it contains.
    text = "Oh wait, I need to redo this."
    rates = feature_rates(text, {"backtracking": ["wait", "oh wait"]})
    n_words = len(text.split())
    assert rates["backtracking"] == 1000 * 1 / n_words


def test_terms_other_than_standalone_i_are_case_insensitive():
    text = "PERHAPS this Might work, Clearly."
    rates = feature_rates(text, {"hedging": ["perhaps", "might"], "confidence": ["clearly"]})
    n_words = len(text.split())
    assert rates["hedging"] == 1000 * 2 / n_words
    assert rates["confidence"] == 1000 * 1 / n_words


def test_empty_text_returns_zero_for_every_category_without_error():
    lexicon = {"hedging": ["perhaps"], "questions": ["?"]}
    rates = feature_rates("", lexicon)
    assert rates == {"hedging": 0.0, "questions": 0.0}
