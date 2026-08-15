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
    text = "I know that i is not important, and I agree, unlike Illinois."
    rates = feature_rates(text, {"personal_pronouns": ["I", "we"]})
    n_words = len(text.split())
    assert rates["personal_pronouns"] == 1000 * 2 / n_words


def test_capital_i_not_counted_before_symbol_notation():
    # This corpus uses bare "I" for the identity matrix, moment of inertia,
    # and integral variables. A genuine first-person "I" is always followed
    # by a verb; a symbol reference is followed by "=", ")", ",", or similar
    # - so require a lowercase word (or contraction) right after "I".
    # Real corpus case that used to be a false positive and now isn't
    # (Yi-34B-Chat, item recOYsaYs6RmtlTDy): "I = (Mx + My) * R^2" - moment of
    # inertia, not the model speaking.
    text = "The moment of inertia of the molecule: I = (Mx + My) * R^2."
    rates = feature_rates(text, {"personal_pronouns": ["I"]})
    assert rates["personal_pronouns"] == 0.0

    # Genuine pronoun use in the same shape of sentence still counts.
    text = "First I calculate the moment of inertia of the molecule."
    rates = feature_rates(text, {"personal_pronouns": ["I"]})
    n_words = len(text.split())
    assert rates["personal_pronouns"] == 1000 * 1 / n_words


def test_capital_i_not_counted_when_followed_by_is():
    # "I is" is never valid English for the pronoun ("I am" is), but is
    # exactly how this corpus introduces a symbol definition. Real corpus
    # case (Yi-34B-Chat, item recOYsaYs6RmtlTDy): "I is the moment of
    # inertia of the molecule" - a false positive under the plain
    # followed-by-lowercase-word rule, since "is" is itself lowercase.
    text = "I is the moment of inertia of the molecule."
    rates = feature_rates(text, {"personal_pronouns": ["I"]})
    assert rates["personal_pronouns"] == 0.0


def test_capital_i_contractions_still_count():
    text = "I'll check this. I'm not sure yet."
    rates = feature_rates(text, {"personal_pronouns": ["I"]})
    n_words = len(text.split())
    assert rates["personal_pronouns"] == 1000 * 2 / n_words


def test_capital_i_pronoun_before_punctuation_or_at_end_of_text_still_counts():
    # Regression guard: the tightened rule only looks at the single token
    # right after "I" (a lowercase word or contraction) - it must not start
    # requiring anything beyond that. A genuine "I VERB" use still counts
    # when the clause is followed shortly by punctuation, and still counts
    # when that clause is the last thing in the whole text.
    text = "Let me reconsider. Actually, I agree."
    rates = feature_rates(text, {"personal_pronouns": ["I"]})
    n_words = len(text.split())
    assert rates["personal_pronouns"] == 1000 * 1 / n_words

    text = "Let me reconsider, actually I agree"  # no trailing punctuation at all
    rates = feature_rates(text, {"personal_pronouns": ["I"]})
    n_words = len(text.split())
    assert rates["personal_pronouns"] == 1000 * 1 / n_words


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
