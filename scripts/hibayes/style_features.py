"""Hand-curated style-lexicon feature counting.

feature_rates counts lexicon terms in a transcript and normalises to a rate
per WORDS_PER_RATE words, per category.
"""
import re
from pathlib import Path

import yaml

WORDS_PER_RATE = 1000  # rates are hits per this many words

# "I" is matched case-sensitively as a standalone capitalised word, so it
# doesn't fire on lowercase "i" or "i" inside other words. Every other term
# (including multi-word phrases starting with "I", e.g. "I think") is matched
# case-insensitively against the lowercased text.
STANDALONE_CAPITAL_I = "I"

# Standalone "I" also collides with single-letter symbol notation in this
# corpus's maths/physics/chemistry content: the identity matrix/operator
# ("det(P - lambda I)", "W^2 = I"), moment of inertia ("I = mu R^2"),
# definite-integral variables, the element iodine, and spectroscopic
# notation ("H I", "He I"). Tuned against judge_results/rewrites/originals.jsonl
# (task-4b-report.md): require "I" to be followed by a lowercase word or a
# contraction ("I need", "I'll") - a real first-person "I" is always followed
# by a verb, a symbol reference never is. Also exclude "I is", since "I is"
# is never valid English for the pronoun ("I am" is) but is exactly how these
# transcripts introduce a symbol ("...where I is the moment of inertia").
STANDALONE_I_RE = re.compile(r"\bI(?=(?:'[a-z])|(?:\s+[a-z]))(?!\s+is\b)")


def load_lexicon(path) -> dict[str, list[str]]:
    """Load a style lexicon from a YAML file: category -> list of terms."""
    return yaml.safe_load(Path(path).read_text())


def _term_pattern(term: str) -> str:
    """Regex for one term: word/phrase boundary, whitespace-flexible.

    Terms with no word characters (e.g. "?") get no \\b boundary, since a
    word boundary can't sit next to punctuation.
    """
    escaped = re.escape(term.lower()).replace(r"\ ", r"\s+")
    return escaped if re.search(r"\w", term) is None else rf"\b{escaped}\b"


def _count_category(text: str, terms: list[str]) -> int:
    """Non-overlapping hits for one category.

    Longer terms are tried first so an overlapping shorter term already
    covered by a longer one (e.g. "wait" inside "oh wait") isn't counted
    again for the same span.
    """
    count = 0
    if STANDALONE_CAPITAL_I in terms:
        count += len(STANDALONE_I_RE.findall(text))
    other_terms = [t for t in terms if t != STANDALONE_CAPITAL_I]
    if other_terms:
        ordered = sorted(other_terms, key=len, reverse=True)
        pattern = "|".join(_term_pattern(t) for t in ordered)
        count += len(re.findall(pattern, text.lower()))
    return count


def feature_rates(text: str, lexicon: dict) -> dict[str, float]:
    """Per-1,000-word lexicon-hit rate for each category in lexicon."""
    n_words = len(text.split())
    if n_words == 0:
        return {category: 0.0 for category in lexicon}
    return {
        category: WORDS_PER_RATE * _count_category(text, terms) / n_words
        for category, terms in lexicon.items()
    }
