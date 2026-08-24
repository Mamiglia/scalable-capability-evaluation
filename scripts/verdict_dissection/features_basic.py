"""Deterministic per-transcript features: length, term counts (pronouns,
hedging, backtracking, ...), formatting. Operates on the assistant spans of a
rendered `[Role]: text` transcript; bare text counts as one assistant span.

Usage (spot check):
  uv run python scripts/verdict_dissection/features_basic.py "<text>"
"""
import math
import re
import sys

# Term lists, hits per 1000 assistant words, matched case-insensitively as whole
# words/phrases (standalone "I" excepted, see STANDALONE_I). Started from the
# lexicon validated on real GPQA transcripts for the earlier GLM and extended by
# hand. Left out on purpose because they fire on subject vocabulary rather than
# the writer's style: "error", "correction", "approximately", "unknown", "test",
# "clearly", bare "appears".
TERMS = {
    "personal_pronouns": ["I", "me", "my", "mine", "myself", "we", "our", "ours", "us", "ourselves"],
    "hedging": ["perhaps", "possibly", "might", "may", "maybe", "seems", "seem", "seemed",
                "seemingly", "appears to", "appear to", "appeared to", "likely", "unlikely",
                "presumably", "arguably", "probably", "potentially", "I think", "I believe",
                "I suspect", "I assume", "could be", "may be", "tends to", "generally",
                "typically", "in general", "sort of", "kind of"],
    "backtracking": ["wait", "actually", "on second", "let me reconsider", "reconsider",
                     "rethink", "re-examine", "scratch that", "I made a mistake", "mistake",
                     "oh wait", "hold on", "hmm", "hmmm", "no wait", "never mind", "I was wrong",
                     "that's wrong", "that's not right", "not right", "let me redo", "try again",
                     "start over", "go back", "backtrack"],
    "uncertainty": ["unsure", "not sure", "not certain", "uncertain", "unclear", "ambiguous",
                    "guess", "guessing", "hard to say", "hard to tell", "difficult to say",
                    "I don't know", "don't know", "not confident", "can't be sure",
                    "cannot be sure", "doubt", "doubtful", "questionable"],
    "confidence": ["obviously", "certainly", "definitely", "undoubtedly", "of course", "surely",
                   "without doubt", "no doubt", "indeed", "evidently", "unquestionably",
                   "must be", "it is clear", "it's clear", "absolutely", "for sure",
                   "straightforward", "trivially", "trivial"],
    "questions": ["?"],
    # new, unvalidated
    "verification": ["check", "checks", "checked", "checking", "recheck", "re-check",
                     "confirm", "confirms", "confirmed", "confirming",
                     "double-check", "double-checks", "double-checked", "double-checking",
                     "cross-check", "verify", "verifies", "verified", "verifying",
                     "validate", "validated", "validating", "make sure", "making sure",
                     "sanity check", "plug back", "plugging back", "substitute back",
                     "substituting back", "plug in", "plugging in", "let me see if"],
    "connectives": ["therefore", "thus", "hence", "because", "since", "it follows",
                    "follows that", "which means", "this means", "means that", "this implies",
                    "implies", "implying", "consequently", "so that", "as a result",
                    "given that", "leads to", "in turn", "accordingly", "thereby"],
    "contrast": ["however", "but", "although", "though", "on the other hand", "nevertheless",
                 "nonetheless", "whereas", "yet", "in contrast", "conversely", "instead",
                 "rather", "despite", "even though", "still", "otherwise", "alternatively",
                 "unlike"],
}
WORDS_PER_RATE = 1000

# Standalone "I" is case-sensitive and must be followed by a lowercase word or
# a contraction, so the identity matrix, moment of inertia, iodine, "H I" and
# symbol definitions ("where I is ...") don't count as the pronoun.
STANDALONE_I = re.compile(r"\bI(?=(?:'[a-z])|(?:\s+[a-z]))(?!\s+is\b)")
MARKER = re.compile(r"(?:^|\n\n)\[(\w+)\]: ")
NUMBERED = re.compile(r"^\s*\d+[.)]\s", re.M)
BULLET = re.compile(r"^\s*[-*•]\s", re.M)
CODE = re.compile(r"```.*?```", re.S)
LATEX = re.compile(r"\$\$.*?\$\$|\$[^$\n]*\$|\\\[.*?\\\]", re.S)


def _assistant_text(text):
    """
    Assistant blocks of a `[Role]: text` transcript; no markers = all assistant.
    Especially important for agentic/multi-turn evals, where user messages
    can be interspersed with model's.
    """
    hits = list(MARKER.finditer(text))
    if not hits:
        return text
    parts = []
    for i, h in enumerate(hits):
        end = hits[i + 1].start() if i + 1 < len(hits) else len(text)
        if h.group(1).lower() == "assistant":
            parts.append(text[h.end():end])
    return "\n\n".join(parts)


def _term_pattern(term):
    """Regex for one term: lowercased, literal, matched as a whole word or
    phrase (\\b on both sides) with any whitespace allowed between the words.
    A term with no word characters ("?") gets no \\b, since a word boundary
    can't sit between punctuation and a space."""
    escaped = re.escape(term.lower()).replace(r"\ ", r"\s+")
    return escaped if re.search(r"\w", term) is None else rf"\b{escaped}\b"


def _count(terms, text):
    """Non-overlapping hits, longest term first so "wait" inside "oh wait" counts once."""
    n = len(STANDALONE_I.findall(text)) if "I" in terms else 0
    others = sorted((t for t in terms if t != "I"), key=len, reverse=True)
    if others:
        n += len(re.findall("|".join(_term_pattern(t) for t in others), text.lower()))
    return n


def basic_features(rendered, terms=TERMS):
    text = _assistant_text(rendered)
    n_chars = max(len(text), 1)
    n_words = max(len(text.split()), 1)

    out = {"log_chars": math.log(n_chars), "n_words": float(n_words)}
    for name, words in terms.items():
        out[f"{name}_per_kword"] = WORDS_PER_RATE * _count(words, text) / n_words
    out["numbered_steps_per_kword"] = WORDS_PER_RATE * len(NUMBERED.findall(text)) / n_words
    out["bullets_per_kword"] = WORDS_PER_RATE * len(BULLET.findall(text)) / n_words
    out["code_density"] = sum(m.end() - m.start() for m in CODE.finditer(text)) / n_chars
    out["latex_density"] = sum(m.end() - m.start() for m in LATEX.finditer(CODE.sub("", text))) / n_chars
    return out


if __name__ == "__main__":
    for k, v in basic_features(sys.argv[1]).items():
        print(f"{k:40s} {v:.4f}")
