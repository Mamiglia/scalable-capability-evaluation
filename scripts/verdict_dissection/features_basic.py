"""Deterministic per-transcript features: length, term counts (pronouns,
hedging, backtracking, ...), formatting. Operates on the assistant spans of a
rendered `[Role]: text` transcript; bare text counts as one assistant span.

Usage (spot check):
  uv run python scripts/verdict_dissection/features_basic.py "<text>"
"""
import math
import re
import sys

# Term lists, hits per 1000 assistant words. The first six categories are the
# lexicon validated on real GPQA transcripts for the earlier GLM (trims: "my"
# fired on the mass variable My, "correction" on "Maxwell's correction",
# "clearly" on one item's own question wording, bare "appears" on NMR/colour
# senses). The last three are new and unvalidated.
TERMS = {
    "personal_pronouns": ["I", "me", "we", "our", "us"],
    "hedging": ["perhaps", "possibly", "might", "may", "seems", "appears to", "likely",
                "presumably", "arguably", "I think", "I believe", "could be"],
    "backtracking": ["wait", "actually", "on second thought", "let me reconsider", "oh wait",
                     "scratch that", "I made a mistake"],
    "uncertainty": ["unsure", "not sure", "uncertain", "unclear", "ambiguous", "guess", "hard to say"],
    "confidence": ["obviously", "certainly", "definitely", "undoubtedly", "of course"],
    "questions": ["?"],
    # new, unvalidated
    "verification": ["let me check", "let me verify", "double-check", "double check",
                     "sanity check", "to confirm", "check that"],
    "connectives": ["therefore", "thus", "hence", "because", "since", "it follows",
                    "which means", "this implies", "consequently"],
    "contrast": ["however", "but", "although", "though", "on the other hand",
                 "nevertheless", "whereas"],
}
ANSWER_MARKERS = ["final answer", "the answer is", "answer:", r"\\boxed"]
CONCLUSION_TAIL_FRAC = 0.2   # last fifth of assistant chars = concluding section
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
    """Assistant blocks of a `[Role]: text` transcript; no markers = all assistant."""
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
    """Whole word/phrase, whitespace-flexible; bare punctuation gets no \\b."""
    escaped = re.escape(term.lower()).replace(r"\ ", r"\s+")
    return escaped if re.search(r"\w", term) is None else rf"\b{escaped}\b"


def _count(terms, text):
    """Non-overlapping hits, longest term first so "wait" inside "oh wait" counts once."""
    n = len(STANDALONE_I.findall(text)) if "I" in terms else 0
    others = sorted((t for t in terms if t != "I"), key=len, reverse=True)
    if others:
        n += len(re.findall("|".join(_term_pattern(t) for t in others), text.lower()))
    return n


def basic_features(rendered, terms=TERMS, tail_frac=CONCLUSION_TAIL_FRAC):
    text = _assistant_text(rendered)
    n_chars = max(len(text), 1)
    n_words = max(len(text.split()), 1)
    tail = text[int(len(text) * (1 - tail_frac)):]
    n_tail_words = max(len(tail.split()), 1)

    out = {"log_chars": math.log(n_chars), "n_words": float(n_words)}
    for name, words in terms.items():
        out[f"{name}_per_kword"] = WORDS_PER_RATE * _count(words, text) / n_words
        out[f"{name}_per_kword_conclusion"] = WORDS_PER_RATE * _count(words, tail) / n_tail_words
    out["numbered_steps_per_kword"] = WORDS_PER_RATE * len(NUMBERED.findall(text)) / n_words
    out["bullets_per_kword"] = WORDS_PER_RATE * len(BULLET.findall(text)) / n_words
    out["code_density"] = sum(m.end() - m.start() for m in CODE.finditer(text)) / n_chars
    out["latex_density"] = sum(m.end() - m.start() for m in LATEX.finditer(CODE.sub("", text))) / n_chars
    marker = min((m.start() for pat in ANSWER_MARKERS
                  for m in [re.search(pat, text, re.I)] if m), default=None)
    out["answer_marker_present"] = float(marker is not None)
    out["answer_marker_position"] = marker / n_chars if marker is not None else 1.0
    return out


if __name__ == "__main__":
    for k, v in basic_features(sys.argv[1]).items():
        print(f"{k:40s} {v:.4f}")
