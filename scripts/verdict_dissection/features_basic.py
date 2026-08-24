"""Deterministic per-transcript features: length, term counts (pronouns,
hedging, backtracking, ...), formatting. Operates on the assistant spans of a
rendered `[Role]: text` transcript; bare text counts as one assistant span.

Usage (spot check):
  uv run python scripts/verdict_dissection/features_basic.py "<text>"
"""
import math
import re
import sys

# term lists: matches per 1000 assistant chars, whole words, case-insensitive
TERMS = {
    "personal_pronouns": ["I", "we", "my", "our", "me", "us", "myself", "ourselves"],
    "hedging": ["might", "may", "perhaps", "possibly", "seems", "seem", "likely",
                "probably", "arguably", "roughly", "approximately", "somewhat"],
    "backtracking": ["wait", "actually", "hmm", "on second thought", "let me reconsider",
                     "scratch that", "correction", "rather than", "instead"],
    "uncertainty": ["not sure", "unsure", "unclear", "uncertain", "hard to say",
                    "I think", "I believe", "I guess"],
    "confidence": ["clearly", "obviously", "certainly", "definitely", "of course",
                   "undoubtedly", "must be", "it is clear"],
    "verification": ["let me check", "let me verify", "double-check", "double check",
                     "sanity check", "to confirm", "verify", "check that"],
    "connectives": ["therefore", "thus", "hence", "because", "since", "so that",
                    "it follows", "which means", "this implies", "consequently"],
    "contrast": ["however", "but", "although", "though", "on the other hand",
                 "nevertheless", "whereas"],
}
ANSWER_MARKERS = ["final answer", "the answer is", "answer:", r"\\boxed"]
CONCLUSION_TAIL_FRAC = 0.2   # last fifth of assistant chars = concluding section

MARKER = re.compile(r"(?:^|\n\n)\[(\w+)\]: ")
NUMBERED = re.compile(r"^\s*\d+[.)]\s", re.M)
BULLET = re.compile(r"^\s*[-*•]\s", re.M)
CODE = re.compile(r"```.*?```", re.S)
LATEX = re.compile(r"\$\$.*?\$\$|\$[^$\n]*\$|\\\[.*?\\\]", re.S)
QUESTION = re.compile(r"\?")


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


def _count(terms, text):
    return sum(len(re.findall(rf"\b{re.escape(t)}\b", text, re.I)) for t in terms)


def basic_features(rendered, terms=TERMS, tail_frac=CONCLUSION_TAIL_FRAC):
    text = _assistant_text(rendered)
    n = max(len(text), 1)
    tail = text[int(len(text) * (1 - tail_frac)):]
    n_tail = max(len(tail), 1)

    out = {"log_chars": math.log(n), "n_words": float(len(text.split()))}
    for name, words in terms.items():
        out[f"{name}_per_kchar"] = 1000.0 * _count(words, text) / n
        out[f"{name}_per_kchar_conclusion"] = 1000.0 * _count(words, tail) / n_tail
    out["questions_per_kchar"] = 1000.0 * len(QUESTION.findall(text)) / n
    out["numbered_steps_per_kchar"] = 1000.0 * len(NUMBERED.findall(text)) / n
    out["bullets_per_kchar"] = 1000.0 * len(BULLET.findall(text)) / n
    out["code_density"] = sum(m.end() - m.start() for m in CODE.finditer(text)) / n
    out["latex_density"] = sum(m.end() - m.start() for m in LATEX.finditer(CODE.sub("", text))) / n
    marker = min((m.start() for pat in ANSWER_MARKERS
                  for m in [re.search(pat, text, re.I)] if m), default=None)
    out["answer_marker_present"] = float(marker is not None)
    out["answer_marker_position"] = marker / n if marker is not None else 1.0
    return out


if __name__ == "__main__":
    for k, v in basic_features(sys.argv[1]).items():
        print(f"{k:36s} {v:.4f}")
