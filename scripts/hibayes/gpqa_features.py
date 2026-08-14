"""Frozen GPQA hand-built feature contract.

This is the minimum source required from gpqa_tfa for the judge GLM. The
definitions come from LASR commit 63605fd2199876cbdfb7cb80bdd2cd315a4d55d8:

* src/features/lexical.py
  SHA-256 96fa1249d8b7ea051ad16384c526b8f72516972dbf7cb604e031b5ad30dc3f32
* src/preprocess/build_views.py
  SHA-256 3f89b0c97668d8786c4b3399da0f9da3fdbd9695e070fbd303be1f1edb3776ca
* src/ingest/parse_eval.py
  SHA-256 34d541ea66e2e4d5547dd2ba7b400b16c01ab7af74f205018d84260152bd66c1

Keep these definitions frozen. Validate any intentional change against the
upstream feature parquets with transcript_features.py --self-check.

The source repositories do not declare a license. Confirm redistribution
permission before publishing this extracted contract outside the project.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter

import numpy as np

WORD_RE = re.compile(r"\b[\w'-]+\b")
SENT_RE = re.compile(r"[.!?]+(?:\s|$)")

# §12.1 "Reasoning lexicons" and §12.7 process markers. Deliberately small,
# hand-auditable cue lists: these are meant to be interpretable, so a reader
# can check what a feature counts. Counted as literal substrings, case-insensitive.
LEXICONS: dict[str, list[str]] = {
    "planning": [
        "let me", "first", "second", "third", "next", "then", "step 1", "step 2",
        "plan", "approach", "strategy", "outline", "break this down",
        "to solve this", "we need to", "start by",
    ],
    "uncertainty": [
        "maybe", "perhaps", "possibly", "might", "could be", "unclear",
        "not sure", "uncertain", "seems", "appears", "likely", "probably",
        "i think", "roughly", "approximately", "ambiguous",
    ],
    "confidence": [
        "clearly", "obviously", "certainly", "definitely", "must be",
        "without doubt", "indeed", "of course", "necessarily", "always",
    ],
    "verification": [
        "check", "verify", "confirm", "double-check", "recompute", "recalculate",
        "sanity check", "let's confirm", "consistent with", "units",
        "substitute back", "cross-check", "validate", "makes sense",
    ],
    "correction": [
        "wait", "actually", "hold on", "correction", "i made an error",
        "that's wrong", "let me redo", "on reflection", "revise", "mistake",
        "scratch that", "rather", "instead",
    ],
    "alternatives": [
        "alternatively", "on the other hand", "another way", "or we could",
        "option", "possibility", "either", "otherwise", "in contrast",
        "a different approach",
    ],
    "causal": [
        "because", "therefore", "thus", "hence", "so that", "as a result",
        "consequently", "since", "due to", "implies", "follows that", "leads to",
    ],
    "contrast": [
        "however", "but", "although", "though", "whereas", "nevertheless",
        "on the contrary", "yet", "despite", "while",
    ],
    "conclusion": [
        "in conclusion", "therefore the answer", "final answer", "to summarise",
        "to summarize", "overall", "in summary", "thus the answer", "so the answer",
    ],
    "elimination": [
        "rule out", "eliminate", "cannot be", "can be excluded", "not correct",
        "incorrect", "doesn't fit", "does not fit", "inconsistent", "too small",
        "too large", "we can discard",
    ],
    "quantitative": [
        "calculate", "compute", "equation", "formula", "solve for", "value of",
        "magnitude", "order of", "ratio", "estimate",
    ],
}

# Closed-class function words: the classic authorship/stylometry signal, which
# §12.1 wants as a family and §16 needs as a style-fingerprint baseline.
FUNCTION_WORDS = [
    "the", "of", "and", "to", "a", "in", "that", "is", "was", "it", "for",
    "as", "with", "be", "by", "on", "not", "this", "are", "or", "an", "we",
    "which", "have", "from", "at", "can", "will", "would", "there", "been",
    "if", "so", "these", "then", "than", "its", "also", "such", "each",
]


def safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def lexical_features(text: str) -> dict:
    t = text or ""
    n_chars = len(t)
    words = WORD_RE.findall(t)
    lower = [w.lower() for w in words]
    n_words = len(words)
    sentences = [s for s in SENT_RE.split(t) if s.strip()]
    n_sent = len(sentences)
    paragraphs = [p for p in t.split("\n\n") if p.strip()]
    lines = t.split("\n")

    f: dict[str, float] = {}

    # --- length and structure -------------------------------------------
    f["n_chars"] = n_chars
    f["n_words"] = n_words
    f["n_sentences"] = n_sent
    f["n_paragraphs"] = len(paragraphs)
    f["n_lines"] = len(lines)
    f["mean_word_len"] = safe_div(sum(len(w) for w in words), n_words)
    f["mean_sentence_words"] = safe_div(n_words, n_sent)
    f["mean_paragraph_words"] = safe_div(n_words, len(paragraphs))
    f["log_n_words"] = float(np.log1p(n_words))

    # --- formatting -------------------------------------------------------
    f["n_headings"] = len(re.findall(r"^#{1,6}\s", t, re.M))
    f["n_bullets"] = len(re.findall(r"^\s*[-*+]\s", t, re.M))
    f["n_numbered"] = len(re.findall(r"^\s*\d+[.)]\s", t, re.M))
    f["n_bold"] = len(re.findall(r"\*\*[^*]+\*\*", t))
    f["n_italic"] = len(re.findall(r"(?<!\*)\*[^*\n]+\*(?!\*)", t))
    f["n_code_blocks"] = t.count("<CODEBLOCK>") + len(re.findall(r"```", t)) // 2
    f["n_inline_code"] = t.count("<CODE>") + len(re.findall(r"`[^`\n]+`", t))
    f["n_math"] = (t.count("<MATH>") + t.count("<MATHDISPLAY>")
                   + len(re.findall(r"\$[^$\n]+\$", t)))
    f["n_tables"] = t.count("<TABLEROW>") + len(re.findall(r"^\s*\|.*\|", t, re.M))
    f["n_blockquote"] = len(re.findall(r"^\s*>", t, re.M))
    f["frac_blank_lines"] = safe_div(sum(1 for l in lines if not l.strip()), len(lines))
    f["has_any_markdown"] = float(
        f["n_headings"] + f["n_bullets"] + f["n_bold"] + f["n_numbered"] > 0
    )
    for k in ("n_headings", "n_bullets", "n_numbered", "n_bold", "n_italic",
              "n_code_blocks", "n_inline_code", "n_math", "n_tables"):
        f[f"{k}_per_kword"] = safe_div(f[k] * 1000, n_words)

    # --- punctuation and casing -------------------------------------------
    for name, ch in [("comma", ","), ("period", "."), ("colon", ":"),
                     ("semicolon", ";"), ("qmark", "?"), ("exclam", "!"),
                     ("lparen", "("), ("dash", "-"), ("quote", '"')]:
        f[f"{name}_per_kword"] = safe_div(t.count(ch) * 1000, n_words)
    f["ellipsis_per_kword"] = safe_div(t.count("...") * 1000, n_words)
    f["frac_upper_chars"] = safe_div(sum(1 for c in t if c.isupper()), n_chars)
    f["n_allcaps_words"] = sum(1 for w in words if len(w) > 2 and w.isupper())
    f["allcaps_per_kword"] = safe_div(f["n_allcaps_words"] * 1000, n_words)
    f["n_digits"] = sum(1 for c in t if c.isdigit())
    f["digit_frac"] = safe_div(f["n_digits"], n_chars)

    # --- lexical diversity -------------------------------------------------
    counts = Counter(lower)
    n_types = len(counts)
    f["n_types"] = n_types
    f["ttr"] = safe_div(n_types, n_words)
    # Root TTR is far less length-dependent than raw TTR; §12.1 asks for variants.
    f["root_ttr"] = safe_div(n_types, np.sqrt(n_words)) if n_words else 0.0
    f["hapax_frac"] = safe_div(sum(1 for c in counts.values() if c == 1), n_types)
    f["top_word_frac"] = safe_div(counts.most_common(1)[0][1], n_words) if counts else 0.0
    f["repetition"] = safe_div(n_words - n_types, n_words)
    for w in FUNCTION_WORDS:
        f[f"fw_{w}"] = safe_div(counts.get(w, 0) * 1000, n_words)

    # --- reasoning lexicons -------------------------------------------------
    low_t = t.lower()
    for name, cues in LEXICONS.items():
        hits = sum(low_t.count(c) for c in cues)
        f[f"lex_{name}"] = hits
        f[f"lex_{name}_per_kword"] = safe_div(hits * 1000, n_words)

    # --- answer behaviour ----------------------------------------------------
    f["n_answer_mask_tokens"] = t.count("<ANSWERMASK>")
    f["n_option_tokens"] = t.count("<OPTION>")
    f["option_tokens_per_kword"] = safe_div(f["n_option_tokens"] * 1000, n_words)
    f["mentions_all_four_options"] = float(
        len(set(re.findall(r"\b([A-D])\)", t))) == 4
    )
    return f


def process_features(text: str, n_words: int) -> dict:
    """§12.7 — trajectory shape, only where reasoning text exists."""
    t = text or ""
    low = t.lower()
    words = WORD_RE.findall(t)
    n = len(words) or 1
    paragraphs = [p for p in t.split("\n\n") if p.strip()]

    f: dict[str, float] = {}

    def first_pos(cues: list[str]) -> float:
        """Relative position of the first cue occurrence, 1.0 if never."""
        idx = [low.find(c) for c in cues if low.find(c) >= 0]
        return min(idx) / len(t) if idx and t else 1.0

    f["proc_plan_position"] = first_pos(LEXICONS["planning"])
    f["proc_first_verification_position"] = first_pos(LEXICONS["verification"])
    f["proc_first_correction_position"] = first_pos(LEXICONS["correction"])
    f["proc_first_conclusion_position"] = first_pos(LEXICONS["conclusion"])

    f["proc_n_steps"] = len(re.findall(r"^\s*(?:\d+[.)]|step\s+\d+)", t, re.M | re.I))
    f["proc_n_paragraph_transitions"] = max(len(paragraphs) - 1, 0)
    f["proc_verification_events"] = sum(low.count(c) for c in LEXICONS["verification"])
    f["proc_correction_events"] = sum(low.count(c) for c in LEXICONS["correction"])
    f["proc_alternative_events"] = sum(low.count(c) for c in LEXICONS["alternatives"])
    f["proc_elimination_events"] = sum(low.count(c) for c in LEXICONS["elimination"])
    f["proc_has_correction"] = float(f["proc_correction_events"] > 0)
    f["proc_has_verification"] = float(f["proc_verification_events"] > 0)

    # Efficiency: how much text precedes the concluding move.
    f["proc_words_to_conclusion"] = f["proc_first_conclusion_position"] * n
    f["proc_conclusion_tail_frac"] = 1.0 - f["proc_first_conclusion_position"]

    # Density measures, normalised so they are not length proxies.
    for k in ("proc_verification_events", "proc_correction_events",
              "proc_alternative_events", "proc_elimination_events",
              "proc_n_steps"):
        f[f"{k}_per_kword"] = safe_div(f[k] * 1000, n)

    # Repeated content: a crude loop/redundancy signal (§12.7 "repeated content").
    sents = [s.strip().lower() for s in SENT_RE.split(t) if len(s.strip()) > 20]
    f["proc_repeated_sentence_frac"] = (
        safe_div(len(sents) - len(set(sents)), len(sents)) if sents else 0.0
    )
    return f


# Fixed at import so the column set never depends on which record is seen
# first — an answer-only record appearing before any full response must still
# produce the full NaN-filled process block.
PROCESS_KEYS: list[str] = list(process_features("placeholder text.", 2).keys())
PROCESS_NAN = {k: np.nan for k in PROCESS_KEYS}

# Only the answer_masked view is part of the judge GLM contract.
ANSWER_LINE_RE = re.compile(
    r"[^\S\n]*(?:\*\*|__)?\s*ANSWER\s*[:：]\s*[\*_\x60\s]*[A-D]\b[\*_\x60\.]*",
    re.IGNORECASE,
)
OPTION_REF_RE = re.compile(
    r"(?<![A-Za-z])(?:option\s+|choice\s+|answer\s+)?\(?\b([A-D])\b\)(?![A-Za-z])"
    r"|(?<![A-Za-z])option\s+([A-D])\b(?![A-Za-z])",
    re.IGNORECASE,
)
NL_RE = re.compile(r"\n{3,}")


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    return text.replace("\r\n", "\n").replace("\r", "\n")


def strip_question_echo(body: str, prompt: str) -> tuple[str, bool]:
    """Remove verbatim copies of prompt lines that contain at least 40 chars."""
    if not prompt:
        return body, False
    echoed = False
    lines = [line.strip() for line in prompt.split("\n") if len(line.strip()) >= 40]
    for line in lines:
        if line and line in body:
            body = body.replace(line, " ")
            echoed = True
    return body, echoed


def answer_masked(response: str, prompt: str) -> str:
    """Return the exact gpqa_tfa answer_masked view used by this analysis."""
    body, _ = strip_question_echo(normalise(response), normalise(prompt))
    body = NL_RE.sub("\n\n", body).strip()
    masked = ANSWER_LINE_RE.sub(" <ANSWERMASK> ", body)
    return OPTION_REF_RE.sub(" <OPTION> ", masked)


VISIBILITY_ANSWER_RE = re.compile(
    r"ANSWER\s*[:：]\s*[\*_\x60\s]*([A-D])\b",
    re.IGNORECASE,
)
TEMPLATE_JUNK_RE = re.compile(
    r"<\|(?:im_end|im_start|eot_id|end_of_text|endoftext|assistant|user)\|>"
)


def classify_visibility(response: str, reasoning: str = "") -> str:
    """Classify whether a transcript contains enough visible text for process features."""
    body = TEMPLATE_JUNK_RE.sub("", response).strip()
    if reasoning.strip():
        return "reasoning_summary"
    if not body:
        return "empty"
    stripped = VISIBILITY_ANSWER_RE.sub("", body).strip(" *_\x60\n\t.")
    return "answer_only" if len(stripped) < 20 else "full_response"
