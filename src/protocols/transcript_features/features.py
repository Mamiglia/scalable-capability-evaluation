"""Compact feature sets: 3 length, 9 lexical, and 12 tool/structure features.

Lexicon/counting rules adapted from small-judge's judge-dissect branch,
697f1e725c328b5b170fdd23dca01da47beb5767, scripts/verdict_dissection/features_basic.py.
Lexical values are hits per 1,000 prose words; lengths and tool-call count use log1p.
These are lexical cues, not annotations of actual reasoning quality.
Definitions and suggested changes: docs/transcript-feature-table.md.
"""

import re
from collections import Counter
from collections.abc import Sequence

import numpy as np
from inspect_ai.model import ChatMessage

from evals.sample import format_reasoning, format_text

# These are all whole-word counted.
TERMS = {
    "personal_pronouns": ["I", "me", "my", "mine", "myself", "we", "our", "ours", "us", "ourselves"],
    "hedging": ["perhaps", "possibly", "might", "may", "maybe", "seems", "seem", "seemed",
                "seemingly", "appears to", "appear to", "appeared to", "likely", "unlikely",
                "presumably", "arguably", "probably", "potentially", "I think", "I believe",
                "I suspect", "could be", "may be", "tends to", "sort of", "kind of"],
    "backtracking": ["wait", "on second", "let me reconsider", "reconsider",
                     "rethink", "re-examine", "scratch that", "I made a mistake", "mistake",
                     "oh wait", "hold on", "hm", "hmm", "hmmm", "no wait",
                     "never mind", "I was wrong",
                     "that's wrong", "that's not right", "not right", "let me redo", "try again",
                     "start over", "backtrack"],
    "uncertainty": ["unsure", "not sure", "not certain", "uncertain", "unclear", "ambiguous",
                    "guess", "guessing", "hard to say", "hard to tell", "difficult to say",
                    "I don't know", "don't know", "not confident", "can't be sure",
                    "cannot be sure", "doubt", "doubtful", "questionable"],
    "confidence": ["obviously", "certainly", "definitely", "undoubtedly", "of course", "surely",
                   "without doubt", "no doubt", "indeed", "evidently", "unquestionably",
                   "must be", "it is clear", "it's clear", "absolutely", "for sure",
                   "trivially"],
    "questions": ["?"],
    "verification": ["check", "checks", "checked", "checking", "recheck", "re-check",
                     "confirm", "confirms", "confirmed", "confirming",
                     "double-check", "double-checks", "double-checked", "double-checking",
                     "cross-check", "verify", "verifies", "verified", "verifying",
                     "validate", "validated", "validating", "make sure", "making sure",
                     "sanity check", "plug back", "plugging back", "substitute back",
                     "substituting back", "plug in", "plugging in", "let me see if"],
    "connectives": ["therefore", "thus", "hence", "because", "it follows",
                    "follows that", "which means", "this means", "means that", "this implies",
                    "implies", "implying", "consequently", "as a result",
                    "given that", "leads to", "accordingly", "thereby"],
    "contrast": ["however", "but", "although", "though", "on the other hand", "nevertheless",
                 "nonetheless", "whereas", "in contrast", "conversely", "instead",
                 "despite", "even though", "otherwise", "alternatively", "unlike"],
}

LENGTH_FEATURES = ["log_chars", "log_reasoning_chars", "visible_reasoning"]
REGEX_FEATURES = list(TERMS)
FEATURES = LENGTH_FEATURES + REGEX_FEATURES + ["log_tool_calls"]

RST_RELATIONS = ["elaboration", "joint", "organization", "restatement", "causal", "explanation",
                 "contingency", "purpose", "adversative", "context", "evaluation", "attribution", "mode", "topic"]
CLAUSE_TYPES = RST_RELATIONS + ["continuation", "same_unit", "root", "equation", "heading", "answer", "code",
                              "tool_inspect", "tool_act", "tool_output", "feedback"]
# A fixed starting shortlist: mathematical exposition and transitions around tool use.
# All clause types still contribute to denominators, even when a pair is not selected.
TRANSITION_PAIRS = [
    ("organization", "equation"),
    ("explanation", "equation"),
    ("equation", "equation"),
    ("equation", "explanation"),
    ("tool_output", "tool_inspect"),
    ("tool_output", "tool_act"),
]
TRANSITION_SOURCES = list(dict.fromkeys(a for a, _ in TRANSITION_PAIRS))
STRUCTURE_FEATURES = [f"transition__{a}__{b}" for a, b in TRANSITION_PAIRS]
# Distinguish an unobserved source type from an observed source that never leads to B.
STRUCTURE_FEATURES += [f"source_observed__{a}" for a in TRANSITION_SOURCES]
STRUCTURE_FEATURES += ["tool_action_share"]

FEATURE_SETS = {
    "length": LENGTH_FEATURES,
    "regex": LENGTH_FEATURES + REGEX_FEATURES + ["log_tool_calls"],  # the paper's "deterministic features": tool use counts
    "structure": FEATURES + STRUCTURE_FEATURES,
}

# A pronoun is followed by a verb/contraction; exclude symbol definitions such as "I is ...".
STANDALONE_I = re.compile(r"\bI(?=(?:'[a-z])|(?:\s+[a-z]))(?!\s+is\b)")
# ponytail: Markdown/LaTeX regexes cover common spans; use a parser if nested syntax matters.
CODE = re.compile(r"(?P<fence>`{3,}|~{3,})[^\n]*\n.*?(?:(?P=fence)|\Z)|`+[^`\n]*`+", re.S)
# Inline $...$ must not start or end with a space, so "$5 to $10" is prose, not maths.
MATH = re.compile(r"\$\$.*?\$\$|\$(?!\s)[^$\n]*?(?<!\s)\$|\\\[.*?\\\]|\\\(.*?\\\)", re.S)
ANSWER = re.compile(r"^[ \t]*(?:\*\*)?ANSWER:(?:\*\*)?[^\n]*$", re.I | re.M)


def _term_pattern(term: str) -> str:
    """Convert a literal term string to a regex string with flexible spaces and word boundaries."""
    escaped = re.escape(term.lower()).replace(r"\ ", r"\s+")
    return rf"\b{escaped}\b" if re.search(r"\w", term) else escaped


PATTERNS = {
    name: re.compile("|".join(_term_pattern(t) for t in sorted(terms, key=len, reverse=True) if t != "I"), re.I)
    for name, terms in TERMS.items()
}


def length_features(assistant_text: str, reasoning_text: str) -> dict[str, float]:
    """Calculate length features for one transcript, before model-level averaging.

    Args:
        assistant_text: Joined assistant-text string, already including visible
            reasoning and separators. Code and the final answer remain; prompts,
            tool payloads/outputs and renderer markers must already be excluded.
        reasoning_text: String containing only the visible reasoning subset.

    Returns:
        Dict with log_chars = ln(1 + len(assistant_text)), log_reasoning_chars
        = ln(1 + len(reasoning_text)), and visible_reasoning = 1.0 for a nonempty
        reasoning string, else 0.0. Empty strings have log length zero.
    """
    return {
        "log_chars": np.log1p(len(assistant_text)),
        "log_reasoning_chars": np.log1p(len(reasoning_text)),
        "visible_reasoning": float(bool(reasoning_text)),
    }


def lexical_features(text: str) -> dict[str, float]:
    """Convert one assistant-text string into a dict of lexical rates.

    The caller supplies assistant-authored text only. Code, maths spans and
    ANSWER lines are removed before counting; words are whitespace-separated.
    Returns one float per TERMS category: 1,000 * non-overlapping hits / words.
    With no remaining words, all rates are zero. No capability label is used.
    """
    text = ANSWER.sub(" ", MATH.sub(" ", CODE.sub(" ", text))).replace("’", "'").replace("‘", "'")
    words = len(text.split())
    counts = {name: len(pattern.findall(text)) for name, pattern in PATTERNS.items()}
    counts["personal_pronouns"] += len(STANDALONE_I.findall(text))
    return {name: 1000 * count / words if words else 0.0 for name, count in counts.items()}


def transition_features(types: Sequence[str]) -> dict[str, float]:
    """Convert one ordered clause-type sequence into transition rates and tool action share.

    Args:
        types: Sequence of strings from CLAUSE_TYPES, in transcript order. RST
            supplies prose labels; tool nodes use tool_inspect, tool_act and
            tool_output. The initial input-question node must be excluded.

    Returns:
        Dict with six transition__A__B rates from TRANSITION_PAIRS and four
        source_observed__A flags. Each rate is count(A -> B) / count(outgoing A),
        counting all destinations in the denominator. A source flag is 1.0 when
        A has any outgoing transition. The final node has no outgoing transition.
        Unobserved sources have zero rates and a zero flag. tool_action_share is
        tool_act / (tool_act + tool_inspect), counting all call nodes, including
        the final node. No calls gives share zero; log_tool_calls distinguishes
        that from observed inspection-only behaviour. There are 11 returned
        features; log_tool_calls is counted separately from structured messages.

    Raises ValueError for labels outside CLAUSE_TYPES. This function counts
    supplied labels; it does not run the RST parser or assign tool types.
    """
    if unknown := set(types) - set(CLAUSE_TYPES):
        raise ValueError(f"unknown clause types: {sorted(unknown)}")
    pairs = Counter(zip(types, types[1:]))
    totals = Counter(types[:-1])
    nodes = Counter(types)
    calls = nodes["tool_act"] + nodes["tool_inspect"]
    return {
        **{f"transition__{a}__{b}": pairs[a, b] / totals[a] if totals[a] else 0.0
           for a, b in TRANSITION_PAIRS},
        **{f"source_observed__{a}": float(totals[a] > 0) for a in TRANSITION_SOURCES},
        "tool_action_share": nodes["tool_act"] / calls if calls else 0.0,
    }


def extract_features(messages: Sequence[ChatMessage], include_reasoning: bool = True) -> dict[str, float]:
    """Extract base features from a sequence of Inspect ChatMessage objects.

    Reads assistant text and visible reasoning blocks (none when include_reasoning is
    False: the reasoning-hidden view), and counts structured tool
    calls. Tool arguments, user/system/tool text and redacted reasoning are excluded
    from text features. The caller is
    responsible for selecting the item and checking correctness/attempt number.

    Returns a dict with 13 FEATURES keys: log lengths, a reasoning indicator,
    lexical rates per 1,000 prose words, and log_tool_calls = ln(1 + total calls).
    Repeated calls count separately. glm_regex and glm_structure use the call count.
    RST/tool transition
    features are calculated separately. No assistant message gives all NaNs;
    an observed empty assistant message has zero text lengths/rates.

    Raises ValueError if messages is None; rendered transcript strings are not
    supported.
    """
    if messages is None:
        raise ValueError("transcript features require structured messages")
    assistants = [m for m in messages if m.role == "assistant"]
    if not assistants:
        return dict.fromkeys(FEATURES, np.nan)

    text, reasoning = [], []
    for message in assistants:
        if isinstance(message.content, str):
            text.append(format_text(message.content))
        else:
            for block in message.content:
                if block.type == "text":
                    text.append(format_text(block.text))
                elif block.type == "reasoning" and include_reasoning and format_reasoning(block):
                    text.append(block.reasoning)
                    reasoning.append(block.reasoning)

    body = "\n".join(part for part in text if part)
    thoughts = "\n".join(reasoning)
    return {
        **length_features(body, thoughts),
        **lexical_features(body),
        "log_tool_calls": np.log1p(sum(len(m.tool_calls or []) for m in assistants)),
    }
