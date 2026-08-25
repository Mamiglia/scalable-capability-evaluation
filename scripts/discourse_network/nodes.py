"""Nodes of a transcript graph, built from workstream A's structured parts.

Kinds: question, assistant_edu (one clause), equation, code, heading, answer,
tool_call, tool_output, user_feedback. Clause boundaries come from the RST
segmenter (segment.py); without them prose is split at sentence ends.
"""
import re

PROTECTED = re.compile(r"```[\s\S]*?```|\$\$[\s\S]*?\$\$|\\\[[\s\S]*?\\\]")   # fenced code, display maths
HEADING = re.compile(r"^#{1,6} .*$", re.M)
ANSWER = re.compile(r"^ANSWER:.*$", re.M)
INLINE = re.compile(r"\\\(.*?\\\)|\$[^$\n]+\$|`[^`\n]+`")                     # inline maths/code stays in its clause
SENTENCE = re.compile(r"[^.!?;]+(?:[.!?;]+|$)")
TARGETS = {"assistant_edu", "equation", "code", "answer", "tool_call"}        # kinds that receive parents


def protected_spans(text):
    """(start, end, kind) of spans never split into clauses: code, equation, heading, answer."""
    spans = [(m.start(), m.end(), "code" if m.group().startswith("```") else "equation")
             for m in PROTECTED.finditer(text)]
    for pattern, kind in ((HEADING, "heading"), (ANSWER, "answer")):
        spans += [(m.start(), m.end(), kind) for m in pattern.finditer(text)
                  if not any(s <= m.start() < e for s, e, _ in spans)]
    return sorted(spans)


def mask_prose(text):
    """Text for the segmenter: protected spans become the word EQUATION, inline maths/code
    the word MATH. Returns (masked, cmap, protected); cmap[i] is the original offset of
    masked character i, with one extra entry for the end."""
    protected = protected_spans(text)
    spans = [(s, e, "EQUATION") for s, e, _ in protected]
    spans += [(m.start(), m.end(), "MATH") for m in INLINE.finditer(text)
              if not any(s <= m.start() < e for s, e, _ in spans)]
    out, cmap, cursor = [], [], 0
    for s, e, word in sorted(spans):
        out += [text[cursor:s], word]
        cmap += list(range(cursor, s)) + [s] * len(word)
        cursor = e
    out.append(text[cursor:])
    cmap += list(range(cursor, len(text) + 1))
    return "".join(out), cmap, protected


def leaf_to_clauses(start, end, cmap, text, protected):
    """Clause spans (original offsets) inside one segmenter leaf [start, end) of the masked
    text: the leaf minus any protected spans in it, whitespace-trimmed."""
    a, b = cmap[start], cmap[end]
    cuts = [a] + [x for s, e, _ in protected for x in (s, e) if a < x < b] + [b]
    clauses = []
    for lo, hi in zip(cuts, cuts[1:]):
        if any(s <= lo and hi <= e for s, e, _ in protected):
            continue
        lo, hi = _trim(text, lo, hi)
        if lo < hi:
            clauses.append((lo, hi))
    return clauses


def sentence_clauses(text):
    """Fallback when no segmenter output exists: sentences between protected spans."""
    spans, cursor = [], 0
    for s, e, _ in protected_spans(text) + [(len(text), len(text), None)]:
        for m in SENTENCE.finditer(text[cursor:s]):
            lo, hi = _trim(text, cursor + m.start(), cursor + m.end())
            if lo < hi:
                spans.append((lo, hi))
        cursor = e
    return spans


def _trim(text, lo, hi):
    while lo < hi and text[lo].isspace():
        lo += 1
    while hi > lo and text[hi - 1].isspace():
        hi -= 1
    return lo, hi


def tool_type(function, content, lexicon, python_write_markers=()):
    """'inspect' (read-only) or 'act' (changes state), from the function name and call text.
    Python is inspect unless a write marker appears; otherwise the first matching lexicon
    group wins and unmatched calls count as act."""
    text = f"{function or ''} {content}"
    if function == "python":                          # python reads unless it writes or shells out
        return "act" if any(m in text for m in python_write_markers) else "inspect"
    for group in ("inspect", "act"):
        if any(term in text for term in lexicon.get(group, ())):
            return group
    return "act"


def output_outcome(content, error_markers):
    return "error" if any(m in content for m in error_markers) else "ok"


def build_nodes(question, transcript, parts, clauses=None, tool_lexicon=None,
                python_write_markers=(), error_markers=()):
    """Nodes of one transcript in character order.
    parts: A's transcript_parts_json (list of parts with subparts and exact offsets).
    clauses: {(part_id, subpart_id): [(start, end), ...]} in transcript offsets, from
    segment.py; subparts without an entry are split at sentence ends.
    Node fields: node_id, part_id, role, kind, start, end, text, call_id, function,
    tool_type, outcome, is_final_sink."""
    lexicon = tool_lexicon or {}
    nodes = [_node(0, None, "question", "question", None, None, question)]
    last_assistant = max((i for i, p in enumerate(parts) if p["role"] == "assistant"), default=None)
    call_types = {}                                   # call_id -> inspect/act, inherited by the output
    for part_id, part in enumerate(parts):
        for subpart_id, sp in enumerate(part.get("subparts", [])):
            base, content = sp["start"], sp["content"]
            assert transcript[base:base + len(content)] == content, "A subpart offsets do not match the transcript"
            if sp["kind"] == "content" and part["role"] == "assistant":
                pieces = [(base + s, base + e, kind) for s, e, kind in protected_spans(content)]
                local = (clauses or {}).get((part_id, subpart_id))
                pieces += ([(s, e, "assistant_edu") for s, e in local] if local is not None
                           else [(base + s, base + e, "assistant_edu") for s, e in sentence_clauses(content)])
            elif sp["kind"] == "content":             # user text after the first assistant message
                pieces = [(base, base + len(content), "user_feedback")]
            else:                                     # tool_call / tool_output
                pieces = [(base, base + len(content), sp["kind"])]
            for start, end, kind in sorted(pieces):
                text = transcript[start:end]
                if not text.strip():
                    continue
                node = _node(len(nodes), part_id, part["role"], kind, start, end, text,
                             call_id=sp.get("call_id"), function=sp.get("function"))
                if kind == "tool_call":
                    node["tool_type"] = call_types[sp.get("call_id")] = tool_type(
                        sp.get("function"), text, lexicon, python_write_markers)
                elif kind == "tool_output":
                    node["tool_type"] = call_types.get(sp.get("call_id"))
                    node["outcome"] = output_outcome(text, error_markers)
                nodes.append(node)
    final = [n for n in nodes if n["part_id"] == last_assistant]
    if final:
        final[-1]["is_final_sink"] = True
    return nodes


def _node(node_id, part_id, role, kind, start, end, text, call_id=None, function=None):
    return {"node_id": node_id, "part_id": part_id, "role": role, "kind": kind, "start": start,
            "end": end, "text": text, "call_id": call_id, "function": function,
            "tool_type": None, "outcome": None, "is_final_sink": False}
