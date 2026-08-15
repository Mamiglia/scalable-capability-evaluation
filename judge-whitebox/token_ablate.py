"""Token-group mapping for a rendered verdict prompt: structural spans (from char-offsets
of the row's own text and the template's fixed literals) plus marker-word spans (from a
config lexicon), for position-resolved ablation."""
import re

import wbprompts


def _find_span(haystack, needle, start=0, end=None):
    """Char span (start, end) of the first `needle` in haystack[start:end]. None if absent
    or needle is empty."""
    if not needle:
        return None
    idx = haystack.find(needle, start, end)
    if idx == -1:
        return None
    return idx, idx + len(needle)


def _overlapping_tokens(offsets, span):
    """Token indices whose (start, end) offset overlaps span."""
    start, end = span
    return {i for i, (s, e) in enumerate(offsets) if s < end and e > start}


def _marker_tokens(text, base, lexicon, offsets):
    """Token indices of lexicon words found in `text` (a span starting at `base` in the
    prompt), matched whole-word, case-insensitive."""
    pattern = re.compile(r"\b(" + "|".join(re.escape(w) for w in lexicon) + r")\b", re.IGNORECASE)
    idx = set()
    for m in pattern.finditer(text):
        idx |= _overlapping_tokens(offsets, (base + m.start(), base + m.end()))
    return idx


def token_groups(row, tokenizer, c, invert=False):
    """Map a rendered verdict prompt's tokens to interpretable groups.

    Structural groups (instr_format, slot_labels, question, transcript_1, transcript_2,
    answer_readout) partition every token index exactly once. Marker groups (one per
    token_ablate.markers class in config) are word hits inside transcript_1/transcript_2 and
    are not required disjoint from the structural groups.
    """
    prompt = wbprompts.render_verdict_prompt(row, tokenizer, invert=invert)
    enc = tokenizer(prompt, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    n = len(offsets)

    text_1_span = _find_span(prompt, row["text_1"])
    assert text_1_span is not None, "text_1 not found in rendered prompt"
    start1, end1 = text_1_span
    text_2_span = _find_span(prompt, row["text_2"], end1)
    assert text_2_span is not None, "text_2 not found after text_1 in rendered prompt"
    start2, end2 = text_2_span
    question_span = _find_span(prompt, row["question"])
    assert question_span is not None, "question not found in rendered prompt"

    label1 = prompt.rfind("Transcript 1:", 0, start1)
    assert label1 != -1, "'Transcript 1:' label not found before text_1"
    digit1 = label1 + len("Transcript ")
    label2 = prompt.find("Transcript 2:", end1, start2)
    assert label2 != -1, "'Transcript 2:' label not found between text_1 and text_2"
    digit2 = label2 + len("Transcript ")

    slot_spans = [(digit1, digit1 + 1), (digit2, digit2 + 1)]
    for span in (_find_span(prompt, row["options_1"], 0, label1),
                 _find_span(prompt, row["options_2"], end1, label2)):
        if span is not None:
            slot_spans.append(span)

    # priority-claimed so structural groups partition the sequence even if a token straddles
    # a span boundary (e.g. merges trailing/leading whitespace into an adjacent span)
    claimed = set()

    def claim(spans):
        idx = set()
        for span in spans:
            idx |= _overlapping_tokens(offsets, span)
        idx -= claimed
        claimed.update(idx)
        return idx

    transcript_1_idx = claim([(start1, end1)])
    transcript_2_idx = claim([(start2, end2)])
    question_idx = claim([question_span])
    slot_labels_idx = claim(slot_spans)
    answer_idx = {n - 1}
    claimed.update(answer_idx)
    instr_format_idx = set(range(n)) - claimed

    groups = {
        "instr_format": sorted(instr_format_idx),
        "slot_labels": sorted(slot_labels_idx),
        "question": sorted(question_idx),
        "transcript_1": sorted(transcript_1_idx),
        "transcript_2": sorted(transcript_2_idx),
        "answer_readout": sorted(answer_idx),
    }

    for name, lexicon in c["token_ablate"]["markers"].items():
        idx = _marker_tokens(row["text_1"], start1, lexicon, offsets) \
            | _marker_tokens(row["text_2"], start2, lexicon, offsets)
        groups[name] = sorted(idx)

    return groups
