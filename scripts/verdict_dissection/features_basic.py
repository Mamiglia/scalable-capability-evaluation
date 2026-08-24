"""Deterministic per-transcript features: length, style rates, formatting,
discourse-connective structure. Operates on assistant spans of a rendered
`[Role]: text` transcript; bare text counts as one assistant span.

Usage (spot check):
  uv run python scripts/verdict_dissection/features_basic.py "<text>"
"""
import math
import re
import sys
from pathlib import Path

import yaml

CONFIG = Path(__file__).resolve().parents[2] / "configs" / "verdict_dissection.yaml"

MARKER = re.compile(r"(?:^|\n\n)\[(\w+)\]: ")
ABBREV = re.compile(r"\b(e\.g|i\.e|etc|vs|cf|Fig|Eq)\.$")
PROTECTED = re.compile(r"\$\$.*?\$\$|\$[^$\n]*\$|```.*?```", re.S)
NUMBERED = re.compile(r"^\s*\d+[.)]\s", re.M)
BULLET = re.compile(r"^\s*[-*•]\s", re.M)
CODE = re.compile(r"```.*?```", re.S)
LATEX = re.compile(r"\$\$.*?\$\$|\$[^$\n]*\$|\\\[.*?\\\]", re.S)
REASONING_CLASSES = ("premise", "inference", "assumption")


def _spans(text):
    """(role, start, end) per block; text without markers is one assistant span."""
    hits = list(MARKER.finditer(text))
    if not hits:
        return [("assistant", 0, len(text))]
    return [(h.group(1).lower(), h.end(),
             hits[i + 1].start() if i + 1 < len(hits) else len(text))
            for i, h in enumerate(hits)]


def _assistant_text(text):
    return "\n\n".join(text[s:e] for r, s, e in _spans(text) if r == "assistant")


def _sentences(text):
    """(start, end) offsets; splits on .?! and newline, not inside $..$ or code
    fences, not on decimals or common abbreviations."""
    protected = bytearray(len(text))
    for m in PROTECTED.finditer(text):
        protected[m.start():m.end()] = b"\x01" * (m.end() - m.start())
    bounds, start = [], 0
    for i, ch in enumerate(text):
        if protected[i]:
            continue
        is_break = ch == "\n" or (
            ch in ".?!"
            and not (ch == "." and i + 1 < len(text) and text[i + 1].isdigit())
            and not ABBREV.search(text[max(0, i - 8):i + 1]))
        if is_break:
            if text[start:i + 1].strip():
                bounds.append((start, i + 1))
            start = i + 1
    if text[start:].strip():
        bounds.append((start, len(text)))
    return bounds


def _rate(patterns, text):
    """Matches per 1000 chars, case-insensitive."""
    n = sum(len(re.findall(p, text, re.I)) for p in patterns)
    return 1000.0 * n / max(len(text), 1)


def _classify(sentence, classes):
    low = sentence.lower()
    for name, cues in classes.items():
        if any(cue in low for cue in cues):
            return name
    return None


def basic_features(rendered, cfg):
    text = _assistant_text(rendered)
    n_chars = max(len(text), 1)
    tail = text[int(len(text) * (1 - cfg["conclusion_tail_frac"])):]
    out = {"log_chars": math.log(n_chars)}

    for name, patterns in cfg["style_lexicons"].items():
        out[f"{name}_rate"] = _rate(patterns, text)
        out[f"{name}_rate_conclusion"] = _rate(patterns, tail)

    out["numbered_steps_per_kchar"] = 1000.0 * len(NUMBERED.findall(text)) / n_chars
    out["bullets_per_kchar"] = 1000.0 * len(BULLET.findall(text)) / n_chars
    out["code_density"] = sum(m.end() - m.start() for m in CODE.finditer(text)) / n_chars
    out["latex_density"] = sum(m.end() - m.start() for m in LATEX.finditer(CODE.sub("", text))) / n_chars
    marker = min((m.start() for pat in cfg["answer_markers"]
                  for m in [re.search(pat, text, re.I)] if m), default=None)
    out["answer_marker_present"] = float(marker is not None)
    out["answer_marker_position"] = marker / n_chars if marker is not None else 1.0

    labels = [_classify(text[s:e], cfg["connective_classes"]) for s, e in _sentences(text)]
    n_sent = max(len(labels), 1)
    for name in cfg["connective_classes"]:
        out[f"{name}_sentence_frac"] = labels.count(name) / n_sent
    chain = best = 0
    for lab in labels:
        chain = chain + 1 if lab in REASONING_CLASSES else 0
        best = max(best, chain)
    out["inference_chain_max"] = float(best)
    ver_pos = [i / n_sent for i, lab in enumerate(labels) if lab == "verification"]
    out["verification_present"] = float(bool(ver_pos))
    out["verification_mean_position"] = sum(ver_pos) / len(ver_pos) if ver_pos else 1.0
    return out


if __name__ == "__main__":
    cfg = yaml.safe_load(CONFIG.read_text())
    for k, v in basic_features(sys.argv[1], cfg).items():
        print(f"{k:32s} {v:.4f}")
