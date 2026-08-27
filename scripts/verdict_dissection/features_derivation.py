"""Derivation features: how much of the reasoning chain is written out.

    uv run python scripts/verdict_dissection/features_derivation.py

Reads the canonical transcript store and the surface feature table (row set and keys) and writes
<features_root>/raw/derivation.parquet, one row per canonical-correct transcript. Everything comes
from the assistant prose with the ANSWER line removed (tool calls and outputs excluded), so the
features are ~0 for agentic transcripts. Counts of stated relations (=, ≈, ⇒ ...), chained
equalities (a = b = c), consequence markers (therefore, hence ...), named methods (Stewart's theorem,
power of a point ...), intermediate numbers (numbers that are in neither the question nor the final
answer), symbols defined on a left-hand side and reused in later equations, and whether the final
answer's value appears on the right of an equation before the ANSWER line.
"""
import json
import re
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
KEYS = ["bench", "model", "item", "epoch", "log", "sample_epoch", "content_hash"]

ANSWER_LINE = re.compile(r"^\s*\**ANSWER:\s*(.*?)\**\s*$", re.M)
RELATION = re.compile(r"(?<![=<>!:])=(?![=>])|≈|≡|⇒|⟹|→|\\Rightarrow|\\implies|\\approx|\\equiv|\\to\b|\\longrightarrow")
CONSEQUENCE = re.compile(r"\b(?:therefore|thus|hence|so that|it follows|implies|consequently|which gives|giving|"
                         r"yields|we get|we obtain|we find|this means|so\b)", re.I)
JUSTIFICATION = re.compile(r"\b(?:since|because|given that|by definition|by symmetry|recall that|note that)\b", re.I)
NAMED_METHOD = re.compile(r"\b[A-Z][a-z]+(?:['’]s)?\s+(?:theorem|law|formula|identity|lemma|rule|principle|inequality|"
                          r"equation|relation|criterion|method)\b|\b(?:power of (?:a|the) point|divergence theorem|"
                          r"chain rule|law of (?:cosines|sines)|Vieta|Bayes|Stokes|Gauss|Pythagor\w+|time dilation|"
                          r"conservation of \w+)\b")
NUMBER = re.compile(r"(?<![\w.^_\\])-?\d+(?:,\d{3})*(?:\.\d+)?(?!\w|\.\d)")
MATH_SPAN = re.compile(r"\\\[.*?\\\]|\$\$.*?\$\$|\\\(.*?\\\)|(?<!\$)\$(?!\$)[^$\n]+?\$(?!\$)|"
                       r"\\begin\{(?:align|equation|aligned|cases)\*?\}.*?\\end\{(?:align|equation|aligned|cases)\*?\}", re.S)
CLAUSE = re.compile(r"\n|[,;]\s+|\.\s+")
SYMBOL = re.compile(r"(?<![A-Za-z\\])(?:\\[A-Za-z]+|[A-Za-z])(?:_\{?\w+\}?)?(?![A-Za-z])")
LATEX_WORDS = {"\\frac", "\\sqrt", "\\cdot", "\\times", "\\left", "\\right", "\\text", "\\mathrm", "\\quad", "\\qquad",
               "\\displaystyle", "\\bigl", "\\bigr", "\\Bigl", "\\Bigr", "\\begin", "\\end", "\\pmod", "\\boxed",
               "\\implies", "\\Rightarrow", "\\to", "\\approx", "\\equiv", "\\sum", "\\int", "\\log", "\\ln", "\\sin",
               "\\cos", "\\tan", "\\hat", "\\vec", "\\bar", "\\tag", "\\dots", "\\ldots", "\\cdots"}


def prose_of(parts_json):
    parts = json.loads(parts_json)
    return "\n".join(sub["content"] or "" for part in parts for sub in part["subparts"] if sub["kind"] == "content")


def numbers_in(text):
    return {n.replace(",", "").lstrip("-") for n in NUMBER.findall(text)}


def equation_lines(body):
    """Text units that state an equation: each math span, plus each plain line containing a relation."""
    units = [m.group(0) for m in MATH_SPAN.finditer(body)]
    rest = MATH_SPAN.sub(" ", body)
    units += CLAUSE.split(rest)          # plain text: one clause per unit, so a chain is within one expression
    return [u for u in units if RELATION.search(u)]


def symbols_of(text):
    return {s for s in SYMBOL.findall(text) if s not in LATEX_WORDS}


def symbol_chain(units):
    """Symbols defined on a left-hand side, and later equations that reuse one of them."""
    defined, links = set(), 0
    for unit in units:
        used = symbols_of(unit)
        if used & defined:
            links += 1
        left = unit.split("=", 1)[0] if "=" in unit else ""
        defined |= symbols_of(left)
    return len(defined), links


def answer_targets(answer, options_json):
    """Numbers whose appearance after '=' would show the answer was computed, not just stated."""
    answer = answer.replace("$", "").replace("\\boxed{", "").replace("}", "").strip()
    if re.fullmatch(r"[A-J]", answer) and options_json:
        options = json.loads(options_json)
        index = ord(answer) - ord("A")
        return numbers_in(options[index]) if index < len(options) else set()
    return numbers_in(answer)


def derivation_of(parts_json, question, options_json):
    prose = prose_of(parts_json)
    match = ANSWER_LINE.search(prose)
    answer = match.group(1) if match else ""
    body = ANSWER_LINE.sub("", prose)
    units = equation_lines(body)
    given = numbers_in(question) | (numbers_in(" ".join(json.loads(options_json))) if options_json else set())
    targets = answer_targets(answer, options_json)
    intermediate = numbers_in(body) - given - targets
    after_equals = " ".join(u.split("=", 1)[1] for u in units if "=" in u)
    n_symbols, links = symbol_chain(units)
    words = max(1, len(body.split()))
    return {"stated_relations": len(RELATION.findall(body)),
            "equation_lines": len(units),
            "chained_equalities": sum(len(RELATION.findall(u)) >= 2 for u in units),
            "consequence_markers": len(CONSEQUENCE.findall(body)),
            "justification_markers": len(JUSTIFICATION.findall(body)),
            "named_methods": len(NAMED_METHOD.findall(body)),
            "intermediate_numbers": len(intermediate),
            "defined_symbols": n_symbols,
            "symbol_links": links,
            "answer_derived": int(bool(targets) and bool(targets & numbers_in(after_equals))),
            "relations_per_100_words": 100 * len(RELATION.findall(body)) / words}


def main():
    config = yaml.safe_load((ROOT / "configs/design_matrices.yaml").read_text())
    features_root = Path(config["features_root"])
    rows = pd.read_parquet(features_root / "raw/surface-textual.parquet", columns=KEYS)
    store = pd.read_parquet(ROOT / config["freeze"] / "transcript_store.parquet",
                            columns=KEYS + ["question", "options_json", "transcript_parts_json"])
    table = rows.merge(store, on=KEYS, how="left", validate="one_to_one")
    if table.transcript_parts_json.isna().any():
        raise ValueError(f"{int(table.transcript_parts_json.isna().sum())} transcripts missing from the store")
    features = pd.DataFrame([derivation_of(p, q, o) for p, q, o in
                             zip(table.transcript_parts_json, table.question, table.options_json)])
    out = pd.concat([table[KEYS], features], axis=1)
    out.to_parquet(features_root / "raw/derivation.parquet", index=False)
    print(out.drop(columns=KEYS).groupby(out.bench).mean().round(2).T.to_string())


if __name__ == "__main__":
    main()
