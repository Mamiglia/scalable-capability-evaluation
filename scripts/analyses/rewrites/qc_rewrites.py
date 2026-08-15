"""Check the paraphrase sets against the originals they were made from.

Per rewrite: word ratio inside bounds, the original's final answer line survives
byte-identical, text non-empty. In aggregate: style-marker rates per category must
fall or hold - if hedging or backtracking went up, the instruction did not take and
the batch is redone rather than patched.

Rows that fail are listed; --drop-failures removes them from the paraphrase file so a
re-run of run_rewrites.py regenerates them (its resume skips whatever is still there).
Rows that fail twice are recorded in dropped.json and excluded downstream from every
condition, so the three condition sets cover identical (model, item) cells.

Usage:
  uv run python scripts/analyses/rewrites/qc_rewrites.py haiku
  uv run python scripts/analyses/rewrites/qc_rewrites.py haiku --drop-failures
"""
import argparse
import json
import re
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
REWRITE_DIR = ROOT / "judge_results/rewrites"
ORIGINALS = REWRITE_DIR / "originals.jsonl"
EXCLUDED = REWRITE_DIR / "excluded.json"

MIN_WORD_RATIO = 0.4
MAX_WORD_RATIO = 1.05  # slack for tokenisation noise when measuring against the original
MAX_EXPANSION_RATIO = 1.0  # the instruction is compression only: never longer than the original

# Minimal per-category style lexicon, counted as hits per 1000 words. A shared
# style-feature module is owned elsewhere; this stands in until it exists.
STYLE_PATTERNS = {
    "hedging": r"\b(perhaps|maybe|possibly|probably|likely|might|seems?|appears?|"
               r"I think|I believe|arguably|roughly|approximately|somewhat|fairly|"
               r"presumably|suppose|guess)\b",
    "backtracking": r"\b(wait|actually|hmm|hold on|but no|on second thought|let me reconsider|"
                    r"scratch that|correction|rethink|reconsider|oh)\b",
    "discourse_markers": r"\b(so|well|now|okay|ok|right|alright|anyway|basically|essentially|"
                         r"let's|let me|first off|note that|recall that)\b",
    "emphasis": r"(\*\*[^*]+\*\*|\*[^*\n]+\*|__[^_]+__|[A-Z]{4,}|!)",
    "formatting_quirks": r"(^\s*[-*•]\s|^\s*#{1,6}\s|^\s*\d+[.)]\s|\\\[|\\\(|\\frac|\$\$)",
    "self_reference": r"\b(I|we|my|our|us)\b",
}
ANSWER_LINE_RE = re.compile(r"\bANSWER\s*:|\bFinal Answer\s*:", re.IGNORECASE)
# Batch markers from the rewriter prompt. None may survive into a stored rewrite.
DELIMITER_RE = re.compile(r"<<<\s*(?:REWRITE|END|TRANSCRIPT)?\s*\d*\s*>>>|<<<|>>>")


def answer_value(line: str) -> str:
    """What the answer actually is, with the wrapping the models vary on removed.

    Originals present the same answer as 'ANSWER: B', '**ANSWER: B**',
    '\\[ \\text{ANSWER: B} \\]', 'Final Answer: C' or inline at the end of a paragraph.
    The load-bearing part is the value after the colon.
    """
    tail = re.split(r"(?i)\b(?:final\s+)?answer\s*:", line)[-1]
    tail = re.sub(r"<\|[^|]*\|>", "", tail)
    tail = tail.replace("\u00d7", "x").replace("\u2212", "-")  # scientific notation varies
    return re.sub(r"[\\{}\[\]$*`.\s]", "", tail).strip().upper()


def final_answer_line(text: str) -> str | None:
    """The last line that reads as a final answer - whatever prefix this model used."""
    for line in reversed([l for l in text.splitlines() if l.strip()]):
        if ANSWER_LINE_RE.search(line):
            return line.strip()
    return None


def feature_hits(texts: list[str]) -> dict[str, list[str]]:
    """The actual matched tokens per category, so a flag can be read rather than trusted.

    The lexicon is a crude stand-in and does produce false positives (LaTeX \\right,
    "quantum corrections"), so any verdict has to be checked against these tokens.
    """
    hits = {}
    for name, pattern in STYLE_PATTERNS.items():
        rx = re.compile(pattern, re.MULTILINE if "^" in pattern else 0)
        hits[name] = [m if isinstance(m, str) else m[0] for t in texts for m in rx.findall(t)]
    return hits


def feature_counts(texts: list[str]) -> dict[str, int]:
    """Raw style-marker hits per category, pooled over texts."""
    return {k: len(v) for k, v in feature_hits(texts).items()}


def feature_rates(texts: list[str]) -> dict[str, float]:
    """Style-marker hits per 1000 words, per category, pooled over texts."""
    words = sum(len(t.split()) for t in texts) or 1
    return {k: 1000 * v / words for k, v in feature_counts(texts).items()}


def write_atomic(path: Path, text: str) -> None:
    """Write via a temp file and rename, so a concurrent reader never sees a half file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def merge_exclusions(previous: dict, failures: list, rewriter: str) -> list[dict]:
    """Union this rewriter's failures with the other rewriters' already on file.

    Replacing rather than merging would discard the other rewriter's exclusions. An older
    file named the rewriter once at the top level instead of per cell, so those cells are
    attributed to it rather than left with no rewriter to sort on.
    """
    was = previous.get("rewriter") or ""
    cells = {}
    for cell in previous.get("cells", []):
        cell = {**cell, "rewriter": cell.get("rewriter") or was}
        if cell["rewriter"] != rewriter:
            cells[(cell["model"], cell["item_id"], cell["rewriter"])] = cell
    for k, reason, ratio in failures:
        cells[(k[0], k[1], rewriter)] = {
            "model": k[0], "item_id": k[1], "rewriter": rewriter,
            "word_ratio": round(ratio, 3), "reason": reason,
        }
    return [cells[k] for k in sorted(cells)]


def check(original: dict, rewrite: str, min_ratio: float, max_ratio: float,
          max_expansion: float = MAX_EXPANSION_RATIO) -> str | None:
    """Reason this rewrite fails, or None."""
    if not rewrite.strip():
        return "empty"
    if DELIMITER_RE.search(rewrite):
        # batch markers in the text mean the reply was mis-parsed and the row may carry
        # another transcript's content, so its answer cannot be trusted
        return f"delimiter_leak({DELIMITER_RE.search(rewrite).group(0)!r})"
    ratio = len(rewrite.split()) / max(original["n_words"], 1)
    if ratio > max_expansion:
        # Reported apart from a ratio miss: expansion is the direction that can mask a weak
        # original, so these are excluded rather than retried - the rewriter already chose
        # to expand this transcript once.
        return f"expanded={ratio:.2f}"
    if not min_ratio <= ratio <= max_ratio:
        return f"word_ratio={ratio:.2f}"
    want = final_answer_line(original["transcript"])
    if want is None:
        return None  # original has no answer line to preserve
    # Always compare the rewrite's own final answer line. Testing whether the wanted line
    # appears anywhere passes rows that carry the right answer mid-text and a different
    # one at the end, which is exactly how a leaked second transcript slips through.
    got = final_answer_line(rewrite)
    if got is None:
        return "answer_line_missing"
    if answer_value(got) == answer_value(want):
        return None  # same answer, only the wrapping differs
    return f"answer_changed({answer_value(want)[:40]!r}->{answer_value(got)[:40]!r})"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("rewriter", choices=["haiku", "gpt"])
    ap.add_argument("--min-ratio", type=float, default=MIN_WORD_RATIO)
    ap.add_argument("--max-ratio", type=float, default=MAX_WORD_RATIO)
    ap.add_argument("--max-expansion", type=float, default=MAX_EXPANSION_RATIO)
    ap.add_argument("--drop-failures", action="store_true",
                    help="rewrite the paraphrase file without failing rows, so a re-run redoes them")
    ap.add_argument("--write-exclusions", action="store_true",
                    help="name the still-failing cells in excluded.json; changes no transcript file")
    args = ap.parse_args()

    path = REWRITE_DIR / f"paraphrase_{args.rewriter}.jsonl"
    originals = {(r["model"], r["item_id"]): r
                 for r in map(json.loads, ORIGINALS.read_text().splitlines())}
    rewrites = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]

    failures, ratios, ok_pairs = [], [], []
    for row in rewrites:
        key = (row["model"], row["item_id"])
        orig = originals[key]
        reason = check(orig, row["rewritten"], args.min_ratio, args.max_ratio, args.max_expansion)
        if reason:
            failures.append((key, reason, len(row["rewritten"].split()) / max(orig["n_words"], 1)))
        else:
            ratios.append(len(row["rewritten"].split()) / max(orig["n_words"], 1))
            ok_pairs.append((orig, row))

    print(f"{args.rewriter}: {len(rewrites)}/{len(originals)} rewritten, "
          f"{len(failures)} fail ({100 * len(failures) / max(len(rewrites), 1):.1f}%)")
    if ratios:
        qs = statistics.quantiles(ratios, n=20) if len(ratios) > 1 else [ratios[0]] * 19
        print(f"word ratio: min {min(ratios):.2f}  p5 {qs[0]:.2f}  median "
              f"{statistics.median(ratios):.2f}  p95 {qs[-1]:.2f}  max {max(ratios):.2f}  "
              f"mean {statistics.fmean(ratios):.2f}")
        print(f"longer than original: {sum(r > 1.0 for r in ratios)} rows")

    by_reason: dict[str, int] = {}
    for _, reason, _ratio in failures:
        by_reason[reason.split("=")[0].split("(")[0]] = by_reason.get(reason.split("=")[0].split("(")[0], 0) + 1
    for reason, n in sorted(by_reason.items(), key=lambda kv: -kv[1]):
        print(f"  fail {reason}: {n}")

    style_failed = False
    if ok_pairs:
        orig_texts = [o["transcript"] for o, _ in ok_pairs]
        new_texts = [r["rewritten"] for _, r in ok_pairs]
        before, after = feature_rates(orig_texts), feature_rates(new_texts)
        n_before, n_after = feature_counts(orig_texts), feature_counts(new_texts)
        # A rate can rise purely because the rewrite is shorter, so a category only counts
        # as risen if its raw count rose too.
        print("\nstyle markers, rate per 1000 words (raw count) original -> rewrite:")
        rose = []
        for name in STYLE_PATTERNS:
            up = after[name] > before[name] and n_after[name] > n_before[name]
            flag = "UP" if up else ("rate up, count down" if after[name] > before[name] else "down/hold")
            print(f"  {name:20s} {before[name]:7.2f} ({n_before[name]:5d}) -> "
                  f"{after[name]:7.2f} ({n_after[name]:5d})   {flag}")
            if up:
                rose.append(name)
        bad = sorted(set(rose) & {"hedging", "backtracking"})
        if rose:
            after_hits = feature_hits(new_texts)
            print("\ntokens behind each risen category (check these before acting on a flag):")
            for name in rose:
                top = Counter(h.lower() for h in after_hits[name]).most_common(6)
                print(f"  {name:20s} {top}")
        if bad:
            print(f"\nFAIL: {bad} rose in both rate and count - redo the batch")
            style_failed = True
        elif rose:
            print(f"\nnote: rose but not hedging/backtracking: {rose}")
        else:
            print("\nstyle check passed: no category rose in both rate and count")

    if args.drop_failures and failures:
        keep = {k for k, _, _ in failures}
        kept = [r for r in rewrites if (r["model"], r["item_id"]) not in keep]
        write_atomic(path, "".join(json.dumps(r) + "\n" for r in kept))
        print(f"\ndropped {len(rewrites) - len(kept)} failing rows from {path.name}; re-run to redo")

    if args.write_exclusions:
        # Merge, never replace: each rewriter contributes its own failures and the union is
        # excluded from every condition. Overwriting here would silently discard the other
        # rewriter's exclusions.
        previous = json.loads(EXCLUDED.read_text()) if EXCLUDED.exists() else {}
        merged = merge_exclusions(previous, failures, args.rewriter)
        write_atomic(EXCLUDED, json.dumps({
            "why": (
                "Cells excluded from the judging campaign. These rewrites fall outside the "
                "word-ratio bounds, and below the lower bound is where rewrites stop "
                "paraphrasing and start summarising: reading them found originals that "
                "flounder or contradict themselves turned into clean, confident, completed "
                "reasoning. Excluding them removes rows known to contain repaired reasoning, "
                "which would make a weak model look strong. This is not threshold "
                "bookkeeping. Reading above the lower bound found no such repair."
            ),
            "excluded_from": "all conditions - originals, both rewriter sets, truncations - "
                             "so every condition covers identical (model, item) cells",
            "caveat": "exclusion is uneven across models, because compression varies by model; "
                      "the surviving corpus is not a uniform sample of the original one",
            "min_word_ratio": args.min_ratio,
            "max_word_ratio": args.max_ratio,
            "n_excluded": len(merged),
            "n_excluded_by_rewriter": {r: sum(1 for c in merged if c["rewriter"] == r)
                                       for r in sorted({c["rewriter"] for c in merged})},
            "excluded_cells": sorted({(c["model"], c["item_id"]) for c in merged}).__len__(),
            "cells": merged,
        }, indent=2) + "\n")
        print(f"\nwrote {len(merged)} excluded rows ({args.rewriter}: {len(failures)}) to "
              f"{EXCLUDED.name}; no transcript file changed")

    return 1 if style_failed else 0


def _self_check() -> None:
    orig = {"n_words": 10, "transcript": "Some reasoning here.\nANSWER: B"}
    assert check(orig, "a b c d e f g h\nANSWER: B", 0.4, 1.05) is None  # 10 words, ratio 1.0
    assert check(orig, "a b c d e f g h i\nANSWER: B", 0.4, 1.05).startswith("expanded")  # 1.1
    # one word longer than the original is still expansion, even inside the 1.05 slack
    assert check({"n_words": 100, "transcript": "x\nANSWER: B"},
                 " ".join(["w"] * 101) + "\nANSWER: B", 0.4, 1.05).startswith("expanded")
    assert check({"n_words": 100, "transcript": "x\nANSWER: B"},
                 " ".join(["w"] * 98) + "\nANSWER: B", 0.4, 1.05) is None
    assert check(orig, "", 0.4, 1.05) == "empty"
    assert check(orig, "one\nANSWER: B", 0.4, 1.05).startswith("word_ratio")  # 0.3, under floor
    assert check(orig, "a b c d e f g h\nANSWER: C", 0.4, 1.05).startswith("answer_changed")
    bold = {"n_words": 10, "transcript": "x\n**ANSWER: B**"}
    assert check(bold, "a b c d e f g h\nANSWER: B", 0.4, 1.05) is None  # markup stripped, same answer
    assert check(bold, "a b c d e f g h\nANSWER: C", 0.4, 1.05).startswith("answer_changed")
    tmpl = {"n_words": 10, "transcript": "x\nANSWER: D<|im_end|>"}
    assert check(tmpl, "a b c d e f g h\nANSWER: D", 0.4, 1.05) is None
    assert check(bold, "a b c d e f g h", 0.4, 1.05) == "answer_line_missing"
    latex = {"n_words": 10, "transcript": "x\n\\[ \\text{ANSWER: B} \\]"}
    assert check(latex, "a b c d e f g h\nANSWER: B", 0.4, 1.05) is None
    # old schema: rewriter named once at the top level, absent from each cell
    old = {"rewriter": "haiku", "cells": [{"model": "m", "item_id": "i", "word_ratio": 0.3,
                                           "reason": "word_ratio=0.30"}]}
    merged = merge_exclusions(old, [(("m", "j"), "word_ratio=0.36", 0.36)], "gpt")
    assert [c["rewriter"] for c in merged] == ["haiku", "gpt"], merged
    # a rewriter's own rows are replaced by its fresh run, other rewriters' are kept
    new = {"cells": merged}
    again = merge_exclusions(new, [], "gpt")
    assert [(c["model"], c["rewriter"]) for c in again] == [("m", "haiku")], again
    assert merge_exclusions({}, [], "haiku") == []

    leak = {"n_words": 10, "transcript": "x\nANSWER: B"}
    assert check(leak, "a b c\nANSWER: B\n<<<END 4>>>\n<<<REWRITE 4>>>\nother\nANSWER: C",
                 0.4, 1.05).startswith("delimiter_leak")
    # the wanted line appearing mid-text must not excuse a different final answer
    assert check(leak, "a b c\nANSWER: B\nmore text\nANSWER: C", 0.4, 1.05).startswith("answer_changed")
    sci = {"n_words": 10, "transcript": "q\nANSWER: A) 6.3x10^-7 M"}
    assert check(sci, "a b c d e f\nANSWER: A) 6.3\u00d710^-7 M", 0.4, 1.05) is None
    gem = {"n_words": 10, "transcript": "x\nFinal Answer: C"}
    assert check(gem, "a b c d e f g h\nANSWER: C", 0.4, 1.05) is None
    assert check(gem, "a b c d e f g h\nANSWER: D", 0.4, 1.05).startswith("answer_changed")
    quirk = {"n_words": 10, "transcript": "x\nFinal Answer: 42"}
    assert check(quirk, "a b c d e f g\nFinal Answer: 42", 0.4, 1.05) is None
    assert final_answer_line("a\nANSWER: B\ntrailing note") == "ANSWER: B"
    hedged = feature_rates(["it might perhaps be so"])
    assert hedged["hedging"] > feature_rates(["the value is two"])["hedging"]
    print("self-check OK")


if __name__ == "__main__":
    import sys
    if "--self-check" in sys.argv:
        _self_check()
    else:
        sys.exit(main())
