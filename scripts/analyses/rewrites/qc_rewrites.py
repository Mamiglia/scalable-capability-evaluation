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
DROPPED = REWRITE_DIR / "dropped.json"

MIN_WORD_RATIO = 0.4
MAX_WORD_RATIO = 1.05

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


def check(original: dict, rewrite: str, min_ratio: float, max_ratio: float) -> str | None:
    """Reason this rewrite fails, or None."""
    if not rewrite.strip():
        return "empty"
    ratio = len(rewrite.split()) / max(original["n_words"], 1)
    if not min_ratio <= ratio <= max_ratio:
        return f"word_ratio={ratio:.2f}"
    want = final_answer_line(original["transcript"])
    if want is None:
        return None  # original has no answer line to preserve
    if want not in rewrite:
        return f"answer_line_lost({want[:40]!r})"
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("rewriter", choices=["haiku", "gpt"])
    ap.add_argument("--min-ratio", type=float, default=MIN_WORD_RATIO)
    ap.add_argument("--max-ratio", type=float, default=MAX_WORD_RATIO)
    ap.add_argument("--drop-failures", action="store_true",
                    help="rewrite the paraphrase file without failing rows, so a re-run redoes them")
    ap.add_argument("--record-dropped", action="store_true",
                    help="record the still-failing rows in dropped.json for downstream exclusion")
    args = ap.parse_args()

    path = REWRITE_DIR / f"paraphrase_{args.rewriter}.jsonl"
    originals = {(r["model"], r["item_id"]): r
                 for r in map(json.loads, ORIGINALS.read_text().splitlines())}
    rewrites = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]

    failures, ratios, ok_pairs = [], [], []
    for row in rewrites:
        key = (row["model"], row["item_id"])
        orig = originals[key]
        reason = check(orig, row["rewritten"], args.min_ratio, args.max_ratio)
        if reason:
            failures.append((key, reason))
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
    for _, reason in failures:
        by_reason[reason.split("=")[0].split("(")[0]] = by_reason.get(reason.split("=")[0].split("(")[0], 0) + 1
    for reason, n in sorted(by_reason.items(), key=lambda kv: -kv[1]):
        print(f"  fail {reason}: {n}")

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
        elif rose:
            print(f"\nnote: rose but not hedging/backtracking: {rose}")
        else:
            print("\nstyle check passed: no category rose in both rate and count")

    if args.drop_failures and failures:
        keep = {tuple(k) for k, _ in failures}
        kept = [r for r in rewrites if (r["model"], r["item_id"]) not in keep]
        path.write_text("".join(json.dumps(r) + "\n" for r in kept))
        print(f"\ndropped {len(rewrites) - len(kept)} failing rows from {path.name}; re-run to redo")

    if args.record_dropped and failures:
        existing = json.loads(DROPPED.read_text()) if DROPPED.exists() else []
        seen = {tuple(d["key"]) for d in existing}
        existing += [{"key": list(k), "rewriter": args.rewriter, "reason": r}
                     for k, r in failures if k not in seen]
        DROPPED.write_text(json.dumps(existing, indent=2) + "\n")
        print(f"recorded {len(existing)} dropped cells in {DROPPED.name}")


def _self_check() -> None:
    orig = {"n_words": 10, "transcript": "Some reasoning here.\nANSWER: B"}
    assert check(orig, "a b c d e f g h\nANSWER: B", 0.4, 1.05) is None  # 10 words, ratio 1.0
    assert check(orig, "a b c d e f g h i\nANSWER: B", 0.4, 1.05).startswith("word_ratio")  # 1.1
    assert check(orig, "", 0.4, 1.05) == "empty"
    assert check(orig, "one\nANSWER: B", 0.4, 1.05).startswith("word_ratio")  # 0.3, under floor
    assert check(orig, "a b c d e f g h\nANSWER: C", 0.4, 1.05).startswith("answer_line_lost")
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
        main()
