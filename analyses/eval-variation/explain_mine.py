"""What judges cite in their EXPLANATION, per condition.

Walks the same judge logs as analyze.py, pulls the EXPLANATION text of every
verdict, and counts mentions per category (plain keyword lexicons). Rates are
per-verdict shares; a verdict can hit several categories.

Usage: explain_mine.py [--out judge_results/eval_prompt_variation/explanations.csv]
"""

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

from inspect_ai.log import read_eval_log

COND_ROOT = Path("logs/eval-variation")
BASELINE_ROOT = Path("logs/data-expansion/gpqa")

LEXICON = {
    "length/detail": ["detail", "thorough", "comprehensive", "longer", "concise",
                      "brief", "in-depth", "elaborate", "extensive", "short"],
    "confidence/tone": ["confident", "confidence", "hedg", "uncertain", "assured",
                        "hesitant", "authorit", "tentative"],
    "structure/clarity": ["structur", "organiz", "clear", "format", "step-by-step",
                          "systematic", "logical flow", "presentation"],
    "correctness/errors": ["error", "mistake", "incorrect", "flaw", "wrong",
                           "misappl", "slip", "invalid", "miscalcul"],
    "depth of reasoning": ["deeper", "rigor", "justif", "derivation", "explains why",
                           "reasoning process", "underlying"],
}
EXPL_RE = re.compile(r"EXPLANATION:\s*(.*?)(?:\n\s*ANSWER:|$)", re.DOTALL | re.IGNORECASE)


def mine(root: Path, cond_of) -> dict:
    counts = defaultdict(lambda: defaultdict(int))
    for lp in root.rglob("*.eval"):
        cond = cond_of(lp)
        if cond is None:
            continue
        log = read_eval_log(str(lp))
        if log.status != "success":
            continue
        for s in log.samples or []:
            m = EXPL_RE.search(s.output.completion or "")
            if not m:
                continue
            text = m.group(1).lower()
            counts[cond]["_verdicts"] += 1
            for cat, words in LEXICON.items():
                if any(w in text for w in words):
                    counts[cond][cat] += 1
    return counts


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("judge_results/eval_prompt_variation/explanations.csv"))
    args = ap.parse_args()

    counts = mine(COND_ROOT, lambda p: p.relative_to(COND_ROOT).parts[0])
    for cond, c in mine(BASELINE_ROOT, lambda p: "baseline").items():
        for k, v in c.items():
            counts[cond][k] += v

    cats = list(LEXICON)
    order = ["baseline", "words200", "words50-100", "lenmatch-pooled", "confident", "matteroffact"]
    conds = [c for c in order if c in counts]
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "verdicts"] + cats)
        print(f"{'condition':18} {'verdicts':>8} " + " ".join(f"{c:>19}" for c in cats))
        for cond in conds:
            n = counts[cond]["_verdicts"]
            rates = [counts[cond][c] / n for c in cats]
            w.writerow([cond, n] + [f"{r:.4f}" for r in rates])
            print(f"{cond:18} {n:8d} " + " ".join(f"{r:19.3f}" for r in rates))
    print("->", args.out)


if __name__ == "__main__":
    main()
