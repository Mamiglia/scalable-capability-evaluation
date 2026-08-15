"""How the within-pair length difference changes when transcripts are paraphrased.

Judges here favour the longer transcript with capability held constant, so the paraphrase
condition only stays interpretable if rewriting preserves which member of a pair is longer.
For every (comparison, item) cell this compares the original word gap with the rewritten one
and reports how often the longer member changes.

Usage:
  uv run python scripts/analyses/rewrites/length_gap.py
  uv run python scripts/analyses/rewrites/length_gap.py --rewriter gpt
"""
import argparse
import json
import statistics
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
SUBSET_CONFIG = ROOT / "scripts/pairwise-newjudges/configs/manipulations/trunc_first_10.yaml"
ORIGINALS = ROOT / "judge_results/rewrites/originals.jsonl"
NEAR_TIE_WORDS = 10  # gaps this small are noise, not a length cue


def load(path: Path, field: str) -> dict[tuple[str, str], int]:
    return {(r["model"], r["item_id"]): len(r[field].split())
            for r in map(json.loads, path.read_text().splitlines()) if r}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rewriter", default="haiku")
    ap.add_argument("--near-tie-words", type=int, default=NEAR_TIE_WORDS)
    args = ap.parse_args()

    cfg = yaml.safe_load(SUBSET_CONFIG.read_text())
    orig = load(ORIGINALS, "transcript")
    rw = load(ROOT / f"judge_results/rewrites/paraphrase_{args.rewriter}.jsonl", "rewritten")

    rows = []
    for major, minor in cfg["comparisons"]:
        for item in cfg["sample_ids"][f"{major}-vs-{minor}"]:
            a, b = (major, item), (minor, item)
            if a not in rw or b not in rw:
                continue
            rows.append((orig[a] - orig[b], rw[a] - rw[b]))

    print(f"{args.rewriter}: {len(rows)} pair-item cells covered")
    o_abs = [abs(o) for o, _ in rows]
    r_abs = [abs(r) for _, r in rows]
    print(f"|gap| original: median {statistics.median(o_abs):.0f} words, mean {statistics.fmean(o_abs):.0f}")
    print(f"|gap| rewrite : median {statistics.median(r_abs):.0f} words, mean {statistics.fmean(r_abs):.0f}")
    print(f"gap shrank in {sum(abs(r) < abs(o) for o, r in rows)} cells, grew in "
          f"{sum(abs(r) > abs(o) for o, r in rows)}")

    flips = [(o, r) for o, r in rows if o * r < 0]
    real = [(o, r) for o, r in flips
            if abs(o) >= args.near_tie_words and abs(r) >= args.near_tie_words]
    print(f"\nlonger member changes in {len(flips)} of {len(rows)} cells "
          f"({100 * len(flips) / len(rows):.1f}%)")
    print(f"  of those, {len(real)} flip a gap of >={args.near_tie_words} words both sides "
          f"({100 * len(real) / len(rows):.1f}% of all cells) - the ones that matter")
    if rows:
        print(f"\ncorrelation(original gap, rewritten gap) = {statistics.correlation(*zip(*rows)):.2f}")


def _self_check() -> None:
    # sign product < 0 is exactly the "longer member changed" test
    assert (120 - 100) * (90 - 95) < 0
    assert not (120 - 100) * (95 - 90) < 0
    print("self-check OK")


if __name__ == "__main__":
    import sys
    if "--self-check" in sys.argv:
        _self_check()
    else:
        main()
