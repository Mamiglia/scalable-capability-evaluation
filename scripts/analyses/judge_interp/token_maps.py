"""What the preference axes read in a transcript.

Two views of the recorded token maps: sentence occlusion (how the
last-token score moves when one sentence is dropped) and per-token
projections. Sentences are grouped into plain categories by surface
form, so the output is "the axis rewards worked steps" rather than a
list of one-off sentences.

Writes out_dir/token_map_summary.csv.

Usage:
  uv run python scripts/analyses/judge_interp/token_maps.py [--axis approval]
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]

# Surface categories; first match wins, so order matters.
CATEGORIES = [
    ("answer line", r"^\s*(\[Assistant\]:\s*)?(\*\*)?ANSWER:"),
    ("final-answer statement", r"\b(final answer|the answer is|thus,? the|therefore,? the answer)\b"),
    ("section heading", r"^\s*(\*\*|#{1,4}|###\s*\*\*)?\s*(step\s*\d|reaction\s*\d|part\s*\d)"),
    ("bulleted point", r"^\s*[-*+]\s"),
    ("numbered step", r"^\s*\d+[.)]\s"),
    ("equation or formula", r"[=∝≈]|\\frac|\\times|\^|_\{"),
    ("option evaluation", r"\b(option|choice)\s*[A-D]\b|\b[A-D]\)\s"),
    ("plan statement", r"\b(let's|we will|we need to|to determine|to solve|first,)\b"),
    ("conclusion", r"\b(therefore|thus|hence|so,|in summary|conclusion)\b"),
]


def categorize(sentence: str) -> str:
    s = sentence.strip()
    for name, pattern in CATEGORIES:
        if re.search(pattern, s, re.IGNORECASE):
            return name
    return "other prose"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "config.yaml")
    ap.add_argument("--maps", type=Path,
                    default=ROOT / "judge_results/judge_interp/pod_runs_260806/token_maps")
    ap.add_argument("--axis", default="approval")
    ap.add_argument("--min-count", type=int, default=8,
                    help="categories rarer than this are not reported")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    out = ROOT / cfg["out_dir"]

    maps = [json.loads(p.read_text()) for p in sorted(args.maps.glob("*.json"))]
    print(f"{len(maps)} token maps, axis {args.axis}")

    recs = []
    for m in maps:
        for occ in m["occlusion"]:
            delta = occ["delta_last"].get(args.axis)
            if delta is None:
                continue
            # delta = score(without sentence) - score(with it); a sentence
            # the axis likes has a negative delta.
            recs.append(dict(category=categorize(occ["sentence"]),
                             contribution=-delta, length=len(occ["sentence"])))
    df = pd.DataFrame(recs)

    # Dropping any sentence shortens the transcript, and the axis responds to
    # length, so raw contributions are mostly a length readout. Report the part
    # that is not explained by how much text the sentence removed.
    slope, intercept = np.polyfit(df["length"], df["contribution"], 1)
    df["excess"] = df["contribution"] - (slope * df["length"] + intercept)
    print(f"\nlength effect: {slope:+.4f} score per character removed "
          f"(r = {np.corrcoef(df['length'], df['contribution'])[0, 1]:+.3f})")

    rows = []
    for cat, g in df.groupby("category"):
        if len(g) < args.min_count:
            continue
        rows.append(dict(category=cat, n=len(g),
                         mean_contribution=g["contribution"].mean(),
                         excess=g["excess"].mean(),
                         se=g["excess"].std() / np.sqrt(len(g))))
    table = pd.DataFrame(rows).sort_values("excess", ascending=False)
    print("\ncontribution beyond length, by sentence type "
          "(positive = the axis likes it more than its length alone explains)")
    for r in table.itertuples():
        print(f"  {r.category:24s} n={r.n:4d} excess {r.excess:+7.3f} +- {r.se:.3f} "
              f"(raw {r.mean_contribution:+.3f})")

    # Token view: which tokens sit highest and lowest on the axis.
    tok = defaultdict(list)
    for m in maps:
        proj = m["token_projections"].get(args.axis)
        if proj is None:
            continue
        for t, p in zip(m["tokens"], proj):
            s = t.strip().replace("Ġ", "")
            if s and s.isascii():
                tok[s].append(p)
    agg = sorted(((np.mean(v), len(v), t) for t, v in tok.items() if len(v) >= 20))
    print("\nper-token projections (tokens seen >= 20 times)")
    print("  lowest :", ", ".join(f"{t}({m:.0f})" for m, _, t in agg[:10]))
    print("  highest:", ", ".join(f"{t}({m:.0f})" for m, _, t in agg[-10:]))

    table.insert(0, "axis", args.axis)
    path = out / "token_map_summary.csv"
    table.to_csv(path, index=False)
    print(f"-> {path}")


if __name__ == "__main__":
    main()
