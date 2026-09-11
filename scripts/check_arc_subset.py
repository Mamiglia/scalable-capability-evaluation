"""Is the fixed ARC-Challenge subset skewed on difficulty?

ai2_arc carries no difficulty label; the original Feb2018 release does (`schoolGrade`,
3-9), joined here on question id from external_data/arc/arc_challenge_test_meta.csv.
Chi-square the subset's grade histogram against the full 1172-item test set.

    uv run python scripts/check_arc_subset.py [--subset 200] [--seed 0]
"""

import argparse
import csv
from collections import Counter

from scipy.stats import chisquare

from collect.tasks.arc import arc_challenge_cot

p = argparse.ArgumentParser()
p.add_argument("--subset", type=int, default=200)
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()

meta = {r["questionID"]: r for r in csv.DictReader(open("external_data/arc/arc_challenge_test_meta.csv"))}
full = [s.id for s in arc_challenge_cot().dataset]
drawn = [s.id for s in arc_challenge_cot(subset=a.subset, seed=a.seed).dataset]
assert set(full) <= set(meta), "id mismatch between ai2_arc and the Feb2018 release"

grades = sorted({meta[i]["schoolGrade"] for i in full}, key=int)
fc, dc = Counter(meta[i]["schoolGrade"] for i in full), Counter(meta[i]["schoolGrade"] for i in drawn)
expected = [fc[g] / len(full) * len(drawn) for g in grades]
observed = [dc[g] for g in grades]
stat, pval = chisquare(observed, expected)

print(f"subset={a.subset} seed={a.seed}  ({len(drawn)}/{len(full)} items)")
print(f"{'grade':>6} {'full %':>8} {'drawn %':>8} {'drawn n':>8} {'expected':>9}")
for g, o, e in zip(grades, observed, expected):
    print(f"{g:>6} {fc[g] / len(full):>8.1%} {o / len(drawn):>8.1%} {o:>8} {e:>9.1f}")
print(f"\nchi2={stat:.2f} p={pval:.3f} -> {'balanced' if pval > 0.05 else 'SKEWED'}")
print(f"diagram-referencing items: {sum(meta[i]['includesDiagram'] == '1' for i in drawn)}/{len(drawn)} "
      f"drawn vs {sum(meta[i]['includesDiagram'] == '1' for i in full)}/{len(full)} full")
