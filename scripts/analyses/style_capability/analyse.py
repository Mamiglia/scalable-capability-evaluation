"""Counterbalanced % chosen (picked-major) from the style_capability verdict
logs, per (comparison, judge) and pooled across judges, for the default vs
style-blind judge prompt.

major = the strong-styled / spoofing side of each pair, so a value > 0.50 means
the judge preferred it. Counterbalancing = average a verdict over the item's two
slot orders first (cancels position bias), then average over items.

Run (main checkout): .venv/bin/python <worktree>/.../analyse.py
"""

import glob
import statistics
from collections import defaultdict
from pathlib import Path

from inspect_ai.log import read_eval_log

ROOT = "logs/style_capability"
VARIANTS = ["default", "styleblind"]


def item_picks(logpath):
    """Per-item counterbalanced picked-major (0/0.5/1) for one verdict log."""
    log = read_eval_log(logpath)
    by = defaultdict(list)
    for s in log.samples:
        sc = list(s.scores.values())[0]
        if sc.answer in ("1", "2"):
            by[s.id.rsplit("_", 1)[0]].append(1.0 if sc.answer == s.target else 0.0)
    return [statistics.mean(v) for v in by.values() if v]


def fmt(per):
    n = len(per)
    if not n:
        return f"{'-':>16}"
    m = statistics.mean(per)
    h = 1.96 * statistics.pstdev(per) / (n ** 0.5) if n > 1 else 0.0
    return f"{m:.2f}±{h:.2f} (n{n})".rjust(16)


def main():
    comps = sorted({Path(p).name for p in glob.glob(f"{ROOT}/default/*/*")})
    for comp in comps:
        print("=" * 78)
        print(comp)
        print("=" * 78)
        print(f"{'judge':28}{'default':>16}{'style-blind':>16}")
        judges = sorted({Path(p).parent.name
                         for v in VARIANTS for p in glob.glob(f"{ROOT}/{v}/*/{comp}")})
        pooled = {v: [] for v in VARIANTS}
        for judge in judges:
            row = f"{judge:28}"
            for v in VARIANTS:
                logs = glob.glob(f"{ROOT}/{v}/{judge}/{comp}/*.eval")
                per = item_picks(sorted(logs)[-1]) if logs else []
                pooled[v] += per
                row += fmt(per)
            print(row)
        row = f"{'POOLED':28}"
        for v in VARIANTS:
            row += fmt(pooled[v])
        print(row)
        print()


if __name__ == "__main__":
    main()
