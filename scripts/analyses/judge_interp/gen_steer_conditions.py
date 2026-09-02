"""Emit the coarse Phase-2 reader-steering condition grid as a JSON list.

5 real directions x {single, block} x alpha ladder (both signs) x slot {1,2},
plus a matched random_a control single+block at +-{.5,1}. alpha=0 is the pod's
built-in unsteered baseline (identical to steering any axis by 0), so it is the
shared anchor for every titration curve and is not re-emitted per axis.

Usage:
  uv run python gen_steer_conditions.py [--out conditions.json]
"""

import argparse
import json
import sys
from pathlib import Path

NAMES5 = ["v_cap", "v_pref_gen", "v_pref_gen_orth", "v_pref_spec", "v_pref_spec_orth"]
MODES = ["single", "block"]
SLOTS = [1, 2]
ALPHAS = [0.01, 0.05, 0.1, 0.5, 1.0]   # nonzero ladder; 0 is the baseline forward
RANDOM_ALPHAS = [0.5, 1.0]
SIGNS = [1, -1]


def conditions():
    out = []
    for name in NAMES5:
        for mode in MODES:
            for slot in SLOTS:
                for a in ALPHAS:
                    for s in SIGNS:
                        out.append({"axis": name, "mode": mode,
                                    "alpha": round(s * a, 4), "slot": slot})
    for mode in MODES:
        for slot in SLOTS:
            for a in RANDOM_ALPHAS:
                for s in SIGNS:
                    out.append({"axis": "random_a", "mode": mode,
                                "alpha": round(s * a, 4), "slot": slot})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, help="write JSON here; default stdout")
    args = ap.parse_args()
    conds = conditions()
    js = json.dumps(conds)
    if args.out:
        args.out.write_text(js)
        print(f"{len(conds)} conditions -> {args.out}", file=sys.stderr)
    else:
        print(js)


if __name__ == "__main__":
    main()
