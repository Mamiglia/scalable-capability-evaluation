"""How often judges pick the steered answer over the unsteered one.

Each comparison shows a judge two answers to the same question - one
written under steering, one not - with matched outcomes, in both slot
orders. The steered answer is the "major" side, so a judge marked
correct on a sample is a judge that picked the steered answer, and the
mean score is the share chosen.

Reported per judge and dose against the 50% null, with the
unsteered-replicate arm (two unsteered draws) as the sanity check and
the random direction as the control. A dose worth expanding shows the
positive arm above 50% and its mirror below; both arms on the same side
is generic damage, not steering.

Writes out_dir/steered_verdicts.csv.

Usage:
  uv run python scripts/analyses/judge_interp/steered_verdicts.py \
      --logs logs/judge-interp-steered
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from inspect_ai.log import read_eval_log

ROOT = Path(__file__).resolve().parents[3]


def parse_condition(name: str):
    """v_pref_judge_p0.05-vs-unsteered -> (axis, +0.05)."""
    cond = name.split("-vs-")[0]
    m = re.match(r"(.+?)_([pm])([0-9.]+)$", cond)
    if not m:
        return cond, 0.0
    axis, sign, mag = m.groups()
    return axis, (1 if sign == "p" else -1) * float(mag)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "config.yaml")
    ap.add_argument("--logs", type=Path, required=True)
    ap.add_argument("--boots", type=int, default=5000)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    out = ROOT / cfg["out_dir"]

    rows = []
    for eval_file in sorted(args.logs.rglob("*.eval")):
        judge = eval_file.relative_to(args.logs).parts[0]
        pair = eval_file.relative_to(args.logs).parts[1]
        axis, alpha = parse_condition(pair)
        try:
            log = read_eval_log(str(eval_file))
        except Exception:
            continue
        if not log.samples:
            continue
        for s in log.samples:
            score = getattr(s, "score", None)
            if score is None or score.value not in ("C", "I"):
                continue
            rows.append(dict(judge=judge, axis=axis, alpha=alpha,
                             chose_steered=score.value == "C"))
    if not rows:
        raise SystemExit(f"no scored verdicts under {args.logs}")
    df = pd.DataFrame(rows)
    print(f"{len(df)} verdicts, {df['judge'].nunique()} judges, "
          f"{df.groupby(['axis', 'alpha']).ngroups} conditions")

    rng = np.random.default_rng(0)
    table = []
    print(f"\n{'judge':22s} {'axis':14s} {'alpha':>7} {'n':>4} {'chose steered':>22}")
    for (judge, axis, alpha), g in df.groupby(["judge", "axis", "alpha"]):
        v = g["chose_steered"].to_numpy().astype(float)
        boot = [rng.choice(v, len(v)).mean() for _ in range(args.boots)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        table.append(dict(judge=judge, axis=axis, alpha=alpha, n=len(v),
                          share=v.mean(), lo=lo, hi=hi))
        print(f"{judge:22s} {axis:14s} {alpha:+7.3f} {len(v):4d} "
              f"{v.mean():10.1%} [{lo:.0%}, {hi:.0%}]")

    # Pooled dose trend: per-condition counts are small, so the question the
    # sweep can answer is whether the share rises with signed dose.
    print("\npooled across doses, steered axis only:")
    for judge, g in df[df["axis"] == "v_pref_judge"].groupby("judge"):
        pos = g[g["alpha"] > 0]["chose_steered"].mean()
        neg = g[g["alpha"] < 0]["chose_steered"].mean()
        r = np.corrcoef(g["alpha"], g["chose_steered"].astype(float))[0, 1]
        print(f"  {judge:22s} alpha>0 {pos:.1%} (n={sum(g['alpha'] > 0)})  "
              f"alpha<0 {neg:.1%} (n={sum(g['alpha'] < 0)})  "
              f"dose correlation {r:+.3f}")
    path = out / "steered_verdicts.csv"
    pd.DataFrame(table).to_csv(path, index=False)
    print(f"-> {path}")


if __name__ == "__main__":
    main()
