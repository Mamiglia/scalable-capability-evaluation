"""Summarise gpqa-pilot-rd2 verdicts: per-run stats, controls, per-judge gap fits.

Reads verdicts.csv and models.csv, writes summary.csv and judge_fits.csv,
prints a console report. Score is the counterbalanced per-item mean: the two
A/B orders of an item average to 0, 0.5 or 1, which cancels position bias.

Reported per run:
  * score with mean +/- 1.96*SEM CI and a permutation p against chance
  * slot-A pick rate (Wilson CI) and order-disagreement rate
  * no-answer rate and mean confidence
  * self-pair runs: slot-A CI must cover 0.5 (no position bias)

Per judge: score = sigmoid(a + d * gap) fitted over the gap_ladder runs;
sigma(a) above 0.5 with clean self-pairs indicates fingerprinting.

Usage:
  uv run python scripts/analyses/analyse.py
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from comparisons import COMPARISONS
from stats import (counterbalanced, drop_degenerate, perm_p, sem_ci, sigmoid_fit,
                   stars, wilson)


def run_groups() -> dict[tuple, list[str]]:
    g: dict[tuple, list[str]] = {}
    for name, triplets in COMPARISONS.items():
        for t in triplets:
            g.setdefault(t, []).append(name)
    return g


def summarise_run(run_df: pd.DataFrame) -> dict:
    scores = counterbalanced(run_df)
    m, lo, hi = sem_ci(scores)
    p = perm_p(scores)

    parsed = run_df.dropna(subset=["picked_major"])
    k_a = int((parsed["picked"] == "1").sum())
    sa_lo, sa_hi = wilson(k_a, len(parsed))

    # Disagreement = the two parsed orders of an item pick different slots'
    # models (counterbalanced score exactly 0.5).
    two = parsed.groupby("item").size()
    both = parsed[parsed["item"].isin(two[two == 2].index)]
    pair_scores = counterbalanced(both)
    disagree = float(np.isclose(pair_scores, 0.5).mean()) if len(pair_scores) else np.nan

    # Score a pick-the-longer-transcript strategy would get on this run.
    lens = run_df.dropna(subset=["len_major", "len_minor"]).groupby("item")[["len_major", "len_minor"]].first()
    len_baseline = float(np.where(lens.len_major == lens.len_minor, 0.5,
                                  lens.len_major > lens.len_minor).mean()) if len(lens) else np.nan

    return dict(
        n_items=len(scores), n_rows=len(run_df),
        score=m, score_lo=lo, score_hi=hi,
        perm_p=p, stars=stars(p),
        slot_a=k_a / len(parsed) if len(parsed) else np.nan,
        slot_a_lo=sa_lo, slot_a_hi=sa_hi,
        disagree_rate=disagree,
        noanswer_rate=float(run_df["picked_major"].isna().mean()),
        mean_conf=float(run_df["confidence"].mean()),
        len_baseline=len_baseline,
    )


def judge_fit(df: pd.DataFrame, judge: str, gaps: pd.Series) -> dict | None:
    """Sigmoid fit over the judge's neutral cross-model Arm A runs (ladder +
    panel + weak-judge), one point per item. Temporal and wide-gap runs are
    displayed on the curve but not fitted."""
    panel = {t for k in ("gap_ladder", "judge_strength", "weak_judge")
             for t in COMPARISONS[k]}
    ladder = [t for t in panel if t[0] == judge]
    g, y = [], []
    for j, maj, mnr in ladder:
        run = df[(df.judge == j) & (df.major == maj) & (df.minor == mnr)]
        s = counterbalanced(run)
        g.extend([gaps[(maj, mnr)]] * len(s))
        y.extend(s)
    fit = sigmoid_fit(g, y)
    if fit is None:
        return None
    a, d, se_a, se_d = fit
    return dict(judge=judge, n_runs=len(ladder), n_items=len(y),
                a=a, d=d, se_a=se_a, se_d=se_d,
                acc_at_gap0=1 / (1 + np.exp(-a)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, default=Path("judge_results/gpqa-pilot-rd2"))
    args = ap.parse_args()

    df = pd.read_csv(args.dir / "verdicts.csv")
    df, n_degenerate = drop_degenerate(df)
    if n_degenerate:
        print(f"dropped {n_degenerate} rows on items where a transcript is answer-only")
    models = pd.read_csv(args.dir / "models.csv").set_index("model")
    groups = run_groups()

    # Gaps on both scales; raw accuracy is the headline axis, logit kept for
    # gap-matching against the run plan.
    acc, logit = models["accuracy"], models["logit"]
    gaps = {(maj, mnr): (0.0 if maj == mnr else acc[maj] - acc[mnr])
            for maj, mnr in df.groupby(["major", "minor"]).groups}

    # Truncation-control runs are separate rows, keyed by truncate_frac.
    rows = []
    for (judge, maj, mnr, tf), run in df.groupby(
            ["judge", "major", "minor", "truncate_frac"], dropna=False):
        entry = dict(judge=judge, major=maj, minor=mnr, truncate_frac=tf,
                     groups=",".join(groups.get((judge, maj, mnr), [])),
                     d_acc=gaps[(maj, mnr)],
                     d_logit=0.0 if maj == mnr else logit[maj] - logit[mnr])
        entry.update(summarise_run(run))
        is_self = (judge, maj, mnr) in set(map(tuple, COMPARISONS["self_pairs"]))
        entry["slot_ok"] = (entry["slot_a_lo"] <= 0.5 <= entry["slot_a_hi"]) if is_self else None
        rows.append(entry)
    summary = pd.DataFrame(rows).sort_values(["judge", "d_acc"])

    full = df[df["truncate_frac"].isna()]
    fits = pd.DataFrame([f for j in summary["judge"].unique()
                         if (f := judge_fit(full, j, gaps)) is not None])

    summary.to_csv(args.dir / "summary.csv", index=False)
    fits.to_csv(args.dir / "judge_fits.csv", index=False)

    report = summary[summary["truncate_frac"].isna()]
    n_trunc_runs = len(summary) - len(report)
    print("=" * 78)
    print(f"GPQA-PILOT-RD2 RESULTS ({n_trunc_runs} truncation-control runs in "
          "summary.csv, not printed)")
    print("=" * 78)
    for judge, block in report.groupby("judge", sort=True):
        print(f"\n{judge}")
        for _, e in block.iterrows():
            tag = f" [{e.groups}]" if e.groups else ""
            print(f"  {e.major} vs {e.minor}{tag}")
            print(f"    score {e.score:.3f} [{e.score_lo:.3f}, {e.score_hi:.3f}] "
                  f"(n={e.n_items}, dAcc {e.d_acc:+.3f}) {e.stars}")
            flag = ""
            if e.slot_ok is not None:
                flag = "  ok (CI covers 0.5)" if e.slot_ok else "  *** slot bias: CI excludes 0.5 ***"
            print(f"    slot-A {e.slot_a:.2f} [{e.slot_a_lo:.2f}, {e.slot_a_hi:.2f}]"
                  f" | disagree {e.disagree_rate:.2f} | no-answer {e.noanswer_rate:.2f}"
                  f" | conf {e.mean_conf:.1f}{flag}")
        f = fits[fits.judge == judge]
        if len(f):
            f = f.iloc[0]
            print(f"  gap fit ({f.n_runs} ladder runs, {f.n_items} items): "
                  f"d={f.d:.2f}, acc at gap 0 = {f.acc_at_gap0:.3f}")

    print(f"\n{len(summary)} runs -> {args.dir}/summary.csv; "
          f"{len(fits)} judge fits -> {args.dir}/judge_fits.csv")


if __name__ == "__main__":
    main()
