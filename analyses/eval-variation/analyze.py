"""Collect eval-variation verdicts into one csv + a per-condition summary.

Conditions read from logs/eval-variation/<cond>/<judge>/<major>|<minor>/.
The baseline row set comes from the cheap-7 judge logs
(logs/data-expansion/gpqa/<judge>/<A>-vs-<B>/phase-*/), reoriented by ECI and
subset to the same +-50-word length-matched items.

Usage: analyze.py [--out judge_results/eval_prompt_variation/verdicts.csv]
"""

import argparse
import csv
import math
import sys
from pathlib import Path

import yaml
from inspect_ai.log import read_eval_log

sys.path.insert(0, str(Path(__file__).parent))
from gen_configs import (CONDITIONS, REWRITE_CONDITIONS, TONE_CONDITIONS,  # noqa: E402
                         SUBJECTS_BY_ECI, resolve_log)
from run import first_correct_samples, words  # noqa: E402

# pooled matching uses CONDITIONS only
ALL_CONDITIONS = CONDITIONS | TONE_CONDITIONS | REWRITE_CONDITIONS
ECI = dict(SUBJECTS_BY_ECI)
BASELINE_ROOT = "logs/data-expansion/gpqa"
COND_ROOT = "logs/eval-variation"
MAX_WORD_DIFF = 50


def wordcounts(cond: str, model: str) -> dict:
    return {i: words(s) for i, s in first_correct_samples(resolve_log(ALL_CONDITIONS[cond], model)).items()}


def pooled_lengths(major: str, minor: str) -> dict:
    """item -> (wc_major, wc_minor) under the pooled combo-selection rule
    (min diff, same-condition preferred - mirrors run.pooled_task)."""
    conds = list(CONDITIONS)
    a = {c: wordcounts(c, major) for c in conds}
    b = {c: wordcounts(c, minor) for c in conds}
    out = {}
    for i in set().union(*a.values()) & set().union(*b.values()):
        combos = [(abs(a[ca][i] - b[cb][i]), ca != cb, ca, cb)
                  for ca in conds for cb in conds if i in a[ca] and i in b[cb]]
        diff, _, ca, cb = min(combos)
        if diff <= MAX_WORD_DIFF:
            out[i] = (a[ca][i], b[cb][i])
    return out


def read_pair_rows(pair_dir: Path, cond, judge, major, minor, lengths) -> list:
    rows = []
    for lp in pair_dir.rglob("*.eval"):
        log = read_eval_log(str(lp))
        if log.status != "success":
            continue
        for s in log.samples or []:
            sc = list(s.scores.values())[0] if s.scores else None
            if not (sc and sc.answer in ("1", "2")):
                continue
            item = s.id.rsplit("_", 1)[0]
            if item not in lengths:
                continue
            picked_major = sc.answer == s.target
            wc_maj, wc_min = lengths[item]
            picked_len = wc_maj if picked_major else wc_min
            other_len = wc_min if picked_major else wc_maj
            rows.append({
                "condition": cond, "judge": judge, "major": major, "minor": minor,
                "item": item, "major_position": s.target,
                "picked_major": int(picked_major),
                "picked_t1": int(sc.answer == "1"),
                "picked_longer": int(picked_len > other_len) if picked_len != other_len else "",
                "confidence": (sc.metadata or {}).get("confidence", ""),
            })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("judge_results/eval_prompt_variation/verdicts.csv"))
    args = ap.parse_args()

    names = [m for m, _ in SUBJECTS_BY_ECI]
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
    rows = []

    # instructed + pooled conditions (dirs are stronger-first already)
    for cond_dir in sorted(Path(COND_ROOT).iterdir()):
        cond = cond_dir.name
        for judge_dir in sorted(p for p in cond_dir.iterdir() if p.is_dir()):
            for major, minor in pairs:
                pd = judge_dir / f"{major}|{minor}"
                if not pd.exists():
                    continue
                if cond == "lenmatch-pooled":
                    lengths = pooled_lengths(major, minor)
                else:
                    wa, wb = wordcounts(cond, major), wordcounts(cond, minor)
                    lengths = {i: (wa[i], wb[i]) for i in wa.keys() & wb.keys()
                               if abs(wa[i] - wb[i]) <= MAX_WORD_DIFF}
                rows += read_pair_rows(pd, cond, judge_dir.name, major, minor, lengths)

    # baseline from the cheap-7 logs, ECI-reoriented, matched subset only
    wa_all = {m: wordcounts("baseline", m) for m in names}
    for judge_dir in sorted(p for p in Path(BASELINE_ROOT).iterdir() if p.is_dir()):
        for major, minor in pairs:  # major = stronger by ECI
            wa, wb = wa_all[major], wa_all[minor]
            lengths = {i: (wa[i], wb[i]) for i in wa.keys() & wb.keys()
                       if abs(wa[i] - wb[i]) <= MAX_WORD_DIFF}
            for dirname, flipped in ((f"{major}-vs-{minor}", False), (f"{minor}-vs-{major}", True)):
                pd = judge_dir / dirname
                if not pd.exists():
                    continue
                if not flipped:
                    rows += read_pair_rows(pd, "baseline", judge_dir.name, major, minor, lengths)
                else:
                    flipped_lengths = {i: (b, a) for i, (a, b) in lengths.items()}
                    got = read_pair_rows(pd, "baseline", judge_dir.name, minor, major, flipped_lengths)
                    for r in got:  # picked_major currently means picked-the-weaker
                        r["major"], r["minor"] = major, minor
                        r["picked_major"] = 1 - r["picked_major"]
                        r["major_position"] = "2" if r["major_position"] == "1" else "1"
                    rows += got

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} verdict rows -> {args.out}\n")

    def agg(rs, key):
        n = len(rs)
        p = sum(r[key] for r in rs if r[key] != "") / max(1, sum(1 for r in rs if r[key] != ""))
        return n, p

    print(f"{'condition':18} {'judge':28} {'n':>6} {'P(stronger)':>11} {'P(T1)':>6} {'P(longer)':>9}")
    conds = sorted({r["condition"] for r in rows})
    for cond in conds:
        crows = [r for r in rows if r["condition"] == cond]
        for judge in sorted({r["judge"] for r in crows}):
            jr = [r for r in crows if r["judge"] == judge]
            n, ps = agg(jr, "picked_major")
            _, p1 = agg(jr, "picked_t1")
            _, pl = agg(jr, "picked_longer")
            print(f"{cond:18} {judge:28} {n:6d} {ps:11.3f} {p1:6.3f} {pl:9.3f}")
        n, ps = agg(crows, "picked_major")
        _, p1 = agg(crows, "picked_t1")
        _, pl = agg(crows, "picked_longer")
        se = math.sqrt(max(ps * (1 - ps), 1e-9) / max(n, 1))
        print(f"{cond:18} {'ALL':28} {n:6d} {ps:11.3f} {p1:6.3f} {pl:9.3f}  (+-{1.96*se:.3f})\n")

    # same-judge pivot: each judge compared with itself across conditions
    order = [c for c in ("baseline", "words200", "words50-100", "lenmatch-pooled",
                         "confident", "matteroffact", "rewrite-self",
                         "rewrite-standard") if c in conds]
    judges = sorted({r["judge"] for r in rows})
    by = {(r["judge"], r["condition"]): [] for r in rows}
    for r in rows:
        by[(r["judge"], r["condition"])].append(r)
    for metric, label in (("picked_major", "P(pick stronger)"), ("picked_longer", "P(pick longer)")):
        print(f"same-judge {label}:")
        print(f"{'judge':28} " + " ".join(f"{c:>16}" for c in order))
        for j in judges:
            cells = []
            for c in order:
                jr = by.get((j, c), [])
                cells.append(f"{agg(jr, metric)[1]:6.3f} (n={len(jr):5d})" if jr else f"{'-':>16}")
            print(f"{j:28} " + " ".join(f"{x:>16}" for x in cells))
        shared = [j for j in judges if all(by.get((j, c)) for c in order)]
        cells = []
        for c in order:
            sr = [r for j in shared for r in by[(j, c)]]
            cells.append(f"{agg(sr, metric)[1]:6.3f} (n={len(sr):5d})")
        print(f"{'ALL shared (' + str(len(shared)) + ' judges)':28} " + " ".join(f"{x:>16}" for x in cells) + "\n")


if __name__ == "__main__":
    main()
