"""Reader-steering: does moving the judge's residual stream move its verdict?

Reads the pod_judge_steer.py verdict shards and, for every steered record,
measures the ANSWER logit-gap shift toward the steered transcript against that
(pair, order)'s own unsteered baseline. Reports per (axis, mode, alpha): mean
shift with a bootstrap CI, % flips, % chose steered; each fitted direction sits
against random_a at the matched dose. Runs on whatever shards are present.

Metric (per steered record, vs its baseline; gap = logit_1 - logit_2):
  slot 1: shift = gap_steered - gap_baseline ; chose = gap_steered > 0
  slot 2: shift = gap_baseline - gap_steered ; chose = gap_steered < 0
  flip   = argmax moved from not-steered-slot at baseline to steered-slot now.
Positive shift = steering pulled the verdict toward the steered slot.

Usage:
  uv run python scripts/analyses/judge_interp/readsteer_verdicts.py
  uv run python scripts/analyses/judge_interp/readsteer_verdicts.py --self-test
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from plots import MUTED, PALETTE, chance, legend, save, style  # noqa: E402  (sets Agg)
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_VERDICTS = ROOT / "judge_results/judge_interp/readsteer_ml_260807"
FITTED = ["v_cap", "v_pref_gen", "v_pref_gen_orth", "v_pref_spec", "v_pref_spec_orth"]


def record_metric(rec, base):
    """(shift_toward_steered, chose_steered, flip_toward_steered) for one record."""
    gap_b = base["logit_1"] - base["logit_2"]
    gap_s = rec["logit_1"] - rec["logit_2"]
    if rec["slot"] == 1:
        shift, chose, chose_b = gap_s - gap_b, gap_s > 0, gap_b > 0
    else:
        shift, chose, chose_b = gap_b - gap_s, gap_s < 0, gap_b < 0
    return shift, int(chose), int(chose and not chose_b)


def build_long(records):
    """Records -> per-steered-record rows joined to their (pair, order) baseline."""
    base = {}
    for r in records:
        if r["axis"] == "none":
            base.setdefault((r["item"], r["text_id_1"], r["text_id_2"], r["order"]), r)
    rows, orphan = [], 0
    for r in records:
        if r["axis"] == "none":
            continue
        b = base.get((r["item"], r["text_id_1"], r["text_id_2"], r["order"]))
        if b is None:
            orphan += 1
            continue
        shift, chose, flip = record_metric(r, b)
        rows.append(dict(axis=r["axis"], mode=r.get("mode", "single"),
                         alpha=r["alpha"], slot=r["slot"],
                         shift=shift, chose=chose, flip=flip))
    return pd.DataFrame(rows), orphan, len(base)


def boot_ci(vals, rng, n_boot, min_n):
    """2000-sample bootstrap 95% CI of the mean; (nan, nan) if under min_n."""
    vals = np.asarray(vals, float)
    if len(vals) < min_n:
        return np.nan, np.nan
    idx = rng.integers(0, len(vals), size=(n_boot, len(vals)))
    means = vals[idx].mean(axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]))


def aggregate(df, n_boot=2000, min_n=3, seed=0):
    """Group by (axis, mode, alpha); mean shift + CI, % flip, % chose. Control matched."""
    rng = np.random.default_rng(seed)
    recs = []
    for (axis, mode, alpha), g in sorted(df.groupby(["axis", "mode", "alpha"]),
                                         key=lambda kv: kv[0]):
        lo, hi = boot_ci(g["shift"].to_numpy(), rng, n_boot, min_n)
        recs.append(dict(axis=axis, mode=mode, alpha=alpha, n=len(g),
                         shift=g["shift"].mean(), lo=lo, hi=hi,
                         pct_flip=g["flip"].mean(), pct_chose=g["chose"].mean()))
    agg = pd.DataFrame(recs)
    # matched random_a control at the same (mode, signed alpha)
    ctrl = {(r.mode, r.alpha): r.shift for r in agg[agg.axis == "random_a"].itertuples()}
    agg["ctrl_shift"] = [ctrl.get((m, a), np.nan) for m, a in zip(agg["mode"], agg["alpha"])]
    agg["_abs"] = agg["alpha"].abs()
    return (agg.sort_values(["mode", "_abs", "alpha", "axis"])
            .drop(columns="_abs").reset_index(drop=True))


def print_table(agg):
    """Gate-result shape: fitted rows carry their matched random_a shift inline."""
    print(f"\n{'axis':16s} {'mode':6s} {'alpha':>6} {'n':>4}  "
          f"{'shift toward steered':>26}  {'%flip':>6} {'%chose':>7} {'rand':>7}")
    for r in agg.itertuples():
        ci = "  (n<ci)" if np.isnan(r.lo) else f"[{r.lo:+.3f}, {r.hi:+.3f}]"
        ctrl = "   -   " if np.isnan(r.ctrl_shift) else f"{r.ctrl_shift:+.3f}"
        print(f"{r.axis:16s} {r.mode:6s} {r.alpha:+6.2f} {r.n:4d}  "
              f"{r.shift:+8.3f} {ci:17s}  {r.pct_flip:5.1%} {r.pct_chose:6.1%} {ctrl:>7}")


def fig_readsteer(agg, out_stub):
    """Grid: rows = {shift, %chose}, cols = 5 directions; single/block + control."""
    fig, axes = plt.subplots(2, len(FITTED), figsize=(2.5 * len(FITTED), 5.2),
                             sharex=True)
    rand = agg[agg.axis == "random_a"]

    def draw(ax, sub, color, ls, label, ycol, err):
        sub = sub.sort_values("alpha")
        if sub.empty:
            return
        y = sub[ycol].to_numpy()
        yerr = None
        if err:
            e = np.vstack([y - sub["lo"], sub["hi"] - y])
            yerr = np.nan_to_num(e, nan=0.0)
        ax.errorbar(sub["alpha"], y, yerr=yerr, color=color, ls=ls, marker="o",
                    ms=3.5, lw=1.6, capsize=2, label=label)

    for c, name in enumerate(FITTED):
        d = agg[agg.axis == name]
        for rowi, (ycol, err, ylab, ch) in enumerate(
                [("shift", True, "shift toward steered (logits)", 0.0),
                 ("pct_chose", False, "chose steered", 0.5)]):
            ax = axes[rowi, c]
            draw(ax, d[d["mode"] == "single"], PALETTE[0], "-", "fitted, single", ycol, err)
            draw(ax, d[d["mode"] == "block"], PALETTE[0], "--", "fitted, block", ycol, err)
            draw(ax, rand[rand["mode"] == "single"], MUTED, "-", "random_a, single", ycol, err)
            draw(ax, rand[rand["mode"] == "block"], MUTED, "--", "random_a, block", ycol, err)
            ax.set_xscale("symlog", linthresh=0.05)
            ax.set_xticks([-1, -0.1, 0, 0.1, 1])
            ax.set_xticklabels(["-1", "-.1", "0", ".1", "1"])
            chance(ax, ch)
            style(ax)
            if rowi == 0:
                ax.set_title(name, fontsize=8)
            if c == 0:
                ax.set_ylabel(ylab)
            if rowi == 1:
                ax.set_xlabel(r"$\alpha$ (weak$\to$strong steps, symlog)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        legend(axes[0, -1], loc="best")
    fig.suptitle("Reader-steering: verdict shift and % chose steered vs dose",
                 fontsize=9)
    save(fig, out_stub)
    print(f"-> {out_stub}.png")


def verdict(agg):
    """Per-direction headline: dose-monotone, right sign, clears random control?"""
    print("\nVerdict (does the corrected direction causally steer the judge?):")
    rand = {(r.mode, r.alpha): r for r in agg[agg.axis == "random_a"].itertuples()}
    move = {}
    for name in FITTED:
        d = agg[agg.axis == name]
        if d.empty:
            print(f"  {name:16s} no data")
            continue
        for mode in ["single", "block"]:
            dm = d[d["mode"] == mode].sort_values("alpha")
            if len(dm) < 2:
                continue
            corr = np.corrcoef(dm["alpha"], dm["shift"])[0, 1]
            hi = dm.loc[dm["alpha"].abs().idxmax()]
            move[(name, mode)] = abs(hi["shift"])
            # clears control: fitted CI at strongest matched dose excludes rand mean
            note = ""
            r = rand.get((mode, hi["alpha"]))
            if r is not None and not np.isnan(hi["lo"]):
                clears = hi["lo"] > r.shift if hi["alpha"] > 0 else hi["hi"] < r.shift
                note = f"vs random {r.shift:+.3f} at a={hi['alpha']:+.2f}: " \
                       f"{'CLEARS' if clears else 'within'}"
            sign = "+" if corr > 0 else "-"
            print(f"  {name:16s} {mode:6s} dose-corr {corr:+.2f} ({sign}sign)  "
                  f"peak |shift| {abs(hi['shift']):.3f}  {note}")
    if move:
        s = sum(v for (n, m), v in move.items() if m == "single")
        b = sum(v for (n, m), v in move.items() if m == "block")
        print(f"  single vs block (summed peak |shift|): "
              f"single {s:.2f}  block {b:.2f}  -> "
              f"{'single' if s > b else 'block'} moves more")


def load_records(verdicts_dir):
    files = sorted(glob.glob(str(Path(verdicts_dir) / "*.jsonl")))
    records, bad = [], 0
    for f in files:
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                bad += 1
    return records, files, bad


def run(verdicts_dir, out_dir, n_boot, min_n, seed):
    records, files, bad = load_records(verdicts_dir)
    print(f"shards present: {[Path(f).name for f in files] or 'NONE'}"
          f"  ({len(records)} records{f', {bad} unparsed' if bad else ''})")
    if not records:
        print(f"no verdict data under {verdicts_dir} yet — nothing to analyze.")
        return
    df, orphan, n_pairs = build_long(records)
    print(f"{len(df)} steered records over {n_pairs} (pair, order) baselines"
          f"{f'; {orphan} steered records had no baseline (skipped)' if orphan else ''}")
    if df.empty:
        print("only baselines present so far — no steered conditions to score yet.")
        return
    agg = aggregate(df, n_boot=n_boot, min_n=min_n, seed=seed)
    print_table(agg)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv = out_dir / "readsteer_ml_summary.csv"
    agg.to_csv(csv, index=False)
    print(f"\n-> {csv}")
    fig_readsteer(agg, str(out_dir / "readsteer_ml"))
    verdict(agg)


def self_test():
    """Inject a known +0.8 shift for one axis; assert the metric recovers it."""
    import tempfile
    recs = []
    for pair in ("A", "B"):
        for order in (0, 1):
            key = dict(item="q", text_id_1=pair + "1", text_id_2=pair + "2", order=order)
            recs.append(dict(**key, axis="none", mode="single", alpha=0, slot=0,
                             logit_1=0.1, logit_2=-0.1))            # baseline gap +0.2
            # slot 1 steered: shift +0.8 -> gap 1.0
            recs.append(dict(**key, axis="v_test", mode="single", alpha=0.5, slot=1,
                             logit_1=0.5, logit_2=-0.5))
            # slot 2 steered: shift +0.8 -> gap -0.6
            recs.append(dict(**key, axis="v_test", mode="single", alpha=0.5, slot=2,
                             logit_1=-0.3, logit_2=0.3))
    with tempfile.TemporaryDirectory() as d:
        Path(d, "verdicts_shard0.jsonl").write_text(
            "\n".join(json.dumps(r) for r in recs))
        records, _, _ = load_records(d)
    df, orphan, _ = build_long(records)
    assert orphan == 0, orphan
    agg = aggregate(df, min_n=3)
    row = agg[(agg.axis == "v_test") & (agg.alpha == 0.5)].iloc[0]
    assert abs(row["shift"] - 0.8) < 1e-9, row["shift"]      # magnitude + sign
    assert row["shift"] > 0
    assert abs(row["pct_chose"] - 1.0) < 1e-9, row["pct_chose"]   # both slots chose steered
    assert abs(row["pct_flip"] - 0.5) < 1e-9, row["pct_flip"]     # slot-2 rows flip only
    assert row["n"] == 8, row["n"]                                # 4 pair-orders x 2 slots
    print(f"self-test OK: recovered shift {row['shift']:+.3f} (injected +0.800), "
          f"%chose {row['pct_chose']:.0%}, %flip {row['pct_flip']:.0%}, n={row['n']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verdicts", type=Path, default=DEFAULT_VERDICTS,
                    help="dir of verdicts_shard*.jsonl")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "judge_results/judge_interp/figures",
                    help="csv + figure destination")
    ap.add_argument("--boots", type=int, default=2000)
    ap.add_argument("--min-ci-n", type=int, default=3, help="skip CI below this n")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return
    run(args.verdicts, args.out_dir, args.boots, args.min_ci_n, args.seed)


if __name__ == "__main__":
    main()
