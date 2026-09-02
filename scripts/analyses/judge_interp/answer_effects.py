"""What steering does to answers the judge writes itself.

Reads the steered-answering round and reports, per condition: how far
the answer moved along the steered axis, that movement as a fraction of
the natural top-decile-vs-bottom-decile elo gap on the same axis
(steering strength in units of a real quality difference), benchmark
correctness, and naturalness. Movement is paired within item and seed,
so sampling noise cancels; the unsteered condition supplies the
seed-to-seed spread that any effect has to beat.

Writes out_dir/answer_effects.csv.

Usage:
  uv run python scripts/analyses/judge_interp/answer_effects.py \
      --answers <dir with answers_shard*.jsonl>
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]


def unit(v):
    return v / np.linalg.norm(v)


def degeneration(text: str, n: int = 8):
    """Repetition, not improbability, is how steered text breaks.

    Mean log-prob cannot see it: a loop of one repeated line is highly
    predictable and scores near zero. These do see it - the share of
    n-gram positions that are repeats, and the longest run of identical
    lines. Also flags reasoning blocks, which analysis excludes.
    """
    words = text.split()
    grams = [tuple(words[i:i + n]) for i in range(max(len(words) - n + 1, 0))]
    repeat = 1 - len(set(grams)) / len(grams) if grams else 0.0
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    run = best = 1
    for a, b in zip(lines, lines[1:]):
        run = run + 1 if a == b else 1
        best = max(best, run)
    return dict(repeat_frac=repeat, max_line_run=best if lines else 0,
                has_think="</think>" in text or "<think>" in text)


def elo_gaps(cfg, out: Path):
    """Top-decile minus bottom-decile projection, per axis, on real transcripts."""
    meta = json.loads((out / "acts_qwen3-32b.meta.json").read_text())
    bt = np.load(out / "bt_direction.npz")
    capd = np.load(out / "capability_directions.npz")
    li = meta["layers"].index(int(bt["layer"]))
    pi = meta["poolings"].index(str(bt["pooling"]))
    acts = np.load(out / "acts_qwen3-32b.npy", mmap_mode="r")
    z = (np.asarray(acts[:, li, pi, :], dtype=np.float32) - bt["mean"]) / bt["std"]
    v_cap, w_qwen = unit(capd["v_cap"]), bt["w_qwen"]
    axes = {"approval": unit(bt["w"]),
            "v_pref_judge": unit(w_qwen - (v_cap @ w_qwen) * v_cap),
            "v_cap": v_cap}
    elo = pd.read_csv(out / "elo.csv")
    dropped = set(meta["dropped_entities"])
    elo = elo[[f"{t}_{i}" not in dropped
               for t, i in zip(elo["text_id"], elo["item"])]].reset_index(drop=True)
    e = elo["elo"].to_numpy()
    hi, lo = e >= np.percentile(e, 90), e <= np.percentile(e, 10)
    return {name: float((z @ ax)[hi].mean() - (z @ ax)[lo].mean())
            for name, ax in axes.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "config.yaml")
    ap.add_argument("--answers", type=Path, required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    out = ROOT / cfg["out_dir"]

    rows = []
    for f in glob.glob(str(args.answers / "answers_shard*.jsonl")):
        rows += [json.loads(l) for l in open(f)]
    if not rows:
        raise SystemExit(f"no answers under {args.answers}")
    gaps = elo_gaps(cfg, out)
    print(f"{len(rows)} generations, {len(set(r['sample_id'] for r in rows))} items")
    print("natural top-decile vs bottom-decile gap: "
          + ", ".join(f"{k} {v:.2f}" for k, v in gaps.items()))

    base = {(r["sample_id"], r["seed"]): r for r in rows if r["axis"] == "none"}
    unsteered = [r for r in rows if r["axis"] == "none"]
    # Seed-to-seed spread of the unsteered condition: the noise floor.
    spread = {}
    for axis in gaps:
        per_item = {}
        for r in unsteered:
            per_item.setdefault(r["sample_id"], []).append(r["reread_scores"][axis])
        diffs = [max(v) - min(v) for v in per_item.values() if len(v) > 1]
        spread[axis] = float(np.mean(diffs)) if diffs else float("nan")
    print("unsteered seed-to-seed spread: "
          + ", ".join(f"{k} {v:.2f}" for k, v in spread.items()))

    recs = []
    for r in rows:
        if r["axis"] == "none":
            continue
        b = base.get((r["sample_id"], r["seed"]))
        if b is None:
            continue
        read = ("v_pref_judge" if r["axis"] in ("random_a", "none")
                else ("v_pref_orth" if r["axis"] == "v_pref_gen" else r["axis"]))
        deg = degeneration(r["answer"])
        recs.append(dict(axis=r["axis"], alpha=r["alpha"],
                         layers="26+28+30" if r.get("multi") else "26",
                         read_axis=read,
                         d_proj=r["reread_scores"][read] - b["reread_scores"][read],
                         correct=r["correct"], parsed=r["parsed"],
                         n_chars=r["n_chars"],
                         base_correct=b["correct"], **deg,
                         broken=deg["repeat_frac"] > 0.5 or deg["max_line_run"] >= 5))
    df = pd.DataFrame(recs)
    print("\nintact answers only (dropping repetition loops and reasoning blocks);"
          " 'broken' column counts what was dropped")
    print(f"\n{'axis':14s} {'alpha':>6} {'layers':>9} {'n':>4} {'score shift':>16} "
          f"{'x elo gap':>10} {'correct':>8} {'base':>6} {'broken':>7}")
    table = []
    for (axis, alpha, layers), g in df.groupby(["axis", "alpha", "layers"]):
        ok = g[~g["broken"] & ~g["has_think"]]
        gap = gaps.get(g["read_axis"].iloc[0], np.nan)
        se = ok["d_proj"].std() / np.sqrt(max(len(ok), 1))
        row = dict(axis=axis, alpha=alpha, layers=layers, n_all=len(g), n_intact=len(ok),
                   d_proj=ok["d_proj"].mean(), d_proj_se=se,
                   gap_fraction=ok["d_proj"].mean() / gap,
                   correct=ok["correct"].mean(), base_correct=ok["base_correct"].mean(),
                   broken_frac=g["broken"].mean(), think_frac=g["has_think"].mean())
        table.append(row)
        print(f"{axis:14s} {alpha:+6.2f} {layers:>9} {len(ok):4d} {row['d_proj']:+8.2f} "
              f"+-{se:5.2f} {row['gap_fraction']:+9.2f}x {row['correct']:7.1%} "
              f"{row['base_correct']:5.1%} {row['broken_frac']:6.1%}")
    path = out / "answer_effects.csv"
    pd.DataFrame(table).to_csv(path, index=False)
    print(f"-> {path}")


if __name__ == "__main__":
    main()
