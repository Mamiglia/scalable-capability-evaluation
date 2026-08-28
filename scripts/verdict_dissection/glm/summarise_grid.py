"""Summarise a screening grid: held-out accuracy per wave and benchmark, per-judge share of the
judges' self-consistency range, and the drop-one-block ablation table.

    uv run python scripts/verdict_dissection/glm/summarise_grid.py judge_results/verdict_dissection/glm_screen_all/fits [--designs new-models-and-items-0 ...]

Reads predictions_<model>.csv under <fits>/<model>/<wave>/<design>/ (pair models) and
held_out_<model>.csv (capability, model-level centred Spearman row). Writes <fits>/../summary/<model>.csv
and prints the tables. The self-consistency ceilings are the rollout-0 -> rollout-1 agreement per judge
and benchmark (within_model contrasts, 2026-08-27).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CEILING = pd.DataFrame(
    {"aime": [.765, np.nan, np.nan, .718, .694, np.nan, .645], "gpqa": [.785, .741, .741, .737, .736, .657, .620],
     "intercode": [.708, .709, .724, .698, .696, .612, .626], "swe_bench_mini": [.826, .828, .768, .778, .754, .626, .546]},
    index=["gpt-oss-120b", "deepseek-r1-0528", "o4-mini", "gemma-4-31b-it", "deepseek-v4-flash-0731", "Llama-3.1-8B-Instruct", "ling-2.6-flash"])
STRONG = CEILING.index[:5]


def pair_table(fits, model, designs):
    rows = []
    for wave_dir in sorted((fits / model).iterdir()):
        frames = [pd.read_csv(p) for d in designs for p in [wave_dir / d / f"predictions_{model}.csv"] if p.exists()]
        if not frames:
            continue
        p = pd.concat(frames)
        p["hit"] = ((p.predicted > 0.5).astype(int) == p.observed)
        by_bench = p.groupby("bench").hit.mean()
        row = {"wave": wave_dir.name, "folds": len(frames), **by_bench.round(3).to_dict(), "mean": round(by_bench.mean(), 3)}
        if model == "judge_verdict_from_pair":
            acc = p.pivot_table(index="judge", columns="bench", values="hit", aggfunc="mean").reindex(CEILING.index)
            share = ((acc - 0.5) / (CEILING - 0.5)).clip(upper=1)
            row["share_strong"] = round(100 * share.loc[STRONG].mean().mean())
            row |= {f"share_{b}": round(100 * share[b].mean()) for b in share.columns}
        rows.append(row)
    return pd.DataFrame(rows).set_index("wave")


def capability_table(fits, model, designs):
    rows = []
    for wave_dir in sorted((fits / model).iterdir()):
        scores = []
        for d in designs:
            path = wave_dir / d / f"held_out_{model}.csv"
            if path.exists():
                h = pd.read_csv(path)
                centred = h[h.group.astype(str).str.contains("centred", case=False)] if "group" in h else h
                scores.append(centred.iloc[0].get("spearman", np.nan) if len(centred) else np.nan)
        if scores:
            rows.append({"wave": wave_dir.name, "folds": len(scores), "spearman_mean": round(np.nanmean(scores), 3)})
    return pd.DataFrame(rows).set_index("wave")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fits", type=Path)
    parser.add_argument("--designs", nargs="+", default=[f"new-models-and-items-{r}" for r in range(4)])
    args = parser.parse_args()
    out = args.fits.parent / "summary"
    out.mkdir(exist_ok=True)
    for model in ["judge_verdict_from_pair", "stronger_model_from_pair", "capability_from_transcript"]:
        if not (args.fits / model).exists():
            continue
        table = capability_table(args.fits, model, args.designs) if model == "capability_from_transcript" else pair_table(args.fits, model, args.designs)
        table.to_csv(out / f"{model}.csv")
        print(f"\n===== {model} ({', '.join(args.designs)})")
        order = [w for w in table.index if not w.startswith("all-minus")] + [w for w in table.index if w.startswith("all-minus")]
        print(table.loc[order].to_string())


if __name__ == "__main__":
    main()
