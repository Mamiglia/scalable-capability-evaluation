"""Gather held-out scores across waves and designs into one table per model.

    uv run python scripts/verdict_dissection/glm/collate.py <fits_dir> <output_dir>

Reads <fits_dir>/<model>/<wave>/<design>/held_out_<model>.csv. Writes <output_dir>/<model>.csv
with one row per (wave, design) holding the overall and per-group scores, and prints a summary
per wave: each held-out family as mean ± sd over its four folds.
"""
import argparse
from pathlib import Path

import pandas as pd


def collate(fits_dir, model):
    rows = []
    for path in sorted(fits_dir.glob(f"{model}/*/*/held_out_{model}.csv")):
        scores = pd.read_csv(path).set_index("group")
        metric = "spearman" if "spearman" in scores else "accuracy"
        rows.append({"wave": path.parents[1].name, "design": path.parent.name, "metric": metric,
                     **scores[metric].to_dict()})
    return pd.DataFrame(rows)


def summary(table):
    """Overall score per wave: folds averaged with their sd, bands as they are."""
    table = table.copy()
    table["family"] = table.design.str.replace(r"-\d+$", "", regex=True)       # model-fold-2 -> model-fold
    grouped = table.groupby(["wave", "family"])["all"].agg(["mean", "std", "size"])
    grouped["score"] = grouped.apply(lambda r: f"{r['mean']:.3f}" + (f" ± {r['std']:.3f}" if r["size"] > 1 else ""), axis=1)
    return grouped.score.unstack("family")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fits_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for model_dir in sorted(p for p in args.fits_dir.iterdir() if p.is_dir()):
        table = collate(args.fits_dir, model_dir.name)
        if table.empty:
            continue
        table.to_csv(args.output_dir / f"{model_dir.name}.csv", index=False)
        print(f"\n{model_dir.name} ({table.metric.iloc[0]}), {len(table)} fits")
        print(summary(table).to_string())


if __name__ == "__main__":
    main()
