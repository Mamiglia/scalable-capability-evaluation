"""Collect the ablation grid's held-out scores into one table per model.

    uv run python scripts/verdict_dissection/glm/collate.py <fits_dir> <output_dir>

For each model: one row per ablation with the overall held-out score and its change from the
full model (accuracy for the Bernoulli models, Spearman for the capability model), plus the
per-group scores as columns.
"""
import argparse
from pathlib import Path

import pandas as pd


def collate(fits_dir, model):
    rows = []
    for path in sorted(fits_dir.glob(f"{model}/*/held_out_{model}.csv")):
        scores = pd.read_csv(path).set_index("group")
        metric = "spearman" if "spearman" in scores else "accuracy"
        rows.append({"ablation": path.parent.name, "metric": metric, **scores[metric].to_dict()})
    table = pd.DataFrame(rows)
    if table.empty or "full" not in set(table.ablation):
        return table
    full = table.loc[table.ablation == "full", "all"].iloc[0]
    table.insert(2, "change_from_full", table["all"] - full)
    return table.sort_values("change_from_full")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fits_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for model_dir in sorted(p for p in args.fits_dir.iterdir() if p.is_dir()):
        table = collate(args.fits_dir, model_dir.name)
        table.to_csv(args.output_dir / f"{model_dir.name}.csv", index=False)
        print(f"{model_dir.name}: {len(table)} ablations")
        if "change_from_full" in table:
            print(table[["ablation", "all", "change_from_full"]].head(8).to_string(index=False))


if __name__ == "__main__":
    main()
