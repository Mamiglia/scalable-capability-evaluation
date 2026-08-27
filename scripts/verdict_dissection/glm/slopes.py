"""Population slopes with posterior intervals from the hibayes fits, averaged over held-out designs.

    uv run python scripts/verdict_dissection/glm/slopes.py <grid_dir> [--designs ...]

Reads <grid_dir>/fits/<model>/<wave>/<design>/communicate/model_<model>_summary.csv and the config's
predictor list; writes <grid_dir>/summary/slopes_<model>_<wave>.csv with the posterior mean and 94 %
HDI bounds averaged over the designs, plus the worst r_hat and total divergences across them.
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

SLOPES = {"capability_from_transcript": "benchmark_slopes_population",
          "stronger_model_from_pair": "benchmark_slopes_population",
          "judge_verdict_from_pair": "judge_slopes_population"}


def read_fit(fit_dir, model, predictors):
    """One design: predictor -> mean, hdi_3%, hdi_97%, r_hat (plus the per-judge gap slopes when fitted)."""
    summary = pd.read_csv(fit_dir / "communicate" / f"model_{model}_summary.csv", index_col=0)
    coords = json.loads((fit_dir / "coords.json").read_text())
    rows = {}
    for name, value in summary.iterrows():
        if name.startswith(SLOPES[model] + "["):
            rows[predictors[int(name.split("[")[1].rstrip("]"))]] = value
        elif name.startswith("judge_accuracy_gap_slopes["):
            rows["accuracy gap, " + coords["judge"][int(name.split("[")[1].rstrip("]"))]] = value
    table = pd.DataFrame(rows).T[["mean", "hdi_3%", "hdi_97%", "r_hat"]]
    diagnostics = json.loads((fit_dir / "models" / model / "diagnostics" / "diagnostics.json").read_text())
    return table, diagnostics["divergences"], summary.r_hat.max()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("grid_dir", type=Path)
    parser.add_argument("--designs", nargs="+", default=[f"new-models-and-items-{r}" for r in range(4)])
    args = parser.parse_args()
    out_dir = args.grid_dir / "summary"
    out_dir.mkdir(exist_ok=True)
    for model_dir in sorted((args.grid_dir / "fits").glob("*")):
        model = model_dir.name
        for wave_dir in sorted(model_dir.glob("*")):
            config = yaml.safe_load((args.grid_dir / "configs" / model / f"{wave_dir.name}/{args.designs[0]}.yaml").read_text())
            predictors = config["model"]["models"][0]["config"]["predictors"]
            fits = [read_fit(wave_dir / design, model, predictors) for design in args.designs
                    if (wave_dir / design / "coords.json").exists()]
            if not fits:
                continue
            table = pd.concat([t for t, _, _ in fits]).groupby(level=0).mean()
            table["r_hat"] = max(r for _, _, r in fits)
            table = table.rename(columns={"r_hat": "worst r_hat in fit"}).assign(
                divergences=sum(d for _, d, _ in fits), designs=len(fits))
            table = table.reindex(table["mean"].abs().sort_values(ascending=False).index)
            table.round(3).to_csv(out_dir / f"slopes_{model}_{wave_dir.name}.csv", index_label="predictor")
            print(f"\n{model} / {wave_dir.name}: {len(fits)} designs, worst r_hat {table['worst r_hat in fit'].iloc[0]:.2f}, "
                  f"divergences {table.divergences.iloc[0]}")
            print(table[["mean", "hdi_3%", "hdi_97%"]].head(12).round(2).to_string())


if __name__ == "__main__":
    main()
