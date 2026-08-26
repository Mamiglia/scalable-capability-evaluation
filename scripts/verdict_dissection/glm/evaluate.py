"""Held-out evaluation of a fitted hibayes GLM on the confirmation items.

    uv run python scripts/verdict_dissection/glm/evaluate.py <config.yaml> <hibayes_out_dir>

Draws posterior-predictive outcomes for the test rows (hibayes stores them but never scores
them), averages over draws, and reports accuracy and log loss for the Bernoulli models or
RMSE and Spearman correlation for the Gaussian one, overall and per benchmark or judge. For
the capability model it also scores at model level: predictions averaged over each held-out
model's transcripts, Spearman against true accuracy across models within each benchmark (rows
"models: <benchmark>") and across all (model, benchmark) pairs after centring by benchmark.
"""
import argparse
import pickle
import sys
from pathlib import Path

import arviz as az
import jax
import numpy as np
import pandas as pd
import yaml
from numpyro.infer import Predictive
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
import model as glm_models  # noqa: E402


def predict(model_name, predictors, model_dir, num_draws=200):
    """Mean posterior-predictive outcome per test row."""
    posterior = az.from_netcdf(model_dir / "inference_data.nc").posterior
    draws = {name: posterior[name].values.reshape(-1, *posterior[name].shape[2:])[:num_draws]
             for name in posterior.data_vars}
    test_features = pickle.load(open(model_dir / "test_features.pkl", "rb"))
    observed = np.asarray(test_features["obs"])
    statistical_model = getattr(glm_models, model_name)(predictors=predictors)
    predictive = Predictive(statistical_model, posterior_samples=draws)
    predicted = predictive(jax.random.PRNGKey(0), test_features | {"obs": None})["obs"]
    return observed, np.asarray(predicted).mean(axis=0), test_features


def report(observed, predicted, groups, gaussian):
    rows = []
    for name, mask in [("all", np.ones_like(observed, bool))] + list(groups.items()):
        if gaussian:
            error = observed[mask] - predicted[mask]
            rows.append({"group": name, "n": int(mask.sum()), "rmse": float(np.sqrt(np.mean(error ** 2))),
                         "spearman": float(spearmanr(observed[mask], predicted[mask]).statistic)})
        else:
            probability = np.clip(predicted[mask], 1e-6, 1 - 1e-6)
            rows.append({"group": name, "n": int(mask.sum()),
                         "accuracy": float(np.mean((probability > 0.5) == observed[mask])),
                         "log_loss": float(-np.mean(observed[mask] * np.log(probability)
                                                    + (1 - observed[mask]) * np.log(1 - probability)))})
    return pd.DataFrame(rows)


def model_level(observed, predicted, test_rows):
    """Spearman between per-model mean prediction and truth, within benchmark and benchmark-centred overall."""
    frame = pd.DataFrame({"bench": test_rows.bench.to_numpy(), "model": test_rows.model.to_numpy(),
                          "observed": observed, "predicted": predicted})
    means = frame.groupby(["bench", "model"], as_index=False)[["observed", "predicted"]].mean()
    rows = []
    for bench, group in means.groupby("bench"):
        rows.append({"group": f"models: {bench}", "n": len(group),
                     "spearman": float(spearmanr(group.observed, group.predicted).statistic) if len(group) > 2 else np.nan})
    centred = means[["observed", "predicted"]] - means.groupby("bench")[["observed", "predicted"]].transform("mean")
    rows.append({"group": "models: all benchmarks, centred", "n": len(means),
                 "spearman": float(spearmanr(centred.observed, centred.predicted).statistic)})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())["model"]["models"][0]
    model_name, predictors = config["name"], config["config"]["predictors"]
    coords = yaml.safe_load((args.out_dir / "coords.json").read_text())

    observed, predicted, test_features = predict(model_name, predictors, args.out_dir / "models" / model_name)
    group_column = "judge" if model_name == "judge_verdict_from_pair" else "bench"
    group_index = np.asarray(test_features[f"{group_column}_index"])
    groups = {label: group_index == i for i, label in enumerate(coords[group_column])}
    table = report(observed, predicted, groups, gaussian=model_name == "capability_from_transcript")
    if model_name == "capability_from_transcript":
        input_path = Path(yaml.safe_load(args.config.read_text())["data_loader"]["paths"]["extracted_data"][0])
        test_rows = pd.read_parquet(input_path, columns=["bench", "model", "test"]).query("test")   # hibayes keeps this order
        if len(test_rows) != len(observed):
            raise ValueError(f"{len(test_rows)} test rows in the input but {len(observed)} test outcomes")
        table = pd.concat([table, model_level(observed, predicted, test_rows)], ignore_index=True)
    table.to_csv(args.out_dir / f"held_out_{model_name}.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
