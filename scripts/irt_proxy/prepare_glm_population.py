"""Prepare bias-correction inputs for the judging-variations population.

Votes come from the fitted GLM frame (verdict features already measured),
responses from the Epoch GPQA log, and anchors from scoring each model
against the frozen fixed-3PL item bank. The frozen reference is never
refitted.
"""
import argparse
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

from inputs import canonical_benchmark, canonical_model  # noqa: E402

STYLE_FEATURES = (
    "personal_pronouns", "hedging", "backtracking",
    "uncertainty", "confidence", "questions",
)


def build_votes(frame: pd.DataFrame) -> pd.DataFrame:
    votes = pd.DataFrame({
        "source": "judging-variations",
        "benchmark": "gpqa",
        "condition": frame["transcript_condition"],
        "artifact": frame["model_pair"],
        "round": 0,
        "judge": frame["judge"],
        "item_id": frame["item"],
        "order": frame["order"],
        "left_model": frame["left_model"].map(canonical_model),
        "right_model": frame["right_model"].map(canonical_model),
        "left_epoch": 0,
        "right_epoch": 0,
        "chosen_left": frame["chosen_left"].astype(float),
        "delta_log_chars": frame["length_difference"],
    })
    for name in STYLE_FEATURES:
        votes[f"delta_{name}"] = frame[f"{name}_difference"]
    return votes


def build_responses(epoch: pd.DataFrame, models: set[str]) -> pd.DataFrame:
    epoch = epoch.copy()
    epoch["model"] = epoch["model"].map(canonical_model)
    epoch["benchmark"] = epoch["benchmark"].map(canonical_benchmark)
    responses = epoch[epoch["model"].isin(models)].copy()
    responses["source"] = "epoch"
    responses["source_priority"] = 0
    responses["source_path"] = ""
    responses["transcript"] = np.nan
    return responses


def score_anchors(
    responses: pd.DataFrame, items: pd.DataFrame,
    warmup: int, samples: int, seed: int,
) -> pd.DataFrame:
    """Posterior theta per model with the item bank frozen (not an IRT fit)."""
    import jax
    import numpyro
    from jax import random
    from numpyro import distributions as dist
    from numpyro.diagnostics import summary
    from numpyro.infer import MCMC, NUTS

    bank = items.set_index("item")
    scored = responses[responses["item"].isin(bank.index)]
    models = sorted(scored["model"].unique())
    model_index = scored["model"].map(
        {name: index for index, name in enumerate(models)},
    ).to_numpy()
    discrimination = bank["discrimination"].loc[scored["item"]].to_numpy()
    difficulty = bank["difficulty"].loc[scored["item"]].to_numpy()
    floor = bank["lower_asymptote"].loc[scored["item"]].to_numpy()
    correct = scored["correct"].astype(float).to_numpy()

    def probability_model():
        theta = numpyro.sample("theta", dist.Normal(0, 1).expand([len(models)]))
        logit = discrimination * (theta[model_index] - difficulty)
        probability = floor + (1 - floor) * jax.nn.sigmoid(logit)
        numpyro.sample("obs", dist.Bernoulli(probs=probability), obs=correct)

    mcmc = MCMC(
        NUTS(probability_model), num_warmup=warmup, num_samples=samples,
        num_chains=4, chain_method="parallel",
    )
    mcmc.run(random.PRNGKey(seed))
    grouped = mcmc.get_samples(group_by_chain=True)
    stats = summary(grouped, group_by_chain=True)["theta"]
    print(f"anchor scoring: max r_hat {np.nanmax(stats['r_hat']):.5f}, "
          f"min n_eff {np.nanmin(stats['n_eff']):.1f}")
    theta = np.asarray(mcmc.get_samples()["theta"])
    return pd.DataFrame({
        "irt_model": "fixed_3pl", "fit": "gpqa", "model": models,
        "theta": theta.mean(axis=0),
        "theta_p5": np.percentile(theta, 5, axis=0),
        "theta_p95": np.percentile(theta, 95, axis=0),
    })


def assign_folds(responses: pd.DataFrame, items: pd.DataFrame, n_folds: int = 5) -> pd.DataFrame:
    """Deal models into folds round-robin by easy-half accuracy for balance."""
    easy_items = set(items.loc[items["half"].eq("easy"), "item"])
    easy = responses[responses["item"].isin(easy_items)]
    accuracy = easy.groupby("model")["correct"].mean().sort_values()
    return pd.DataFrame({
        "model": accuracy.index,
        "fold": [index % n_folds for index in range(len(accuracy))],
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glm-frame", type=Path, required=True)
    parser.add_argument("--epoch-responses", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True,
                        help="frozen reference dir (items.csv, posterior.npz)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=2000)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    frame = pd.read_csv(args.glm_frame)
    votes = build_votes(frame)
    models = set(votes["left_model"]) | set(votes["right_model"])
    print(f"votes: {len(votes)} rows, {len(models)} models, "
          f"{votes['judge'].nunique()} judges")

    epoch = pd.read_parquet(args.epoch_responses)
    responses = build_responses(epoch, models)
    missing = models - set(responses["model"])
    if missing:
        raise ValueError(f"models without responses: {sorted(missing)}")
    print(f"responses: {len(responses)} rows")

    all_items = pd.read_csv(args.reference / "items.csv")
    items = all_items[all_items["irt_model"].eq("fixed_3pl")]
    anchors = score_anchors(
        responses, items, args.warmup, args.samples, args.seed,
    )
    folds = assign_folds(responses, items)
    print("fold sizes:", folds.groupby("fold")["model"].count().tolist())

    input_dir = args.output / "input"
    reference_dir = args.output / "reference"
    input_dir.mkdir(parents=True, exist_ok=True)
    reference_dir.mkdir(parents=True, exist_ok=True)
    votes.to_parquet(input_dir / "votes.parquet", index=False)
    responses.to_parquet(input_dir / "responses.parquet", index=False)
    folds.to_parquet(input_dir / "folds.parquet", index=False)
    anchors.to_csv(reference_dir / "anchors.csv", index=False)
    for name in ("items.csv", "posterior.npz", "diagnostics.csv"):
        shutil.copy(args.reference / name, reference_dir / name)
    print(f"written to {args.output}")


if __name__ == "__main__":
    main()
