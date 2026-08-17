#!/usr/bin/env python3
"""Fit the existing HiBayes judge-feature GLM to the new HF frame on SWC."""

import argparse
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from numpyro.diagnostics import summary
from numpyro.infer import MCMC, NUTS

PREDICTORS = ["length_difference_sd", "capability_difference_sd", "same_model_family"]


def judge_feature_model(features, predictors=PREDICTORS):
    """The model in model.py, without HiBayes' orchestration wrappers."""
    num_judges = features["num_judges"]
    judge = features["judge_index"]

    position_mean = numpyro.sample("position_effect_mean", dist.Normal(0, 1.5))
    position_scale = numpyro.sample("position_effect_scale", dist.HalfNormal(0.5))
    position_raw = numpyro.sample(
        "judge_position_effects_raw", dist.Normal(0, 1).expand([num_judges]).to_event(1)
    )
    intercepts = position_mean + position_scale * position_raw
    numpyro.deterministic("judge_intercepts", intercepts)

    effect_mean = numpyro.sample(
        "predictor_effect_mean", dist.Normal(0, 0.5).expand([len(predictors)]).to_event(1)
    )
    effect_scale = numpyro.sample(
        "predictor_effect_scale", dist.HalfNormal(0.5).expand([len(predictors)]).to_event(1)
    )
    effect_raw = numpyro.sample(
        "judge_predictor_effects_raw",
        dist.Normal(0, 1).expand([num_judges, len(predictors)]).to_event(2),
    )
    effects = effect_mean + effect_scale * effect_raw
    numpyro.deterministic("judge_predictor_effects", effects)

    condition_free = numpyro.sample(
        "judge_condition_effects_free",
        dist.Normal(0, 0.5).expand([num_judges, features["num_conditions"] - 1]).to_event(2),
    )
    condition_effects = jnp.concatenate([jnp.zeros((num_judges, 1)), condition_free], axis=1)

    item_sigma = numpyro.sample("item_sigma", dist.HalfNormal(0.5))
    item_raw = numpyro.sample(
        "item_effects_raw", dist.Normal(0, 1).expand([features["num_items"]]).to_event(1)
    )
    design = jnp.stack([features[name] for name in predictors], axis=1)
    logits = (
        intercepts[judge]
        + jnp.sum(effects[judge] * design, axis=1)
        + condition_effects[judge, features["condition_index"]]
        + item_sigma * item_raw[features["item_index"]]
    )
    numpyro.sample("obs", dist.Bernoulli(logits=logits), obs=features["obs"])


def encode(frame):
    judge, judges = pd.factorize(frame["judge"], sort=True)
    item, items = pd.factorize(frame["item"], sort=True)
    condition, conditions = pd.factorize(frame["transcript_condition"], sort=True)
    features = {
        "obs": jnp.asarray(frame["chosen_left"].to_numpy(np.int8)),
        "judge_index": jnp.asarray(judge),
        "item_index": jnp.asarray(item),
        "condition_index": jnp.asarray(condition),
        "num_judges": len(judges),
        "num_items": len(items),
        "num_conditions": len(conditions),
    }
    features.update({name: jnp.asarray(frame[name].to_numpy(float)) for name in PREDICTORS})
    return features, list(judges), list(conditions)


def effect_row(level, judge, predictor, draws):
    values = np.asarray(draws).reshape(-1)
    lo, hi = np.quantile(values, [0.025, 0.975])
    return {
        "level": level,
        "judge": judge,
        "predictor": predictor,
        "mean_log_odds": values.mean(),
        "hdi_2.5%": lo,
        "hdi_97.5%": hi,
        "p_effect_gt_0": (values > 0).mean(),
        "odds_ratio": np.exp(values.mean()),
    }


def atomic_csv(frame, path):
    tmp = path.with_suffix(".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def atomic_npz(samples, path):
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **{name: np.asarray(values) for name, values in samples.items()})
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=2000)
    parser.add_argument("--samples", type=int, default=2500)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "DONE").exists():
        print(f"Already complete: {args.out}")
        return

    frame = pd.read_csv(args.data)
    features, judges, conditions = encode(frame)
    print(f"rows={len(frame):,} judges={judges} conditions={conditions} devices={jax.devices()}")
    if len(jax.devices("cpu")) < 4:
        raise RuntimeError("four CPU devices are required for parallel chains")

    kernel = NUTS(judge_feature_model, target_accept_prob=0.95)
    mcmc = MCMC(
        kernel, num_warmup=args.warmup, num_samples=args.samples,
        num_chains=4, chain_method="parallel", progress_bar=True,
    )
    mcmc.run(jax.random.key(0), features)
    samples = mcmc.get_samples(group_by_chain=True)
    atomic_npz(samples, args.out / "posterior.npz")
    population = np.asarray(samples["predictor_effect_mean"])
    judge_effects = (
        population[:, :, None, :]
        + np.asarray(samples["predictor_effect_scale"])[:, :, None, :]
        * np.asarray(samples["judge_predictor_effects_raw"])
    )
    rows = [
        effect_row("population", "", predictor, population[:, :, index])
        for index, predictor in enumerate(PREDICTORS)
    ]
    rows += [
        effect_row("judge", judge, predictor, judge_effects[:, :, judge_index, predictor_index])
        for judge_index, judge in enumerate(judges)
        for predictor_index, predictor in enumerate(PREDICTORS)
    ]
    atomic_csv(pd.DataFrame(rows), args.out / "effects.csv")

    diagnostics = summary(samples, prob=0.95, group_by_chain=True)
    r_hats = np.concatenate([np.asarray(value["r_hat"]).reshape(-1) for value in diagnostics.values()])
    n_eff = np.concatenate([np.asarray(value["n_eff"]).reshape(-1) for value in diagnostics.values()])
    report = {
        "rows": len(frame), "judges": judges, "conditions": conditions,
        "warmup": args.warmup, "samples_per_chain": args.samples, "chains": 4,
        "divergences": int(np.asarray(mcmc.get_extra_fields(group_by_chain=True)["diverging"]).sum()),
        "max_r_hat": float(np.nanmax(r_hats)), "min_n_eff": float(np.nanmin(n_eff)),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    tmp = args.out / "diagnostics.tmp"
    tmp.write_text(json.dumps(report, indent=2))
    tmp.replace(args.out / "diagnostics.json")
    (args.out / "DONE").touch()
    print(json.dumps(report, indent=2))
    print(pd.DataFrame(rows).query("level == 'population'").to_string(index=False))


if __name__ == "__main__":
    main()
