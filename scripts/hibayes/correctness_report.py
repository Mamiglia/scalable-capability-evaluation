#!/usr/bin/env python3
"""Summarise the correctness-model arms: coefficients, LOO accuracy, ablation shifts.

Run with the hibayes venv, which has arviz and h5netcdf:
  analysis/hibayes/.venv/bin/python scripts/hibayes/correctness_report.py \
    analysis/hibayes/run_correctness_full analysis/hibayes/run_correctness_minus_capability ...

The first run given is the reference every other arm is compared against.
Reads each run's posterior group lazily; the log-likelihood in the same file is
tens of gigabytes and is never needed here.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

INTERVAL = (0.025, 0.975)
QUARTILES = 4


def load_arm(run: Path) -> dict:
    """Posterior draws, predictor names and pointwise elpd for the one model in a run."""
    import xarray as xr  # only the hibayes venv has it; nothing else in this module needs it

    models = sorted((run / "models").iterdir())
    if len(models) != 1:
        raise ValueError(f"{run} holds {len(models)} models; expected exactly one")
    model = models[0]
    config = json.loads((model / "model_config.json").read_text())
    predictors = list(config["extra_kwargs"]["predictors"])
    coords = json.loads((model / "coords.json").read_text())
    posterior = xr.open_dataset(model / "inference_data.nc", group="posterior", engine="h5netcdf")
    elpd = pd.read_csv(model / "diagnostics" / "inference_data_loo_i.csv")["loo_i"].to_numpy()
    draws = {
        name: posterior[name].to_numpy().reshape(-1, *posterior[name].shape[2:])
        for name in ("judge_intercepts", "judge_predictor_effects", "item_effects",
                     "predictor_effect_mean", "predictor_effect_scale")
    }
    if draws["judge_predictor_effects"].shape[-1] != len(predictors):
        raise ValueError(
            f"{run}: {draws['judge_predictor_effects'].shape[-1]} fitted effects but "
            f"{len(predictors)} predictor names; every label would be wrong"
        )
    posterior.close()
    # older runs saved a null tag, so fall back to the directory name rather than
    # labelling a whole arm NaN
    return {"run": run, "name": model.name, "tag": config.get("tag") or model.name,
            "predictors": predictors, "judges": coords["judge"], "items": coords["item"],
            "elpd": elpd, **draws}


def summarise(samples: np.ndarray) -> dict:
    low, high = np.quantile(samples, INTERVAL)
    return {"mean": float(samples.mean()), "low": float(low), "high": float(high),
            "p_gt_0": float((samples > 0).mean())}


def effects(arm: dict) -> pd.DataFrame:
    """Population mean, between-judge scale and each judge's effect, per predictor."""
    rows = []
    for index, predictor in enumerate(arm["predictors"]):
        rows.append({"arm": arm["tag"], "predictor": predictor, "level": "population",
                     **summarise(arm["predictor_effect_mean"][:, index])})
        rows.append({"arm": arm["tag"], "predictor": predictor, "level": "between_judge_scale",
                     **summarise(arm["predictor_effect_scale"][:, index])})
        for judge_index, judge in enumerate(arm["judges"]):
            rows.append({"arm": arm["tag"], "predictor": predictor, "level": judge,
                         **summarise(arm["judge_predictor_effects"][:, judge_index, index])})
    for judge_index, judge in enumerate(arm["judges"]):
        rows.append({"arm": arm["tag"], "predictor": "intercept", "level": judge,
                     **summarise(arm["judge_intercepts"][:, judge_index])})
    frame = pd.DataFrame(rows)
    frame["credible_nonzero"] = (frame["low"] > 0) | (frame["high"] < 0)
    return frame


def loo_accuracy(arm: dict) -> float:
    """Leave-one-observation-out accuracy: exp(elpd_i) is P(observed outcome)."""
    return float((np.exp(arm["elpd"]) > 0.5).mean())


def arm_summary(arms: list[dict]) -> pd.DataFrame:
    reference = arms[0]
    rows = []
    for arm in arms:
        difference = arm["elpd"] - reference["elpd"]
        n = len(difference)
        rows.append({
            "arm": arm["tag"],
            "predictors": len(arm["predictors"]),
            "dropped": ",".join(p for p in reference["predictors"] if p not in arm["predictors"]) or "-",
            "elpd_loo": float(arm["elpd"].sum()),
            "loo_accuracy": loo_accuracy(arm),
            "delta_elpd_vs_reference": float(difference.sum()),
            # Vehtari, Gelman & Gabry (2017): the paired standard error is sqrt(n)
            # times the standard deviation of the pointwise gaps.
            "delta_elpd_se": float(np.sqrt(n) * difference.std(ddof=1)) if arm is not reference else 0.0,
            "delta_accuracy_vs_reference": loo_accuracy(arm) - loo_accuracy(reference),
        })
    return pd.DataFrame(rows)


def shifts(arms: list[dict]) -> pd.DataFrame:
    """Population coefficient of every surviving predictor against the reference arm."""
    reference = arms[0]
    baseline = {p: reference["predictor_effect_mean"][:, i].mean()
                for i, p in enumerate(reference["predictors"])}
    rows = []
    for arm in arms[1:]:
        for index, predictor in enumerate(arm["predictors"]):
            value = float(arm["predictor_effect_mean"][:, index].mean())
            rows.append({
                "arm": arm["tag"], "predictor": predictor,
                "reference": baseline[predictor], "arm_value": value,
                "shift": value - baseline[predictor],
                "sign_change": bool(np.sign(value) != np.sign(baseline[predictor])),
            })
    return pd.DataFrame(rows)


def quartile_pattern(arm: dict, frame: pd.DataFrame, draws: int, seed: int) -> pd.DataFrame:
    """Model-implied accuracy when the stronger model is shown second minus shown first.

    Per judge and total-length quartile, so it lines up with the raw marginal table.
    Only the position terms are flipped; every other column is order-invariant.
    """
    if "truth_side" not in arm["predictors"]:
        return pd.DataFrame()
    design = frame[arm["predictors"]].to_numpy(dtype=float)
    position = arm["predictors"].index("truth_side")
    # Both configurations are built from scratch rather than by flipping each row away
    # from the order it happens to be stored in: half the rows are already stored with
    # the stronger model second, and flipping those would label them backwards. The two
    # halves then cancel and the difference comes out exactly zero.
    stronger_first, stronger_second = design.copy(), design.copy()
    stronger_first[:, position], stronger_second[:, position] = 1.0, -1.0
    if "truth_side_x_total_length" in arm["predictors"]:
        interaction = arm["predictors"].index("truth_side_x_total_length")
        length = design[:, interaction] * design[:, position]  # truth_side is +/-1
        stronger_first[:, interaction], stronger_second[:, interaction] = length, -length
    judge_index = pd.Index(arm["judges"]).get_indexer(frame["judge"])
    item_index = pd.Index(arm["items"]).get_indexer(frame["item"])
    if (judge_index < 0).any() or (item_index < 0).any():
        raise ValueError("The frame holds judges or items the fit never saw")

    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(arm["judge_intercepts"]), size=draws, replace=False)
    first = np.empty((draws, len(frame)))
    second = np.empty((draws, len(frame)))
    for row, draw in enumerate(chosen):
        effect = arm["judge_predictor_effects"][draw][judge_index]
        base = (arm["judge_intercepts"][draw][judge_index]
                + arm["item_effects"][draw][item_index])
        first[row] = 1.0 / (1.0 + np.exp(-(base + np.einsum("ij,ij->i", effect, stronger_first))))
        second[row] = 1.0 / (1.0 + np.exp(-(base + np.einsum("ij,ij->i", effect, stronger_second))))

    quartile = pd.qcut(frame["total_length"], QUARTILES, labels=[f"Q{i + 1}" for i in range(QUARTILES)])
    rows = []
    for judge in arm["judges"]:
        for label in quartile.cat.categories:
            mask = ((frame["judge"] == judge) & (quartile == label)).to_numpy()
            if not mask.any():
                continue
            # every row appears in both orders, so the two arms average over the same cells
            per_draw = second[:, mask].mean(axis=1) - first[:, mask].mean(axis=1)
            rows.append({"arm": arm["tag"], "judge": judge, "quartile": label,
                         "n": int(mask.sum()), **summarise(per_draw)})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path, help="Run directories; the first is the reference")
    parser.add_argument("--frame", type=Path, default=Path("analysis/hibayes/gpqa_correctness.csv"))
    parser.add_argument("--out", type=Path, default=Path("analysis/hibayes"))
    parser.add_argument("--prefix", default="correctness")
    parser.add_argument("--quartile-draws", type=int, default=200,
                        help="Posterior draws used for the model-implied quartile table")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    arms = [load_arm(run) for run in args.runs]
    frame = pd.read_csv(args.frame)
    if len(frame) != len(arms[0]["elpd"]):
        raise ValueError(f"{len(frame)} frame rows but {len(arms[0]['elpd'])} fitted observations")

    tables = {
        "arms": arm_summary(arms),
        "effects": pd.concat([effects(arm) for arm in arms], ignore_index=True),
        "shifts": shifts(arms),
        "quartiles": pd.concat(
            [quartile_pattern(arm, frame, args.quartile_draws, args.seed) for arm in arms],
            ignore_index=True,
        ),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        if table.empty:
            continue
        path = args.out / f"{args.prefix}_{name}.csv"
        table.to_csv(path, index=False)
        print(f"\n=== {name} -> {path} ===")
        print(table.to_string(index=False, max_rows=200))


if __name__ == "__main__":
    main()
