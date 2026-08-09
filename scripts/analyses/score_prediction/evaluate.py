"""Evaluate all score-prediction models on the shared CV folds.

Reads preds/{regime}_{fold}.parquet; writes metrics.csv (per regime x
fold x model, plus pooled rows with item-bootstrap CIs); prints the
console report: pooled nll/acc/ece per model per regime, delta-nll vs
the two baselines, and the b_j cross-check against self-pair slot-1
rates (skipped with a note if checkpoints are not there yet).

Usage:
  uv run python scripts/analyses/score_prediction/evaluate.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from ml_suite import log_loss

sys.path.append(str(Path(__file__).resolve().parents[1]))
from stats import wilson

ROOT = Path(__file__).resolve().parents[3]


def ece(y, p, n_bins: int) -> float:
    """Expected calibration error over equal-count bins."""
    bins = np.array_split(np.argsort(p), n_bins)
    return float(np.mean([abs(y[b].mean() - p[b].mean()) for b in bins if len(b)]))


def bootstrap_mean_ci(items, values, n_boot: int, seed: int) -> tuple[float, float]:
    """95% percentile CI for the mean of a per-row value (loss, correctness),
    resampling items (the cluster unit)."""
    per_item = (pd.DataFrame(dict(item=items, loss=values))
                .groupby("item")["loss"].agg(["sum", "count"]))
    sums, counts = per_item["sum"].to_numpy(), per_item["count"].to_numpy()
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(sums), size=(n_boot, len(sums)))
    means = sums[draw].sum(1) / counts[draw].sum(1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def bj_crosscheck(ckpt_dir: Path, rows: pd.DataFrame) -> pd.DataFrame | None:
    """MLP sigma(b_j), averaged over items-regime folds, vs the measured
    self-pair slot-1 rate with a Wilson interval."""
    ckpts = sorted(ckpt_dir.glob("items_*.pt"))
    if not ckpts:
        return None
    loaded = [torch.load(f, weights_only=False) for f in ckpts]
    judges = loaded[0]["judge_names"]
    b = np.mean([c["state_dict"]["b"].numpy() for c in loaded], axis=0)
    sigma_b = 1 / (1 + np.exp(-b))

    self_rows = rows[rows["major"] == rows["minor"]]
    out = []
    for judge, sb in zip(judges, sigma_b):
        judge_self = self_rows[self_rows["judge"] == judge]
        if not len(judge_self):
            continue
        k, n = int(judge_self["y"].sum()), len(judge_self)
        lo, hi = wilson(k, n)
        out.append(dict(judge=judge, sigma_b=sb, selfpair_rate=k / n,
                        wilson_lo=lo, wilson_hi=hi, n=n,
                        within=bool(lo <= sb <= hi)))
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "config.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    out = ROOT / cfg["out_dir"]
    seed = cfg["seeds"][0]
    n_boot, ece_bins = cfg["evaluate"]["n_boot"], cfg["evaluate"]["ece_bins"]

    rows = pd.read_parquet(out / "rows.parquet")
    item_of_row = rows.set_index("row_id")["item"]
    preds = pd.concat([pd.read_parquet(f)
                       for f in sorted((out / "preds").glob("*.parquet"))],
                      ignore_index=True)
    preds["item"] = preds["row_id"].map(item_of_row)
    p = np.clip(preds["p_slot1"].to_numpy(dtype=np.float64), 1e-12, 1 - 1e-12)
    preds["loss"] = -(preds["y"] * np.log(p) + (1 - preds["y"]) * np.log(1 - p))

    per_fold = (preds.groupby(["regime", "fold", "model"])
                .apply(lambda g: pd.Series(dict(
                    nll=float(g["loss"].mean()),
                    acc=float((g["p_slot1"].round() == g["y"]).mean()),
                    ece=ece(g["y"].to_numpy(), g["p_slot1"].to_numpy(), ece_bins),
                    n=len(g))))
                .reset_index())

    pooled_rows = []
    for (regime, model), g in preds.groupby(["regime", "model"]):
        lo, hi = bootstrap_mean_ci(g["item"].to_numpy(), g["loss"].to_numpy(),
                                   n_boot, seed)
        pooled_rows.append(dict(
            regime=regime, fold=np.nan, model=model, nll=float(g["loss"].mean()),
            nll_lo=lo, nll_hi=hi,
            acc=float((g["p_slot1"].round() == g["y"]).mean()),
            ece=ece(g["y"].to_numpy(), g["p_slot1"].to_numpy(), ece_bins), n=len(g)))
    pooled = pd.DataFrame(pooled_rows)

    metrics = pd.concat([per_fold, pooled], ignore_index=True)
    metrics.to_csv(out / "metrics.csv", index=False)

    print(f"pooled held-out metrics (chance nll {np.log(2):.4f}); "
          f"CIs bootstrap {n_boot} item resamples")
    for regime, g in pooled.groupby("regime"):
        g = g.sort_values("nll")
        base = {m: g.loc[g["model"] == m, "nll"].squeeze()
                for m in ("position", "dlogit") if m in set(g["model"])}
        print(f"\n{regime}:")
        for r in g.itertuples():
            deltas = "  ".join(f"vs {m} {r.nll - v:+.4f}" for m, v in base.items()
                               if m != r.model)
            print(f"  {r.model:9s} nll {r.nll:.4f} [{r.nll_lo:.4f}, {r.nll_hi:.4f}]"
                  f"  acc {r.acc:.3f}  ece {r.ece:.3f}  {deltas}")

    check = bj_crosscheck(out / "checkpoints", rows)
    if check is None:
        print("\nb_j cross-check skipped: no checkpoints yet")
    else:
        print("\nb_j cross-check (MLP sigma(b_j) vs self-pair slot-1 rate, "
              "Wilson 95%):")
        print(check.round(3).to_string(index=False))
    print(f"\n-> {out}/metrics.csv")


if __name__ == "__main__":
    main()
