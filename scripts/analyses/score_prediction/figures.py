"""Figures for the score-prediction analysis (house style).

Reads metrics.csv, checkpoints/, rows.parquet; writes
figures/N_name.{png,svg}. Rerun after training/evaluation to refresh.

Usage:
  uv run python scripts/analyses/score_prediction/figures.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt

from plots import (COL, INK2, MUTED, PALETTE, SURFACE, WIDE, chance, dot_ci,
                   legend, save, style)

ROOT = Path(__file__).resolve().parents[3]

# Fixed model -> colour assignment (identity, never rank). Baselines are
# open ink-tone markers so they read as inline controls.
MODEL_ORDER = ["position", "dlogit", "logreg", "gbm", "mlp"]
MODEL_STYLE = {  # colour, open_marker
    "position": (MUTED, True),
    "dlogit": (INK2, True),
    "logreg": (PALETTE[0], False),
    "gbm": (PALETTE[1], False),
    "mlp": (PALETTE[2], False),
}
REGIME_TITLE = {"items": "held-out items", "pairs": "held-out pairs",
                "judges": "held-out judges"}


def fig_model_comparison(pooled: pd.DataFrame, out_stub: str):
    """Dot+CI per model, one panel per regime, shared NLL axis."""
    regimes = [r for r in ("items", "pairs", "judges") if r in set(pooled["regime"])]
    fig, axes = plt.subplots(1, len(regimes), figsize=(WIDE, 1.9), sharex=True)
    for ax, regime in zip(np.atleast_1d(axes), regimes):
        rows = pooled[pooled["regime"] == regime].set_index("model")
        present = [m for m in MODEL_ORDER if m in rows.index]
        for y, model in enumerate(reversed(present)):
            r = rows.loc[model]
            colour, open_marker = MODEL_STYLE[model]
            dot_ci(ax, y, r["nll"], r["nll_lo"], r["nll_hi"], colour,
                   note=f"acc {r['acc']:.2f}", open_marker=open_marker)
        ax.set_yticks(range(len(present)), list(reversed(present)))
        chance(ax, np.log(2), axis="x")
        ax.set_title(REGIME_TITLE[regime], fontsize=8, color=INK2)
        ax.set_ylim(-0.6, len(present) - 0.4)
        style(ax, grid_axis="x")
    np.atleast_1d(axes)[0].figure.supxlabel(
        "held-out NLL, nats (dots: mean; whiskers: 95% item-bootstrap CI; "
        "dashed: chance ln 2; open markers: baselines)", fontsize=7, color=MUTED)
    save(fig, out_stub)


def fig_bj_crosscheck(check: pd.DataFrame, out_stub: str):
    """MLP sigma(b_j) against measured self-pair slot-1 rate."""
    fig, ax = plt.subplots(figsize=(COL, COL))
    lo = min(check["wilson_lo"].min(), check["sigma_b"].min()) - 0.05
    hi = max(check["wilson_hi"].max(), check["sigma_b"].max()) + 0.05
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=0.9, ls=(0, (4, 3)), zorder=1)
    for r in check.itertuples():
        ax.plot([r.wilson_lo, r.wilson_hi], [r.sigma_b, r.sigma_b],
                color=PALETTE[2], lw=1.2, zorder=3)
        ax.plot(r.selfpair_rate, r.sigma_b, "o", ms=5.5, color=PALETTE[2], zorder=4)
        ax.annotate(f"{r.judge} (n={r.n})", (r.selfpair_rate, r.sigma_b),
                    textcoords="offset points", xytext=(5, 5),
                    fontsize=6.5, color=INK2)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("self-pair slot-1 rate (whiskers: Wilson 95%)")
    ax.set_ylabel("MLP sigma(b_j), items-fold mean")
    style(ax, grid_axis="both")
    save(fig, out_stub)


def fig_heldout_judge(per_fold: pd.DataFrame, judge_names: list[str],
                      slot1_rates: pd.Series, out_stub: str):
    """Per held-out judge: NLL of each content model vs the dlogit control,
    judges ordered by position-bias extremity."""
    judges_frame = per_fold[per_fold["regime"] == "judges"].copy()
    judges_frame["judge"] = judges_frame["fold"].astype(int).map(dict(enumerate(judge_names)))
    order = sorted(judge_names, key=lambda j: abs(slot1_rates[j] - 0.5))
    fig, ax = plt.subplots(figsize=(COL * 1.35, 2.6))
    shown = [m for m in ("dlogit", "logreg", "gbm", "mlp")
             if m in set(judges_frame["model"])]
    for y, judge in enumerate(order):
        for model in shown:
            r = judges_frame.query("judge == @judge and model == @model")
            if not len(r):
                continue
            colour, open_marker = MODEL_STYLE[model]
            ax.plot(r["nll"].squeeze(), y, "o", ms=5,
                    mfc="#ffffff" if open_marker else colour,
                    mec=colour, mew=1.2, zorder=4,
                    label=model if y == 0 else None)
        ax.annotate(f"slot-1 rate {slot1_rates[judge]:.2f}",
                    xy=(1.01, y), xycoords=("axes fraction", "data"),
                    va="center", ha="left", color=MUTED, fontsize=6.5,
                    annotation_clip=False)
    ax.set_yticks(range(len(order)), order)
    chance(ax, np.log(2), axis="x")
    ax.set_xlabel("held-out NLL, nats (judge unseen in training; dashed: chance)")
    legend(ax, loc="lower right")
    style(ax, grid_axis="x")
    save(fig, out_stub)


def fig_pair_transfer(preds: pd.DataFrame, rows: pd.DataFrame, acc: pd.Series,
                      out_stub: str, n_boot: int, seed: int, n_labels: int = 3):
    """Held-out NLL per pairing (logreg, pairs regime) against the pair's
    raw accuracy gap. Tests 'near-tie pairings are the hard transfers';
    self-pairs sit at gap 0 as open markers."""
    from evaluate import bootstrap_nll_ci
    frame = preds.query("regime == 'pairs' and model == 'logreg'").copy()
    meta = rows.set_index("row_id")
    frame["pair_key"] = frame["row_id"].map(meta["pair_key"])
    frame["item"] = frame["row_id"].map(meta["item"])

    fig, ax = plt.subplots(figsize=(COL * 1.35, 2.5))
    points = []
    for pair_key, pair_rows in frame.groupby("pair_key"):
        model_a, model_b = pair_key.split("|")
        gap = abs(acc[model_a] - acc[model_b])
        lo, hi = bootstrap_nll_ci(pair_rows["item"].to_numpy(),
                                  pair_rows["loss"].to_numpy(), n_boot, seed)
        nll = pair_rows["loss"].mean()
        self_pair = model_a == model_b
        ax.plot([gap, gap], [lo, hi], color=PALETTE[0], lw=1.1, zorder=3)
        ax.plot(gap, nll, "o", ms=5, mfc=SURFACE if self_pair else PALETTE[0],
                mec=PALETTE[0], mew=1.2, zorder=4)
        points.append((nll, gap, pair_key))
    for nll, gap, pair_key in sorted(points, reverse=True)[:n_labels]:
        ax.annotate(pair_key.replace("|", " vs "), (gap, nll),
                    textcoords="offset points", xytext=(4, 4),
                    fontsize=5.5, color=INK2)
    chance(ax, np.log(2), axis="y")
    ax.set_xlabel("pair accuracy gap |d_acc| (open markers: self-pairs)")
    ax.set_ylabel("held-out NLL, nats (logreg;\nwhiskers: 95% item-bootstrap CI)")
    style(ax)
    save(fig, out_stub)


def fig_reliability(preds: pd.DataFrame, out_stub: str, n_bins: int):
    """Reliability per model, items regime: mean predicted p vs empirical
    slot-1 rate over equal-count bins; diagonal = calibrated."""
    frame = preds.query("regime == 'items'")
    fig, ax = plt.subplots(figsize=(COL, COL))
    ax.plot([0, 1], [0, 1], color=MUTED, lw=0.9, ls=(0, (4, 3)), zorder=1)
    for model in MODEL_ORDER:
        model_rows = frame[frame["model"] == model]
        if not len(model_rows):
            continue
        p = model_rows["p_slot1"].to_numpy()
        y = model_rows["y"].to_numpy()
        bins = np.array_split(np.argsort(p), n_bins)
        colour, open_marker = MODEL_STYLE[model]
        ax.plot([p[b].mean() for b in bins], [y[b].mean() for b in bins],
                "-o", ms=3.5, lw=1.1, color=colour,
                mfc=SURFACE if open_marker else colour, mec=colour, mew=1.0,
                label=model)
    ax.set_xlabel(f"mean predicted P(slot 1), {n_bins} equal-count bins")
    ax.set_ylabel("empirical slot-1 rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    legend(ax, loc="upper left")
    style(ax, grid_axis="both")
    save(fig, out_stub)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "config.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    out = ROOT / cfg["out_dir"]
    fig_dir = out / "figures"
    fig_dir.mkdir(exist_ok=True)

    metrics = pd.read_csv(out / "metrics.csv")
    pooled = metrics[metrics["fold"].isna()]
    per_fold = metrics[metrics["fold"].notna()]
    rows = pd.read_parquet(out / "rows.parquet")
    preds = pd.concat([pd.read_parquet(f)
                       for f in sorted((out / "preds").glob("*.parquet"))],
                      ignore_index=True)
    p = np.clip(preds["p_slot1"].to_numpy(), 1e-12, 1 - 1e-12)
    preds["loss"] = -(preds["y"] * np.log(p) + (1 - preds["y"]) * np.log(1 - p))
    acc = pd.read_csv(ROOT / cfg["models_csv"]).set_index("model")["accuracy"]

    fig_model_comparison(pooled, str(fig_dir / "1_model_comparison"))
    fig_pair_transfer(preds, rows, acc, str(fig_dir / "4_pair_transfer"),
                      cfg["evaluate"]["n_boot"], cfg["seeds"][0])
    fig_reliability(preds, str(fig_dir / "5_reliability"),
                    cfg["evaluate"]["ece_bins"])

    ckpts = sorted((out / "checkpoints").glob("items_*.pt"))
    if ckpts:
        sys.path.append(str(Path(__file__).resolve().parent))
        from evaluate import bj_crosscheck
        check = bj_crosscheck(out / "checkpoints", rows)
        fig_bj_crosscheck(check, str(fig_dir / "2_bj_crosscheck"))
        judge_names = torch.load(ckpts[0], weights_only=False)["judge_names"]
        fig_heldout_judge(per_fold, judge_names,
                          rows.groupby("judge")["y"].mean(),
                          str(fig_dir / "3_heldout_judge"))
    else:
        print("checkpoints missing: figures 2-3 skipped")
    print(f"-> {fig_dir}/")


if __name__ == "__main__":
    main()
