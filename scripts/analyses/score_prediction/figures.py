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

from plots import COL, INK2, MUTED, PALETTE, WIDE, chance, dot_ci, legend, save, style
from stats import wilson

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

    fig_model_comparison(pooled, str(fig_dir / "1_model_comparison"))

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
