"""Figures for the judge-interp analysis (house style).

Reads probe_metrics.csv, repeat_ceiling.csv and the pod-run rewrite
files; writes figures/1_probe_curve, 2_verdict_ladder,
3_steering_doses (.png/.svg each). Rerun after new results to refresh.

Usage:
  uv run python scripts/analyses/judge_interp/figures.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt

from plots.style import COL, INK2, MUTED, PALETTE, WIDE, chance, legend, save, style

ROOT = Path(__file__).resolve().parents[3]

POOLING_STYLE = {"last_token": (PALETTE[0], "last token"),
                 "mean_transcript": (PALETTE[1], "mean over transcript")}

# Held-out verdict-ordering accuracies from the probe runs (findings
# doc has provenance); ceilings are read from repeat_ceiling.csv.
LADDER = [
    ("coin flip", 0.500, True),
    ("label-shuffle null", 0.532, True),
    ("true capability gap", 0.60, True),
    ("embedding model (8B)", 0.695, True),
    ("elo-probe direction", 0.708, False),
    ("BT direction + judge gains", 0.732, False),
    ("pooled elo (9 judges)", 0.770, True),
]

AXIS_STYLE = {"approval": (PALETTE[0], "approval"),
              "v_cap": (PALETTE[1], "v_capability"),
              "v_pref": (PALETTE[2], "v_pref_gen (residual)"),
              "v_pref_orth": (PALETTE[3], "v_pref_gen (orthogonal)"),
              "random_a": (MUTED, "random control")}


def fig_probe_curve(metrics: pd.DataFrame, out_stub: str):
    """Held-out r over depth, one line per pooling, controls annotated."""
    fig, ax = plt.subplots(figsize=(COL * 1.6, 2.6))
    for pooling, (colour, label) in POOLING_STYLE.items():
        rows = metrics[metrics["pooling"] == pooling].sort_values("layer")
        ax.plot(rows["layer"], rows["r"], color=colour, lw=2, label=label)
    best = metrics.loc[metrics["r"].idxmax()]
    ax.scatter([best.layer], [best.r], color=POOLING_STYLE[best.pooling][0],
               s=28, zorder=4)
    ax.annotate(f"layer {int(best.layer)}: r = {best.r:.2f}",
                (best.layer, best.r), textcoords="offset points",
                xytext=(8, 6), color=INK2, fontsize=8)
    control = metrics[(metrics["layer"] == 0)
                      & (metrics["pooling"] == "mean_transcript")]["r"].iloc[0]
    ax.annotate(f"layer 0 mean pooling: r = {control:.2f}\n"
                "(surface token statistics, before any computation)",
                (0, control), textcoords="offset points", xytext=(8, -16),
                color=MUTED, fontsize=7.5)
    ax.set_xlabel("layer (residual stream, 0 = embeddings)", color=INK2)
    ax.set_ylabel("held-out r vs elo", color=INK2)
    ax.set_ylim(-0.05, 0.7)
    style(ax)
    legend(ax, loc="lower right")
    save(fig, out_stub)


def fig_verdict_ladder(ceilings: pd.DataFrame, out_stub: str):
    """Dot per predictor on the verdict-ordering accuracy axis, with the
    repeat-ceiling bands the predictors are trying to reach."""
    fig, ax = plt.subplots(figsize=(COL * 1.7, 2.4))
    qwen = ceilings[ceilings["model"] == "qwen3-32b"].iloc[0]
    pooled = ceilings[ceilings["model"] == "all models pooled"].iloc[0]
    for row, colour, label in (
            (qwen, PALETTE[1], "ceiling, qwen3-32b"),
            (pooled, PALETTE[0], "ceiling, judges pooled")):
        ax.axvspan(row["acc_ceiling_lo"], row["acc_ceiling_hi"],
                   color=colour, alpha=0.18, lw=0)
        ax.text(row["acc_ceiling_lo"] + 0.002, len(LADDER) - 0.4, label,
                color=colour, fontsize=7.5, rotation=90, va="top")
    for y, (name, acc, is_reference) in enumerate(LADDER):
        colour = MUTED if is_reference else PALETTE[0]
        ax.plot([0.48, acc], [y, y], color="#eceae2", lw=0.8, zorder=1)
        ax.scatter([acc], [y], s=26, color=colour,
                   facecolor="white" if is_reference else colour,
                   zorder=3, lw=1.2, edgecolor=colour)
        ax.annotate(f"{acc:.2f}", (acc, y), textcoords="offset points",
                    xytext=(0, 6), ha="center", color=INK2, fontsize=7)
    ax.set_yticks(range(len(LADDER)), [n for n, _, _ in LADDER])
    ax.set_xlim(0.48, 0.95)
    ax.set_xlabel("verdict-ordering accuracy (held out)", color=INK2)
    chance(ax, 0.5, axis="x")
    style(ax, grid_axis="x")
    save(fig, out_stub)


def load_rewrites(pod_dir: Path):
    recs = [json.loads(l) for f in sorted(pod_dir.glob("rewrites*.jsonl"))
            for l in open(f)]
    return pd.DataFrame(recs)


def fig_steering_doses(rewrites: pd.DataFrame, selection: pd.DataFrame,
                       out_stub: str):
    """Left: the rewrite-alone effect. Right: steering dose-response
    (change in own-axis re-read score vs the alpha=0 rewrite)."""
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(WIDE, 2.6),
                                   gridspec_kw={"width_ratios": [1, 2.4]})
    has_main_keys = rewrites["reread_scores"].map(lambda s: "approval" in s)
    main = rewrites[has_main_keys & (rewrites["alpha_frac"].abs() <= 0.1)]
    base = {r.text_id: r.reread_scores for r in main.itertuples()
            if r.axis == "none"}
    orig = selection.set_index("text_id")["approval_score"]
    deltas = [base[t]["approval"] - orig[t] for t in base if t in orig.index]
    ax0.scatter(np.zeros(len(deltas)) + np.linspace(-.12, .12, len(deltas)),
                deltas, s=22, color=PALETTE[0])
    ax0.axhline(0, color=MUTED, lw=0.8, ls="--")
    ax0.set_xticks([])
    ax0.set_ylabel("approval score change", color=INK2)
    ax0.set_title("rewriting alone\n(alpha = 0 rewrite vs original)",
                  fontsize=8.5, color=INK2)
    style(ax0)

    for axis, (colour, label) in AXIS_STYLE.items():
        rows = main[(main["axis"] == axis)]
        key = "approval" if axis == "random_a" else axis
        points = {}
        for a in sorted(rows["alpha_frac"].unique()):
            sub = rows[rows["alpha_frac"] == a]
            ds = [r.reread_scores[key] - base[r.text_id][key]
                  for r in sub.itertuples() if r.text_id in base]
            if ds:
                points[a] = np.mean(ds)
        if points:
            ax1.plot(list(points), list(points.values()), color=colour,
                     lw=1.6, marker="o", ms=3.5,
                     ls="--" if axis == "random_a" else "-", label=label)
    ax1.axhline(0, color=MUTED, lw=0.8, ls="--")
    ax1.set_xlabel("steering strength (fraction of median residual norm)",
                   color=INK2)
    ax1.set_ylabel("own-axis score change vs alpha=0", color=INK2)
    ax1.set_title("steering dose-response (all-positions, layer 26; "
                  "greedy decoding)", fontsize=8.5, color=INK2)
    style(ax1)
    legend(ax1, loc="lower left", fontsize=7)
    save(fig, out_stub)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "config.yaml")
    ap.add_argument("--pod-dir", type=Path,
                    default=ROOT / "judge_results/judge_interp/pod_runs_260806")
    ap.add_argument("--selection", type=Path, required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    out = ROOT / cfg["out_dir"]
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    metrics = pd.read_csv(out / "probe_metrics.csv")
    fig_probe_curve(metrics, str(figures / "1_probe_curve"))
    ceilings = pd.read_csv(ROOT / "judge_results/gpqa-pilot-rd2/repeat_ceiling.csv")
    fig_verdict_ladder(ceilings, str(figures / "2_verdict_ladder"))
    rewrites = load_rewrites(args.pod_dir)
    selection = pd.read_parquet(args.selection)
    fig_steering_doses(rewrites, selection, str(figures / "3_steering_doses"))
    print(f"-> {figures}/")


if __name__ == "__main__":
    main()
