"""Observed pick rate vs a length-only judge's prediction, per gpqa grid pair.

The length-only judge is a 1-parameter logistic on the log-length difference,
fitted on self-pair verdicts (same model, two epochs), where there is no
capability signal by construction. Points above the diagonal = judges beat
the length-only baseline toward the stronger member.

Usage:
  uv run python scripts/judge/fig_length_only_judge.py --out judge_results/multi-eval-pilot/figures/16_length_only_judge
"""

import argparse
import glob
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DATA = Path("interactive_dashboards/transcript_viewer/data/data-expansion-gpqa")
DSK = "deepseek-v4-flash-0731"


def load_transcripts(idx):
    tr, ep1 = {}, {}
    for m in idx["models"]:
        f0, f1 = DATA / f"transcripts/{m}.json", DATA / f"transcripts/{m}__ep1.json"
        if f0.exists():
            tr[m] = json.load(open(f0))
        if f1.exists():
            ep1[m] = json.load(open(f1))
    return tr, ep1


def fit_beta(tr, ep1):
    """1-param logistic MLE on self-pair verdicts."""
    X, Y = [], []
    for pf in glob.glob(str(DATA / "pairs/*__full.json")):
        a, b, _ = Path(pf).stem.split("__")
        if a != b or a not in tr or a not in ep1:
            continue
        for item, judges in json.load(open(pf)).items():
            if item not in tr[a] or item not in ep1[a]:
                continue
            x = math.log(len(tr[a][item]) + 1) - math.log(len(ep1[a][item]) + 1)
            for j, cell in judges.items():
                for o in cell["orders"]:
                    if o and o["picked_major"] is not None:
                        X.append(x)
                        Y.append(o["picked_major"])
    X, Y = np.array(X), np.array(Y)
    beta = 0.0
    for _ in range(50):
        p = 1 / (1 + np.exp(-beta * X))
        beta -= np.sum((Y - p) * X) / -np.sum(p * (1 - p) * X ** 2)
    return beta, len(Y)


def pair_points(idx, tr, beta):
    pts = []
    for pf in sorted(glob.glob(str(DATA / "pairs/*__full.json"))):
        a, b, _ = Path(pf).stem.split("__")
        if a == b or a not in tr or b not in tr:
            continue
        if idx["models"][a]["arm"] != "A" or idx["models"][b]["arm"] != "A":
            continue
        pred = obs = n = 0
        for item, judges in json.load(open(pf)).items():
            if item not in tr[a] or item not in tr[b]:
                continue
            x = math.log(len(tr[a][item]) + 1) - math.log(len(tr[b][item]) + 1)
            pp = 1 / (1 + math.exp(-beta * x))
            for j, cell in judges.items():
                for o in cell["orders"]:
                    if o and o["picked_major"] is not None:
                        pred += pp
                        obs += o["picked_major"]
                        n += 1
        if n:
            pts.append({"a": idx["models"][a]["label"], "b": idx["models"][b]["label"],
                        "pred": pred / n, "obs": obs / n, "n": n, "dsk": DSK in (a, b)})
    return pts


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    idx = json.load(open(DATA / "index.json"))
    tr, ep1 = load_transcripts(idx)
    beta, n_fit = fit_beta(tr, ep1)
    pts = pair_points(idx, tr, beta)

    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.plot([0, 1], [0, 1], color="#8a8a86", lw=1.2)
    ax.axhline(0.5, ls="--", c="#ccc", lw=0.8)
    ax.axvline(0.5, ls="--", c="#ccc", lw=0.8)
    for p in pts:
        ax.scatter(p["pred"], p["obs"], s=12 + p["n"] / 40, color="#999", alpha=0.4,
                   edgecolors="white", linewidths=0.5, zorder=3)
    # pooled per stronger member, Wilson CI on observed; colour = ECI
    lab2eci = {v["label"]: v["eci"] for v in idx["models"].values() if v.get("eci")}
    bymodel = {}
    for p in pts:
        g = bymodel.setdefault(p["a"], {"pred": 0.0, "obs": 0.0, "n": 0})
        g["pred"] += p["pred"] * p["n"]
        g["obs"] += p["obs"] * p["n"]
        g["n"] += p["n"]
    ecis = [lab2eci[l] for l in bymodel]
    norm = plt.Normalize(min(ecis), max(ecis))
    cmap = plt.cm.viridis
    for i, (label, g) in enumerate(sorted(bymodel.items(), key=lambda kv: kv[1]["pred"] / kv[1]["n"])):
        x, y, n = g["pred"] / g["n"], g["obs"] / g["n"], g["n"]
        p_, d = y, 1 + 1.96 ** 2 / n
        center = (p_ + 1.96 ** 2 / (2 * n)) / d
        half = 1.96 * math.sqrt(p_ * (1 - p_) / n + 1.96 ** 2 / (4 * n * n)) / d
        c = cmap(norm(lab2eci[label]))
        ax.errorbar(x, y, yerr=[[y - (center - half)], [(center + half) - y]],
                    color=c, marker="D", ms=11, capsize=3, lw=1.8, zorder=5,
                    markeredgecolor="white", markeredgewidth=0.8)
        dsk = label == "DeepSeek V4 Flash"
        if x > 0.68:  # rightmost point: label above, clear of the colorbar
            xy, ha = (0, 11), "center"
        else:
            xy, ha = (9, (7, -14)[i % 2]), "left"
        ax.annotate(f"{label} ({lab2eci[label]:.0f})", (x, y), textcoords="offset points",
                    xytext=xy, fontsize=9, color="#333", ha=ha,
                    fontweight="bold" if dsk else "normal")
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                      shrink=0.6, pad=0.02)
    cb.set_label("ECI (stronger member)", fontsize=10)
    cb.ax.tick_params(labelsize=9)
    h = [plt.Line2D([], [], marker="o", ls="", c="#999", alpha=0.5, label="pair"),
         plt.Line2D([], [], marker="D", ls="", c="#555", label="stronger member pooled (Wilson 95% CI), colour = ECI")]
    ax.legend(handles=h, frameon=False, fontsize=9, loc="upper left")
    ax.set_xlabel("predicted by length-only judge", fontsize=12)
    ax.set_ylabel("observed P(pick stronger)", fontsize=12)
    ax.tick_params(labelsize=9.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(f"{args.out}.png", dpi=150)
    fig.savefig(f"{args.out}.svg")
    print(f"wrote {args.out}.png/.svg  (beta {beta:.2f})")


if __name__ == "__main__":
    main()
