"""Train-loss curves for the cross-domain cells, one panel per (cell, label),
overlaying adapter ranks. Reads train_log.jsonl under
<results-dir>/<root>/<size>/<cell>_<label>/adapters/final/.

  python plot_train_loss.py --size Qwen3-0.6B
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

COLORS = ["#3B6FB6", "#E07B39", "#5B9E5B", "#9A5BA5"]


def load(path):
    # a restarted run appends to the old log; keep only the latest segment
    steps, losses = [], []
    for line in open(path):
        r = json.loads(line)
        if "step" in r:
            if steps and r["step"] < steps[-1]:
                steps, losses = [], []
            steps.append(r["step"])
            losses.append(r["loss"])
    return steps, losses


def smooth(xs, k):
    return [sum(xs[max(0, i - k + 1):i + 1]) / len(xs[max(0, i - k + 1):i + 1])
            for i in range(len(xs))]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size", required=True, help="Qwen3-0.6B / Qwen3-8B")
    ap.add_argument("--results-dir", default="judge_results/judge_ft")
    ap.add_argument("--roots", nargs="+", default=["cross=16", "cross_r4=4", "cross_r2=2", "cross_r1=1"],
                    help="root=rank pairs; roots without data are skipped")
    ap.add_argument("--out-dir", default="judge_results/judge_ft/cross_ranks")
    ap.add_argument("--config", default=str(Path(__file__).parent / "cross_domain.yaml"))
    ap.add_argument("--smooth-window", type=int, default=5, help="points (20 rows each)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    cells = list(cfg["cells"])
    labels = cfg["labels"]
    roots = [(r.split("=")[0], r.split("=")[1]) for r in args.roots]

    fig, axes = plt.subplots(len(cells), len(labels), figsize=(4.5 * len(labels), 2.2 * len(cells)),
                             sharey=True, squeeze=False)
    for i, cell in enumerate(cells):
        for j, lab in enumerate(labels):
            ax = axes[i][j]
            ax.axhline(math.log(2), color="#999", lw=0.8, ls="--")
            for (root, rank), color in zip(roots, COLORS):
                p = (Path(args.results_dir) / root / args.size / f"{cell}_{lab}"
                     / "adapters" / "final" / "train_log.jsonl")
                if not p.exists():
                    continue
                s, l = load(p)
                ax.plot(s, l, color=color, alpha=0.15, lw=0.7)
                ax.plot(s, smooth(l, args.smooth_window), color=color, lw=1.6, label=f"rank {rank}")
            ax.set_ylim(0, 1.15)
            ax.text(0.02, 0.04, f"{cell} / {lab}", transform=ax.transAxes, fontsize=8, color="#444")
            if i == 0 and j == 0:
                ax.legend(fontsize=8, loc="upper right", frameon=False)
                ax.text(0.98, math.log(2) + 0.03, "chance (ln 2)", fontsize=7, color="#999",
                        ha="right", transform=ax.get_yaxis_transform())
            if i == len(cells) - 1:
                ax.set_xlabel("rows seen", fontsize=8)
            if j == 0:
                ax.set_ylabel("train CE (verdict token)", fontsize=8)
            ax.tick_params(labelsize=7)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
    fig.tight_layout()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(out / f"loss_curves_{args.size}.{ext}", dpi=150)
    print(f"wrote {out}/loss_curves_{args.size}.png + .svg")


if __name__ == "__main__":
    main()
