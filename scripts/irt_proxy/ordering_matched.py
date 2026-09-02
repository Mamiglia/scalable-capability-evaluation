"""Matched pre/post ordering-accuracy gap curve: raw judge Elo vs corrected, SAME pairs.

Both panels are computed on the identical set of candidate pairs from gap_pairs.csv
(method raw_elo vs corrected); n per gap bin is shown so the matching is explicit.
"""
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "font.family": "sans-serif", "font.sans-serif": ["Arial"], "svg.fonttype": "none",
})
GREY, BLUE = "#7f7f7f", "#2a78d6"
XHI = 0.7  # beyond here bins hold <10 pairs; drawn faint


def band(ax, rows, colour, seed=0):
    rng = np.random.default_rng(seed)
    centres, mean, lo, hi, ns = [], [], [], [], []
    for centre, b in rows.groupby("gap_bin_center"):
        y = b["ordering_correct"].to_numpy(float)
        draws = rng.choice(y, size=(1000, len(y))).mean(1)
        centres.append(centre); mean.append(y.mean()); ns.append(len(y))
        lo.append(np.percentile(draws, 2.5)); hi.append(np.percentile(draws, 97.5))
    centres, mean, lo, hi, ns = map(np.array, (centres, mean, lo, hi, ns))
    solid = centres <= XHI
    ax.fill_between(centres, lo, hi, color=colour, alpha=0.2, lw=0)
    ax.plot(centres[solid], mean[solid], lw=1.2, c=colour)
    ax.plot(centres, mean, lw=1.2, c=colour, alpha=0.35)  # faint continuation into sparse tail
    return centres, ns


def dress(ax, tag):
    ax.set_title(tag, loc="left", fontsize=7, color="dimgrey")
    ax.axhline(0.5, ls="--", lw=0.8, c="grey", zorder=0)
    ax.set_xticks([0, 0.35, 0.7]); ax.set_yticks([0, 0.5, 1])
    ax.set_xlim(-0.02, 0.85); ax.set_ylim(-0.02, 1.02)
    for side, (lo, hi) in (("bottom", (0, 0.8)), ("left", (0, 1))):
        ax.spines[side].set_position(("outward", 3)); ax.spines[side].set_bounds(lo, hi)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parents[2] / "judge_results/glm_correction"
    p.add_argument("--gap-pairs", type=Path, default=here / "tables/gap_pairs.csv")
    p.add_argument("--out", type=Path, default=here)
    args = p.parse_args()

    d = pd.read_csv(args.gap_pairs)
    d = d[d.benchmark.eq("gpqa") & d.irt_model.eq("fixed_3pl")]
    raw, cor = d[d.method.eq("raw_elo")], d[d.method.eq("corrected")]
    assert raw.model_left.equals(cor.model_left) and raw.true_gap.round(6).equals(cor.true_gap.round(6))

    fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.4), sharex=True, sharey=True)
    centres, ns = band(axes[0], raw, GREY)
    band(axes[1], cor, BLUE)
    for ax, tag in zip(axes, ["original judge Elo (pre)", "corrected (post)"]):
        dress(ax, tag)
    axes[0].set_ylabel("Ordering accuracy")
    fig.supxlabel("True accuracy gap  (same pairs both panels)", fontsize=8, y=0.02)
    # n per bin along the top of the left panel, so matching is explicit
    for c, n in zip(centres, ns):
        if c <= 0.8:
            axes[0].text(c, 1.06, str(n), fontsize=4.5, ha="center", color="dimgrey")
    axes[0].text(-0.02, 1.10, "n:", fontsize=4.5, color="dimgrey")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    for suffix in ("png", "svg"):
        fig.savefig(args.out / f"ordering_matched_pre_post.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"pairs per method: {len(raw)} | written to {args.out}/ordering_matched_pre_post.{{png,svg}}")


if __name__ == "__main__":
    main()
