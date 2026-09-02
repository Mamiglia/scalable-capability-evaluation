"""Position bias vs context fullness.

Top row: P(pick transcript 1) on a transcript-1 x transcript-2 length grid
(chars as the judge saw them), one panel per judge serving, pooled over benches
and conditions. Dashed arc = the serving's context limit (max-model-len tokens
x CHARS_PER_TOKEN), ignoring fixed prompt overhead. Pairs beyond the arc
errored out and carry no verdict, so cells there are empty by construction.

Bottom row: the diagonal of the grid — length-matched pairs (lengths within
MATCH_RATIO of each other), P(pick transcript 1) vs total pair length, dashed
vertical line = same context limit. CIs are 95% cluster-robust, clustered on
the matchup (bench x major x minor): verdicts within a matchup are correlated,
so naive binomial CIs would be ~2-3x too tight in the dense bins.

Also prints P(pick transcript 1) by context fullness (pair chars / limit chars)
per serving.

Usage: uv run python fig_position_context.py
"""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7})
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
OUT_DIR = ROOT / "judge_results" / "judging_variations"
POS_DIR = OUT_DIR / "position_context"

CHARS_PER_TOKEN = 4  # same working ratio as the config generator
MIN_BIN_N = 30
BINS = np.geomspace(300, 260_000, 13)
MATCH_RATIO = 2  # length-matched = longer/shorter <= this
TOTAL_BINS = np.geomspace(600, 560_000, 10)
JUDGE_COLOR = {"qwen3-0.6b": "#2a78d6", "qwen3-8b": "#1baf7a",
               "llama-3.2-3b": "#eb6834"}
FULLNESS_EDGES = [0, 0.25, 0.5, 0.75, 1.01]
# grey midpoint so no-bias (0.5) is grey, empty bins stay white
CMAP = LinearSegmentedColormap.from_list("bias", ["#2166ac", "#d9d9d9", "#b2182b"])
# serving name in verdicts.csv -> max-model-len tokens (slurm serve args)
SERVINGS = [("qwen3-0.6b", 32_768), ("qwen3-8b", 32_768),
            ("llama-3.2-3b", 131_072),
            ("qwen3-0.6b-yarn", 131_072), ("qwen3-8b-yarn", 131_072)]


def load():
    """serving -> (t1_chars, t2_chars, picked1, matchup_id) arrays."""
    out = defaultdict(list)
    matchups = {}
    for r in csv.DictReader(open(OUT_DIR / "verdicts.csv")):
        if r["parse_ok"] != "True" or not r["len_major"] or not r["len_minor"]:
            continue
        lm, ln = int(r["len_major"]), int(r["len_minor"])
        t1, t2 = (lm, ln) if r["order"] == "1" else (ln, lm)
        key = (r["bench"], r["major"], r["minor"])
        out[r["judge"]].append((t1, t2, r["picked"] == "1",
                                matchups.setdefault(key, len(matchups))))
    return {k: np.array(v, dtype=float) for k, v in out.items()}


def cluster_se(y, cluster_ids, p):
    """SE of mean(y) robust to within-cluster correlation."""
    c = cluster_ids.astype(int)
    sums, ns = np.bincount(c, weights=y), np.bincount(c)
    return float(((sums - ns * p) ** 2).sum()) ** 0.5 / len(y)


def main():
    data = load()
    POS_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, len(SERVINGS), figsize=(2.1 * len(SERVINGS), 4.0),
                             sharex="row", sharey="row",
                             gridspec_kw={"height_ratios": [1.55, 1]})
    mesh = None
    for ax, (serving, limit_tok) in zip(axes[0], SERVINGS):
        d = data.get(serving)
        if d is None:
            ax.set_axis_off()
            continue
        t1, t2, p1 = d[:, 0], d[:, 1], d[:, 2]
        n = np.histogram2d(t1, t2, bins=[BINS, BINS])[0]
        wins = np.histogram2d(t1[p1 == 1], t2[p1 == 1], bins=[BINS, BINS])[0]
        rate = np.ma.masked_where(n < MIN_BIN_N, wins / np.maximum(n, 1))
        mesh = ax.pcolormesh(BINS, BINS, rate.T, cmap=CMAP, vmin=0, vmax=1)

        limit_chars = limit_tok * CHARS_PER_TOKEN
        xs = np.geomspace(BINS[0], min(limit_chars - BINS[0], BINS[-1]), 200)
        ys = limit_chars - xs
        ok = (ys > BINS[0]) & (ys < BINS[-1])
        ax.plot(xs[ok], ys[ok], color="#52514e", lw=1.2, ls="--")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_aspect("equal")
        ax.text(0.02, 1.02, serving, transform=ax.transAxes, fontsize=9,
                color="#52514e")
        ax.set_xlabel("transcript 1 length (chars)")
    axes[0][0].set_ylabel("transcript 2 length (chars)")

    # diagonal: length-matched pairs, P(pick 1) vs total length
    for ax, (serving, limit_tok) in zip(axes[1], SERVINGS):
        d = data.get(serving)
        if d is None:
            ax.set_axis_off()
            continue
        t1, t2, p1 = d[:, 0], d[:, 1], d[:, 2]
        m = np.maximum(t1, t2) / np.minimum(t1, t2) <= MATCH_RATIO
        tot, pk, cl = t1[m] + t2[m], p1[m], d[m, 3]
        xs, ps, cis = [], [], []
        for lo, hi in zip(TOTAL_BINS, TOTAL_BINS[1:]):
            sel = (tot >= lo) & (tot < hi)
            n = sel.sum()
            if n < MIN_BIN_N:
                continue
            p = pk[sel].mean()
            xs.append((lo * hi) ** 0.5)
            ps.append(p)
            cis.append(1.96 * cluster_se(pk[sel], cl[sel], p))
        color = JUDGE_COLOR[serving.removesuffix("-yarn")]
        ax.axhline(0.5, color="#b0b0b0", lw=1, ls=":")
        ax.axvline(limit_tok * CHARS_PER_TOKEN, color="#52514e", lw=1.2, ls="--")
        ax.errorbar(xs, ps, yerr=cis, fmt="o-", color=color, ms=4.5, lw=1.2,
                    elinewidth=1.1, capsize=2.5)
        ax.set_xscale("log")
        ax.set_xlim(TOTAL_BINS[0], TOTAL_BINS[-1] * 1.15)
        ax.set_ylim(0, 1)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("total pair length (chars)")
    axes[1][0].set_ylabel("P(pick transcript 1)")

    handles = [plt.Line2D([], [], color="#52514e", ls="--",
                          label="context full (4 chars/token)"),
               plt.Line2D([], [], color="#b0b0b0", ls=":", label="chance")]
    fig.legend(handles=handles, loc="upper right", frameon=False, ncol=2,
               bbox_to_anchor=(0.99, 1.04))
    cbar = fig.colorbar(mesh, ax=axes, fraction=0.015, pad=0.01)
    cbar.set_label("P(pick transcript 1)")
    for ext in ("png", "svg"):
        fig.savefig(POS_DIR / f"position_vs_length.{ext}", dpi=150,
                    bbox_inches="tight")
    print(f"-> {POS_DIR}/position_vs_length.png")

    print("\nP(pick transcript 1) by context fullness (pair chars / limit chars)")
    for serving, limit_tok in SERVINGS:
        d = data.get(serving)
        if d is None:
            continue
        full = (d[:, 0] + d[:, 1]) / (limit_tok * CHARS_PER_TOKEN)
        parts = []
        for lo, hi in zip(FULLNESS_EDGES, FULLNESS_EDGES[1:]):
            m = (full >= lo) & (full < hi)
            parts.append(f"{lo:.2f}-{min(hi, 1):.2f}: "
                         + (f"{d[m, 2].mean():.3f} (n={int(m.sum())})"
                            if m.sum() else "-"))
        print(f"  {serving:18s} " + "  ".join(parts))


if __name__ == "__main__":
    main()
