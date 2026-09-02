"""Which (major, minor) author pairs to judge to fill the capability-gap /
release-gap plane. All 17 pool models already have GPQA transcripts
(judge-pilot/models.csv), so each proposed pair is just a judge run.

Enumerates every pair (major = higher accuracy, so cap_gap >= 0), drops the 10
already run in rd2, bins the plane into a cap x era grid, and picks up to 2
pairs per cell preferring cross-family and family variety. The matched-cap row
and same-era column are the explicit capability-matched / year-matched combos.

Usage: uv run python scripts/analyses/heatmap_fill_plan.py
"""
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from plots import INK, INK2, MUTED, plt, save, style

POOL = Path("judge_results/judge-pilot/models.csv")
RD2 = Path("judge_results/gpqa-pilot-rd2/summary.csv")
OUT = Path("judge_results/gpqa-pilot-rd2/figures")
DOCS = Path("docs")

CAP_EDGES = [0.0, 0.04, 0.10, 0.18, 0.28, 1.0]
CAP_LBL = ["matched", "small", "medium", "large", "x-large"]
REL_EDGES = [-9, -0.05, 0.15, 0.6, 1.1, 9]
REL_LBL = ["older-abler", "same-era", "~half-yr", "~1-yr", "1.5-2-yr"]

FAMS = {"gemma": "Gemma", "gemini": "Gemini", "qwen": "Qwen", "gpt": "GPT",
        "llama": "Llama", "claude": "Claude", "mistral": "Mistral"}


def family(m):
    return next((v for k, v in FAMS.items() if m.startswith(k)), m)


def rel_frac(s):
    d = pd.to_datetime(s)
    return d.year + (d.dayofyear - 1) / 365.25


def binlbl(v, edges, lbls):
    return lbls[max(0, np.searchsorted(edges, v, side="right") - 1)]


def load():
    m = pd.read_csv(POOL).set_index("model")
    m["rel"] = m["release"].map(rel_frac)
    m["fam"] = [family(x) for x in m.index]
    s = pd.read_csv(RD2)
    run = {frozenset((a, b)) for a, b in zip(s.major, s.minor) if a != b}
    return m, run


def main():
    m, run = load()
    acc, rel, fam = m["accuracy"], m["rel"], m["fam"]
    pairs = []
    for a, b in combinations(m.index, 2):
        maj, mnr = (a, b) if acc[a] >= acc[b] else (b, a)
        pairs.append(dict(
            major=maj, minor=mnr,
            cap_gap=acc[maj] - acc[mnr], rel_gap=rel[maj] - rel[mnr],
            fam_maj=fam[maj], fam_mnr=fam[mnr],
            cross=fam[maj] != fam[mnr], run=frozenset((maj, mnr)) in run))
    P = pd.DataFrame(pairs)
    P["cap_bin"] = [binlbl(v, CAP_EDGES, CAP_LBL) for v in P.cap_gap]
    P["rel_bin"] = [binlbl(v, REL_EDGES, REL_LBL) for v in P.rel_gap]

    # Pick up to 2 unrun pairs per (cap_bin, rel_bin), cross-family first, then
    # widen family variety within the cell.
    picks = []
    cand = P[~P.run].copy()
    cand["prio"] = (~cand.cross).astype(int)  # 0 = cross-family, sorts first
    for (cb, rb), block in cand.groupby(["cap_bin", "rel_bin"]):
        seen, taken = set(), 0
        for _, r in block.sort_values(["prio", "cap_gap"]).iterrows():
            combo = frozenset((r.fam_maj, r.fam_mnr))
            if combo in seen and taken:
                continue
            seen.add(combo)
            picks.append(r)
            taken += 1
            if taken == 2:
                break
    sel = pd.DataFrame(picks)

    cols = ["cap_bin", "rel_bin", "major", "minor", "cap_gap", "rel_gap",
            "fam_maj", "fam_mnr", "cross"]
    sel = (sel[cols].assign(cap_gap=sel.cap_gap.round(3), rel_gap=sel.rel_gap.round(2))
           .sort_values(["cap_bin", "rel_bin"]))
    sel.to_csv(DOCS / "heatmap_fill_plan.csv", index=False)

    pd.set_option("display.width", 170)
    print(f"pool {len(m)} models, {len(P)} possible pairs, {P.run.sum()} already run\n")
    print("PROPOSED PAIRS TO JUDGE (fills the plane):")
    print(sel.to_string(index=False))
    print(f"\n{len(sel)} new pairs.  matched-cap row = capability-matched combos;"
          f" same-era column = year-matched combos.")
    grid = (P.assign(n=1).pivot_table(index="rel_bin", columns="cap_bin", values="n",
            aggfunc="sum", fill_value=0).reindex(index=REL_LBL, columns=CAP_LBL, fill_value=0))
    print("\nPairs available per cell (structural coverage of the pool):")
    print(grid.to_string())

    # Overlay: current pairs filled grey, proposed hollow, sized by nothing.
    fig, ax = plt.subplots(figsize=(7.2, 5.2), layout="constrained")
    cur = P[P.run]
    ax.scatter(cur.cap_gap, cur.rel_gap, s=90, c=MUTED, edgecolors="white",
               linewidths=0.8, zorder=3, label="already run (10)")
    csel = sel[sel.cross]
    ssel = sel[~sel.cross]
    ax.scatter(csel.cap_gap, csel.rel_gap, s=95, marker="o", facecolors="none",
               edgecolors="#2a78d6", linewidths=1.6, zorder=4,
               label="proposed, cross-family")
    ax.scatter(ssel.cap_gap, ssel.rel_gap, s=95, marker="s", facecolors="none",
               edgecolors="#e34948", linewidths=1.6, zorder=4,
               label="proposed, same-family")
    for _, r in sel.iterrows():
        ax.annotate(f"{r.fam_maj[:3]}/{r.fam_mnr[:3]}", (r.cap_gap, r.rel_gap),
                    xytext=(0, 7), textcoords="offset points", ha="center",
                    fontsize=5.0, color=MUTED, zorder=5)
    for e in CAP_EDGES[1:-1]:
        ax.axvline(e, color="#e8e6df", lw=0.8, zorder=1)
    for e in REL_EDGES[1:-1]:
        ax.axhline(e, color="#e8e6df", lw=0.8, zorder=1)
    ax.axhline(0, color=MUTED, lw=0.7, ls=":", zorder=1)
    ax.set_xlabel("capability gap (GPQA accuracy diff)")
    ax.set_ylabel("release gap (years, major - minor)")
    ax.set_title("Filling the capability / release plane  ·  proposed judge runs",
                 color=INK2, fontsize=9.5, loc="left")
    ax.legend(frameon=False, loc="upper left", fontsize=7, labelcolor=INK2)
    style(ax, grid_axis="both")
    save(fig, OUT / "heatmap_fill_plan")
    print(f"\nwrote {OUT/'heatmap_fill_plan'}.png/.svg and {DOCS/'heatmap_fill_plan.csv'}")


if __name__ == "__main__":
    # self-check: orientation gives non-negative cap_gap, bins cover the data
    _m = pd.DataFrame({"a": [0.44, 0.93]})
    assert binlbl(0.0, CAP_EDGES, CAP_LBL) == "matched"
    assert binlbl(0.49, CAP_EDGES, CAP_LBL) == "x-large"
    assert binlbl(-0.2, REL_EDGES, REL_LBL) == "older-abler"
    main()
