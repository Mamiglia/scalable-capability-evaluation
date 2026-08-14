"""Baseline-vs-intervention contrast versions of the suite figure types.

Per comparison condition (cot on matched cells; the three prompt conditions),
per bench:
  <dir>/contrasts/<cond>_pair_heatmap.png   fig-3 analog: judge x pair grid of
                                            delta P(pick stronger), diverging map
  <dir>/contrasts/<cond>_bias_delta.png     fig-9/10 analog: delta position- and
                                            length-pick per judge, 95% CI
  <dir>/contrasts/<cond>_item_scatter.png   fig-5 analog: per-item accuracy,
                                            baseline vs condition, one column
                                            per judge; the cot condition also
                                            splits rows by which pair member
                                            carries real CoT (both / stronger
                                            only / weaker only / neither -
                                            neither is the null control)

Member-level figures (0 model accuracy, 11 capability-vs-length) have no
condition contrast - members don't change across conditions.

Usage: uv run python fig_contrasts.py
"""
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7})
import numpy as np

from fig_gap_delta import wilson, E_SERVING
from pool import cot_ok, load_members

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
BASE = ROOT / "judge_results" / "judging_variations"

JUDGES = ["qwen3-0.6b", "llama-3.2-3b", "qwen3-8b", "qwen3-14b", "qwen3-32b", "gpt-oss-20b", "gpt-oss-120b"]
COMPARISONS = {  # cond key -> (verdicts condition, output folder)
    "cot": ("e_with_reasoning", "cot_inclusion"),
    "ignore-style": ("b_ignore_superficial", "ignore_instructions"),
    "ignore-style-conf": ("c_ignore_superficial_confidence", "ignore_instructions"),
    "pick-weaker": ("d_pick_weaker", "ignore_instructions"),
}
MIN_N = 20


def norm_judge(j):
    return E_SERVING.get(j, j)


def cell_rates(rows):
    """(judge, bench, pair) -> (p, n) and (judge, bench) slot/length tallies."""
    acc = defaultdict(lambda: [0, 0])
    slot = defaultdict(lambda: [0, 0])
    longer = defaultdict(lambda: [0, 0])
    item = defaultdict(lambda: [0, 0])
    for r in rows:
        j = norm_judge(r["judge"])
        pair = (r["bench"], r["major"], r["minor"])
        acc[(j,) + pair][1] += 1
        acc[(j,) + pair][0] += r["picked_major"] == "True"
        slot[(j,) + pair][1] += 1
        slot[(j,) + pair][0] += r["picked"] == "1"
        if r["picked_longer"] != "":
            longer[(j,) + pair][1] += 1
            longer[(j,) + pair][0] += r["picked_longer"] == "True"
        item[(r["bench"], r["item_id"])][1] += 1
        item[(r["bench"], r["item_id"])][0] += r["picked_major"] == "True"
    return acc, slot, longer, item


def _flags_cache():
    import json
    p = BASE / "cot_flags.json"
    return json.load(open(p)) if p.exists() else {}


def _tkey(cache, bench, major, minor):
    a, b = cache.get(f"{bench}|{major}"), cache.get(f"{bench}|{minor}")
    if a is None or b is None:
        return 9
    return 0 if a and b else 1 if a else 2 if b else 3


TYPE_TAG = {0: "[both] ", 1: "[s-cot] ", 2: "[w-cot] ", 3: "[plain] ", 9: ""}


def pair_heatmap(cond, out, acc_b, acc_c, benches, gaps):
    cache = _flags_cache()
    # one row list shared across benches, so the left panel's labels serve all
    seen = {}
    for k in acc_c:
        if k[0] in JUDGES:
            seen.setdefault((k[2], k[3]), []).append(k[1])
    def rkey(mm):
        b0 = sorted(seen[mm])[0]
        gap = np.mean([gaps.get((b, *mm), 0) for b in seen[mm]])
        return (_tkey(cache, b0, mm[0], mm[1]), gap)
    row_pairs = sorted(seen, key=rkey)
    fig, axes = plt.subplots(1, len(benches), figsize=(4.6 * len(benches), 6),
                             squeeze=False)
    for bi, bench in enumerate(benches):
        ax = axes[0][bi]
        grid = np.full((len(row_pairs), len(JUDGES)), np.nan)
        for pi, mm in enumerate(row_pairs):
            for ji, j in enumerate(JUDGES):
                c = acc_c.get((j, bench) + mm)
                b = acc_b.get((j, bench) + mm)
                if c and b and c[1] >= MIN_N and b[1] >= MIN_N:
                    grid[pi, ji] = c[0] / c[1] - b[0] / b[1]
        im = ax.imshow(grid, cmap="RdBu_r", vmin=-0.4, vmax=0.4, aspect="auto")
        ax.set_xticks(range(len(JUDGES)))
        ax.set_xticklabels(JUDGES, rotation=25, ha="right")
        ax.set_yticks(range(len(row_pairs)))
        if bi == 0:
            ax.set_yticklabels(
                [f"{TYPE_TAG[rkey(mm)[0]]}{mm[0][:14]}–{mm[1][:14]} ({rkey(mm)[1]:.0f})"
                 for mm in row_pairs], fontsize=7)
        else:
            ax.set_yticklabels([])
        ax.set_title(bench, fontsize=8, color="#52514e", loc="left")
    fig.colorbar(im, ax=axes[0].tolist(), label="Δ P(pick stronger) vs baseline",
                 shrink=0.7)
    for ext in ("png", "svg"):
        fig.savefig(out / f"{cond}_pair_heatmap.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)


def bias_delta(cond, out, tallies_b, tallies_c, benches):
    cache = _flags_cache()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, (name, tb, tc) in zip(axes, [("Δ P(picked slot 1)", tallies_b[0], tallies_c[0]),
                                         ("Δ P(picked longer)", tallies_b[1], tallies_c[1])]):
        ax.axvline(0, ls="--", c="#8a8a86", lw=1)

        def by_type(tallies):
            agg = defaultdict(lambda: [0, 0])
            for (j, bench, major, minor), v in tallies.items():
                ty = _tkey(cache, bench, major, minor)
                agg[(j, bench, ty)][0] += v[0]
                agg[(j, bench, ty)][1] += v[1]
            return agg

        ab, ac = by_type(tb), by_type(tc)
        ys, labels = [], []
        for bench in benches:
            for j in JUDGES:
                for ty in (0, 1, 2, 3, 9):
                    cb, cc = ab.get((j, bench, ty)), ac.get((j, bench, ty))
                    if not cb or not cc or cb[1] < MIN_N or cc[1] < MIN_N:
                        continue
                    pb, lob, hib = wilson(cb[0], cb[1])
                    pc, loc, hic = wilson(cc[0], cc[1])
                    d = pc - pb
                    se = math.sqrt(((hib - lob) / 3.92) ** 2 + ((hic - loc) / 3.92) ** 2)
                    y = len(labels)
                    ax.plot([d - 1.96 * se, d + 1.96 * se], [y, y], color="#2a78d6", lw=2)
                    ax.scatter([d], [y], color="#2a78d6", s=40, zorder=5)
                    tag = TYPE_TAG[ty].strip("[] ") or "all"
                    labels.append(f"{j} · {bench} · {tag}")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel(name)
        ax.set_xlim(-0.35, 0.35)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(out / f"{cond}_bias_delta.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)


MIN_ITEM_N = 4  # verdicts per item x judge (x pair type) on each side

PAIR_TYPES = ["both CoT", "stronger CoT only", "weaker CoT only", "neither"]


def item_scatter(cond, out, rows_b, rows_c, benches, cot_flags=None):
    """One column per judge; with cot_flags, one row per pair CoT-type."""
    def ptype(r):
        if cot_flags is None:
            return ""
        a = cot_flags.get((r["bench"], r["major"]))
        b = cot_flags.get((r["bench"], r["minor"]))
        if a is None or b is None:
            return None
        return PAIR_TYPES[0 if a and b else 1 if a else 2 if b else 3]

    def tally(rows):
        t = defaultdict(lambda: [0, 0])
        for r in rows:
            pt = ptype(r)
            if pt is None:
                continue
            k = (norm_judge(r["judge"]), r["bench"], r["item_id"], pt)
            t[k][1] += 1
            t[k][0] += r["picked_major"] == "True"
        return t

    item_b, item_c = tally(rows_b), tally(rows_c)
    types = [t for t in PAIR_TYPES if any(k[3] == t for k in item_c)] \
        if cot_flags else [""]
    judges = [j for j in JUDGES if any(k[0] == j for k in item_c)]
    colors = {"gpqa": "#2a78d6", "otis-aime": "#eb6834"}
    fig, axes = plt.subplots(len(types), len(judges),
                             figsize=(2.1 * len(judges), 2.1 * len(types) + 0.7),
                             sharex=True, sharey=True, squeeze=False)
    for ri, pt in enumerate(types):
        for ci, judge in enumerate(judges):
            ax = axes[ri][ci]
            ax.plot([0, 1], [0, 1], color="#b0b0b0", lw=1, ls="--")
            for bench in benches:
                xs, ys = [], []
                for (j, bn, item, t), (c, n) in item_c.items():
                    if (j, bn, t) != (judge, bench, pt) or n < MIN_ITEM_N:
                        continue
                    b = item_b.get((j, bn, item, t))
                    if not b or b[1] < MIN_ITEM_N:
                        continue
                    xs.append(b[0] / b[1])
                    ys.append(c / n)
                ax.scatter(xs, ys, s=9, alpha=0.4, color=colors[bench],
                           edgecolors="white", linewidths=0.3)
                if len(xs) >= 3:
                    mx, my = np.mean(xs), np.mean(ys)
                    ex = 1.96 * np.std(xs, ddof=1) / math.sqrt(len(xs))
                    ey = 1.96 * np.std(ys, ddof=1) / math.sqrt(len(ys))
                    ax.errorbar([mx], [my], xerr=[ex], yerr=[ey], fmt="D",
                                ms=6, color=colors[bench], markeredgecolor="black",
                                markeredgewidth=0.8, ecolor="black",
                                elinewidth=1.1, capsize=2, zorder=6)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xticks([0, 0.5, 1.0])
            ax.set_yticks([0, 0.5, 1.0])
            ax.set_aspect("equal")
            ax.spines[["top", "right"]].set_visible(False)
            if ri == 0:
                ax.text(0.5, 1.03, judge, transform=ax.transAxes, ha="center",
                        fontsize=9, color="#52514e")
        label = f"{pt}\n" if pt else ""
        axes[ri][0].set_ylabel(label + "P(pick stronger),\ncondition", fontsize=8)
    fig.supxlabel("per-item P(pick stronger),\nbaseline", fontsize=8)
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=b)
               for b, c in colors.items() if b in benches]
    handles.append(plt.Line2D([], [], marker="D", ls="", color="#52514e",
                              markeredgecolor="black", label="mean ± 95% CI"))
    fig.legend(handles=handles, frameon=False, fontsize=8, ncol=3,
               loc="lower center", bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    for ext in ("png", "svg"):
        fig.savefig(out / f"{cond}_item_scatter.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    rows = [r for r in csv.DictReader(open(BASE / "verdicts.csv"))
            if r["parse_ok"] == "True"]
    benches = sorted({r["bench"] for r in rows})
    gaps = {}
    paths = {}
    for bench in benches:
        members = load_members(ROOT / "external_data" / f"epoch-{bench}",
                               ROOT / "model-sampler" / "model_pool.json")
        eci = {m["short"]: m["eci"] for m in members}
        for m in members:
            paths[(bench, m["short"])] = m["path"]
        for r in rows:
            if r["bench"] == bench and eci.get(r["major"]) and eci.get(r["minor"]):
                gaps[(bench, r["major"], r["minor"])] = abs(eci[r["major"]] - eci[r["minor"]])

    e_models = {(r["bench"], m) for r in rows if r["condition"].startswith("e_")
                for m in (r["major"], r["minor"])}
    cot_flags = {k: cot_ok(paths[k]) for k in e_models if k in paths}

    base_rows = [r for r in rows if r["condition"] == "a_baseline"]
    for cond, (key, folder) in COMPARISONS.items():
        cond_rows = [r for r in rows if r["condition"] == key]
        if not cond_rows:
            continue
        cells = {(norm_judge(r["judge"]), r["bench"], r["major"], r["minor"])
                 for r in cond_rows}
        matched_base = [r for r in base_rows
                        if (norm_judge(r["judge"]), r["bench"], r["major"], r["minor"]) in cells]
        acc_b, slot_b, long_b, _ = cell_rates(matched_base)
        acc_c, slot_c, long_c, _ = cell_rates(cond_rows)
        out = BASE / folder / "contrasts"
        out.mkdir(parents=True, exist_ok=True)
        pair_heatmap(cond, out, acc_b, acc_c, benches, gaps)
        bias_delta(cond, out, (slot_b, long_b), (slot_c, long_c), benches)
        item_scatter(cond, out, matched_base, cond_rows, benches,
                     cot_flags if cond == "cot" else None)
        print(f"-> {out}/{cond}_{{pair_heatmap,bias_delta,item_scatter}}.png")


if __name__ == "__main__":
    main()
