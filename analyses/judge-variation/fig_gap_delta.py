"""Gap curves with a delta panel: baseline vs intervention, per bench.

Top row: P(pick stronger) vs ECI gap, across-judge bin means with t-based 95%
CIs, one series per condition (baseline + interventions). Bins with fewer than
MIN_UNITS judges show the per-judge points and a CI-less mean instead.
Bottom row: intervention minus baseline per gap bin (matched judge x pair
cells), zero line = no effect.

Outputs:
  cot_inclusion/gap_delta.png                (baseline vs cot, both benches)
  ignore_instructions/gap_delta.png          (baseline vs the 3 prompt conditions)

Usage: uv run python fig_gap_delta.py
"""
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "axes.linewidth": 0.5, "xtick.major.width": 0.5,
                     "ytick.major.width": 0.5})

plt.rcParams.update({"font.size": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7})

from pool import load_members

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
BASE = ROOT / "judge_results" / "judging_variations"

BINS = [(4, 8), (8, 12), (12, 16), (16, 20), (20, 26), (26, 40)]
E_SERVING = {"qwen3-0.6b-yarn": "qwen3-0.6b", "qwen3-8b-yarn": "qwen3-8b",
             "qwen3-14b-yarn": "qwen3-14b", "qwen3-32b-yarn": "qwen3-32b"}
JUDGE_COLORS = {"qwen3-0.6b": "#2a78d6", "llama-3.2-3b": "#eb6834",
                "qwen3-8b": "#1baf7a", "qwen3-14b": "#e87ba4",
                "qwen3-32b": "#eda100", "gpt-oss-20b": "#9085e9",
                "gpt-oss-120b": "#4a3aa7"}
# conditions are neutral so the palette stays exclusive to judges
COND_STYLE = {"baseline": ("#9a9a97", "-"), "cot": ("#222222", "-"),
              "ignore-style": ("#222222", "--"),
              "ignore-style-conf": ("#222222", ":"),
              "pick-weaker": ("#222222", "-.")}
COND_COLORS = {k: v[0] for k, v in COND_STYLE.items()}
COND_KEYS = {"baseline": "a_baseline", "cot": "e_with_reasoning",
             "ignore-style": "b_ignore_superficial",
             "ignore-style-conf": "c_ignore_superficial_confidence",
             "pick-weaker": "d_pick_weaker"}


def wilson(c, n, z=1.96):
    if n == 0:
        return 0.5, 0.0, 0.0
    p, d = c / n, 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, center - half, center + half


def gap_of(bench, ecis, major, minor):
    a, b = ecis[bench].get(major), ecis[bench].get(minor)
    return abs(a - b) if a is not None and b is not None else None


def cell_key(r):
    judge = E_SERVING.get(r["judge"], r["judge"])
    return (judge, r["bench"], r["major"], r["minor"])


def binned(rows, bench, ecis, cells=None):
    """bin index -> [correct, n] over verdicts (optionally cell-restricted)."""
    out = defaultdict(lambda: [0, 0])
    for r in rows:
        if r["bench"] != bench or (cells is not None and cell_key(r) not in cells):
            continue
        g = gap_of(bench, ecis, r["major"], r["minor"])
        if g is None:
            continue
        for i, (lo, hi) in enumerate(BINS):
            if lo <= g < hi:
                out[i][1] += 1
                out[i][0] += r["picked_major"] == "True"
                break
    return out


def binned_judges(rows, bench, ecis, cells):
    """bin index -> base-judge -> [picked-stronger, n] (yarn folds into its judge)."""
    out = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for r in rows:
        if r["bench"] != bench or (cells is not None and cell_key(r) not in cells):
            continue
        g = gap_of(bench, ecis, r["major"], r["minor"])
        if g is None:
            continue
        for i, (lo, hi) in enumerate(BINS):
            if lo <= g < hi:
                j = E_SERVING.get(r["judge"], r["judge"])
                out[i][j][1] += 1
                out[i][j][0] += r["picked_major"] == "True"
                break
    return out


MIN_JUDGE_N = 20   # verdicts a judge needs in a bin to count as a unit
MIN_UNITS = 3      # judges needed for a CI


def _tci(vals):
    from scipy.stats import t as tdist
    m = float(np.mean(vals))
    if len(vals) < 2:
        return m, 0.0
    half = float(tdist.ppf(0.975, len(vals) - 1) * np.std(vals, ddof=1) / math.sqrt(len(vals)))
    return m, half


def draw(fig_conds, out_path, rows_by_cond, benches, ecis):
    """Unit of replication = judge model: bin points are across-judge means with
    t-based 95% CIs; the delta panel uses paired per-judge deltas."""
    fig, axes = plt.subplots(2, len(benches), figsize=(2.2 * len(benches), 2.9),
                             sharex=True, squeeze=False,
                             gridspec_kw={"height_ratios": [2, 1]})
    centers = [(lo + hi) / 2 for lo, hi in BINS]
    n_units = set()
    seen_judges = {}
    d_extent = [0.05]
    for bi, bench in enumerate(benches):
        top, bot = axes[0][bi], axes[1][bi]
        for ax in (top, bot):
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", color="#eeeeee", lw=0.6, zorder=-1)
        top.axhline(0.5, ls="--", c="#8a8a86", lw=0.8)
        bot.axhline(0.0, ls="--", c="#8a8a86", lw=0.8)

        base_rows = rows_by_cond["baseline"]
        common = None
        for cond in fig_conds:
            cells = {cell_key(r) for r in rows_by_cond[cond] if r["bench"] == bench}
            common = cells if common is None else common & cells
        bj_base = binned_judges(base_rows, bench, ecis, common)
        judge_traj = defaultdict(list)
        for ci_, cond in enumerate(fig_conds):
            xoff = (ci_ - (len(fig_conds) - 1) / 2) * 0.55
            bj = binned_judges(rows_by_cond[cond], bench, ecis, common)
            xs, ys, es, dxs, dys, des = [], [], [], [], [], []
            for i in range(len(BINS)):
                rates = {j: c / n for j, (c, n) in bj.get(i, {}).items() if n >= MIN_JUDGE_N}
                if not rates:
                    continue
                n_units.add(len(rates))
                if len(rates) >= MIN_UNITS:
                    m, half = _tci(list(rates.values()))
                else:
                    # too few judges for a CI: show the units, mean has no bar
                    m, half = float(np.mean(list(rates.values()))), 0.0
                    for jj, v in rates.items():
                        seen_judges[jj] = JUDGE_COLORS.get(jj, "#999999")
                        top.scatter([centers[i] + xoff], [v], s=4,
                                    color=JUDGE_COLORS.get(jj, "#999999"),
                                    alpha=0.85, zorder=3, linewidths=0)
                xs.append(centers[i] + xoff); ys.append(m); es.append(half)
                if cond != "baseline":
                    braw = {j: c / n for j, (c, n) in bj_base.get(i, {}).items() if n >= MIN_JUDGE_N}
                    paired = {j: rates[j] - braw[j] for j in rates if j in braw}
                    if paired:
                        dm, dh = (_tci(list(paired.values()))
                                  if len(paired) >= MIN_UNITS
                                  else (float(np.mean(list(paired.values()))), 0.0))
                        dxs.append(centers[i] + xoff); dys.append(dm); des.append(dh)
                        d_extent.append(abs(dm) + dh)
                        for jj, dv in paired.items():
                            seen_judges[jj] = JUDGE_COLORS.get(jj, "#999999")
                            judge_traj[jj].append((centers[i] + xoff + 0.35, dv))
                            d_extent.append(abs(dv))
                            bot.scatter([centers[i] + xoff + 0.35], [dv], s=4,
                                        color=JUDGE_COLORS.get(jj, "#999999"),
                                        alpha=0.85, zorder=3, linewidths=0)
            col, ls = COND_STYLE[cond]
            top.errorbar(xs, ys, yerr=es, color=col, ls=ls, marker="o", ms=3,
                         lw=1, capsize=0, elinewidth=1, label=cond)
            if cond != "baseline" and dxs:
                bot.errorbar(dxs, dys, yerr=des, color="#404040", ls=ls, marker="o",
                             ms=3, lw=1, capsize=0, elinewidth=1)
        for jj, pts in judge_traj.items():
            pts = sorted(pts)
            bot.plot([x for x, _ in pts], [d for _, d in pts], lw=0.5, alpha=0.35,
                     color=JUDGE_COLORS.get(jj, "#999999"), zorder=2)
        top.set_title(bench, fontsize=8, color="#52514e", loc="left")
        top.set_ylim(0.3, 1.0)
        bot.set_xlabel("capability gap (ECI)")
        if bi == 0:
            top.set_ylabel("P(pick stronger)")
            bot.set_ylabel("Δ vs baseline")
    dlim = max(d_extent) * 1.08
    for bi in range(len(benches)):
        axes[1][bi].set_ylim(-dlim, dlim)
    handles, labels = axes[0][0].get_legend_handles_labels()
    handles += [plt.Line2D([], [], marker="o", ls="", ms=3, color=c, label=j)
                for j, c in JUDGE_COLORS.items() if j in seen_judges]
    labels += [j for j in JUDGE_COLORS if j in seen_judges]
    fig.legend(handles, labels, frameon=False, ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, 1.0), fontsize=6, columnspacing=0.9,
               handletextpad=0.3)
    if n_units:
        lo, hi = min(n_units), max(n_units)
        n_txt = f"N={lo}" if lo == hi else f"N={lo}–{hi}"
        desc = "95% CI across judges" if hi >= MIN_UNITS else "per-judge points, no CI"
        fig.text(0.99, 1.0, f"{desc} ({n_txt}/bin)",
                 ha="right", fontsize=6, color="#8a8a86")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(str(out_path) + f".{ext}", dpi=300, bbox_inches="tight")
    print(f"-> {out_path}.png")


def main():
    rows = [r for r in csv.DictReader(open(BASE / "verdicts.csv"))
            if r["parse_ok"] == "True" and r["condition"] != "attribution"]
    benches = sorted({r["bench"] for r in rows})
    ecis = {}
    for bench in benches:
        members = load_members(ROOT / "external_data" / f"epoch-{bench}",
                               ROOT / "model-sampler" / "model_pool.json")
        ecis[bench] = {m["short"]: m["eci"] for m in members}

    rows_by_cond = {name: [r for r in rows if r["condition"] == key]
                    for name, key in COND_KEYS.items()}

    draw(["baseline", "cot"], BASE / "cot_inclusion" / "gap_delta",
         rows_by_cond, benches, ecis)

    # typed variant: one (curve+delta) row block per pair reasoning-type
    import json
    cache_p = BASE / "cot_flags.json"
    cache = json.load(open(cache_p)) if cache_p.exists() else {}
    TYPES = {0: "both reasoning", 1: "stronger reasoning only",
             2: "weaker reasoning only", 3: "neither (control)"}

    def rtype(r):
        a = cache.get(f"{r['bench']}|{r['major']}")
        b = cache.get(f"{r['bench']}|{r['minor']}")
        if a is None or b is None:
            return None
        return 0 if a and b else 1 if a else 2 if b else 3

    # split by whether the JUDGE is a reasoning model (thinks before answering).
    # Caveat: only llama-3.2-3b is non-reasoning, so that class is one judge.
    REASONING_JUDGES = {"qwen3-0.6b", "qwen3-0.6b-yarn", "qwen3-8b", "qwen3-8b-yarn",
                        "qwen3-14b", "qwen3-14b-yarn", "qwen3-32b", "qwen3-32b-yarn",
                        "gpt-oss-20b", "gpt-oss-120b"}
    # one combined figure: class = linestyle (solid reasoning / dashed plain),
    # condition = grey baseline / black cot; deltas per class below
    CLS = {"reasoning": ("-", lambda j: j in REASONING_JUDGES),
           "non-reasoning": ("--", lambda j: j not in REASONING_JUDGES)}
    fig, axes = plt.subplots(2, len(benches), figsize=(2.2 * len(benches), 2.9),
                             sharex=True, squeeze=False,
                             gridspec_kw={"height_ratios": [2, 1]})
    centers = [(lo + hi) / 2 for lo, hi in BINS]
    n_units = set()
    for bi, bench in enumerate(benches):
        top, bot = axes[0][bi], axes[1][bi]
        for ax in (top, bot):
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", color="#eeeeee", lw=0.6, zorder=-1)
        top.axhline(0.5, ls="--", c="#c9c9c6", lw=0.8)
        bot.axhline(0.0, ls="--", c="#c9c9c6", lw=0.8)
        cells_cot = {cell_key(r) for r in rows_by_cond["cot"] if r["bench"] == bench}
        cells_base = {cell_key(r) for r in rows_by_cond["baseline"] if r["bench"] == bench}
        common = cells_cot & cells_base
        for cli, (cls, (ls, keep)) in enumerate(CLS.items()):
            sub = {c: [r for r in rs if keep(r["judge"])] for c, rs in rows_by_cond.items()}
            bj_base = binned_judges(sub["baseline"], bench, ecis, common)
            bj_cot = binned_judges(sub["cot"], bench, ecis, common)
            for cond, bj, col in [("baseline", bj_base, "#9a9a97"), ("cot", bj_cot, "#222222")]:
                xs, ys, es = [], [], []
                xoff = (cli - 0.5) * 0.55 + (0.25 if cond == "cot" else -0.25)
                for i in range(len(BINS)):
                    rates = {j: c / n for j, (c, n) in bj.get(i, {}).items() if n >= MIN_JUDGE_N}
                    if not rates:
                        continue
                    n_units.add(len(rates))
                    m, half = _tci(list(rates.values())) if len(rates) >= 2 else (list(rates.values())[0], 0)
                    xs.append(centers[i] + xoff); ys.append(m); es.append(half)
                if xs:
                    top.errorbar(xs, ys, yerr=es, color=col, ls=ls, marker="o", ms=2.5,
                                 lw=1, capsize=0, elinewidth=1,
                                 label=f"{cls} · {cond}" if bi == 0 else None)
            # class delta with per-judge dots (reasoning left, non-reasoning right)
            dxs, dys, des = [], [], []
            side = -0.45 if cls == "reasoning" else 0.45
            traj = defaultdict(list)
            for i in range(len(BINS)):
                rb = {j: c / n for j, (c, n) in bj_base.get(i, {}).items() if n >= MIN_JUDGE_N}
                rc = {j: c / n for j, (c, n) in bj_cot.get(i, {}).items() if n >= MIN_JUDGE_N}
                paired = {j: rc[j] - rb[j] for j in rc if j in rb}
                if not paired:
                    continue
                vals = list(paired.values())
                dm, dh = _tci(vals) if len(vals) >= 2 else (vals[0], 0)
                x0 = centers[i] + (cli - 0.5) * 0.55
                dxs.append(x0); dys.append(dm); des.append(dh)
                for jj, dv in paired.items():
                    cj = JUDGE_COLORS.get(jj, "#999999")
                    bot.scatter([x0 + side], [dv], s=4, color=cj, alpha=0.85,
                                zorder=3, linewidths=0)
                    traj[jj].append((x0 + side, dv))
            for jj, pts in traj.items():
                pts = sorted(pts)
                bot.plot([x for x, _ in pts], [d for _, d in pts], lw=0.5,
                         alpha=0.35, color=JUDGE_COLORS.get(jj, "#999999"), zorder=2)
            if dxs:
                bot.errorbar(dxs, dys, yerr=des, color="#222222", ls=ls, marker="o",
                             ms=2.5, lw=1, capsize=0, elinewidth=1,
                             label=f"{cls}" if bi == 0 else None)
        top.set_title(bench, fontsize=8, color="#52514e", loc="left")
        top.set_ylim(0.3, 1.0)
        bot.set_ylim(-0.3, 0.3)
        bot.set_xlabel("capability gap (ECI)")
        if bi == 0:
            top.set_ylabel("P(pick stronger)")
            bot.set_ylabel("Δ vs baseline")
    h1, l1 = axes[0][0].get_legend_handles_labels()
    fig.legend(h1, l1, frameon=False, ncol=2, loc="lower center",
               bbox_to_anchor=(0.5, 1.0), fontsize=6)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(BASE / "cot_inclusion" / f"gap_delta_by_judge_class.{ext}",
                    dpi=300, bbox_inches="tight")
    print(f"-> {BASE}/cot_inclusion/gap_delta_by_judge_class.png")

    for t, label in TYPES.items():
        sub = {c: [r for r in rs if rtype(r) == t] for c, rs in rows_by_cond.items()}
        if sum(len(v) for c, v in sub.items() if c == "cot") < 500:
            print(f"typed gap_delta: skipping '{label}' (too little cot data yet)")
            continue
        draw(["baseline", "cot"],
             BASE / "cot_inclusion" / f"gap_delta_type{t}_{label.split()[0]}",
             sub, benches, ecis)
    draw(["baseline", "ignore-style", "ignore-style-conf"],
         BASE / "ignore_instructions" / "gap_delta", rows_by_cond, benches, ecis)
    draw(["baseline", "pick-weaker"],
         BASE / "ignore_instructions" / "gap_delta_pick_weaker", rows_by_cond, benches, ecis)


if __name__ == "__main__":
    main()
