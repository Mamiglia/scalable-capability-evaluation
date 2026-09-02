"""Figures for the eval-variation conditions, from verdicts.csv + eval logs.

Nature-style: small panels, 8pt, CIs at the judge level (the replication
unit), legends outside.

Usage: figures.py [--verdicts judge_results/eval_prompt_variation/verdicts.csv]
                  [--out judge_results/eval_prompt_variation/figures]
"""

import argparse
import csv
import glob
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from scipy import stats
from inspect_ai.log import read_eval_log

sys.path.insert(0, str(Path(__file__).parent))
from gen_configs import CONDITIONS, TONE_CONDITIONS, SUBJECTS_BY_ECI  # noqa: E402

plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7})

ORDER = ["baseline", "words200", "words50-100", "lenmatch-pooled", "confident", "matteroffact",
         "rewrite-self", "rewrite-standard"]
XLAB = ["baseline", "words200", "words50-100", "pooled", "confident", "matter-of-fact", "rewrite-self", "rewrite-std"]
JUDGE_COLORS = {"qwen3-8b": "#2a78d6", "qwen3-32b": "#eb6834",
                "qwen3-vl-30b-a3b-instruct": "#1baf7a", "ling-2.6-flash": "#eda100",
                "glm-4.6": "#e87ba4", "deepseek-v4-flash-0731": "#008300",
                "gemini-3.1-flash-lite": "#4a3aa7",
                # SWC local fleet (vllm serving; Qwen3-8B = same weights as
                # qwen3-8b but different serving, kept as its own series)
                "Qwen3-0.6B": "#9aa0a6", "Llama-3.2-3B-Instruct": "#8c6d4f",
                "Qwen3-8B": "#77aadd"}
SHORT = {"qwen3-8b": "Qwen3 8B", "qwen3-32b": "Qwen3 32B",
         "qwen3-vl-30b-a3b-instruct": "Qwen3 VL 30B", "ling-2.6-flash": "Ling 2.6",
         "glm-4.6": "GLM-4.6", "deepseek-v4-flash-0731": "DeepSeek V4F",
         "gemini-3.1-flash-lite": "Gemini 3.1 FL",
         "Qwen3-0.6B": "Qwen3 0.6B (swc)", "Llama-3.2-3B-Instruct": "Llama 3B (swc)",
         "Qwen3-8B": "Qwen3 8B (swc)"}
GAP_CONDS = {"baseline": "#5b5b57", "words200": "#2a78d6", "matteroffact": "#eb6834"}


def ax_house(ax):
    ax.grid(True, alpha=0.25, linewidth=0.4)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def chance(ax, y=0.5):
    ax.axhline(y, ls="--", c="#8a8a86", lw=0.8)


def save(fig, path):
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(f"{path}.{ext}", dpi=300)
    plt.close(fig)


def rate_plot(rows, key, ylabel, path, delta_panel=False):
    """Per-judge rates + cross-judge mean (t-based 95% CI, N = judges).
    delta_panel adds paired per-judge deltas vs baseline below, same cells."""
    if delta_panel:
        fig, (ax, axd) = plt.subplots(2, 1, figsize=(4.6, 4.2), sharex=True,
                                      height_ratios=[2, 1.3])
    else:
        fig, ax = plt.subplots(figsize=(4.6, 2.8))
        axd = None
    ax_house(ax)
    chance(ax)
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r[key] != "":
            by[r["judge"]][r["condition"]].append(int(r[key]))
    # the mean is only comparable across columns on a FIXED judge set: the
    # six api judges present in every completed condition
    core = {"deepseek-v4-flash-0731", "gemini-3.1-flash-lite", "glm-4.6",
            "ling-2.6-flash", "qwen3-32b", "qwen3-vl-30b-a3b-instruct"}
    rates_by_cond, deltas_by_cond = defaultdict(list), defaultdict(list)
    for judge, conds in sorted(by.items()):
        xs = [i for i, c in enumerate(ORDER) if c in conds]
        ys = [sum(conds[ORDER[i]]) / len(conds[ORDER[i]]) for i in xs]
        if judge in core:
            for x, y in zip(xs, ys):
                rates_by_cond[x].append(y)
        ax.plot(xs, ys, "-o", ms=2, lw=0.7, alpha=0.4, color=JUDGE_COLORS[judge], label=SHORT[judge])
        if axd is not None and judge in core and "baseline" in conds:
            base = sum(conds["baseline"]) / len(conds["baseline"])
            dxs = [x for x in xs if ORDER[x] != "baseline"]
            dys = [y - base for x, y in zip(xs, ys) if ORDER[x] != "baseline"]
            for x, d in zip(dxs, dys):
                deltas_by_cond[x].append(d)
            axd.plot(dxs, dys, "-o", ms=2, lw=0.7, alpha=0.4, color=JUDGE_COLORS[judge])
    mx = sorted(x for x in rates_by_cond if len(rates_by_cond[x]) >= 3)
    my = [sum(rates_by_cond[x]) / len(rates_by_cond[x]) for x in mx]
    ns = [len(rates_by_cond[x]) for x in mx]
    me = [stats.sem(rates_by_cond[x]) * stats.t.ppf(0.975, n - 1) for x, n in zip(mx, ns)]
    ax.errorbar(mx, my, yerr=me, fmt="-o", ms=3.5, lw=1.3, capsize=0, color="black",
                label=f"mean of {len(core)} shared judges ±95% CI", zorder=5)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              columnspacing=0.8, handlelength=1.2)
    if axd is not None:
        ax_house(axd)
        axd.axhline(0, ls="--", c="#8a8a86", lw=0.8)
        dx = sorted(x for x in deltas_by_cond if len(deltas_by_cond[x]) >= 3)
        dy = [sum(deltas_by_cond[x]) / len(deltas_by_cond[x]) for x in dx]
        dn = [len(deltas_by_cond[x]) for x in dx]
        de = [stats.sem(deltas_by_cond[x]) * stats.t.ppf(0.975, n - 1) for x, n in zip(dx, dn)]
        axd.errorbar(dx, dy, yerr=de, fmt="-o", ms=3.5, lw=1.3, capsize=0, color="black", zorder=5)
        axd.set_ylabel("Δ vs baseline")
        axd.set_xticks(range(len(ORDER)), XLAB, rotation=90)
    else:
        ax.set_xticks(range(len(ORDER)), XLAB, rotation=90)
    save(fig, path)


def pool_plot(rows, path):
    items = defaultdict(set)
    for r in rows:
        items[r["condition"]].add((r["major"], r["minor"], r["item"]))
    fig, ax = plt.subplots(figsize=(3.0, 2.2))
    ax_house(ax)
    xs = [i for i, c in enumerate(ORDER) if c in items]
    vals = [len(items[ORDER[i]]) for i in xs]
    ax.bar(xs, vals, color="#2a78d6", width=0.6)
    for x, v in zip(xs, vals):
        ax.annotate(f"{v:,}", (x, v), ha="center", va="bottom", fontsize=6)
    ax.set_xticks(range(len(ORDER)), XLAB, rotation=90)
    ax.set_ylabel("length-matched items")
    save(fig, path)


def accuracy_change_plot(path):
    conds = {**CONDITIONS, **TONE_CONDITIONS}
    accs = {}
    for cond, d in conds.items():
        for m, _ in SUBJECTS_BY_ECI:
            sub = f"{m}-rescored" if m == "gemini-3.1-flash-lite" else m
            # from samples, never the header: the 08-08 gemini baseline rescore
            # fixed sample scores only and left the header accuracy stale
            log = read_eval_log(sorted(glob.glob(f"{d}/{sub}/0/*.eval"))[0])
            accs[(cond, m)] = sum(
                1 for s in log.samples if s.scores and s.scores["choice"].value == "C"
            ) / len(log.samples)
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    ax_house(ax)
    ax.axhline(0, ls="--", c="#8a8a86", lw=0.8)
    plot_conds = [c for c in ORDER if c in conds and c != "baseline"]
    deltas = defaultdict(list)
    for m, _ in SUBJECTS_BY_ECI:
        ys = [accs[(c, m)] - accs[("baseline", m)] for c in plot_conds]
        for x, y in enumerate(ys):
            deltas[x].append(y)
        ax.plot(range(len(plot_conds)), ys, "-o", ms=2, lw=0.7, alpha=0.4,
                color=JUDGE_COLORS[m], label=SHORT[m])
    dx = sorted(deltas)
    dy = [sum(deltas[x]) / len(deltas[x]) for x in dx]
    de = [stats.sem(deltas[x]) * stats.t.ppf(0.975, len(deltas[x]) - 1) for x in dx]
    ax.errorbar(dx, dy, yerr=de, fmt="-o", ms=3.5, lw=1.3, capsize=0, color="black",
                label=f"mean ±95% CI (N={len(SUBJECTS_BY_ECI)} models)", zorder=5)
    ax.set_xticks(range(len(plot_conds)), [XLAB[ORDER.index(c)] for c in plot_conds], rotation=90)
    ax.set_ylabel("Δ accuracy vs baseline")
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              columnspacing=0.8, handlelength=1.2)
    save(fig, path)


# judge ECI for the oversight filter; SWC small judges sit below everything
JUDGE_ECI = dict(SUBJECTS_BY_ECI) | {"Qwen3-8B": 127.9, "Qwen3-0.6B": 0.0,
                                     "Llama-3.2-3B-Instruct": 0.0}


def gap_plot(rows, path, n_bins=6, weak_judges_only=False):
    """Gap-binned mean with t-based 95% CI across judges (the replication
    unit). Only pairs judged in every plotted condition, so bins compare the
    same matchups (baseline lacks e.g. the deepseek pairs otherwise).
    weak_judges_only keeps verdicts where judge ECI < the stronger member's."""
    eci = dict(SUBJECTS_BY_ECI)
    if weak_judges_only:
        rows = [r for r in rows if JUDGE_ECI[r["judge"]] < eci[r["major"]]]
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    ax_house(ax)
    chance(ax)
    pair_gap = lambda r: eci[r["major"]] - eci[r["minor"]]
    per_cond_pairs = {c: {(r["major"], r["minor"]) for r in rows if r["condition"] == c}
                      for c in GAP_CONDS}
    common = set.intersection(*per_cond_pairs.values())
    rows = [r for r in rows if (r["major"], r["minor"]) in common]
    print(f"gap curve: {len(common)} pairs present in all of {list(GAP_CONDS)}")
    gaps_all = sorted({round(pair_gap(r), 2) for r in rows})
    edges = [gaps_all[len(gaps_all) * k // n_bins] for k in range(n_bins)] + [gaps_all[-1] + 1]
    for cond, color in GAP_CONDS.items():
        crows = [r for r in rows if r["condition"] == cond]
        xs, ys, es = [], [], []
        for k in range(n_bins):
            binned = [r for r in crows if edges[k] <= pair_gap(r) < edges[k + 1]]
            per_judge = defaultdict(list)
            for r in binned:
                per_judge[r["judge"]].append(int(r["picked_major"]))
            rates = [sum(v) / len(v) for v in per_judge.values()]
            if len(rates) < 3:
                continue
            xs.append(sum(pair_gap(r) for r in binned) / len(binned))
            m = sum(rates) / len(rates)
            se = stats.sem(rates)
            ys.append(m)
            es.append(se * stats.t.ppf(0.975, len(rates) - 1))
        ax.errorbar(xs, ys, yerr=es, fmt="-o", ms=3.5, lw=1, capsize=0, color=color,
                    label=cond, zorder=4)
    ax.set_xlabel("pair ECI gap")
    ax.set_ylabel("P(pick stronger)")
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    save(fig, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verdicts", type=Path, default=Path("judge_results/eval_prompt_variation/verdicts.csv"))
    ap.add_argument("--out", type=Path, default=Path("judge_results/eval_prompt_variation/figures"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(args.verdicts)))
    rate_plot(rows, "picked_major", "P(pick stronger)", args.out / "1_pick_stronger_by_condition", delta_panel=True)
    rate_plot(rows, "picked_longer", "P(pick longer)", args.out / "2_pick_longer_by_condition")
    pool_plot(rows, args.out / "3_matched_pool_by_condition")
    accuracy_change_plot(args.out / "4_accuracy_change")
    gap_plot(rows, args.out / "5_gap_curve_matteroffact")
    gap_plot(rows, args.out / "6_gap_curve_weak_judges", weak_judges_only=True)
    print("figures ->", args.out)


if __name__ == "__main__":
    main()
