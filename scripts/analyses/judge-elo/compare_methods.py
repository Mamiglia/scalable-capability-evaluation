# scripts/analyses/judge-elo/compare_methods.py
"""Pairwise vs direct-scoring, one consistent set of definitions across all views.

Every view uses item-matched data, drops single-order verdicts (position-bias
leak), labels the methods plainly, and reports uncertainty at the right unit
(bootstrap over models for a single tau; the benchmark as the unit of
independence for a pairwise-vs-direct claim, since judges share a bench's pool).

Three methods rank models:
  pairwise      Bradley-Terry on counterbalanced verdicts
  direct-elo    BT on induced per-item picks from 0-100 scores (item win-rate)
  direct-score  mean 0-100 score per model  (uses graded magnitude)

Outputs (judge_results/judge_elo/figures/pairwise-vs-direct/ + a CSV):
  ranking_by_judge.png      overall ranking quality (all common models)
  oversight_by_judge.png    models stronger than the judge
  budget_curve.png          tau vs judge calls, from 50 calls up
  headtohead_by_filter.png  per-pair accuracy, matched, across pair filters
  oversight_tau_ci.csv      oversight tau with bootstrap CIs

Reads only the log-verified raw CSVs. Run: uv run python compare_methods.py
"""
import csv
import functools
import math
import random
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kendalltau, t as t_dist

import elo

ROOT = elo.ROOT
OUT = ROOT / "judge_results" / "judge_elo"
FIG = OUT / "figures" / "pairwise-vs-direct"
BENCHES = ["gpqa", "otis-aime"]
# ordered by judge_eci (weak -> strong); gpt-oss-120b (140.48) is the strongest local judge
JUDGES = ["llama-3.2-3b", "qwen3-8b", "qwen3-14b", "qwen3-32b", "ling-2.6-flash",
          "gpt-oss-20b", "gpt-oss-120b"]
# pairwise verdicts: canonical file + the new-judges sidecar (not appended upstream)
VERDICT_FILES = ["judge_results/judging_variations/verdicts.csv",
                 "judge_results/judging_variations/verdicts_newjudges.csv"]
METHODS = ["pairwise", "direct-elo", "direct-score"]
COLORS = {"pairwise": "#2a78d6", "direct-elo": "#eda100", "direct-score": "#1baf7a"}
AS_WIN = {"True": 1.0, "False": 0.0}
CLOSE = 10.0
NBOOT = 3000
RC = {"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
      "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False}
_rng = random.Random(0)


# ---------- data ----------
@functools.lru_cache(maxsize=None)
def load(bench, judge):
    """Return (pw_item, scores). pw_item: (a,b,item)->[sum picked_major, n_orders]."""
    pw_item = defaultdict(lambda: [0.0, 0])
    for vf in VERDICT_FILES:
        p = ROOT / vf
        if not p.exists():
            continue
        for r in csv.DictReader(open(p)):
            if (r["bench"], r["condition"], r["judge"], r["parse_ok"]) == (bench, "a_baseline", judge, "True"):
                pw_item[(r["major"], r["minor"], r["item_id"])][0] += AS_WIN[r["picked_major"]]
                pw_item[(r["major"], r["minor"], r["item_id"])][1] += 1
    scores = defaultdict(dict)
    for r in csv.DictReader(open(ROOT / "judge_results/direct_judge/scores.csv")):
        if (r["bench"], r["judge"], r["parse_ok"]) == (bench, judge, "True"):
            scores[r["model"]][r["item_id"]] = int(r["score"])
    return pw_item, scores


@functools.lru_cache(maxsize=None)
def rankings(bench, judge):
    """Full-data model strengths per method, MATCHED items, single-order dropped."""
    pw_item, scores = load(bench, judge)
    pw_wins = defaultdict(lambda: [0.0, 0])
    elo_wins = defaultdict(lambda: [0.0, 0])
    for (a, b, it), (s, n) in pw_item.items():
        if n < 2 or a not in scores or b not in scores or it not in scores[a] or it not in scores[b]:
            continue                                   # both orders AND both scores present
        pw_wins[(a, b)][0] += s / n; pw_wins[(a, b)][1] += 1
        sa, sb = scores[a][it], scores[b][it]
        elo_wins[(a, b)][0] += 0.5 if sa == sb else float(sa > sb)
        elo_wins[(a, b)][1] += 1
    pw = elo.fit_bt(dict(pw_wins)) if pw_wins else {}
    el = elo.fit_bt(dict(elo_wins)) if elo_wins else {}
    ms = {m: float(np.mean(list(d.values()))) for m, d in scores.items()}
    return {"pairwise": pw, "direct-elo": el, "direct-score": ms}


def tau_ci(rank, cap, models):
    x = np.array([rank[m] for m in models]); y = np.array([cap[m] for m in models])
    t0 = kendalltau(x, y).statistic
    n = len(models); boots = []
    for _ in range(NBOOT):
        idx = [_rng.randrange(n) for _ in range(n)]
        xs, ys = x[idx], y[idx]
        if len(set(xs)) > 1 and len(set(ys)) > 1:
            boots.append(kendalltau(xs, ys).statistic)
    lo, hi = np.percentile(boots, [2.5, 97.5]) if boots else (t0, t0)
    return t0, lo, hi


# ---------- compute tau tables ----------
def tau_table(subset):
    """subset in {'all','strong'}. Returns {(target,bench,judge,method):(tau,lo,hi,n)}."""
    out = {}
    for bench in BENCHES:
        cfg = elo.load_config(bench); jeci = cfg.get("judge_eci", {})
        for judge in JUDGES:
            if judge not in jeci:
                continue
            rank = rankings(bench, judge)
            common = set(rank["pairwise"]) & set(rank["direct-elo"]) & set(rank["direct-score"])
            for target in ("accuracy", "eci"):
                cap = cfg.get(target, {})
                models = sorted(m for m in common if m in cap)
                if subset == "strong":
                    models = [m for m in models if cfg["eci"].get(m, -1) > jeci[judge]]
                if len(models) < 10:
                    continue
                for meth in METHODS:
                    out[(target, bench, judge, meth)] = (*tau_ci(rank[meth], cap, models), len(models))
    return out


def bench_clustered(table, target):
    """pairwise - direct-score, bench means, then t across benches (n=2)."""
    per_bench = defaultdict(list)
    for (t, b, j, m), (tau, *_1) in table.items():
        if t == target and m == "pairwise":
            ds = table.get((t, b, j, "direct-score"))
            if ds:
                per_bench[b].append(tau - ds[0])
    bm = [np.mean(v) for v in per_bench.values()]
    if len(bm) < 2:
        return None
    m, sd = np.mean(bm), np.std(bm, ddof=1)
    tc = t_dist.ppf(0.975, len(bm) - 1); half = tc * sd / np.sqrt(len(bm))
    return dict(benchmeans={b: round(np.mean(v), 3) for b, v in per_bench.items()},
                mean=m, lo=m - half, hi=m + half)


# ---------- figures ----------
def _dyn_ylim(vals, include, pad_frac=0.08):
    """Data-driven y-limits (never clip a CI); always keep the reference line in."""
    lo, hi = min(list(vals) + [include]), max(list(vals) + [include])
    pad = pad_frac * (hi - lo + 1e-9)
    return lo - pad, hi + pad


def fig_tau(table, fname):
    """rows = capability target; cols = bench; x = judge; a dot per method with its
    bootstrap CI, thin grey line linking a judge's three methods."""
    plt.rcParams.update(RC)
    targets = ["accuracy", "eci"]
    allv = [x for v in table.values() for x in (v[0], v[1], v[2])]
    ylo, yhi = _dyn_ylim(allv, include=0.0)
    fig, axes = plt.subplots(len(targets), len(BENCHES), figsize=(4.8, 3.4),
                             sharey=True, sharex="col", squeeze=False)
    for ri, target in enumerate(targets):
        for ci, bench in enumerate(BENCHES):
            ax = axes[ri][ci]
            judges = [j for j in JUDGES if (target, bench, j, "pairwise") in table]
            for k, j in enumerate(judges):                      # grey link within judge
                pts = [(k + (mi - 1) * 0.24, table[(target, bench, j, m)][0])
                       for mi, m in enumerate(METHODS) if (target, bench, j, m) in table]
                if len(pts) > 1:
                    ax.plot(*zip(*pts), color="#cccccc", lw=0.5, zorder=1)
            for mi, meth in enumerate(METHODS):
                off = (mi - 1) * 0.24
                xs, ys, los, his = [], [], [], []
                for k, j in enumerate(judges):
                    v = table.get((target, bench, j, meth))
                    if v:
                        xs.append(k + off); ys.append(v[0]); los.append(v[0] - v[1]); his.append(v[2] - v[0])
                ax.errorbar(xs, ys, yerr=[los, his], fmt="o", ms=3, lw=0.9, capsize=0,
                            color=COLORS[meth], zorder=3)
            ax.axhline(0, lw=0.8, ls="--", color="grey", zorder=0)
            ax.set_ylim(ylo, yhi)
            ax.set_xticks(range(len(judges)), judges if ri == len(targets) - 1 else [], rotation=90)
            if ri == 0:
                ax.set_title(bench, fontsize=7, color="grey", loc="left", pad=3)
            if ci == 0:
                ax.set_ylabel(f"τ vs {'ECI' if target == 'eci' else target}")
    handles = [plt.Line2D([], [], ls="", marker="o", ms=3, color=COLORS[m], label=m) for m in METHODS]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / fname, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"  {FIG / fname}")


# ---------- budget ----------
def budget_curve():
    """per (judge,B), averaged over seeds: ranking tau AND low-gap pair accuracy.
    Band later = 95% t CI across judges (the replication unit)."""
    # even (x2) spacing on the log axis; each line stops where its data runs out
    BUD = {"gpqa": [50, 100, 200, 400, 800, 1600, 3200],
           "otis-aime": [50, 100, 200, 400, 800]}
    SEEDS = 8
    tau_d, lg_d = {}, {}          # (bench,meth,B)->{judge: mean over seeds}
    for bench in BENCHES:
        cfg = elo.load_config(bench); eci = cfg["eci"]

        def is_close(a, b):
            return a in eci and b in eci and eci[a] != eci[b] and abs(eci[a] - eci[b]) < CLOSE

        def oriented(a, b, rate):     # rate = P(rank a above b); correct vs ECI
            return 0.5 if rate == 0.5 else float((rate > 0.5) == (eci[a] > eci[b]))

        for judge in JUDGES:
            pw_item, scores = load(bench, judge)
            verd = [(a, b, it, s / n) for (a, b, it), (s, n) in pw_item.items() if n >= 2]
            srows = [(m, i, s) for m, d in scores.items() for i, s in d.items()]
            by_item = defaultdict(dict)
            for m, i, s in srows:
                by_item[i][m] = s
            base_fs = {frozenset((a, b)) for a, b, _it, _r in verd}   # unordered baseline pairs
            base_pairs = {(a, b) for a, b, _it, _r in verd}
            picks = []
            for i, ms in by_item.items():
                mods = sorted(ms)
                for ai, a in enumerate(mods):
                    for b in mods[ai + 1:]:
                        if frozenset((a, b)) in base_fs:      # both orderings, not just lexical
                            picks.append((a, b, 0.5 if ms[a] == ms[b] else float(ms[a] > ms[b])))
            for B in BUD[bench]:
                tau, lg = defaultdict(list), defaultdict(list)
                for seed in range(SEEDS):
                    rnd = random.Random(seed * 131 + B)
                    if B <= len(verd):
                        sv = rnd.sample(verd, B); w = defaultdict(lambda: [0.0, 0])
                        for a, b, _it, r in sv:
                            w[(a, b)][0] += r; w[(a, b)][1] += 1
                        _tau_add(tau, "pairwise", elo_bt(w), eci)
                        cl = [oriented(a, b, s / n) for (a, b), (s, n) in w.items() if is_close(a, b)]
                        if len(cl) >= 5:
                            lg["pairwise"].append(np.mean(cl))
                    if B <= len(srows):
                        ss = rnd.sample(srows, B); mm = defaultdict(list)
                        for m, i, s in ss:
                            mm[m].append(s)
                        means = {m: np.mean(v) for m, v in mm.items()}
                        _tau_add(tau, "direct-score", means, eci)
                        cl = [oriented(a, b, 1.0 if means[a] > means[b] else (0.0 if means[a] < means[b] else 0.5))
                              for (a, b) in base_pairs if a in means and b in means and is_close(a, b)]
                        if len(cl) >= 5:
                            lg["direct-score"].append(np.mean(cl))
                    nip = B // 2
                    if 5 <= nip <= len(picks):
                        sp = rnd.sample(picks, nip); w = defaultdict(lambda: [0.0, 0])
                        for a, b, r in sp:
                            w[(a, b)][0] += r; w[(a, b)][1] += 1
                        _tau_add(tau, "direct-elo", elo_bt(w), eci)
                        cl = [oriented(a, b, s / n) for (a, b), (s, n) in w.items() if is_close(a, b)]
                        if len(cl) >= 5:
                            lg["direct-elo"].append(np.mean(cl))
                for meth in METHODS:
                    if tau[meth]:
                        tau_d.setdefault((bench, meth, B), {})[judge] = np.mean(tau[meth])
                    if lg[meth]:
                        lg_d.setdefault((bench, meth, B), {})[judge] = np.mean(lg[meth])
    return tau_d, lg_d, BUD


def elo_bt(wins):
    try:
        return elo.fit_bt(dict(wins))
    except Exception:
        return {}


def _tau_add(acc, meth, strengths, eci):
    ms = [m for m in strengths if m in eci]
    if len(ms) >= 10:
        acc[meth].append(kendalltau([strengths[m] for m in ms], [eci[m] for m in ms]).statistic)


def _budget_panels(fig, axes, series, BUD, chance, ylabel):
    """Shared plotter for a budget grid: series (bench,meth)->(xs,mean,lo,hi)."""
    allv = [v for (xs, mean, lo, hi) in series.values() for v in (lo + hi)]
    ylo, yhi = _dyn_ylim(allv, include=chance)
    for ci, bench in enumerate(BENCHES):
        ax = axes[0][ci]
        for meth in METHODS:
            xs, mean, lo, hi = series.get((bench, meth), ([], [], [], []))
            if xs:
                ax.fill_between(xs, lo, hi, color=COLORS[meth], alpha=0.15, lw=0)
                ax.plot(xs, mean, "-o", ms=2.5, lw=1, color=COLORS[meth])
        ax.set_xscale("log"); ax.set_xticks(BUD[bench])
        ax.set_xticklabels([str(x) for x in BUD[bench]], rotation=90)   # raw counts, no ambiguous "k"
        ax.minorticks_off()
        ax.axhline(chance, lw=0.8, ls="--", color="grey", zorder=0)
        ax.set_ylim(ylo, yhi)
        ax.set_title(bench, fontsize=7, color="grey", loc="left", pad=3)
        ax.set_xlabel("judge calls")
        if ci == 0:
            ax.set_ylabel(ylabel)


def _budget_series(data, BUD):
    series = {}
    for bench in BENCHES:
        for meth in METHODS:
            xs, mean, lo, hi = [], [], [], []
            for B in BUD[bench]:
                vals = list(data.get((bench, meth, B), {}).values())
                if len(vals) < 2:
                    continue
                m = np.mean(vals); half = t_dist.ppf(0.975, len(vals) - 1) * np.std(vals, ddof=1) / np.sqrt(len(vals))
                xs.append(B); mean.append(m); lo.append(m - half); hi.append(m + half)
            series[(bench, meth)] = (xs, mean, lo, hi)
    return series


def fig_budget(tau_d, lg_d, BUD):
    plt.rcParams.update(RC)
    for data, chance, ylabel, fname in (
            (tau_d, 0.0, "τ vs ECI\n(mean over judges)", "budget_curve.png"),
            (lg_d, 0.5, "low-gap pair accuracy\n(mean over judges)", "budget_curve_lowgap.png")):
        fig, axes = plt.subplots(1, len(BENCHES), figsize=(5.0, 2.3), sharey=True, squeeze=False)
        _budget_panels(fig, axes, _budget_series(data, BUD), BUD, chance, ylabel)
        handles = [plt.Line2D([], [], marker="o", ms=3, lw=1, color=COLORS[m], label=m) for m in METHODS]
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
        fig.tight_layout()
        fig.savefig(FIG / fname, dpi=300, bbox_inches="tight"); plt.close(fig)
        print(f"  {FIG / fname}")


# ---------- low-gap comparisons (matched, per-pair ordering) ----------
FILTERS = ["all\npairs", "close\n(<10 ECI)", "≥1 above\njudge",
           "both above\njudge", "close & both\nabove judge"]
FKEY = ["all", "close", "ge", "both", "closeboth"]


def low_gap_data():
    per_judge = defaultdict(lambda: defaultdict(dict))   # (bench,fkey)->meth->{judge:acc}
    for bench in BENCHES:
        cfg = elo.load_config(bench); eci, jeci = cfg["eci"], cfg.get("judge_eci", {})
        for judge in JUDGES:
            if judge not in jeci:
                continue
            pw_item, scores = load(bench, judge)
            pw_pair = defaultdict(dict)
            for (a, b, it), (s, n) in pw_item.items():
                if n >= 2:
                    pw_pair[(a, b)][it] = s / n
            jc = jeci[judge]; local = defaultdict(list)
            for (a, b), items in pw_pair.items():
                if a not in eci or b not in eci or eci[a] == eci[b] or a not in scores or b not in scores:
                    continue
                matched = [it for it in items if it in scores[a] and it in scores[b]]
                if len(matched) < 5:
                    continue
                hi = eci[a] > eci[b]
                pw = np.mean([items[it] for it in matched])
                el = np.mean([1.0 if scores[a][it] > scores[b][it] else (0.0 if scores[a][it] < scores[b][it] else 0.5)
                              for it in matched])
                ma = np.mean([scores[a][it] for it in matched]); mb = np.mean([scores[b][it] for it in matched])
                dec = {"pairwise": 0.5 if pw == 0.5 else float((pw > 0.5) == hi),
                       "direct-elo": 0.5 if el == 0.5 else float((el > 0.5) == hi),
                       "direct-score": 0.5 if ma == mb else float((ma > mb) == hi)}
                fl = {"all": True, "close": abs(eci[a] - eci[b]) < CLOSE,
                      "ge": max(eci[a], eci[b]) > jc, "both": min(eci[a], eci[b]) > jc}
                fl["closeboth"] = fl["close"] and fl["both"]
                for fk in FKEY:
                    if fl[fk]:
                        for meth in METHODS:
                            local[(fk, meth)].append(dec[meth])
            for (fk, meth), vals in local.items():
                if len(vals) >= 5:
                    per_judge[(bench, fk)][meth][judge] = np.mean(vals)
    return per_judge


def fig_low_gap(per_judge):
    """x = pair filter; a dot per method with 95% CI across judges; thin grey line
    links the three methods within a filter. Chance = 0.5."""
    plt.rcParams.update(RC)
    cell = {}                                           # (bench,fk,meth) -> (mean, half)
    allv = []
    for bench in BENCHES:
        for fk in FKEY:
            for meth in METHODS:
                vals = list(per_judge[(bench, fk)].get(meth, {}).values())
                if len(vals) < 2:
                    continue
                m = np.mean(vals); half = t_dist.ppf(0.975, len(vals) - 1) * np.std(vals, ddof=1) / np.sqrt(len(vals))
                cell[(bench, fk, meth)] = (m, half); allv += [m - half, m + half]
    ylo, yhi = _dyn_ylim(allv, include=0.5)
    fig, axes = plt.subplots(1, len(BENCHES), figsize=(5.2, 2.6), sharey=True, squeeze=False)
    for ci, bench in enumerate(BENCHES):
        ax = axes[0][ci]
        for fk_i, fk in enumerate(FKEY):                # grey link within filter
            pts = [(fk_i + (mi - 1) * 0.24, cell[(bench, fk, m)][0])
                   for mi, m in enumerate(METHODS) if (bench, fk, m) in cell]
            if len(pts) > 1:
                ax.plot(*zip(*pts), color="#cccccc", lw=0.5, zorder=1)
        for mi, meth in enumerate(METHODS):
            off = (mi - 1) * 0.24
            xs = [fk_i + off for fk_i, fk in enumerate(FKEY) if (bench, fk, meth) in cell]
            ys = [cell[(bench, fk, meth)][0] for fk in FKEY if (bench, fk, meth) in cell]
            es = [cell[(bench, fk, meth)][1] for fk in FKEY if (bench, fk, meth) in cell]
            ax.errorbar(xs, ys, yerr=es, fmt="o", ms=3, lw=0.9, capsize=0, color=COLORS[meth], zorder=3)
        ax.axhline(0.5, lw=0.8, ls="--", color="grey", zorder=0)
        ax.set_ylim(ylo, yhi)
        ax.set_xticks(range(len(FILTERS)), FILTERS, rotation=90)
        ax.set_title(bench, fontsize=7, color="grey", loc="left", pad=3)
        if ci == 0:
            ax.set_ylabel("pair ordering accuracy\n(mean over judges)")
    handles = [plt.Line2D([], [], ls="", marker="o", ms=3, color=COLORS[m], label=m) for m in METHODS]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "low_gap_comparisons.png", dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"  {FIG / 'low_gap_comparisons.png'}")


# ---------- per-model scatter grids (fitted strength vs capability) ----------
YLAB = {"pairwise": "Bradley-Terry strength", "direct-elo": "Bradley-Terry strength",
        "direct-score": "mean score"}


def fig_scatter(method, target, bench):
    """Per-judge grid, one dot per model: the judge's fitted strength vs capability.
    Grey dashed line = the judge's own capability (models to its right are stronger)."""
    plt.rcParams.update(RC)
    cfg = elo.load_config(bench)
    cap, jcap = cfg.get(target, {}), cfg.get(f"judge_{target}", {})
    panels = []
    for judge in JUDGES:
        st = rankings(bench, judge)[method]
        pts = sorted((cap[m], st[m]) for m in st if m in cap)
        if len(pts) >= 3:
            panels.append((judge, pts, jcap.get(judge)))
    if not panels:
        return
    ncols = min(4, len(panels)); nrows = math.ceil(len(panels) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.35 * ncols + 0.4, 1.25 * nrows + 0.5),
                             squeeze=False)
    for ax, (judge, pts, jc) in zip(axes.flat, panels):
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color="#cccccc", lw=0.5, zorder=1)
        ax.scatter(xs, ys, s=6, color=COLORS[method], linewidths=0, zorder=2)
        if jc is not None:
            ax.axvline(jc, ls="--", lw=0.8, color="grey", zorder=0)
        ax.set_title(judge, fontsize=6.5, color="grey", loc="left", pad=2)
        ax.text(0.05, 0.95, f"τ={kendalltau(xs, ys).statistic:.2f}", transform=ax.transAxes,
                fontsize=6, color="grey", va="top")
        ax.xaxis.set_major_locator(plt.MaxNLocator(3)); ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("ECI" if target == "eci" else target)
    for row in axes:
        row[0].set_ylabel(YLAB[method])
    fig.tight_layout()
    fname = f"{method}_vs_{target}_{bench}.png"
    fig.savefig(FIG / fname, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"  {FIG / fname}")


# ---------- single-judgement accuracy (one call, before aggregating) ----------
SJ_FILTERS = [("all", "all pairs"), ("close", "close (<10 ECI)"), ("both", "both above judge")]
SJ_COLORS = {"pairwise": "#2a78d6", "direct": "#1baf7a"}
SJ_LABEL = {"pairwise": "pairwise (one verdict)", "direct": "direct (compare 2 scores)"}


def single_judgement_data():
    """P(a SINGLE judgement is correct): one counterbalanced pairwise verdict, or
    one score-vs-score comparison on an item. (bench,fkey)->meth->{judge:mean}."""
    per_judge = defaultdict(lambda: defaultdict(dict))
    for bench in BENCHES:
        cfg = elo.load_config(bench); eci, jeci = cfg["eci"], cfg.get("judge_eci", {})
        for judge in JUDGES:
            if judge not in jeci:
                continue
            pw_item, scores = load(bench, judge); jc = jeci[judge]
            acc = defaultdict(lambda: defaultdict(list))    # fk -> meth -> [per-item correctness]
            for (a, b, it), (s, n) in pw_item.items():
                if n < 2 or a not in eci or b not in eci or eci[a] == eci[b]:
                    continue
                if a not in scores or b not in scores or it not in scores[a] or it not in scores[b]:
                    continue
                hi = eci[a] > eci[b]
                rate = s / n                                # counterbalanced P(pick major)
                pw_c = rate if hi else 1 - rate             # oriented P(one verdict picks stronger)
                sa, sb = scores[a][it], scores[b][it]
                dsc = 0.5 if sa == sb else float((sa > sb) == hi)
                fl = {"all": True, "close": abs(eci[a] - eci[b]) < CLOSE,
                      "ge": max(eci[a], eci[b]) > jc, "both": min(eci[a], eci[b]) > jc}
                fl["closeboth"] = fl["close"] and fl["both"]
                for fk in FKEY:                          # same 5 filters as low_gap_comparisons
                    if fl[fk]:
                        acc[fk]["pairwise"].append(pw_c); acc[fk]["direct"].append(dsc)
            for fk in FKEY:
                for meth in ("pairwise", "direct"):
                    if len(acc[fk][meth]) >= 20:
                        per_judge[(bench, fk)][meth][judge] = np.mean(acc[fk][meth])
    return per_judge


def fig_single_judgement(per_judge):
    plt.rcParams.update(RC)
    cell, allv = {}, []
    for bench in BENCHES:
        for fk, _lab in SJ_FILTERS:
            for meth in ("pairwise", "direct"):
                for j, v in per_judge[(bench, fk)].get(meth, {}).items():
                    cell[(bench, fk, meth, j)] = v; allv.append(v)
    ylo, yhi = _dyn_ylim(allv, include=0.5)
    fig, axes = plt.subplots(len(SJ_FILTERS), len(BENCHES),
                             figsize=(2.3 * len(BENCHES) + 0.6, 1.5 * len(SJ_FILTERS) + 0.6),
                             sharey=True, sharex="col", squeeze=False)
    for ri, (fk, flabel) in enumerate(SJ_FILTERS):
        for ci, bench in enumerate(BENCHES):
            ax = axes[ri][ci]
            js = [j for j in JUDGES if any((bench, fk, m, j) in cell for m in ("pairwise", "direct"))]
            for k, j in enumerate(js):                       # grey link within judge
                pts = [(k + (mi - 0.5) * 0.3, cell[(bench, fk, m, j)])
                       for mi, m in enumerate(("pairwise", "direct")) if (bench, fk, m, j) in cell]
                if len(pts) > 1:
                    ax.plot(*zip(*pts), color="#cccccc", lw=0.5, zorder=1)
            for mi, meth in enumerate(("pairwise", "direct")):
                xs = [k + (mi - 0.5) * 0.3 for k, j in enumerate(js) if (bench, fk, meth, j) in cell]
                ys = [cell[(bench, fk, meth, j)] for j in js if (bench, fk, meth, j) in cell]
                ax.scatter(xs, ys, s=9, color=SJ_COLORS[meth], linewidths=0, zorder=2)
            ax.axhline(0.5, lw=0.8, ls="--", color="grey", zorder=0); ax.set_ylim(ylo, yhi)
            ax.set_xticks(range(len(js)), js if ri == len(SJ_FILTERS) - 1 else [], rotation=90)
            if ri == 0:
                ax.set_title(bench, fontsize=7, color="grey", loc="left", pad=3)
        axes[ri][0].set_ylabel(f"P(correct)\n{flabel}")
    handles = [plt.Line2D([], [], ls="", marker="o", ms=3, color=SJ_COLORS[m], label=SJ_LABEL[m])
               for m in ("pairwise", "direct")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "single_judgement_accuracy.png", dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"  {FIG / 'single_judgement_accuracy.png'}")


def fig_single_judgement_summary(per_judge):
    """Aggregate mirror of low_gap_comparisons: x = filter, a dot per method =
    mean over judges with 95% CI across judges; thin grey link within a filter."""
    plt.rcParams.update(RC)
    methods = ("pairwise", "direct")
    cell, allv = {}, []
    for bench in BENCHES:
        for fk in FKEY:
            for meth in methods:
                vals = list(per_judge[(bench, fk)].get(meth, {}).values())
                if len(vals) < 2:
                    continue
                m = np.mean(vals); half = t_dist.ppf(0.975, len(vals) - 1) * np.std(vals, ddof=1) / np.sqrt(len(vals))
                cell[(bench, fk, meth)] = (m, half); allv += [m - half, m + half]
    ylo, yhi = _dyn_ylim(allv, include=0.5)
    fig, axes = plt.subplots(1, len(BENCHES), figsize=(2.7 * len(BENCHES) + 0.5, 2.6),
                             sharey=True, squeeze=False)
    for ci, bench in enumerate(BENCHES):
        ax = axes[0][ci]
        for fk_i, fk in enumerate(FKEY):
            pts = [(fk_i + (mi - 0.5) * 0.3, cell[(bench, fk, m)][0])
                   for mi, m in enumerate(methods) if (bench, fk, m) in cell]
            if len(pts) > 1:
                ax.plot(*zip(*pts), color="#cccccc", lw=0.5, zorder=1)
        for mi, meth in enumerate(methods):
            off = (mi - 0.5) * 0.3
            xs = [fk_i + off for fk_i, fk in enumerate(FKEY) if (bench, fk, meth) in cell]
            ys = [cell[(bench, fk, meth)][0] for fk in FKEY if (bench, fk, meth) in cell]
            es = [cell[(bench, fk, meth)][1] for fk in FKEY if (bench, fk, meth) in cell]
            ax.errorbar(xs, ys, yerr=es, fmt="o", ms=3, lw=0.9, capsize=0, color=SJ_COLORS[meth], zorder=2)
        ax.axhline(0.5, lw=0.8, ls="--", color="grey", zorder=0); ax.set_ylim(ylo, yhi)
        ax.set_xticks(range(len(FILTERS)), FILTERS, rotation=90)
        ax.set_title(bench, fontsize=7, color="grey", loc="left", pad=3)
        if ci == 0:
            ax.set_ylabel("P(single judgement correct)\n(mean over judges)")
    handles = [plt.Line2D([], [], ls="", marker="o", ms=3, color=SJ_COLORS[m], label=SJ_LABEL[m])
               for m in methods]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "single_judgement_summary.png", dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"  {FIG / 'single_judgement_summary.png'}")


# ---------- accuracy vs number of comparisons per pair ----------
def comparisons_curve_data():
    """(bench,filter,meth,K)->{judge: per-pair accuracy using K comparisons/pair}.
    One comparison = one item's verdict (counterbalanced) or one score-vs-score."""
    # gpqa has ~60 items/pair (pairwise cap); otis only ~16 median, so it stops sooner.
    # The pair set is PINNED to pairs with >= max(K) items, so every K uses the SAME
    # pairs -> the curve is monotonic (no selection shift as K grows).
    KS = {"gpqa": [1, 2, 5, 10, 20, 40, 60], "otis-aime": [1, 2, 5, 10]}
    SEEDS = 40
    out = defaultdict(dict)
    for bench in BENCHES:
        eci = elo.load_config(bench)["eci"]; maxK = KS[bench][-1]
        for judge in JUDGES:
            pw_item, scores = load(bench, judge)
            pw_pairs, dr_pairs = defaultdict(list), defaultdict(list)   # (a,b)->[P(correct) per item]
            for (a, b, it), (s, n) in pw_item.items():
                if n < 2 or a not in eci or b not in eci or eci[a] == eci[b]:
                    continue
                r = s / n
                pw_pairs[(a, b)].append(r if eci[a] > eci[b] else 1 - r)
            for (a, b) in {(a, b) for (a, b, _it) in pw_item}:
                if a not in eci or b not in eci or eci[a] == eci[b] or a not in scores or b not in scores:
                    continue
                for it in set(scores[a]) & set(scores[b]):
                    sa, sb = scores[a][it], scores[b][it]
                    sc = 0.5 if sa == sb else float(sa > sb)
                    dr_pairs[(a, b)].append(sc if eci[a] > eci[b] else 1 - sc)
            for filt in ("all", "close"):
                for meth, pairs in (("pairwise", pw_pairs), ("direct", dr_pairs)):
                    elig = [(p, outs) for p, outs in pairs.items()
                            if (filt == "all" or abs(eci[p[0]] - eci[p[1]]) < CLOSE) and len(outs) >= maxK]
                    if len(elig) < 8:                     # need a stable pinned set
                        continue
                    for K in KS[bench]:
                        accs = []
                        for (a, b), outs in elig:         # same pairs at every K
                            cs = []
                            for seed in range(SEEDS):
                                m = sum(random.Random(seed * 131 + K).sample(outs, K)) / K
                                cs.append(0.5 if m == 0.5 else float(m > 0.5))
                            accs.append(np.mean(cs))
                        out[(bench, filt, meth, K)][judge] = np.mean(accs)
    return out, KS


def fig_comparisons_curve(data, KS):
    """rows = pair filter, cols = bench; x = comparisons per pair (log); pairwise vs
    direct, mean over judges with 95% CI band. Interpolates single-call -> aggregate."""
    plt.rcParams.update(RC)
    rows = [("all", "all pairs"), ("close", "close (<10 ECI)")]
    band, allv = {}, []
    for bench in BENCHES:
        for filt, _ in rows:
            for meth in ("pairwise", "direct"):
                xs, mean, lo, hi = [], [], [], []
                for K in KS[bench]:
                    vals = list(data.get((bench, filt, meth, K), {}).values())
                    if len(vals) < 2:
                        continue
                    m = np.mean(vals); half = t_dist.ppf(0.975, len(vals) - 1) * np.std(vals, ddof=1) / np.sqrt(len(vals))
                    xs.append(K); mean.append(m); lo.append(m - half); hi.append(m + half)
                band[(bench, filt, meth)] = (xs, mean, lo, hi); allv += lo + hi
    ylo, yhi = _dyn_ylim(allv, include=0.5)
    fig, axes = plt.subplots(len(rows), len(BENCHES), figsize=(2.5 * len(BENCHES) + 0.4, 1.7 * len(rows) + 0.6),
                             sharey=True, sharex="col", squeeze=False)
    for ri, (filt, flabel) in enumerate(rows):
        for ci, bench in enumerate(BENCHES):
            ax = axes[ri][ci]
            for meth in ("pairwise", "direct"):
                xs, mean, lo, hi = band[(bench, filt, meth)]
                if xs:
                    ax.fill_between(xs, lo, hi, color=SJ_COLORS[meth], alpha=0.15, lw=0)
                    ax.plot(xs, mean, "-o", ms=2.5, lw=1, color=SJ_COLORS[meth])
            ax.set_xscale("log"); ax.set_xticks(KS[bench]); ax.set_xticklabels([str(k) for k in KS[bench]])
            ax.minorticks_off(); ax.axhline(0.5, lw=0.8, ls="--", color="grey", zorder=0); ax.set_ylim(ylo, yhi)
            if ri == 0:
                ax.set_title(bench, fontsize=7, color="grey", loc="left", pad=3)
            if ri == len(rows) - 1:
                ax.set_xlabel("comparisons per pair")
            if ci == 0:
                ax.set_ylabel(f"pair accuracy\n{flabel}")
    handles = [plt.Line2D([], [], marker="o", ms=3, lw=1, color=SJ_COLORS[m], label=SJ_LABEL[m])
               for m in ("pairwise", "direct")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "accuracy_vs_comparisons.png", dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"  {FIG / 'accuracy_vs_comparisons.png'}")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    print("per-model scatter grids...")
    for method in ("pairwise", "direct-score"):
        for target in ("eci", "accuracy"):
            for bench in BENCHES:
                fig_scatter(method, target, bench)
    print("computing ranking taus...")
    all_t = tau_table("all"); strong_t = tau_table("strong")
    fig_tau(all_t, "ranking_by_judge.png")
    fig_tau(strong_t, "oversight_by_judge.png")
    # decisive bench-clustered numbers
    for name, tab in (("ALL MODELS", all_t), ("STRONGER THAN JUDGE", strong_t)):
        for target in ("accuracy", "eci"):
            bc = bench_clustered(tab, target)
            if bc:
                print(f"  [{name} vs {target}] pw-directscore bench means {bc['benchmeans']} "
                      f"-> {bc['mean']:+.3f} 95%CI [{bc['lo']:+.2f},{bc['hi']:+.2f}]  "
                      f"{'CANNOT SIGN' if bc['lo'] < 0 < bc['hi'] else 'SIGNED'}")
    with open(OUT / "oversight_tau_ci.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["target", "bench", "judge", "method", "tau", "ci_lo", "ci_hi", "n_models"])
        for (t, b, j, m), (tau, lo, hi, n) in sorted(strong_t.items()):
            w.writerow([t, b, j, m, round(tau, 4), round(lo, 4), round(hi, 4), n])
    print("computing budget curves (ranking + low-gap)...")
    tau_d, lg_d, BUD = budget_curve()
    fig_budget(tau_d, lg_d, BUD)
    print("computing low-gap comparisons...")
    fig_low_gap(low_gap_data())
    print("computing single-judgement accuracy...")
    sjd = single_judgement_data()
    fig_single_judgement(sjd)
    fig_single_judgement_summary(sjd)
    print("accuracy vs comparisons per pair...")
    cc, KS = comparisons_curve_data()
    fig_comparisons_curve(cc, KS)
    print("done.")


if __name__ == "__main__":
    main()
