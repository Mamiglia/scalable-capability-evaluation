# scripts/analyses/judge-elo/budget_curve.py
"""Judging quality vs budget (judge calls), matched across methods.

Two metrics, SEPARATE plots in figures/budget/:
  tau      - Kendall tau of the model ranking vs ECI (ranking quality).
             methods: pairwise, direct-elo, direct-score.
  accuracy - per-PAIR head-to-head: does the method order two models correctly?
             pairwise = verdict majority, direct-elo = item win-rate,
             direct-score = mean-score comparison. This aggregates over the
             sampled calls, so it CLIMBS with budget (more calls -> firmer
             per-pair prediction). Filters: all / >=1>judge / both>judge.

At budget B each method is subsampled to B judge calls. Averaged over judges,
error bands. No Spearman rho.

Usage: uv run python budget_curve.py
"""
import csv
import random
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.stats import kendalltau, t as t_dist

import elo

HERE = Path(__file__).parent
ROOT = elo.ROOT
OUT = ROOT / "judge_results" / "judge_elo" / "figures" / "budget"
JUDGES = ["gpt-oss-20b", "qwen3-14b", "qwen3-32b", "qwen3-8b"]
BENCHES = ["gpqa", "otis-aime"]
BUDGETS = {"gpqa": [250, 500, 1000, 2000, 4000],
           "otis-aime": [100, 250, 500, 750, 1000]}
SEEDS = 6
CONDS = ["pairwise", "direct-elo", "direct-score"]
COND_COLORS = {"pairwise": "#2a78d6", "direct-elo": "#eb6834", "direct-score": "#1baf7a"}
FILTERS = ["all", "≥1 > judge", "both > judge"]
RC = {"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
      "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False}
AS_WIN = {"True": 1.0, "False": 0.0}


def rank_tau(values, eci):
    ms = [m for m in values if m in eci]
    return kendalltau([values[m] for m in ms], [eci[m] for m in ms]).statistic \
        if len(ms) >= 10 else None


def load(bench):
    cfg = yaml.safe_load((HERE / "configs" / f"{bench}.yaml").read_text())
    eci, jeci = cfg["eci"], cfg.get("judge_eci", {})
    scores, verdicts = defaultdict(list), defaultdict(list)
    for r in csv.DictReader(open(ROOT / "judge_results/direct_judge/scores.csv")):
        if r["bench"] == bench and r["parse_ok"] == "True":
            scores[r["judge"]].append((r["model"], r["item_id"], int(r["score"])))
    for r in csv.DictReader(open(ROOT / "judge_results/judging_variations/verdicts.csv")):
        if r["bench"] == bench and r["condition"] == "a_baseline" and r["parse_ok"] == "True":
            verdicts[r["judge"]].append(r)
    return eci, jeci, scores, verdicts


def compute(bench):
    """tau_data {(j,meth,B):[vals]}, acc_data {(j,meth,filt,B):[vals]}."""
    eci, jeci, scores, verdicts = load(bench)
    tau_data, acc_data = defaultdict(list), defaultdict(list)
    for j in JUDGES:
        if j not in scores or j not in verdicts or j not in jeci:
            continue
        jc = jeci[j]
        baseline = {frozenset((r["major"], r["minor"])) for r in verdicts[j]}
        maxb = min(len(scores[j]), len(verdicts[j]))
        for B in BUDGETS[bench]:
            if B > maxb:
                continue
            for seed in range(SEEDS):
                rnd = random.Random(seed * 997 + B)
                ssc = rnd.sample(scores[j], B)
                spv = rnd.sample(verdicts[j], B)
                model_items, per_item = defaultdict(dict), defaultdict(dict)
                for m, i, s in ssc:
                    model_items[m][i] = s
                    per_item[i][m] = s
                means = {m: np.mean(list(d.values())) for m, d in model_items.items()}
                # ranking tau
                t_score = rank_tau(means, eci)
                wins = defaultdict(lambda: [0.0, 0])
                for ms in per_item.values():
                    mods = sorted(ms)
                    for ai, a in enumerate(mods):
                        for b in mods[ai + 1:]:
                            if frozenset((a, b)) in baseline:
                                w = 1.0 if ms[a] > ms[b] else (0.0 if ms[a] < ms[b] else 0.5)
                                wins[(a, b)][0] += w
                                wins[(a, b)][1] += 1
                try:
                    t_elo = rank_tau(elo.fit_bt(dict(wins)), eci) if wins else None
                except Exception:
                    t_elo = None
                g = elo.collect_pair_wins(spv, ["bench", "condition", "judge"])
                try:
                    t_pw = rank_tau(elo.fit_bt(next(iter(g.values()))), eci)
                except Exception:
                    t_pw = None
                for meth, t in (("direct-score", t_score), ("direct-elo", t_elo),
                                ("pairwise", t_pw)):
                    if t is not None:
                        tau_data[(j, meth, B)].append(t)
                # per-pair head-to-head accuracy
                pw_rate = defaultdict(lambda: [0.0, 0])
                for r in spv:
                    a, b = r["major"], r["minor"]
                    if a in eci and b in eci and eci[a] != eci[b]:
                        v = AS_WIN[r["picked_major"]]
                        pw_rate[frozenset((a, b))][0] += v if eci[a] > eci[b] else 1 - v
                        pw_rate[frozenset((a, b))][1] += 1
                bucket = defaultdict(lambda: defaultdict(list))
                for fs in baseline:
                    a, b = sorted(fs)
                    if a not in eci or b not in eci or eci[a] == eci[b]:
                        continue
                    hi, lo = (a, b) if eci[a] > eci[b] else (b, a)
                    subs = [f for f in FILTERS if f == "all"
                            or (f == "≥1 > judge" and max(eci[a], eci[b]) > jc)
                            or (f == "both > judge" and min(eci[a], eci[b]) > jc)]
                    if pw_rate[fs][1]:
                        rate = pw_rate[fs][0] / pw_rate[fs][1]
                        c = 0.5 if rate == 0.5 else float(rate > 0.5)
                        for s in subs:
                            bucket["pairwise"][s].append(c)
                    shared = set(model_items.get(hi, {})) & set(model_items.get(lo, {}))
                    if shared:
                        wr = np.mean([1.0 if model_items[hi][i] > model_items[lo][i]
                                      else (0.0 if model_items[hi][i] < model_items[lo][i] else 0.5)
                                      for i in shared])
                        c = 0.5 if wr == 0.5 else float(wr > 0.5)
                        for s in subs:
                            bucket["direct-elo"][s].append(c)
                    if hi in means and lo in means:
                        c = 0.5 if means[hi] == means[lo] else float(means[hi] > means[lo])
                        for s in subs:
                            bucket["direct-score"][s].append(c)
                for meth in CONDS:
                    for s in FILTERS:
                        vals = bucket[meth][s]
                        if len(vals) >= 10:
                            acc_data[(j, meth, s, B)].append(np.mean(vals))
    return tau_data, acc_data


def _style_x(ax, bench):
    tk = BUDGETS[bench]
    ax.set_xscale("log")
    ax.set_xticks(tk)
    ax.set_xticklabels([f"{x // 1000}k" if x >= 1000 else str(x) for x in tk])
    ax.minorticks_off()


def _ylim(axes, series, chance=None):
    vals = [y for lo, hi in series for y in (lo, hi)]
    lo, hi = min(vals), max(vals)
    pad = 0.03 * (hi - lo + 0.1)
    for ax in axes.flat:
        ax.set_ylim(lo - pad, hi + pad)
        if chance is not None:
            ax.axhline(chance, ls="--", lw=0.8, color="grey", zorder=0)


def _band(vals, across_judges):
    """(mean, half-width). across_judges: 95% t CI; else +-1 SD over seeds."""
    if across_judges:
        h = (t_dist.ppf(0.975, len(vals) - 1) * np.std(vals, ddof=1) / np.sqrt(len(vals))
             if len(vals) >= 3 else 0.0)
    else:
        h = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
    return np.mean(vals), h


def plot_tau(tau, out_name, average):
    plt.rcParams.update(RC)
    if average:
        fig, axes = plt.subplots(1, len(BENCHES), figsize=(2.4 * len(BENCHES) + 0.4, 2.2),
                                 sharey=True, squeeze=False)
    else:
        fig, axes = plt.subplots(len(BENCHES), len(JUDGES),
                                 figsize=(1.7 * len(JUDGES) + 0.5, 1.7 * len(BENCHES) + 0.6),
                                 sharey=True, squeeze=False)
    series = []
    if average:
        for ax, bench in zip(axes[0], BENCHES):
            for meth, c in COND_COLORS.items():
                xs, mean, lo, hi = [], [], [], []
                for B in BUDGETS[bench]:
                    jm = [np.mean(tau[bench][(j, meth, B)]) for j in JUDGES
                          if (j, meth, B) in tau[bench]]
                    if len(jm) < 2:
                        continue
                    m, h = _band(jm, True)
                    xs.append(B); mean.append(m); lo.append(m - h); hi.append(m + h)
                    series.append((m - h, m + h))
                if xs:
                    ax.fill_between(xs, lo, hi, color=c, alpha=0.15, linewidth=0)
                    ax.plot(xs, mean, "-o", ms=3, lw=1, color=c)
            ax.set_title(bench, fontsize=7, color="grey", loc="left", pad=2)
            ax.set_xlabel("judge calls")
            _style_x(ax, bench)
        axes[0][0].set_ylabel("τ vs ECI  (mean over judges)")
    else:
        for ri, bench in enumerate(BENCHES):
            for ci, j in enumerate(JUDGES):
                ax = axes[ri][ci]
                for meth, c in COND_COLORS.items():
                    xs, mean, lo, hi = [], [], [], []
                    for B in BUDGETS[bench]:
                        v = tau[bench].get((j, meth, B))
                        if not v:
                            continue
                        m, h = _band(v, False)
                        xs.append(B); mean.append(m); lo.append(m - h); hi.append(m + h)
                        series.append((m - h, m + h))
                    if xs:
                        ax.fill_between(xs, lo, hi, color=c, alpha=0.15, linewidth=0)
                        ax.plot(xs, mean, "-o", ms=2.5, lw=1, color=c)
                if ri == 0:
                    ax.set_title(j, fontsize=7, color="grey", loc="left", pad=2)
                if ci == 0:
                    ax.set_ylabel(f"{bench}\nτ vs ECI")
                if ri == len(BENCHES) - 1:
                    ax.set_xlabel("judge calls")
                _style_x(ax, bench)
    _ylim(axes, series)
    _save(fig, out_name)


def plot_accuracy_avg(acc, out_name):
    """Rows = capability filter, cols = bench; 3 method lines, mean over judges."""
    plt.rcParams.update(RC)
    fig, axes = plt.subplots(len(FILTERS), len(BENCHES),
                             figsize=(2.4 * len(BENCHES) + 0.4, 1.7 * len(FILTERS) + 0.6),
                             sharey=True, squeeze=False)
    series = []
    for ri, filt in enumerate(FILTERS):
        for ax, bench in zip(axes[ri], BENCHES):
            for meth, c in COND_COLORS.items():
                xs, mean, lo, hi = [], [], [], []
                for B in BUDGETS[bench]:
                    jm = [np.mean(acc[bench][(j, meth, filt, B)]) for j in JUDGES
                          if (j, meth, filt, B) in acc[bench]]
                    if len(jm) < 2:
                        continue
                    m, h = _band(jm, True)
                    xs.append(B); mean.append(m); lo.append(m - h); hi.append(m + h)
                    series.append((m - h, m + h))
                if xs:
                    ax.fill_between(xs, lo, hi, color=c, alpha=0.15, linewidth=0)
                    ax.plot(xs, mean, "-o", ms=3, lw=1, color=c)
            if ri == 0:
                ax.set_title(bench, fontsize=7, color="grey", loc="left", pad=2)
            if ri == len(FILTERS) - 1:
                ax.set_xlabel("judge calls")
            _style_x(ax, bench)
        axes[ri][0].set_ylabel(f"accuracy\n({filt})")
    _ylim(axes, series, chance=0.5)
    _save(fig, out_name)


def _save(fig, name):
    handles = [plt.Line2D([], [], color=c, marker="o", ms=3, lw=1, label=m)
               for m, c in COND_COLORS.items()]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0),
               ncol=len(handles), frameon=False)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {OUT / name}")


if __name__ == "__main__":
    tau = {b: None for b in BENCHES}
    acc = {b: None for b in BENCHES}
    for b in BENCHES:
        tau[b], acc[b] = compute(b)
    plot_tau(tau, "tau_per_judge.png", average=False)
    plot_tau(tau, "tau_avg.png", average=True)
    plot_accuracy_avg(acc, "accuracy_avg.png")
