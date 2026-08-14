# scripts/analyses/judge-elo/head_to_head.py
"""Head-to-head accuracy: given two models, does the method pick the stronger?

Three methods give three DIFFERENT per-pair predictions:
  pairwise     - majority of the pairwise verdicts on that pair
  direct-elo   - which model scores higher on more items (item win-rate)
  direct-score - which model has the higher mean 0-100 score

Accuracy = P(pick the higher-ECI model), over pairs, per capability filter:
  all           - every baseline pair
  >=1 > judge   - at least one model stronger than the judge
  both > judge  - both models stronger than the judge (the oversight case)

Chance = 0.5. CI is clustered by pair (t-based across pairs). Writes summary
dots + summary lines into figures/head_to_head/.

Usage: uv run python head_to_head.py
"""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.stats import t as t_dist

import elo

HERE = Path(__file__).parent
ROOT = elo.ROOT
OUT = ROOT / "judge_results" / "judge_elo" / "figures" / "head_to_head"
BENCHES = ["gpqa", "otis-aime"]
JUDGES = ["gpt-oss-20b", "llama-3.2-3b", "qwen3-14b", "qwen3-32b", "qwen3-8b"]
CONDS = ["pairwise", "direct-elo", "direct-score"]
COND_COLORS = {"pairwise": "#2a78d6", "direct-elo": "#eb6834", "direct-score": "#1baf7a"}
FILTERS = ["all", "≥1 > judge", "both > judge"]
CLOSE = 10.0
AS_WIN = {"True": 1.0, "False": 0.0}
RC = {"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
      "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False}


def load(bench):
    cfg = yaml.safe_load((HERE / "configs" / f"{bench}.yaml").read_text())
    eci, jeci = cfg["eci"], cfg.get("judge_eci", {})
    scores, verdicts = defaultdict(lambda: defaultdict(dict)), defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
    for r in csv.DictReader(open(ROOT / "judge_results/direct_judge/scores.csv")):
        if r["bench"] == bench and r["parse_ok"] == "True":
            scores[r["judge"]][r["model"]][r["item_id"]] = int(r["score"])
    for r in csv.DictReader(open(ROOT / "judge_results/judging_variations/verdicts.csv")):
        if r["bench"] == bench and r["condition"] == "a_baseline" and r["parse_ok"] == "True":
            verdicts[r["judge"]][(r["major"], r["minor"], r["item_id"])][0] += AS_WIN[r["picked_major"]]
            verdicts[r["judge"]][(r["major"], r["minor"], r["item_id"])][1] += 1
    return eci, jeci, scores, verdicts


def per_pair_correct(bench, judge, eci, jeci, model_scores, verdict_items, min_models=20):
    """Yield (filters, {cond: correct in {0,.5,1}}) per baseline pair."""
    if len({m for m in model_scores}) < min_models:
        return
    # counterbalanced per-item pairwise rate, then per-pair
    pw_pair = defaultdict(list)
    for (a, b, _it), (s, n) in verdict_items.items():
        pw_pair[(a, b)].append(s / n)
    jc = jeci.get(judge)
    for (a, b), rates in pw_pair.items():
        if a not in eci or b not in eci or eci[a] == eci[b]:
            continue
        shared = set(model_scores.get(a, {})) & set(model_scores.get(b, {}))
        if not shared or a not in model_scores or b not in model_scores:
            continue
        hi = eci[a] > eci[b]

        def oriented(higher_a):  # higher_a: does the metric rank a above b
            return 0.5 if higher_a is None else float(higher_a == hi)

        pw_rate = np.mean(rates)
        item_rate = np.mean([1.0 if model_scores[a][i] > model_scores[b][i]
                             else (0.0 if model_scores[a][i] < model_scores[b][i] else 0.5)
                             for i in shared])
        mean_a = np.mean(list(model_scores[a].values()))
        mean_b = np.mean(list(model_scores[b].values()))
        correct = {
            "pairwise": oriented(None if pw_rate == 0.5 else pw_rate > 0.5),
            "direct-elo": oriented(None if item_rate == 0.5 else item_rate > 0.5),
            "direct-score": oriented(None if mean_a == mean_b else mean_a > mean_b),
        }
        flags = {"all": True,
                 "≥1 > judge": jc is not None and max(eci[a], eci[b]) > jc,
                 "both > judge": jc is not None and min(eci[a], eci[b]) > jc}
        yield flags, correct


def accuracy(bench):
    """{(judge, filt, cond): (mean, ci_half, n_pairs)}."""
    eci, jeci, scores, verdicts = load(bench)
    bucket = defaultdict(list)  # (judge, filt, cond) -> per-pair correct
    for j in JUDGES:
        if j not in scores or j not in verdicts:
            continue
        for flags, correct in per_pair_correct(bench, j, eci, jeci, scores[j], verdicts[j]):
            for filt in FILTERS:
                if flags[filt]:
                    for cond in CONDS:
                        bucket[(j, filt, cond)].append(correct[cond])
    out = {}
    for k, v in bucket.items():
        if len(v) < 2:
            continue
        m = np.mean(v)
        h = t_dist.ppf(0.975, len(v) - 1) * np.std(v, ddof=1) / np.sqrt(len(v))
        out[k] = (m, h, len(v))
    return out


def _judges(data):
    return sorted({j for b in data for (j, _f, _c) in data[b]})


def _ylim(axes, data):
    vals = [m for b in data for (m, h, _n) in data[b].values()]
    lo, hi = min(vals) - 0.06, max(vals) + 0.06
    for ax in axes.flat:
        ax.set_ylim(lo, hi)
        ax.axhline(0.5, ls="--", lw=0.8, color="grey", zorder=0)


def plot_lines(data, out_name):
    """Slopegraph: x = method, one line per judge, rows = capability filter."""
    plt.rcParams.update(RC)
    judges = _judges(data)
    fig, axes = plt.subplots(len(FILTERS), len(BENCHES),
                             figsize=(2.1 * len(BENCHES) + 0.5, 1.8 * len(FILTERS) + 0.5),
                             sharey=True, sharex=True, squeeze=False)
    for ri, filt in enumerate(FILTERS):
        for ax, bench in zip(axes[ri], BENCHES):
            for j in judges:
                pts = [(i, data[bench][(j, filt, c)][0]) for i, c in enumerate(CONDS)
                       if (j, filt, c) in data[bench]]
                if len(pts) >= 2:
                    ax.plot(*zip(*pts), "-o", ms=3, lw=1,
                            color=elo.JUDGE_COLORS.get(j, "#999999"))
            ax.set_xlim(-0.3, len(CONDS) - 0.7)
            ax.set_xticks(range(len(CONDS)), CONDS if ri == len(FILTERS) - 1 else [], rotation=90)
            if ri == 0:
                ax.set_title(bench, fontsize=7, color="grey", loc="left", pad=2)
        axes[ri][0].set_ylabel(f"accuracy\n({filt})")
    _ylim(axes, data)
    handles = [plt.Line2D([], [], ls="-", marker="o", ms=3, lw=1,
                          color=elo.JUDGE_COLORS.get(j, "#999999"), label=j) for j in judges]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0),
               ncol=min(5, len(judges)), frameon=False)
    fig.tight_layout()
    _save(fig, out_name)


def plot_dots(data, out_name):
    """Dots: x = judge, one dot per method, rows = capability filter."""
    plt.rcParams.update(RC)
    judges = _judges(data)
    fig, axes = plt.subplots(len(FILTERS), len(BENCHES),
                             figsize=(2.4 * len(BENCHES) + 0.5, 1.8 * len(FILTERS) + 0.5),
                             sharey=True, squeeze=False)
    for ri, filt in enumerate(FILTERS):
        for ax, bench in zip(axes[ri], BENCHES):
            for k, j in enumerate(judges):  # thin grey line connecting a judge's methods
                pts = [(k + (ci - 1) * 0.22, data[bench][(j, filt, c)][0])
                       for ci, c in enumerate(CONDS) if (j, filt, c) in data[bench]]
                if len(pts) > 1:
                    ax.plot(*zip(*pts), color="#cccccc", lw=0.5, zorder=1)
            for ci, cond in enumerate(CONDS):
                xs = [k + (ci - 1) * 0.22 for k, j in enumerate(judges)
                      if (j, filt, cond) in data[bench]]
                ys = [data[bench][(j, filt, cond)][0] for j in judges
                      if (j, filt, cond) in data[bench]]
                es = [data[bench][(j, filt, cond)][1] for j in judges
                      if (j, filt, cond) in data[bench]]
                ax.errorbar(xs, ys, yerr=es, fmt="o", ms=3, lw=1, capsize=0,
                            color=COND_COLORS[cond], zorder=2)
            ax.set_xticks(range(len(judges)), judges if ri == len(FILTERS) - 1 else [], rotation=90)
            if ri == 0:
                ax.set_title(bench, fontsize=7, color="grey", loc="left", pad=2)
        axes[ri][0].set_ylabel(f"accuracy\n({filt})")
    _ylim(axes, data)
    handles = [plt.Line2D([], [], ls="", marker="o", ms=3, color=c, label=m)
               for m, c in COND_COLORS.items()]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0),
               ncol=3, frameon=False)
    fig.tight_layout()
    _save(fig, out_name)


def plot_averaged(data, out_name):
    """Mean over judges per method, 95% CI across judges; rows = filter."""
    plt.rcParams.update(RC)
    judges = _judges(data)
    fig, axes = plt.subplots(len(FILTERS), len(BENCHES),
                             figsize=(1.9 * len(BENCHES) + 0.5, 1.7 * len(FILTERS) + 0.5),
                             sharey=True, sharex=True, squeeze=False)
    for ri, filt in enumerate(FILTERS):
        for ax, bench in zip(axes[ri], BENCHES):
            xs, ys, es, cs = [], [], [], []
            for ci, cond in enumerate(CONDS):
                jm = [data[bench][(j, filt, cond)][0] for j in judges
                      if (j, filt, cond) in data[bench]]
                if len(jm) < 2:
                    continue
                h = (t_dist.ppf(0.975, len(jm) - 1) * np.std(jm, ddof=1) / np.sqrt(len(jm))
                     if len(jm) >= 3 else 0.0)
                xs.append(ci); ys.append(np.mean(jm)); es.append(h); cs.append(COND_COLORS[cond])
            if len(xs) > 1:
                ax.plot(xs, ys, color="#cccccc", lw=0.5, zorder=1)
            for x, y, e, c in zip(xs, ys, es, cs):
                ax.errorbar(x, y, yerr=e, fmt="o", ms=4, capsize=0, color=c, zorder=2)
            ax.set_xlim(-0.3, len(CONDS) - 0.7)
            ax.set_xticks(range(len(CONDS)), CONDS if ri == len(FILTERS) - 1 else [], rotation=90)
            if ri == 0:
                ax.set_title(bench, fontsize=7, color="grey", loc="left", pad=2)
        axes[ri][0].set_ylabel(f"accuracy\n({filt})")
    _ylim(axes, data)
    handles = [plt.Line2D([], [], ls="", marker="o", ms=3, color=c, label=m)
               for m, c in COND_COLORS.items()]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0),
               ncol=3, frameon=False)
    fig.tight_layout()
    _save(fig, out_name)


def _save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {OUT / name}")


if __name__ == "__main__":
    data = {b: accuracy(b) for b in BENCHES}
    plot_dots(data, "by_judge.png")           # x=judge, dots per method + grey connectors
    plot_lines(data, "by_judge_slopegraph.png")  # x=method, one line per judge
    plot_averaged(data, "averaged.png")       # mean over judges, 95% CI
    with open(OUT.parent.parent / "head_to_head.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["bench", "judge", "filter", "method",
                                          "accuracy", "ci_half", "n_pairs"])
        w.writeheader()
        for b in BENCHES:
            for (j, filt, cond), (m, h, n) in sorted(data[b].items()):
                w.writerow(dict(bench=b, judge=j, filter=filt, method=cond,
                                accuracy=round(m, 4), ci_half=round(h, 4), n_pairs=n))
