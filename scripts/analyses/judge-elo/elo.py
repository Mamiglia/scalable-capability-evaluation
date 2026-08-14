# scripts/analyses/judge-elo/elo.py
"""Per-judge Bradley-Terry strengths from pair picks, compared to capability.

One Bradley-Terry fit per group: the fitted strength of each evaluated model
is that judge's capability estimate for it. The main source is the pairwise
judging verdicts (bench x condition x judge); the induced pair picks from
pointwise 0-100 scoring are fit the same way as a baseline to compare against.

Ground truth (--capability) is ECI or benchmark accuracy from our own eval
logs, both in configs/{bench}.yaml; IRT-derived capability may follow.

Outputs elos.csv + elo_summary.csv (Kendall's tau vs the chosen capability
per group) and per-bench scatter figures of fitted strength vs capability.

Usage:
  uv run python elo.py [--capability accuracy]
"""
import argparse
import csv
import functools
import math
import re
import zlib
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import kendalltau, pearsonr, spearmanr, t as t_dist

HERE = Path(__file__).parent
ROOT = HERE.parent.parent.parent
OUT_DIR = ROOT / "judge_results" / "judge_elo"

ELO_FIELDS = ["bench", "source", "condition", "judge", "model", "elo", "n_picks",
              "component"]
SUMMARY_FIELDS = ["bench", "source", "condition", "judge", "capability", "n_models",
                  "n_picks", "n_calls", "ties", "n_components", "n_models_tau",
                  "kendall_elo_vs_capability", "spearman_elo_vs_capability",
                  "pearson_elo_vs_capability", "spearman_top20_vs_capability",
                  "n_models_top20", "split_half_item_agreement"]
TOP_N = 20
MIN_COMPARE_MODELS = 20  # a partial run's rank correlation is too noisy to compare;
                         # keeps in-progress collections from leaking misleading dots
XLABEL = {"eci": "ECI", "accuracy": "Benchmark accuracy"}
CORR_SYMBOL = {"kendall": "τ", "spearman": "ρ", "pearson": "r"}


def correlations(xs, ys):
    """kendall, spearman, pearson — '' if fewer than 3 points."""
    if len(xs) < 3:
        return "", "", ""
    return (round(kendalltau(xs, ys).statistic, 4),
            round(spearmanr(xs, ys).statistic, 4),
            round(pearsonr(xs, ys).statistic, 4))


def top_spearman(values, capability_map, models):
    """Spearman among the TOP_N models by capability — frontier discrimination,
    which the all-model correlation hides. '' below 5 models."""
    top = sorted(models, key=capability_map.get, reverse=True)[:TOP_N]
    if len(top) < 5:
        return "", len(top)
    rho = spearmanr([values[m] for m in top],
                    [capability_map[m] for m in top]).statistic
    return round(rho, 4), len(top)


def item_half(item_id):
    """Stable 0/1 split of items for split-half reliability."""
    return zlib.crc32(item_id.encode()) % 2


@functools.lru_cache(maxsize=None)
def load_config(bench):
    return yaml.safe_load((HERE / "configs" / f"{bench}.yaml").read_text())


def fit_bt(pair_wins, l2=1e-3):
    """Bradley-Terry strengths from {(a, b): [wins_a, n]}. Returns model -> strength.

    Fractional wins allowed (ties = 0.5). l2 is a fixed Gaussian prior on the
    strengths: negligible against real pick counts, it only keeps models with
    all-win or all-loss records finite. Strengths are shifted to mean 0.
    """
    models = sorted({m for ab in pair_wins for m in ab})
    idx = {m: i for i, m in enumerate(models)}
    ia = np.array([idx[a] for a, _ in pair_wins])
    ib = np.array([idx[b] for _, b in pair_wins])
    w = np.array([v[0] for v in pair_wins.values()], dtype=float)
    n = np.array([v[1] for v in pair_wins.values()], dtype=float)

    def nll(theta):
        d = theta[ia] - theta[ib]
        # -log p = softplus(-d), -log(1-p) = softplus(d)
        loss = (w * np.logaddexp(0, -d) + (n - w) * np.logaddexp(0, d)).sum() \
            + l2 * (theta ** 2).sum()
        resid = w - n * expit(d)
        grad = 2 * l2 * theta
        np.subtract.at(grad, ia, resid)
        np.add.at(grad, ib, resid)
        return loss, grad

    res = minimize(nll, np.zeros(len(models)), jac=True, method="L-BFGS-B")
    if not res.success:
        raise RuntimeError(f"Bradley-Terry fit failed: {res.message}")
    theta = res.x - res.x.mean()
    return dict(zip(models, theta))


def components(pairs):
    """Connected components of the comparison graph, largest first.

    Strengths are only comparable within a component; tau and figures use the
    largest one when the graph is disconnected.
    """
    adj = defaultdict(set)
    for a, b in pairs:
        adj[a].add(b)
        adj[b].add(a)
    seen, comps = set(), []
    for start in adj:
        if start in seen:
            continue
        comp, stack = set(), [start]
        while stack:
            m = stack.pop()
            if m in seen:
                continue
            seen.add(m)
            comp.add(m)
            stack.extend(adj[m] - seen)
        comps.append(comp)
    return sorted(comps, key=len, reverse=True)


def collect_pair_wins(rows, group_cols, ties="half"):
    """rows -> {group: {(major, minor): [wins_major, n]}}.

    picked_major is a float in the induced picks (ties 0.5) and True/False in
    the pairwise verdicts. ties: "half" counts a tie as half a win (evidence
    of equal strength), "drop" discards it (no evidence) as a sensitivity.

    Position is counterbalanced first: the two presentation orders of each item
    are averaged into one item vote before aggregation, so position bias
    cancels and a single parse-failed order can't inject an unbalanced vote
    (matches scripts/analyses/stats.py). Induced picks have one row per item,
    so this is a no-op for them.
    """
    as_win = {"True": 1.0, "False": 0.0}
    per_item = defaultdict(lambda: [0.0, 0])
    for r in rows:
        if r["major"] == r["minor"]:
            raise ValueError(f"self-comparison row: {r}")
        v = r["picked_major"]
        v = as_win[v] if v in as_win else float(v)
        if v not in (0.0, 0.5, 1.0):
            raise ValueError(f"bad picked_major in row: {r}")
        gk = tuple(r[c] for c in group_cols)
        pi = per_item[(gk, r["major"], r["minor"], r["item_id"])]
        pi[0] += v
        pi[1] += 1
    groups = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
    for (gk, major, minor, _item), (s, n) in per_item.items():
        v = s / n
        if v == 0.5 and ties == "drop":
            continue
        pw = groups[gk][(major, minor)]
        pw[0] += v
        pw[1] += 1
    return groups


def read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def mean_score_rows(score_rows, capability):
    """Per (bench, judge): mean 0-100 score per model + Kendall tau vs capability.

    Returns (rows, summary_rows) shaped like the elo outputs so the same grid
    plotter works (component is always 0: no comparison graph here).
    """
    by_group = defaultdict(lambda: defaultdict(list))
    for r in score_rows:
        if r["parse_ok"] == "True":
            by_group[(r["bench"], r["judge"])][r["model"]].append(
                (r["item_id"], float(r["score"])))
    rows, summary_rows = [], []
    for (bench, judge), per_model in sorted(by_group.items()):
        cap = load_config(bench)[capability]
        means = {m: sum(s for _, s in v) / len(v) for m, v in per_model.items()}
        for m, s in sorted(means.items()):
            rows.append(dict(bench=bench, source="direct", condition="", judge=judge,
                             model=m, mean_score=round(s, 4),
                             n_scores=len(per_model[m]), component=0))
        both = [m for m in means if m in cap]
        tau, rho, r = correlations([means[m] for m in both], [cap[m] for m in both])
        top_rho, n_top = top_spearman(means, cap, both)
        halves = [{m: [s for i, s in v if item_half(i) == p] for m, v in per_model.items()}
                  for p in (0, 1)]
        common = [m for m in means if all(h[m] for h in halves)]
        rel = ""
        if len(common) >= 3:
            rel = round(spearmanr([sum(halves[0][m]) / len(halves[0][m]) for m in common],
                                  [sum(halves[1][m]) / len(halves[1][m]) for m in common]
                                  ).statistic, 4)
        summary_rows.append(dict(bench=bench, judge=judge, capability=capability,
                                 n_models=len(means), kendall_elo_vs_capability=tau,
                                 spearman_elo_vs_capability=rho,
                                 pearson_elo_vs_capability=r,
                                 spearman_top20_vs_capability=top_rho,
                                 n_models_top20=n_top, split_half_item_agreement=rel))
    return rows, summary_rows


def fit_groups(groups, source, l2, capability, ties="half", calls=None,
               reliability=None):
    """groups: (bench, condition, judge) -> pair_wins. Returns (elo_rows, summary_rows).

    calls: group -> number of judge calls behind the picks. For pairwise every
    pick is one call; for induced picks one scoring call feeds many picks, so
    the caller passes the distinct scored-transcript count.
    reliability: group -> split-half item agreement (rank one item-half vs the other).
    """
    elo_rows, summary_rows = [], []
    for (bench, condition, judge), pair_wins in sorted(groups.items()):
        eci = load_config(bench)[capability]
        strengths = fit_bt(pair_wins, l2=l2)
        comps = components(pair_wins)
        if len(comps) > 1:
            print(f"  WARNING {bench}/{condition or source}/{judge}: comparison graph "
                  f"has {len(comps)} components; tau and figures use the largest "
                  f"({len(comps[0])}/{len(strengths)} models)")
        comp_of = {m: i for i, comp in enumerate(comps) for m in comp}
        picks_per_model = defaultdict(int)
        for (a, b), (_, n) in pair_wins.items():
            picks_per_model[a] += n
            picks_per_model[b] += n
        for m, s in strengths.items():
            elo_rows.append(dict(bench=bench, source=source, condition=condition,
                                 judge=judge, model=m, elo=round(s, 4),
                                 n_picks=picks_per_model[m], component=comp_of[m]))
        both = [m for m in comps[0] if m in eci]
        if len(both) < len(comps[0]):
            print(f"  {bench}/{condition or source}/{judge}: "
                  f"{len(comps[0]) - len(both)} models missing {capability}, "
                  f"dropped from tau")
        tau, rho, r = correlations([strengths[m] for m in both],
                                   [eci[m] for m in both])
        top_rho, n_top = top_spearman(strengths, eci, both)
        n_picks = sum(n for _, n in pair_wins.values())
        summary_rows.append(dict(
            bench=bench, source=source, condition=condition, judge=judge,
            capability=capability, n_models=len(strengths), n_picks=n_picks,
            n_calls=calls[(bench, condition, judge)] if calls else n_picks,
            ties=ties, n_components=len(comps), n_models_tau=len(both),
            kendall_elo_vs_capability=tau, spearman_elo_vs_capability=rho,
            pearson_elo_vs_capability=r, spearman_top20_vs_capability=top_rho,
            n_models_top20=n_top,
            split_half_item_agreement=(reliability or {}).get((bench, condition, judge), "")))
    return elo_rows, summary_rows


def elo_split_half(rows, group_cols, ties, l2):
    """Split items in half, fit each half, spearman between the two rankings.

    High values mean the judge's ranking is reproducible; the gap to the
    capability correlation is then real disagreement, not noise.
    """
    halves = []
    for parity in (0, 1):
        sub = [r for r in rows if item_half(r["item_id"]) == parity]
        groups = collect_pair_wins(sub, group_cols, ties)
        halves.append({g: fit_bt(pw, l2) for g, pw in groups.items()})
    rel = {}
    for g in halves[0].keys() & halves[1].keys():
        common = sorted(halves[0][g].keys() & halves[1][g].keys())
        if len(common) >= 3:
            rel[g] = round(spearmanr([halves[0][g][m] for m in common],
                                     [halves[1][g][m] for m in common]).statistic, 4)
    return rel


RC = {"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
      "xtick.labelsize": 7, "ytick.labelsize": 7,
      "axes.spines.top": False, "axes.spines.right": False}


def linear_fit_local_band(xs, ys, xline, bandwidth_frac=0.2):
    """Global linear fit with a locally-sized 95% prediction band.

    The line is plain least squares. The band around it is kernel-local: at
    each grid position a Gaussian kernel (bandwidth = bandwidth_frac of the
    x range) sets the effective n and the residual scatter, so the band
    tightens where models are dense and widens where they are sparse or
    noisier. Grid positions with effective n < 3 get no band.
    """
    slope, intercept = np.polyfit(xs, ys, 1)
    yline = slope * xline + intercept
    resid = ys - (slope * xs + intercept)
    h = bandwidth_frac * (xs.max() - xs.min())
    lo, hi = (np.full(len(xline), np.nan) for _ in range(2))
    for i, x0 in enumerate(xline):
        w = np.exp(-0.5 * ((xs - x0) / h) ** 2)
        n_eff = w.sum() ** 2 / (w ** 2).sum()
        if n_eff < 3:
            continue
        w /= w.sum()
        s = np.sqrt((w * resid ** 2).sum() * n_eff / (n_eff - 2))
        se = s * np.sqrt(1 + 1 / n_eff)
        tcrit = t_dist.ppf(0.975, n_eff - 2)
        lo[i], hi[i] = yline[i] - tcrit * se, yline[i] + tcrit * se
    return yline, lo, hi


def plot_group_grid(elo_rows, summary_rows, out_path, capability,
                    ylabel="Judge Elo", ykey="elo", corr="kendall"):
    """Wrapped grid of per-judge panels: fitted strength vs capability, one
    point per model.

    Grey dashed vertical line = the judge's own capability; if the judge is
    among the evaluated models (judge_model in the config), its point is
    drawn in orange.
    """
    judges = sorted({r["judge"] for r in elo_rows})
    bench = elo_rows[0]["bench"]
    cfg = load_config(bench)
    eci, judge_eci = cfg[capability], cfg.get(f"judge_{capability}", {})
    judge_model = cfg.get("judge_model", {})
    tau = {r["judge"]: r[f"{corr}_elo_vs_capability"] for r in summary_rows}
    ncols = min(4, len(judges))
    nrows = math.ceil(len(judges) / ncols)
    plt.rcParams.update(RC)
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.5 * ncols + 0.4, 1.35 * nrows + 0.4),
                             sharex=True, sharey=True, squeeze=False)
    any_self = any_vline = False
    for ax, judge in zip(axes.flat, judges):
        if judge in judge_eci:
            ax.axvline(judge_eci[judge], lw=0.8, ls="--", color="grey", zorder=0)
            any_vline = True
        else:
            print(f"  no judge_{capability} for {judge}, no reference line")
        self_model = judge_model.get(judge)
        pts, self_pt = [], None
        for r in elo_rows:
            if r["judge"] != judge or r["model"] not in eci or r["component"] != 0:
                continue
            if r["model"] == self_model:
                self_pt = (eci[r["model"]], r[ykey])
            else:
                pts.append((eci[r["model"]], r[ykey]))
        fit_pts = pts + ([self_pt] if self_pt else [])
        # thin grey line connecting models in capability order (the "connected
        # scatter"); models right of the dashed line are stronger than the judge
        if len(fit_pts) >= 2:
            ax.plot(*zip(*sorted(fit_pts)), color="#cccccc", lw=0.5, zorder=1)
        if pts:
            ax.scatter(*zip(*pts), s=6, color="#2a78d6", linewidths=0, zorder=2)
        if len(fit_pts) >= 3:
            xs, ys = np.array([p[0] for p in fit_pts]), np.array([p[1] for p in fit_pts])
            xline = np.linspace(xs.min(), xs.max(), 50)
            yline, lo, hi = linear_fit_local_band(xs, ys, xline)
            ax.plot(xline, yline, lw=1, color="#2a78d6", alpha=0.6, zorder=1)
        if self_pt:
            ax.scatter(*self_pt, s=10, color="#eb6834", linewidths=0, zorder=3)
            any_self = True
        ax.set_title(judge, loc="left", fontsize=7, color="grey", pad=2)
        label = f"n={len(fit_pts)}"
        if tau.get(judge) != "":
            label = f"{CORR_SYMBOL[corr]}={tau[judge]:.2f}\n{label}"
        ax.text(0.04, 0.96, label, transform=ax.transAxes, fontsize=7,
                color="grey", va="top")
    for ax in axes.flat:
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))  # sparse ticks, no crowding
    for ax in axes.flat[len(judges):]:
        ax.set_visible(False)
    handles = [plt.Line2D([], [], ls="", marker="o", ms=3, color="#2a78d6",
                          label="evaluated model"),
               plt.Line2D([], [], lw=1, color="#2a78d6", alpha=0.6, label="linear fit")]
    if any_vline:
        handles.append(plt.Line2D([], [], ls="--", lw=0.8, color="grey",
                                  label=f"judge's own {XLABEL[capability]}"))
    if any_self:
        handles.append(plt.Line2D([], [], ls="", marker="o", ms=4, color="#eb6834",
                                  label="judge itself"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0),
               ncol=len(handles), frameon=False)
    for ax in axes[-1]:
        ax.set_xlabel(XLABEL[capability])
    for row in axes:
        row[0].set_ylabel(ylabel)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out_path}")


CONDITION_COLORS = {"pairwise": "#2a78d6", "direct-elo": "#eb6834",
                    "direct-score": "#1baf7a"}
CONDITION_ORDER = ["pairwise", "direct-elo", "direct-score"]
# judge identity is the series in the slopegraph; muted, colourblind-safe, fixed
JUDGE_COLORS = {"gpt-oss-120b": "#2a78d6", "gpt-oss-20b": "#eb6834",
                "qwen3-8b": "#1baf7a", "qwen3-14b": "#eda100",
                "qwen3-32b": "#e87ba4", "llama-3.2-3b": "#4a3aa7",
                "api-deepseek-v4-flash": "#8c6d4a", "api-ling-2.6-flash": "#5a5a5a"}


def plot_judge_lines(value_rows, capability, corr, out_path):
    """Slopegraph: x = condition (pairwise-elo, direct-elo, direct-score),
    one line per judge. Same values as plot_condition_summary, transposed so a
    judge's method-to-method change is a single readable slope.
    value_rows: list of (row_label, {(bench, condition, judge) -> value})."""
    all_keys = [k for _, vals in value_rows for k in vals]
    benches = sorted({b for b, _, _ in all_keys})
    judges = sorted({j for _, _, j in all_keys})
    xlabels = ["pairwise-elo", "direct-elo", "direct-score"]
    plt.rcParams.update(RC)
    fig, axes = plt.subplots(len(value_rows), len(benches),
                             figsize=(2.0 * len(benches) + 0.6, 1.9 * len(value_rows) + 0.4),
                             sharey=True, sharex=True, squeeze=False)
    drawn = []
    for row, (label, values) in enumerate(value_rows):
        for ax, bench in zip(axes[row], benches):
            for judge in judges:
                pts = [(i, values[(bench, c, judge)]) for i, c in enumerate(CONDITION_ORDER)
                       if (bench, c, judge) in values]
                if not pts:
                    continue
                ax.plot(*zip(*pts), "-o", ms=3, lw=1,
                        color=JUDGE_COLORS.get(judge, "#999999"), zorder=2)
                if judge not in drawn:
                    drawn.append(judge)
            ax.set_xlim(-0.3, 2.3)
            ax.set_xticks(range(3), xlabels if row == len(value_rows) - 1 else [], rotation=90)
            if row == 0:
                ax.set_title(bench, loc="left", fontsize=7, color="grey", pad=2)
        axes[row][0].set_ylabel(label)
    corr_ylim(axes, [v for _, vals in value_rows for v in vals.values()])
    handles = [plt.Line2D([], [], ls="-", marker="o", ms=3, lw=1,
                          color=JUDGE_COLORS.get(j, "#999999"), label=j)
               for j in sorted(drawn)]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0),
               ncol=min(4, len(handles)), frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out_path}")

ACC_SUBSETS = ["all", "close", "stronger"]
ACC_LABEL = {"all": "all pairs", "close": f"close-ranked (≤{{}} ECI)",
             "stronger": "both stronger than judge"}
PICK_COLORS = {"pairwise": "#2a78d6", "direct": "#eb6834"}


def pick_accuracy(rows, close_eci, min_n=50, min_models=MIN_COMPARE_MODELS):
    """Oriented pick accuracy (P pick the higher-ECI model) over pair subsets.

    rows carry bench, judge, major, minor, item_id, picked_major (True/False for
    pairwise verdicts, float in [0,1] for induced picks; ties contribute 0.5).
    Position is counterbalanced first, same as the BT input: an item's two
    orders are averaged into one vote before scoring. Subsets:
      all      - every pair with an ECI gap
      close    - |ECI gap| <= close_eci (discrimination among near-equals)
      stronger - both models above the judge's own ECI (scalable-oversight case)
    Returns {(bench, judge, subset): (accuracy, n_items)}, dropping cells below
    min_n items.
    """
    as_win = {"True": 1.0, "False": 0.0}
    per_item = defaultdict(lambda: [0.0, 0])
    models_bj = defaultdict(set)
    for r in rows:
        v = r["picked_major"]
        v = as_win[v] if v in as_win else float(v)
        per_item[(r["bench"], r["judge"], r["major"], r["minor"], r["item_id"])][0] += v
        per_item[(r["bench"], r["judge"], r["major"], r["minor"], r["item_id"])][1] += 1
        models_bj[(r["bench"], r["judge"])].update((r["major"], r["minor"]))
    agg = defaultdict(lambda: [0.0, 0])
    for (bench, j, a, b, _item), (s, n) in per_item.items():
        cfg = load_config(bench)
        eci, jeci = cfg["eci"], cfg.get("judge_eci", {})
        if a not in eci or b not in eci or eci[a] == eci[b]:
            continue
        v = s / n
        correct = v if eci[a] > eci[b] else 1 - v
        subsets = ["all"]
        if abs(eci[a] - eci[b]) <= close_eci:
            subsets.append("close")
        if j in jeci and min(eci[a], eci[b]) > jeci[j]:
            subsets.append("stronger")
        for sub in subsets:
            agg[(bench, j, sub)][0] += correct
            agg[(bench, j, sub)][1] += 1
    # drop under-covered (in-progress) judges along with small subsets
    return {k: (round(c / n, 4), n) for k, (c, n) in agg.items()
            if n >= min_n and len(models_bj[(k[0], k[1])]) >= min_models}


PAIRED_SLICES = ["close", "close, ≥1>judge", "close, both>judge"]


def _counterbalanced_items(rows):
    """(bench, judge, major, minor, item) -> mean picked_major over its orders."""
    pi = defaultdict(lambda: [0.0, 0])
    as_win = {"True": 1.0, "False": 0.0}
    for r in rows:
        v = r["picked_major"]
        v = as_win[v] if v in as_win else float(v)
        k = (r["bench"], r["judge"], r["major"], r["minor"], r["item_id"])
        pi[k][0] += v
        pi[k][1] += 1
    return {k: s / n for k, (s, n) in pi.items()}


def paired_diff(pairwise_rows, direct_rows, close_eci=10.0, min_models=MIN_COMPARE_MODELS):
    """Paired pairwise − pointwise oriented-correctness diff on matched items,
    analytic CI clustered by (major, minor) pair.

    Per item (counterbalanced, restricted to items in BOTH sources):
      correct = v if eci[major] > eci[minor] else 1 - v   (v = counterbalanced pick)
      diff = correct_pairwise - correct_pointwise
    Cluster unit = pair: per-pair mean over items, then equal-weight mean over
    pairs; CI = mean ± 1.96·sd(per-pair diffs, ddof=1)/sqrt(n_pairs). Slices all
    require close = |ECI gap| < close_eci: close; close & >=1 above judge; close
    & both above judge. Under-covered judges (< min_models) dropped.
    """
    pw, dr = _counterbalanced_items(pairwise_rows), _counterbalanced_items(direct_rows)
    pair_items = defaultdict(list)  # (bench, judge, major, minor) -> [(diff, cpw, cpt), ...]
    models_bj = defaultdict(set)    # coverage = models in the MATCHED data (the limit)
    for k in pw.keys() & dr.keys():
        bench, judge, a, b, _item = k
        eci = load_config(bench)["eci"]
        if a not in eci or b not in eci or eci[a] == eci[b]:
            continue
        hi = eci[a] > eci[b]
        cpw = pw[k] if hi else 1 - pw[k]
        cpt = dr[k] if hi else 1 - dr[k]
        pair_items[(bench, judge, a, b)].append((cpw - cpt, cpw, cpt))
        models_bj[(bench, judge)].update((a, b))
    pairs_by = defaultdict(list)  # (bench, judge) -> [(a, b, per-pair means, n_items)]
    for (bench, judge, a, b), items in pair_items.items():
        pairs_by[(bench, judge)].append((a, b, np.mean(items, axis=0), len(items)))

    out = []
    for (bench, judge), plist in sorted(pairs_by.items()):
        if len(models_bj[(bench, judge)]) < min_models:
            continue
        cfg = load_config(bench)
        eci, jeci = cfg["eci"], cfg.get("judge_eci", {})
        jc = jeci.get(judge)
        preds = {
            "close": lambda a, b: True,
            "close, ≥1>judge": lambda a, b: jc is not None and max(eci[a], eci[b]) > jc,
            "close, both>judge": lambda a, b: jc is not None and min(eci[a], eci[b]) > jc,
        }
        close = [(a, b, pm, n) for (a, b, pm, n) in plist if abs(eci[a] - eci[b]) < close_eci]
        for name, keep in preds.items():
            sel = [(pm, n) for (a, b, pm, n) in close if keep(a, b)]
            if len(sel) < 2:
                continue
            pm = np.array([m for m, _ in sel])  # rows: (diff, cpw, cpt)
            mean = pm.mean(axis=0)
            se = pm[:, 0].std(ddof=1) / np.sqrt(len(sel))
            tcrit = t_dist.ppf(0.975, len(sel) - 1)  # t, not 1.96: n_pairs can be small
            out.append(dict(bench=bench, judge=judge, slice=name,
                            pairwise_acc=round(mean[1], 4), pointwise_acc=round(mean[2], 4),
                            diff=round(mean[0], 4), ci_lo=round(mean[0] - tcrit * se, 4),
                            ci_hi=round(mean[0] + tcrit * se, 4),
                            n_pairs=len(sel), n_items=int(sum(n for _, n in sel))))
    return out


def plot_paired_diff(records, out_path):
    """Rows = slices, cols = bench; x = judges, y = pairwise − pointwise oriented
    accuracy with 95% CI clustered by pair. Above 0 = pairwise better."""
    benches = sorted({r["bench"] for r in records})
    idx = {(r["bench"], r["slice"], r["judge"]): r for r in records}
    plt.rcParams.update(RC)
    fig, axes = plt.subplots(len(PAIRED_SLICES), len(benches),
                             figsize=(2.4 * len(benches) + 0.4, 1.7 * len(PAIRED_SLICES) + 0.4),
                             sharey=True, sharex="col", squeeze=False)
    span = max((abs(r["ci_lo"]) for r in records), default=0.1)
    span = max(span, max((abs(r["ci_hi"]) for r in records), default=0.1))
    for row, sl in enumerate(PAIRED_SLICES):
        for ax, bench in zip(axes[row], benches):
            judges = sorted({r["judge"] for r in records
                             if r["bench"] == bench and r["slice"] == sl})
            ax.axhline(0, lw=0.8, ls="--", color="grey", zorder=0)
            for i, j in enumerate(judges):
                r = idx.get((bench, sl, j))
                if not r:
                    continue
                ax.errorbar(i, r["diff"], yerr=[[r["diff"] - r["ci_lo"]], [r["ci_hi"] - r["diff"]]],
                            fmt="o", ms=3, color="#2a78d6", capsize=0, lw=1, zorder=2)
            ax.set_xticks(range(len(judges)),
                          judges if row == len(PAIRED_SLICES) - 1 else [], rotation=90)
            if row == 0:
                ax.set_title(bench, loc="left", fontsize=7, color="grey", pad=2)
            ax.set_ylim(-span * 1.15, span * 1.15)
        axes[row][0].set_ylabel(sl)
    fig.supylabel("pairwise − pointwise accuracy", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out_path}")


def plot_pick_accuracy(acc, close_eci, out_path):
    """Rows = pair subsets, cols = benches, x = judges; pairwise vs direct dots,
    0.5 chance line. acc: {(bench, source, judge, subset): (accuracy, n)}."""
    benches = sorted({b for b, _, _, _ in acc})
    plt.rcParams.update(RC)
    fig, axes = plt.subplots(len(ACC_SUBSETS), len(benches),
                             figsize=(2.4 * len(benches) + 0.4, 1.7 * len(ACC_SUBSETS) + 0.4),
                             sharey=True, sharex="col", squeeze=False)
    for row, subset in enumerate(ACC_SUBSETS):
        for ax, bench in zip(axes[row], benches):
            judges = sorted({j for b, _, j, s in acc if b == bench})
            ax.axhline(0.5, lw=0.8, ls="--", color="grey", zorder=0)
            for src, color in PICK_COLORS.items():
                off = -0.12 if src == "pairwise" else 0.12
                xs = [i + off for i, j in enumerate(judges) if (bench, src, j, subset) in acc]
                ys = [acc[(bench, src, j, subset)][0] for j in judges
                      if (bench, src, j, subset) in acc]
                ax.scatter(xs, ys, s=8, color=color, linewidths=0, zorder=2)
            ax.set_xticks(range(len(judges)),
                          judges if row == len(ACC_SUBSETS) - 1 else [], rotation=90)
            if row == 0:
                ax.set_title(bench, loc="left", fontsize=7, color="grey", pad=2)
            ax.set_ylim(0.3, 1)
            ax.set_yticks([0.5, 0.75, 1])
        axes[row][0].set_ylabel(ACC_LABEL[subset].format(close_eci))
    handles = [plt.Line2D([], [], ls="", marker="o", ms=3, color=c, label=s)
               for s, c in PICK_COLORS.items()]
    handles.append(plt.Line2D([], [], ls="--", lw=0.8, color="grey", label="chance"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0),
               ncol=3, frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out_path}")


def corr_ylim(axes, values):
    """Shared y-limits for a correlation grid: include any negatives, cap at 1,
    dashed chance line at 0. Spearman/Kendall are 0 at chance, not 0.5."""
    lo = min([v for v in values], default=0.0)
    bottom = min(0.0, lo) - 0.05
    ticks = [t for t in (-0.5, 0, 0.5, 1) if t >= bottom - 0.01]
    for ax in axes.flat:
        ax.set_ylim(bottom, 1.03)
        ax.set_yticks(ticks)
        ax.axhline(0, lw=0.8, ls="--", color="grey", zorder=0)


def comparison_values(elo_rows, mean_rows, capability, fig_condition):
    """Per (bench, judge): each method's rank correlation vs capability over the
    COMMON set of models all present methods can rank.

    A method's rho over its own model set is not comparable to another's over a
    different set (e.g. a partial direct run ranks fewer models than pairwise).
    Restricting to the intersection makes the pairwise-vs-direct comparison
    apples-to-apples. Returns {metric: {(bench, condition, judge): value}} for
    kendall / spearman / spearman_top20, plus n_common per (bench, judge).
    """
    per_model = defaultdict(dict)  # (bench, judge, condition) -> {model: value}
    for r in elo_rows:
        if str(r["component"]) != "0":
            continue
        if r["source"] == "pairwise":
            if r["condition"] != fig_condition:
                continue
            cond = "pairwise"
        else:
            cond = "direct-elo"
        per_model[(r["bench"], r["judge"], cond)][r["model"]] = float(r["elo"])
    for r in mean_rows:
        per_model[(r["bench"], r["judge"], "direct-score")][r["model"]] = float(r["mean_score"])

    conds_by_bj = defaultdict(list)
    for (bench, judge, cond) in per_model:
        conds_by_bj[(bench, judge)].append(cond)
    out = {"kendall": {}, "spearman": {}, "spearman_top20": {},
           "spearman_stronger": {}, "kendall_stronger": {}, "n_common": {}}
    for (bench, judge), conds in conds_by_bj.items():
        cfg = load_config(bench)
        cap, eci, jeci = cfg[capability], cfg["eci"], cfg.get("judge_eci", {})
        # drop under-covered (in-progress) conditions before intersecting, so a
        # partial direct run can't shrink a complete method's common set
        conds = [c for c in conds
                 if len(set(per_model[(bench, judge, c)]) & set(cap)) >= MIN_COMPARE_MODELS]
        if not conds:
            continue
        common = set.intersection(*(set(per_model[(bench, judge, c)]) for c in conds)) & set(cap)
        if len(common) < 3:
            continue
        out["n_common"][(bench, judge)] = len(common)
        models = sorted(common)
        top = sorted(common, key=cap.get, reverse=True)[:TOP_N]
        jc = jeci.get(judge)
        stronger = sorted(m for m in common if jc is not None and m in eci and eci[m] > jc)
        for c in conds:
            vals = per_model[(bench, judge, c)]
            corr = lambda subset: (round(kendalltau([vals[m] for m in subset],
                                                     [cap[m] for m in subset]).statistic, 4),
                                   round(spearmanr([vals[m] for m in subset],
                                                   [cap[m] for m in subset]).statistic, 4))
            out["kendall"][(bench, c, judge)], out["spearman"][(bench, c, judge)] = corr(models)
            if len(top) >= 5:
                out["spearman_top20"][(bench, c, judge)] = corr(top)[1]
            if len(stronger) >= 10:  # oversight regime: models stronger than the judge
                k, sp = corr(stronger)
                out["kendall_stronger"][(bench, c, judge)] = k
                out["spearman_stronger"][(bench, c, judge)] = sp
    return out


def oversight_values(elo_rows, mean_rows, fig_condition, floor=10):
    """The scalable-oversight cut: Kendall tau over models STRONGER than the
    judge (eci > judge_eci), pairwise vs direct-score, vs both ECI and accuracy.

    Subset = stronger-than-judge models both methods rank (common set). Cells
    below `floor` models are dropped (strong judges have few stronger models).
    Returns {(bench, target, method, judge): (tau, n)}; target in eci/accuracy.
    """
    per_model = defaultdict(dict)  # (bench, judge, method) -> {model: value}
    for r in elo_rows:
        if str(r["component"]) != "0" or r["source"] != "pairwise":
            continue
        if r["condition"] == fig_condition:
            per_model[(r["bench"], r["judge"], "pairwise")][r["model"]] = float(r["elo"])
    for r in mean_rows:
        per_model[(r["bench"], r["judge"], "direct-score")][r["model"]] = float(r["mean_score"])

    methods_by = defaultdict(set)
    for (bench, judge, method) in per_model:
        methods_by[(bench, judge)].add(method)
    out = {}
    for (bench, judge), methods in methods_by.items():
        if methods != {"pairwise", "direct-score"}:
            continue  # need both to compare
        cfg = load_config(bench)
        jeci = cfg.get("judge_eci", {})
        if judge not in jeci:
            continue
        common = set(per_model[(bench, judge, "pairwise")]) & set(per_model[(bench, judge, "direct-score")])
        stronger = {m for m in common if m in cfg["eci"] and cfg["eci"][m] > jeci[judge]}
        for target in ("eci", "accuracy"):
            capmap = cfg.get(target, {})
            models = sorted(m for m in stronger if m in capmap)
            if len(models) < floor:
                continue
            for method in methods:
                vals = per_model[(bench, judge, method)]
                tau = kendalltau([vals[m] for m in models],
                                 [capmap[m] for m in models]).statistic
                out[(bench, target, method, judge)] = (round(tau, 4), len(models))
    return out


def plot_oversight(oversight, out_path):
    """Rows = capability target (ECI, accuracy), cols = bench; x = judges,
    pairwise vs direct-score dots. The stronger-than-judge (oversight) cut."""
    benches = sorted({b for b, _, _, _ in oversight})
    targets = ["eci", "accuracy"]
    colors = {"pairwise": "#2a78d6", "direct-score": "#1baf7a"}
    plt.rcParams.update(RC)
    fig, axes = plt.subplots(len(targets), len(benches),
                             figsize=(2.4 * len(benches) + 0.4, 1.9 * len(targets) + 0.4),
                             sharey=True, sharex="col", squeeze=False)
    for row, target in enumerate(targets):
        for ax, bench in zip(axes[row], benches):
            judges = sorted({j for b, t, _, j in oversight if b == bench and t == target})
            for method, color in colors.items():
                off = -0.12 if method == "pairwise" else 0.12
                xs = [i + off for i, j in enumerate(judges)
                      if (bench, target, method, j) in oversight]
                ys = [oversight[(bench, target, method, j)][0] for j in judges
                      if (bench, target, method, j) in oversight]
                ax.scatter(xs, ys, s=8, color=color, linewidths=0, zorder=2)
            ax.set_xticks(range(len(judges)),
                          judges if row == len(targets) - 1 else [], rotation=90)
            if row == 0:
                ax.set_title(bench, loc="left", fontsize=7, color="grey", pad=2)
        axes[row][0].set_ylabel(f"τ vs {XLABEL[target]}")
    corr_ylim(axes, [v[0] for v in oversight.values()])
    handles = [plt.Line2D([], [], ls="", marker="o", ms=3, color=c, label=m)
               for m, c in colors.items()]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0),
               ncol=2, frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out_path}")


def plot_condition_summary(value_rows, capability, corr, out_path):
    """value_rows: list of (row_label, {(bench, condition, judge) -> correlation}).
    One column per bench, one row per label, judges on x, one dot per condition."""
    all_keys = [k for _, vals in value_rows for k in vals]
    benches = sorted({b for b, _, _ in all_keys})
    plt.rcParams.update(RC)
    fig, axes = plt.subplots(len(value_rows), len(benches),
                             figsize=(2.4 * len(benches) + 0.4, 1.9 * len(value_rows) + 0.4),
                             sharey=True, sharex="col", squeeze=False)
    conds = list(CONDITION_COLORS)
    for row, (label, values) in enumerate(value_rows):
        for ax, bench in zip(axes[row], benches):
            judges = sorted({j for b, _, j in all_keys if b == bench})
            # thin grey line per judge connecting its condition dots
            for jx, judge in enumerate(judges):
                pts = [(jx + (i - 1) * 0.18, values[(bench, c, judge)])
                       for i, c in enumerate(conds) if (bench, c, judge) in values]
                if len(pts) > 1:
                    ax.plot(*zip(*pts), color="#bbbbbb", lw=0.5, zorder=1)
            for i, cond in enumerate(conds):
                xs = [j + (i - 1) * 0.18 for j, judge in enumerate(judges)
                      if (bench, cond, judge) in values]
                ys = [values[(bench, cond, judge)] for judge in judges
                      if (bench, cond, judge) in values]
                ax.scatter(xs, ys, s=8, color=CONDITION_COLORS[cond], linewidths=0, zorder=2)
            ax.set_xticks(range(len(judges)), judges if row == len(value_rows) - 1 else [],
                          rotation=90)
            if row == 0:
                ax.set_title(bench, loc="left", fontsize=7, color="grey", pad=2)
        axes[row][0].set_ylabel(label)
    corr_ylim(axes, [v for _, vals in value_rows for v in vals.values()])
    handles = [plt.Line2D([], [], ls="", marker="o", ms=3, color=c, label=cond)
               for cond, c in CONDITION_COLORS.items()]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0),
               ncol=3, frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--picks", default=str(ROOT / "judge_results" / "direct_judge" / "picks.csv"),
                    help="induced picks from pointwise scores")
    ap.add_argument("--scores", default=str(ROOT / "judge_results" / "direct_judge" / "scores.csv"),
                    help="per-transcript pointwise scores, for the mean-score view")
    ap.add_argument("--pairwise-verdicts",
                    default=str(ROOT / "judge_results" / "judging_variations" / "verdicts.csv"))
    ap.add_argument("--fig-condition", default="a_baseline",
                    help="pairwise condition to plot (all conditions are fit)")
    ap.add_argument("--l2", type=float, default=1e-3,
                    help="ridge strength keeping Bradley-Terry finite")
    ap.add_argument("--exclude-judges", nargs="*",
                    default=["qwen3-0.6b", "qwen3-0.6b-yarn",
                             "api-deepseek-v4-flash", "api-ling-2.6-flash"],
                    help="judges to drop from fits, outputs and figures. Default "
                         "drops qwen3-0.6b (at chance) and the api-* judges (their "
                         "pairwise runs are on other benches, so no head-to-head here)")
    ap.add_argument("--capability", choices=sorted(XLABEL), default="eci",
                    help="ground truth for tau and the x-axis")
    ap.add_argument("--ties", choices=["half", "drop"], default="half",
                    help="induced-pick ties: half a win, or dropped (sensitivity)")
    ap.add_argument("--close-eci", type=float, default=10.0,
                    help="ECI gap defining a close-ranked pair in the accuracy figure "
                         "(canonical 10; try 5 as a sensitivity)")
    ap.add_argument("--detail", action="store_true",
                    help="also write the per-method scatter grids; default emits "
                         "only the summary + decisive figures")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)

    drop = set(args.exclude_judges)
    direct_pick_rows, pairwise_pick_rows = [], []
    elo_rows, summary_rows = [], []
    if Path(args.picks).exists():
        rows = [r for r in read_rows(args.picks) if r["judge"] not in drop]
        print(f"direct: {len(rows)} pick rows")
        direct_pick_rows = rows
        for r in rows:
            r["condition"] = ""
        calls = defaultdict(set)
        for r in rows:
            calls[(r["bench"], r["condition"], r["judge"])].update(
                [(r["major"], r["item_id"]), (r["minor"], r["item_id"])])
        calls = {g: len(s) for g, s in calls.items()}
        groups = collect_pair_wins(rows, ["bench", "condition", "judge"], args.ties)
        rel = elo_split_half(rows, ["bench", "condition", "judge"], args.ties, args.l2)
        elo_rows, summary_rows = fit_groups(groups, "direct", args.l2,
                                            args.capability, args.ties, calls, rel)
    else:
        print(f"no direct picks at {args.picks}, skipped")
    if Path(args.pairwise_verdicts).exists():
        all_rows = [r for r in read_rows(args.pairwise_verdicts)
                    if r["judge"] not in drop]
        rows = [r for r in all_rows if r["parse_ok"] == "True"]
        print(f"pairwise: {len(rows)}/{len(all_rows)} verdict rows parsed")
        if not rows and all_rows:
            print("  WARNING all pairwise rows dropped, check parse_ok serialization")
        pairwise_pick_rows = [r for r in rows if r["condition"] == args.fig_condition]
        # one judge call per verdict row (both orders are real calls); counter-
        # balancing halves the BT observation count but not the call count
        pw_calls = defaultdict(int)
        for r in rows:
            pw_calls[(r["bench"], r["condition"], r["judge"])] += 1
        groups = collect_pair_wins(rows, ["bench", "condition", "judge"])
        rel = elo_split_half(rows, ["bench", "condition", "judge"], "half", args.l2)
        e, s = fit_groups(groups, "pairwise", args.l2, args.capability,  # no ties in pairwise
                          calls=dict(pw_calls), reliability=rel)
        elo_rows += e
        summary_rows += s
    else:
        print(f"no pairwise verdicts at {args.pairwise_verdicts}, skipped")

    mean_rows, mean_summary = [], []
    if Path(args.scores).exists():
        rows = [r for r in read_rows(args.scores) if r["judge"] not in drop]
        print(f"direct: {len(rows)} score rows for the mean-score view")
        mean_rows, mean_summary = mean_score_rows(rows, args.capability)
    else:
        print(f"no direct scores at {args.scores}, no mean-score view")

    out_dir.mkdir(parents=True, exist_ok=True)
    MEAN_FIELDS = ["bench", "source", "condition", "judge", "model", "mean_score",
                   "n_scores", "component"]
    for name, rows, fields in (("elos.csv", elo_rows, ELO_FIELDS),
                               ("elo_summary.csv", summary_rows, SUMMARY_FIELDS),
                               ("mean_scores.csv", mean_rows, MEAN_FIELDS)):
        with open(out_dir / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
    print(f"{len(elo_rows)} elo rows -> {out_dir}")

    fig_groups = sorted({(r["bench"], r["source"], r["condition"]) for r in elo_rows
                         if r["source"] == "direct" or r["condition"] == args.fig_condition})
    for corr, suffix in (("kendall", ""), ("spearman", "_rho")):
        # per-method performance-vs-model scatters (each dot a model; dashed line
        # = judge's own ECI, so models to its right are stronger than the judge).
        # pairwise + direct-score by default; direct-elo only with --detail.
        for bench, source, condition in fig_groups:
            if source == "direct" and not args.detail:
                continue  # direct-elo scatter is detail-only; direct-score handled below
            if source == "pairwise" and condition != args.fig_condition:
                continue
            sel = [r for r in elo_rows if (r["bench"], r["source"], r["condition"])
                   == (bench, source, condition)]
            summ = [r for r in summary_rows if (r["bench"], r["source"], r["condition"])
                    == (bench, source, condition)]
            cond_name = "pairwise" if source == "pairwise" else "direct-elo"
            name = f"{cond_name}_vs_{args.capability}_{bench}{suffix}"
            plot_group_grid(sel, summ, out_dir / "figures" / f"{name}.png",
                            args.capability, corr=corr)
        for bench in sorted({r["bench"] for r in mean_rows}):
            sel = [r for r in mean_rows if r["bench"] == bench]
            summ = [r for r in mean_summary if r["bench"] == bench]
            plot_group_grid(
                sel, summ,
                out_dir / "figures" / f"direct-score_vs_{args.capability}_{bench}{suffix}.png",
                args.capability, ylabel="Mean score", ykey="mean_score", corr=corr)

        def summary_values(key):
            values = {}
            for r in summary_rows:
                if r[key] == "":
                    continue
                if r["source"] == "direct":
                    values[(r["bench"], "direct-elo", r["judge"])] = r[key]
                elif r["condition"] == args.fig_condition:
                    values[(r["bench"], "pairwise", r["judge"])] = r[key]
            for r in mean_summary:
                if r[key] != "":
                    values[(r["bench"], "direct-score", r["judge"])] = r[key]
            return values

        # comparison figures use the common model set across methods (fair
        # apples-to-apples); the per-method scatter grids above keep each
        # method's own full set.
        comp = comparison_values(elo_rows, mean_rows, args.capability, args.fig_condition)
        # comparison figures show only complete head-to-head judges (both
        # pairwise and direct-score present); a pairwise-only judge is nothing
        # to compare, and this drops partial in-progress runs (e.g. gpt-oss-120b)
        complete = ({(b, j) for (b, c, j) in comp[corr] if c == "pairwise"}
                    & {(b, j) for (b, c, j) in comp[corr] if c == "direct-score"})
        keep = lambda vals: {(b, c, j): v for (b, c, j), v in vals.items() if (b, j) in complete}
        # lead with the two the analysis is about: all judgings, then the
        # oversight case (models stronger than the judge)
        sym, cap_name = CORR_SYMBOL[corr], XLABEL[args.capability]
        value_rows = [(f"{sym} vs {cap_name}\n(all models)", keep(comp[corr])),
                      (f"{sym} vs {cap_name}\n(stronger than judge)",
                       keep(comp[f"{corr}_stronger"]))]
        if corr == "spearman":
            value_rows.append((f"ρ vs {cap_name}\n(top {TOP_N})", keep(comp["spearman_top20"])))
            value_rows.append(("split-half\nagreement",
                               keep(summary_values("split_half_item_agreement"))))
        plot_condition_summary(value_rows, args.capability, corr,
                               out_dir / "figures" / f"summary_vs_{args.capability}{suffix}.png")
        plot_judge_lines(value_rows, args.capability, corr,
                         out_dir / "figures" / f"summary_lines_vs_{args.capability}{suffix}.png")

    # scalable-oversight headline: judge ranking stronger-than-itself models,
    # pairwise vs direct-score, vs ECI and accuracy
    oversight = oversight_values(elo_rows, mean_rows, args.fig_condition)
    if oversight:
        plot_oversight(oversight, out_dir / "figures" / "oversight_stronger_than_judge.png")
        with open(out_dir / "oversight_stronger_than_judge.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["bench", "target", "method", "judge",
                                              "kendall", "n_models"])
            w.writeheader()
            for (bench, target, method, judge), (tau, n) in sorted(oversight.items()):
                w.writerow(dict(bench=bench, target=target, method=method,
                                judge=judge, kendall=tau, n_models=n))

    # common-set comparison numbers backing the summary figures
    comp = comparison_values(elo_rows, mean_rows, args.capability, args.fig_condition)
    with open(out_dir / "comparison_common_set.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["bench", "condition", "judge", "n_common",
                                          "kendall", "spearman", "spearman_top20"])
        w.writeheader()
        for key in sorted(comp["spearman"]):
            bench, cond, judge = key
            w.writerow(dict(bench=bench, condition=cond, judge=judge,
                            n_common=comp["n_common"].get((bench, judge), ""),
                            kendall=comp["kendall"].get(key, ""),
                            spearman=comp["spearman"].get(key, ""),
                            spearman_top20=comp["spearman_top20"].get(key, "")))

    # oriented pick accuracy on pair subsets: same picks fed to pairwise vs direct
    pw_acc = pick_accuracy(pairwise_pick_rows, args.close_eci)
    direct_acc = pick_accuracy(direct_pick_rows, args.close_eci)
    acc = {(b, "pairwise", j, s): v for (b, j, s), v in pw_acc.items()}
    acc.update({(b, "direct", j, s): v for (b, j, s), v in direct_acc.items()})
    if acc:
        plot_pick_accuracy(acc, args.close_eci,
                           out_dir / "figures" / "pick_accuracy.png")
        with open(out_dir / "pick_accuracy.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["bench", "source", "judge", "subset",
                                              "accuracy", "n_items"])
            w.writeheader()
            for (b, src, j, s), (a, n) in sorted(acc.items()):
                w.writerow(dict(bench=b, source=src, judge=j, subset=s,
                                accuracy=a, n_items=n))

    # decisive cut: paired pairwise − pointwise on matched items, CI clustered by pair
    paired = paired_diff(pairwise_pick_rows, direct_pick_rows, args.close_eci)
    if paired:
        plot_paired_diff(paired, out_dir / "figures" / "paired_diff.png")
        with open(out_dir / "paired_diff.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["bench", "judge", "slice", "pairwise_acc",
                                              "pointwise_acc", "diff", "ci_lo", "ci_hi",
                                              "n_pairs", "n_items"])
            w.writeheader()
            w.writerows(paired)

    for s in summary_rows + mean_summary:
        print("  ", s)


if __name__ == "__main__":
    main()
