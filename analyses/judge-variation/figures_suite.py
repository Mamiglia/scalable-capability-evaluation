"""Figure suite over transcript-viewer exports - judging-variations copy.

Generalized from scripts/judge/figures_expansion.py: series colors assigned
dynamically from the dataset list (any jv-* condition datasets work), judge
palette covers the local judges.

Default: the data-expansion run with ECI gap axis (the run's design variable).
--datasets extends to any viewer datasets (e.g. gpqa-pilot-rd2); pilot models
have no ECI, so combined runs use --gap acc (per-bench accuracy diff).
Exported scores are P(pick first-named); load() reorients every pair so score
= P(pick the gap-axis-stronger member). Truncation runs are skipped.

Usage:
  uv run python scripts/judge/figures_expansion.py --out-dir judge_results/multi-eval-pilot/figures
  uv run python scripts/judge/figures_expansion.py --out-dir judge_results/combined/figures \
      --datasets data-expansion-gpqa data-expansion-aime2025 data-expansion-aime2026 gpqa-pilot-rd2 \
      --gap acc
"""

import argparse
import csv
import glob
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7})
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import beta as beta_dist

DATA = Path("interactive_dashboards/jv/data")
MODEL_SET_CSV = Path("configs/comparisons/model_set.csv")
EXPANSION_DATASETS = ["data-expansion-gpqa", "data-expansion-aime2025", "data-expansion-aime2026"]
SERIES_PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7", "#008300", "#e34948"]
EVAL_COLORS = {}  # filled in main from --datasets order

# set in main from --datasets / --gap
BENCHES = []       # series names, in dataset order (same bench pools across datasets)
DIRS = {}          # series -> list of viewer dataset dirs
GAP_AXIS = "eci"   # model-meta key the gap and pair orientation use
GAP_LABEL = "capability gap (ECI)"
EXCLUDE = set()    # members whose pairs are dropped everywhere


COND_LABEL = {"baseline": "final text only",
              "cot": "reasoning shown, long context",
              "cot-native": "reasoning shown, 32k context",
              "ignore-style": "ignore superficial style",
              "ignore-style-conf": "+ ignore confidence",
              "pick-weaker": "pick weaker (inverted)"}


def series_name(dataset):
    return dataset.removeprefix("jv-").replace("otis-aime-", "otis-aime ").replace("gpqa-", "gpqa ")
JUDGE_ORDER = ["qwen3-0.6b", "qwen3-0.6b-yarn", "llama-3.2-3b",
               "qwen3-8b", "qwen3-8b-yarn", "qwen3-14b", "qwen3-14b-yarn",
               "qwen3-32b", "qwen3-32b-yarn", "gpt-oss-20b", "gpt-oss-120b"]
JUDGE_COLORS = dict(zip(JUDGE_ORDER, ["#2a78d6", "#86b6ef", "#eb6834",
                                      "#1baf7a", "#66c7a3", "#e87ba4", "#eeaac4",
                                      "#eda100", "#f4c65c", "#9085e9", "#4a3aa7"]))


def load():
    """rows: one per (eval, judge, pair run); models: per-eval metadata."""
    rows, models = [], {}
    for s in BENCHES:
      for d in DIRS[s]:
        if not (d / "index.json").exists():
            continue
        idx = json.load(open(d / "index.json"))
        for m, meta in idx["models"].items():
            models.setdefault(s, {}).setdefault(m, meta)
        for r in idx["runs"]:
            if r.get("frac", "full") != "full":
                continue
            if r["major"] in EXCLUDE or r["minor"] in EXCLUDE:
                continue
            r = dict(r)
            M, N = idx["models"][r["major"]], idx["models"][r["minor"]]
            a, b = M.get(GAP_AXIS), N.get(GAP_AXIS)
            # exported score is P(pick first-named) and pair-name orientation is
            # mixed, so reorient every pair to major = gap-axis-stronger member
            if not r["is_self"] and a is not None and b is not None and a < b:
                r["major"], r["minor"], M, N, a, b = r["minor"], r["major"], N, M, b, a
                if r["score"] is not None:
                    r["score"] = 1 - r["score"]
                    r["score_lo"], r["score_hi"] = 1 - r["score_hi"], 1 - r["score_lo"]
            r["eval"] = s
            r["gap"] = abs(a - b) if a is not None and b is not None else None
            r["rel_gap_months"] = _month_gap(M.get("release"), N.get("release"))
            # ECI gap kept alongside the gap axis; score re-aligned to the
            # ECI-stronger member where the two orderings disagree
            ea, eb = M.get("eci"), N.get("eci")
            if ea is not None and eb is not None:
                r["eci_gap"] = abs(ea - eb)
                flip = ea < eb and not r["is_self"] and r["score"] is not None
                r["score_eci"] = 1 - r["score"] if flip else r["score"]
            else:
                r["eci_gap"] = r["score_eci"] = None
            rows.append(r)
    return rows, models


def _month_gap(a, b):
    """Signed months a minus b; call with (stronger, weaker) post-orientation."""
    def m(x):
        if not x or "-" not in str(x):
            return None
        y, mo = str(x).split("-")[:2]
        return int(y) * 12 + int(mo)
    ma, mb = m(a), m(b)
    return ma - mb if ma is not None and mb is not None else None


def wilson(c, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p, d = c / n, 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return center - half, center + half


def sigmoid(x, a, d):
    return 1 / (1 + np.exp(-(x - d) * a))


def fit_sigmoid(gaps, scores, weights):
    try:
        p0 = [2 / max(float(np.std(gaps)), 1e-9), float(np.median(gaps))]
        (a, d), _ = curve_fit(sigmoid, gaps, scores, p0=p0,
                              sigma=1 / np.sqrt(weights), maxfev=5000)
        return a, d
    except Exception:
        return None


DELTA_EDGES = {"rel_gap_months": [0, 4, 8, 12, 18, 24, 36]}
ECI_DELTA_EDGES = [4, 8, 12, 16, 20, 26, 40]


def binned_deltas(deltas, key):
    """(x, delta) pairs -> [(bin center, mean, 95% ci)] on fixed gap edges."""
    edges = DELTA_EDGES.get(key, ECI_DELTA_EDGES)
    out = []
    for lo, hi in zip(edges, edges[1:]):
        chunk = [d for x, d in deltas if lo <= x < hi]
        if len(chunk) < 2:
            continue
        out.append(((lo + hi) / 2, float(np.mean(chunk)),
                    1.96 * float(np.std(chunk, ddof=1)) / len(chunk) ** 0.5))
    return out


def style(ax):
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def save(fig, out, name):
    fig.savefig(out / f"{name}.png", dpi=150, bbox_inches="tight")
    fig.savefig(out / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)
    print(" ", name)


JUDGE_ONLY = {"qwen3-32b": ("Qwen3 32B", 130.0)}  # off-grid judge, absent from member tables


def jlabel(models, j):
    j = j.removesuffix("-yarn")  # serving variant, same judge; detail lives in methods
    if j in JUDGE_ONLY:
        return f"{JUDGE_ONLY[j][0]} ({JUDGE_ONLY[j][1]:.0f})"
    m = models.get("gpqa", {}).get(j) or next((mm[j] for mm in models.values() if j in mm), {})
    e = m.get("eci")
    return f"{m.get('label', j)} ({e:.0f})" if e else m.get("label", j)


# --- figures -------------------------------------------------------------


def fig_model_accuracy(models, out):
    grid, labels = [m for m in JUDGE_ORDER if m in models.get("gpqa", {})], {}
    for s in BENCHES:
        for m, meta in models.get(s, {}).items():
            if meta["arm"] == "A" and m not in grid:
                grid.append(m)
            labels.setdefault(m, meta["label"])
    fig, ax = plt.subplots(figsize=(0.8 * len(grid) + 2.2, 3.3))
    style(ax)
    W = 0.8 / len(BENCHES)
    for i, b in enumerate(BENCHES):
        xs, ys = [], []
        for k, m in enumerate(grid):
            acc = models.get(b, {}).get(m, {}).get("accuracy")
            if acc is not None:
                xs.append(k + (i - (len(BENCHES) - 1) / 2) * W)
                ys.append(acc)
        ax.bar(xs, ys, width=W, color=EVAL_COLORS[b], label=b, edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(grid)))
    ax.set_xticklabels([labels[m] for m in grid], rotation=20, ha="right", fontsize=8)
    ax.axhline(0.25, ls=":", c="#8a8a86", lw=1)
    ax.text(len(grid) - 0.5, 0.26, "GPQA chance", fontsize=7, color="#8a8a86", ha="right")
    ax.set_ylabel("accuracy (any single epoch)")
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Model accuracy per eval — collected members, 4 epochs pooled", fontsize=11)
    fig.tight_layout()
    save(fig, out, "0_model_accuracy")


def _gap_curve(ax, rows, key, series, colors, labelfn, xlab, fit=True, styles=None):
    style(ax)
    ax.axhline(0.5, ls="--", c="#8a8a86", lw=1)
    for s in series:
        rs = [r for r in rows if r["_series"] == s and not r["is_self"] and r.get(key) is not None
              and r["score"] is not None]
        if not rs:
            continue
        ls = (styles or {}).get(s, "-")
        xs = np.array([r[key] for r in rs], float)
        ys = np.array([r["score"] for r in rs], float)
        ns = np.array([max(r["n_items"], 1) for r in rs], float)
        ax.scatter(xs, ys, s=8 + ns / 4,
                   facecolors=(colors[s] if ls == "-" else "none"),
                   edgecolors=(("white") if ls == "-" else colors[s]),
                   alpha=0.55, linewidths=0.7, label=labelfn(s))
        if fit and len(rs) >= 4:
            f = fit_sigmoid(xs, ys, ns)
            if f:
                gx = np.linspace(0, max(xs) * 1.02, 100)
                ax.plot(gx, sigmoid(gx, *f), color=colors[s], lw=1.8, ls=ls)
    ax.set_xlabel(xlab)
    ax.set_ylabel("judge accuracy")
    ax.set_ylim(0.05, 1.02)


def _cot_flags(rows):
    """(bench, model) -> has real CoT in its epoch log; disk-cached."""
    import json as _json
    from pool import cot_ok, load_members
    cache_p = Path("judge_results/judging_variations/cot_flags.json")
    cache = _json.load(open(cache_p)) if cache_p.exists() else {}
    used = defaultdict(set)
    for r in rows:
        used[r["eval"].partition(" ")[0]] |= {r["major"], r["minor"]}
    flags = {}
    for bench, ms in used.items():
        try:
            members = {m["short"]: m for m in load_members(
                Path("external_data") / f"epoch-{bench}",
                Path("model-sampler") / "model_pool.json")}
        except Exception:
            continue
        for name in ms & set(members):
            k = f"{bench}|{name}"
            if k not in cache:
                cache[k] = cot_ok(members[name]["path"])
            flags[(bench, name)] = cache[k]
    cache_p.parent.mkdir(parents=True, exist_ok=True)
    _json.dump(cache, open(cache_p, "w"))
    return flags


def fig_gap_curves(rows, models, out):
    # 2a by judge (pooled evals), 2a-eval by eval; same pair by release gap (2b)
    specs = [("2a_gap_curve_by_capability", "gap", GAP_LABEL),
             ("2b_gap_curve_by_release", "rel_gap_months",
              "release gap (months, + = stronger member newer)")]
    if GAP_AXIS != "eci":
        specs.append(("2a_gap_curve_by_capability_eci", "eci_gap", "capability gap (ECI)"))
    for name, key, xlab in specs:
        # by judge: grid of pair reasoning-type (rows, accuracy+delta band) x
        # bench (cols); color = judge serving, linestyle = condition
        panel_benches, conds = [], []
        for s_ in EVAL_COLORS:
            b_, _, c_ = s_.partition(" ")
            c_ = c_ or "baseline"
            if b_ not in panel_benches:
                panel_benches.append(b_)
            if c_ not in conds:
                conds.append(c_)
        conds = [c_ for c_ in conds if c_ != "cot-native"]  # too busy on canvas
        cond_ls = dict(zip(conds, ["-", "--", ":", "-."] * 3))
        flags = _cot_flags(rows)
        TYPE_LABEL = {0: "both reasoning", 1: "stronger reasoning only",
                      2: "weaker reasoning only", 3: "neither reasoning"}

        def ptype(bench, major, minor):
            a, b = flags.get((bench, major)), flags.get((bench, minor))
            if a is None or b is None:
                return None
            return 0 if a and b else 1 if a else 2 if b else 3

        rs = [dict(r, score=(r["score_eci"] if key == "eci_gap" else r["score"]))
              for r in rows]
        by_bct = defaultdict(list)
        for r in rs:
            b_, _, c_ = r["eval"].partition(" ")
            if r["is_self"]:
                continue
            t = ptype(b_, r["major"], r["minor"])
            if t is not None:
                by_bct[(b_, c_ or "baseline", t)].append(r)
        types = sorted({t for (_, c_, t) in by_bct if c_ != "baseline"})
        if not types:
            types = [0]

        fig, axes = plt.subplots(2 * len(types), len(panel_benches),
                                 figsize=(3.2 * len(panel_benches),
                                          2.4 * len(types) + 0.6),
                                 sharex=True, squeeze=False,
                                 gridspec_kw={"height_ratios": [2, 1] * len(types)})
        seen_j, seen_c = [], []
        for ti, t in enumerate(types):
            for bi, bench in enumerate(panel_benches):
                top, bot = axes[2 * ti][bi], axes[2 * ti + 1][bi]
                style(top)
                style(bot)
                top.axhline(0.5, ls="--", c="#8a8a86", lw=1)
                top.set_ylim(0.05, 1.02)
                bot.axhline(0.0, ls="--", c="#8a8a86", lw=1)
                bot.set_ylim(-0.45, 0.45)
                base = {(r["judge"], r["major"], r["minor"]): r["score"]
                        for r in by_bct.get((bench, "baseline", t), [])
                        if r["score"] is not None}
                comp_pairs = {(r["major"], r["minor"])
                              for cc in conds if cc != "baseline"
                              for r in by_bct.get((bench, cc, t), [])}
                for c_ in conds:
                    for j in JUDGE_ORDER:
                        live = [r for r in by_bct.get((bench, c_, t), [])
                                if r["judge"] == j and r.get(key) is not None
                                and r["score"] is not None]
                        if c_ == "baseline" and comp_pairs:
                            live = [r for r in live
                                    if (r["major"], r["minor"]) in comp_pairs]
                        lbins = binned_deltas([(r[key], r["score"]) for r in live], key)
                        if not lbins:
                            continue
                        lx, lm, le = zip(*lbins)
                        top.errorbar(lx, lm, yerr=le, fmt="o", ls=cond_ls[c_],
                                     color=JUDGE_COLORS.get(j, "#888888"), ms=3,
                                     lw=1.4, elinewidth=1.0, capsize=2)
                        if j not in seen_j:
                            seen_j.append(j)
                        if c_ not in seen_c:
                            seen_c.append(c_)
                        if c_ == "baseline":
                            continue
                        jb = j.removesuffix("-yarn")  # e runs on long-context serving
                        deltas = [(r[key], r["score"] - base[(jb, r["major"], r["minor"])])
                                  for r in live if (jb, r["major"], r["minor"]) in base]
                        bins = binned_deltas(deltas, key)
                        if not bins:
                            continue
                        bx, bm, be = zip(*bins)
                        bot.errorbar(bx, bm, yerr=be, fmt="o", ls=cond_ls[c_],
                                     color=JUDGE_COLORS.get(j, "#888888"), ms=3.5,
                                     lw=1.1, elinewidth=1.0, capsize=2)
                if ti == 0:
                    top.set_title(bench, fontsize=8, color="#52514e", loc="left")
                if ti == len(types) - 1:
                    bot.set_xlabel(xlab)
            axes[2 * ti][0].set_ylabel(f"{TYPE_LABEL[t]}\njudge accuracy")
            axes[2 * ti + 1][0].set_ylabel("Δ")
        handles = [plt.Line2D([], [], color=JUDGE_COLORS.get(j, "#888888"), lw=1.6,
                              label=jlabel(models, j)
                              + (" · yarn" if j.endswith("-yarn") else ""))
                   for j in JUDGE_ORDER if j in seen_j]
        handles += [plt.Line2D([], [], color="#52514e", lw=1.4, ls=cond_ls[c_],
                               label=COND_LABEL.get(c_, c_)) for c_ in seen_c]
        fig.legend(handles=handles, frameon=False, fontsize=7, ncol=4,
                   loc="lower center", bbox_to_anchor=(0.5, 0.995))
        fig.tight_layout()
        save(fig, out, name)

        # by-eval: grid of pair reasoning-type (rows, accuracy+delta band each)
        # x bench (cols); type = which members carry real CoT in their logs
        flags = _cot_flags(rows)
        ccolors = dict(zip(conds, SERIES_PALETTE))
        TYPE_LABEL = {0: "both reasoning", 1: "stronger reasoning only",
                      2: "weaker reasoning only", 3: "neither reasoning"}

        def ptype(bench, major, minor):
            a, b = flags.get((bench, major)), flags.get((bench, minor))
            if a is None or b is None:
                return None
            return 0 if a and b else 1 if a else 2 if b else 3

        rs = [dict(r, score=(r["score_eci"] if key == "eci_gap" else r["score"]))
              for r in rows]
        by_bct = defaultdict(list)  # (bench, cond, type) -> rows
        for r in rs:
            b_, _, c_ = r["eval"].partition(" ")
            if (c_ or "baseline") not in conds or r["is_self"]:
                continue
            t = ptype(b_, r["major"], r["minor"])
            if t is not None:
                by_bct[(b_, c_ or "baseline", t)].append(r)
        types = sorted({t for (_, c_, t) in by_bct if c_ != "baseline"
                        and by_bct[_, c_, t]})
        if not types:
            types = [0]

        fig, axes = plt.subplots(2 * len(types), len(panel_benches),
                                 figsize=(3.2 * len(panel_benches),
                                          2.4 * len(types) + 0.6),
                                 sharex=True, squeeze=False,
                                 gridspec_kw={"height_ratios": [2, 1] * len(types)})
        seen_conds = []
        for ti, t in enumerate(types):
            for bi, bench in enumerate(panel_benches):
                top, bot = axes[2 * ti][bi], axes[2 * ti + 1][bi]
                style(top)
                style(bot)
                top.axhline(0.5, ls="--", c="#8a8a86", lw=1)
                top.set_ylim(0.05, 1.02)
                bot.axhline(0.0, ls="--", c="#8a8a86", lw=1)
                bot.set_ylim(-0.45, 0.45)
                base_rows_t = [r for r in by_bct.get((bench, "baseline", t), [])
                               if r["score"] is not None]
                base = {(r["judge"].removesuffix("-yarn"), r["major"], r["minor"]): r["score"]
                        for r in base_rows_t}
                comp_pairs = {(r["major"], r["minor"]) for c_ in conds
                              if c_ != "baseline"
                              for r in by_bct.get((bench, c_, t), [])}
                for c_ in conds:
                    live = [r for r in by_bct.get((bench, c_, t), [])
                            if r.get(key) is not None and r["score"] is not None]
                    if c_ == "baseline" and comp_pairs:
                        live = [r for r in live
                                if (r["major"], r["minor"]) in comp_pairs]
                    lbins = binned_deltas([(r[key], r["score"]) for r in live], key)
                    if not lbins:
                        continue
                    lx, lm, le = zip(*lbins)
                    top.errorbar(lx, lm, yerr=le, fmt="o-", color=ccolors[c_],
                                 ms=3.5, lw=1.4, elinewidth=1.0, capsize=2)
                    if c_ not in seen_conds:
                        seen_conds.append(c_)
                    if c_ == "baseline":
                        continue
                    deltas = [(r[key], r["score"]
                               - base[(r["judge"].removesuffix("-yarn"), r["major"], r["minor"])])
                              for r in live
                              if (r["judge"].removesuffix("-yarn"), r["major"], r["minor"]) in base]
                    dbins = binned_deltas(deltas, key)
                    if dbins:
                        dx, dm, de = zip(*dbins)
                        bot.errorbar(dx, dm, yerr=de, fmt="o-", color=ccolors[c_],
                                     ms=3, lw=1.1, elinewidth=1.0, capsize=2)
                if ti == 0:
                    top.set_title(bench, fontsize=9, color="#52514e", loc="left")
                if ti == len(types) - 1:
                    bot.set_xlabel(xlab)
            axes[2 * ti][0].set_ylabel(f"{TYPE_LABEL[t]}\njudge accuracy")
            axes[2 * ti + 1][0].set_ylabel("Δ")
        handles = [plt.Line2D([], [], color=ccolors[c_], lw=1.8,
                              label=COND_LABEL.get(c_, c_)) for c_ in seen_conds]
        fig.legend(handles=handles, frameon=False, fontsize=7, ncol=4,
                   loc="lower center", bbox_to_anchor=(0.5, 0.995))
        fig.tight_layout()
        save(fig, out, name + "_by_eval")


def fig_weak_judge(rows, models, out):
    rs = [dict(r, _series=r["eval"]) for r in rows if r["weak_judge"]]
    fig, ax = plt.subplots(figsize=(6.2, 3.9))
    _gap_curve(ax, rs, "gap", list(EVAL_COLORS), EVAL_COLORS, lambda e: e, GAP_LABEL)
    ax.set_title("Weak-judge subset (judge below both members) by eval — sigmoid fits, dot area ∝ items",
                 fontsize=10)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    save(fig, out, "2d_gap_curve_weak_judge_by_eval")


def fig_judge_pair_heatmap(rows, models, out, bench):
    rs = [r for r in rows if r["eval"] == bench and not r["is_self"] and r["score"] is not None]
    if not rs:
        return
    pairs = {}
    cache = _flags_cache()
    b_ = bench.partition(" ")[0]
    for r in rs:
        pairs.setdefault((r["major"], r["minor"]), r["gap"] or 0)
    def _tkey(p):
        t = _ptype_of(cache, b_, p[0], p[1])
        return 9 if t is None else t
    pair_order = sorted(pairs, key=lambda p: (_tkey(p), -pairs[p]))
    judges = [j for j in JUDGE_ORDER if any(r["judge"] == j for r in rs)] + \
             sorted({r["judge"] for r in rs} - set(JUDGE_ORDER))
    M = np.full((len(pair_order), len(judges)), np.nan)
    for r in rs:
        M[pair_order.index((r["major"], r["minor"])), judges.index(r["judge"])] = r["score"]
    fig, ax = plt.subplots(figsize=(6, 0.24 * len(pair_order) + 1.4))
    im = ax.imshow(M, cmap="RdBu_r", vmin=0, vmax=1, aspect="auto")
    lab = models.get(bench) or next(iter(models.values()))
    TYPE_TAG = {0: "[both] ", 1: "[s-cot] ", 2: "[w-cot] ", 3: "[plain] ", 9: ""}
    def plabel(p):
        gm = lab.get(p[0], {}); gn = lab.get(p[1], {})
        return f"{TYPE_TAG[_tkey(p)]}{gm.get('label', p[0])} v {gn.get('label', p[1])} · Δ{pairs[p]:.3g}"
    ax.set_yticks(range(len(pair_order)))
    ax.set_yticklabels([plabel(p) for p in pair_order], fontsize=7)
    ax.set_xticks(range(len(judges)))
    ax.set_xticklabels([jlabel(models, j) for j in judges], rotation=30, ha="right", fontsize=8)
    for i in range(len(pair_order)):
        for k in range(len(judges)):
            if not np.isnan(M[i, k]):
                ax.text(k, i, f"{M[i,k]:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if abs(M[i, k] - 0.5) > 0.3 else "#222")
    fig.colorbar(im, ax=ax, label="judge accuracy (0.5 = chance)", shrink=0.8)
    ax.set_title(f"Judge × pair accuracy — {bench} (rows sorted by capability gap; blank = not yet judged)",
                 fontsize=10)
    fig.tight_layout()
    save(fig, out, f"3_judge_strength_{bench}")


def _grid_meta(models):
    """shortname -> (eci, release in months, label), grid members only."""
    g = {}
    for mm in models.values():
        for m, meta in mm.items():
            if meta["arm"] == "A" and meta.get("eci") and meta.get("release"):
                y, mo = str(meta["release"]).split("-")[:2]
                g[m] = (meta["eci"], int(y) * 12 + int(mo), meta["label"])
    return g


def _pool_pairs(rows, grid):
    """Cross grid pairs oriented stronger-first, pooled over judges and evals.

    Returns (pairs, run_arrays): pairs maps (hi, lo) -> {cap, rel, score, n_runs,
    judges: {judge: (c, n)}}; run_arrays are per-run (score, cap, rel) for partial r.
    """
    pairs, runs = {}, []
    for r in rows:
        if (r["eval"] not in BENCHES or r["is_self"] or r["score"] is None
                or r["major"] not in grid or r["minor"] not in grid):
            continue
        hi, lo = sorted((r["major"], r["minor"]), key=lambda m: -grid[m][0])
        p = pairs.setdefault((hi, lo), {"cap": grid[hi][0] - grid[lo][0],
                                        "rel": grid[hi][1] - grid[lo][1],
                                        "s": [], "w": [], "judges": defaultdict(lambda: [0.0, 0.0])})
        # rows are oriented by GAP_AXIS; these figures are ECI-oriented, so
        # re-align score to the higher-ECI member when the orderings disagree
        sc = r["score"] if r["major"] == hi else 1 - r["score"]
        n = 2 * r["n_items"] * (1 - r["noanswer_rate"])
        p["s"].append(sc)
        p["w"].append(max(n, 1))
        p["judges"][r["judge"]][0] += sc * n
        p["judges"][r["judge"]][1] += n
        runs.append((r["score"], p["cap"], p["rel"]))
    for p in pairs.values():
        p["score"] = float(np.average(p["s"], weights=p["w"]))
        p["n_runs"] = len(p["s"])
    return pairs, np.array(runs)


def _resid(y, x):
    b, a = np.polyfit(x, y, 1)
    return y - (a + b * x)


PLANE_BINS = 4


def _plane_runs(rows, evals):
    """(score, cap gap, signed release gap, verdict weight) per cross run."""
    return np.array([(r["score"], r["gap"], r["rel_gap_months"],
                      2 * r["n_items"] * (1 - r["noanswer_rate"]))
                     for r in rows if r["eval"] in evals and not r["is_self"]
                     and r["score"] is not None and r["gap"] is not None
                     and r["rel_gap_months"] is not None])


def _binned_panel(ax, x, y, s, w, xl, yl, title, note):
    def edges(v):
        lo, hi = float(v.min()), float(v.max())
        if lo == hi:
            lo, hi = lo - 0.5, hi + 0.5
        return np.linspace(lo, hi, PLANE_BINS + 1)
    xe, ye = edges(x), edges(y)
    num, _, _ = np.histogram2d(x, y, [xe, ye], weights=s * w)
    den, _, _ = np.histogram2d(x, y, [xe, ye], weights=w)
    M = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
    ax.pcolormesh(xe, ye, np.ma.masked_invalid(M.T), cmap="RdBu_r", vmin=0, vmax=1,
                  edgecolors="#ddd", linewidth=0.6)
    for i in range(PLANE_BINS):
        for k in range(PLANE_BINS):
            if den[i, k] > 0:
                ax.text((xe[i] + xe[i + 1]) / 2, (ye[k] + ye[k + 1]) / 2,
                        f"{M[i, k]:.2f}\n{den[i, k]:.0f}v", ha="center", va="center",
                        fontsize=6.5, color="white" if abs(M[i, k] - 0.5) > 0.3 else "#222")
    ax.set_xticks(xe)
    ax.set_yticks(ye)
    ax.set_xticklabels([f"{v:.3g}" for v in xe], fontsize=7)
    ax.set_yticklabels([f"{v:.3g}" for v in ye], fontsize=7)
    ax.set_xlabel(xl, fontsize=8)
    ax.set_ylabel(yl, fontsize=8)

    # marginals: pooled accuracy against each axis alone
    top = ax.inset_axes([0, 1.05, 1, 0.36])
    side = ax.inset_axes([1.05, 0, 0.36, 1])
    for m, eaxis, vals, vertical, mc in ((top, xe, x, True, "#2a78d6"),
                                         (side, ye, y, False, "#eb6834")):
        m.grid(True, alpha=0.25, linewidth=0.4)
        m.set_axisbelow(True)
        for sp in ("top", "right"):
            m.spines[sp].set_visible(False)
        pts = []
        for lo, hi in zip(eaxis[:-1], eaxis[1:]):
            sel = (vals >= lo) & ((vals < hi) if hi < eaxis[-1] else (vals <= hi))
            if sel.any():
                c, n = float((s[sel] * w[sel]).sum()), float(w[sel].sum())
                pts.append(((lo + hi) / 2, c / n, *wilson(c, n)))
        if vertical:
            m.axhline(0.5, ls="--", c="#8a8a86", lw=0.8)
            m.plot([p[0] for p in pts], [p[1] for p in pts], c=mc, lw=2.4, marker="o", ms=5)
            for cx, a, lo, hi in pts:
                m.plot([cx, cx], [lo, hi], c=mc, lw=1.4)
            m.set_xlim(xe[0], xe[-1])
            m.set_ylim(0.15, 0.85)
            m.set_xticks([])
            m.set_yticks([0.3, 0.5, 0.7])
            m.set_ylabel("acc", fontsize=7)
        else:
            m.axvline(0.5, ls="--", c="#8a8a86", lw=0.8)
            m.plot([p[1] for p in pts], [p[0] for p in pts], c=mc, lw=2.4, marker="o", ms=5)
            for cy, a, lo, hi in pts:
                m.plot([lo, hi], [cy, cy], c=mc, lw=1.4)
            m.set_ylim(ye[0], ye[-1])
            m.set_xlim(0.15, 0.85)
            m.set_yticks([])
            m.set_xticks([0.3, 0.5, 0.7])
            m.set_xlabel("acc", fontsize=7)
        m.tick_params(labelsize=6.5)
    top.set_title(f"{title}\n{note}", fontsize=8.5, loc="left", color="#333")


def _plane_row(axs, runs, row_label):
    s, cap, rel, w = runs[:, 0], runs[:, 1], runs[:, 2], runs[:, 3]
    from scipy.stats import pearsonr
    r_cr = pearsonr(cap, rel)[0]
    pr_rel = pearsonr(_resid(s, cap), _resid(rel, cap))[0]
    pr_cap = pearsonr(_resid(s, rel), _resid(cap, rel))[0]
    _binned_panel(axs[0], cap, rel, s, w, GAP_LABEL,
                  "release gap (months, + = stronger newer)",
                  f"{row_label} · raw axes", f"axes r = {r_cr:+.2f}")
    _binned_panel(axs[1], cap, _resid(rel, cap), s, w, GAP_LABEL,
                  "release gap | capability (resid, months)",
                  "release gap, capability regressed out",
                  f"partial r(acc, release | cap) = {pr_rel:+.2f}")
    _binned_panel(axs[2], _resid(cap, rel), rel, s, w,
                  f"{GAP_LABEL} | release (resid)",
                  "release gap (months, + = stronger newer)",
                  "capability gap, release regressed out",
                  f"partial r(acc, cap | release) = {pr_cap:+.2f}")


def _plane_colorbar(fig, x=0.955):
    sm = plt.cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap="RdBu_r")
    cax = fig.add_axes([x, 0.15, 0.011, 0.55])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("P(judge picks stronger member); 0.5 = chance", fontsize=8)
    cb.ax.axhline(0.5, color="#222", lw=0.8)
    cb.ax.tick_params(labelsize=7)


def fig_cap_release_plane(rows, models, out):
    """Binned capability-gap x release-gap grid with marginal accuracy strips,
    judges pooled; cell = weighted mean P(pick stronger), verdict counts shown.
    Raw axes only in the main figures (the member grid was designed to
    decorrelate the axes); residualized panels go to a reference figure."""
    from scipy.stats import pearsonr
    runs = _plane_runs(rows, BENCHES)
    y_lab = "release gap (months, + = stronger newer)"

    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    fig.subplots_adjust(left=0.10, right=0.66, top=0.60, bottom=0.09)
    r_cr = pearsonr(runs[:, 1], runs[:, 2])[0]
    _binned_panel(ax, runs[:, 1], runs[:, 2], runs[:, 0], runs[:, 3],
                  GAP_LABEL, y_lab, "all evals", f"axes r = {r_cr:+.2f}")
    _plane_colorbar(fig, x=0.92)
    fig.suptitle("Judge accuracy over the capability-gap / release-gap plane — judges pooled;\n"
                 "epoch-import pairs excluded (no release dates); strips: accuracy vs each axis alone",
                 fontsize=10, y=0.97)
    save(fig, out, "heatmap_cap_vs_release")

    evals = [b for b in BENCHES if len(_plane_runs(rows, [b]))]
    if len(evals) > 1:
        fig, axs = plt.subplots(1, len(evals), figsize=(4.4 * len(evals), 4.4), squeeze=False)
        fig.subplots_adjust(left=0.07, right=0.82, top=0.60, bottom=0.11, wspace=0.55)
        for ax, b in zip(axs[0], evals):
            rb = _plane_runs(rows, [b])
            rr = pearsonr(rb[:, 1], rb[:, 2])[0]
            _binned_panel(ax, rb[:, 1], rb[:, 2], rb[:, 0], rb[:, 3],
                          GAP_LABEL, y_lab, b, f"axes r = {rr:+.2f}")
        _plane_colorbar(fig, x=0.945)
        fig.suptitle("Capability-gap / release-gap plane per eval — judges pooled;"
                     " strips: accuracy vs each axis alone", fontsize=10, y=0.97)
        save(fig, out, "heatmap_cap_vs_release_by_eval")

    # reference: residualized panels (matters when pooled datasets re-correlate the axes)
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.8))
    fig.subplots_adjust(left=0.05, right=0.88, top=0.62, bottom=0.12, wspace=0.62)
    _plane_row(axs, runs, "all evals")
    _plane_colorbar(fig, x=0.965)
    fig.suptitle("Reference: residualized axes — all evals + judges pooled", fontsize=10, y=0.98)
    save(fig, out, "heatmap_cap_vs_release_residuals")


def _design_grid():
    """display label -> (half, tier) from the model-set design grid."""
    g = {}
    for r in csv.DictReader(open(MODEL_SET_CSV)):
        for tier, label in r.items():
            if tier != "half" and label:
                g[label] = (r["half"], tier)
    return g


def fig_control_pairs(rows, models, out):
    """The design's explicit controls: same-half pairs isolate capability gap,
    same-tier pairs isolate release gap."""
    grid = _grid_meta(models)
    design = _design_grid()
    pairs, _ = _pool_pairs(rows, grid)
    blocks = {"same half": [], "same tier": []}
    for (hi, lo), p in pairs.items():
        try:
            (ha, ta), (hb, tb) = design[grid[hi][2]], design[grid[lo][2]]
        except KeyError:
            continue  # members outside the collection design grid (epoch imports)
        if ha == hb:
            blocks["same half"].append(((hi, lo), p))
        elif ta == tb:
            blocks["same tier"].append(((hi, lo), p))
    blocks["same half"].sort(key=lambda kp: kp[1]["cap"])
    blocks["same tier"].sort(key=lambda kp: kp[1]["rel"])

    heights = [max(len(b), 1) for b in blocks.values()]
    fig, axs = plt.subplots(2, 1, figsize=(6.5, 0.4 * sum(heights) + 2.0), sharex=True,
                            gridspec_kw={"height_ratios": heights})
    titles = {"same half": "capability gap, release matched (same half-year in design grid)",
              "same tier": "release gap, capability matched (same tier in design grid)"}
    for ax, (kind, block) in zip(axs, blocks.items()):
        style(ax)
        ax.axvline(0.5, ls="--", c="#8a8a86", lw=1)
        labels = []
        for y, ((hi, lo), p) in enumerate(block):
            c = sum(v[0] for v in p["judges"].values())
            n = sum(v[1] for v in p["judges"].values())
            wlo, whi = wilson(c, n)
            ax.plot([wlo, whi], [y, y], color="#222", lw=1.6, zorder=4)
            ax.scatter([c / n], [y], marker="D", color="#222", s=50, zorder=5)
            for j, (jc, jn) in p["judges"].items():
                ax.scatter([jc / jn], [y], color=JUDGE_COLORS.get(j, "#999"), s=16,
                           alpha=0.8, zorder=3)
            labels.append(f"{grid[hi][2]} v {grid[lo][2]}  ·  ΔECI {p['cap']:.1f}, Δ{p['rel']:.0f}mo")
        ax.set_yticks(range(len(block)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_ylim(-0.6, len(block) - 0.4)
        ax.set_title(titles[kind], fontsize=9, loc="left", color="#333")
    axs[1].set_xlabel("judge accuracy")
    axs[1].set_xlim(0.15, 1.0)
    h = [plt.Line2D([], [], marker="D", ls="-", c="#222", label="all judges pooled (95% Wilson CI)")] + \
        [plt.Line2D([], [], marker="o", ls="", c=JUDGE_COLORS[j], label=jlabel(models, j))
         for j in JUDGE_ORDER]
    fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, fontsize=7)
    fig.suptitle("Design-grid control pairs — evals pooled; stronger member listed first", fontsize=10)
    fig.tight_layout(rect=[0, 0.09, 1, 0.96])
    save(fig, out, "2f_control_pairs")



TYPE_LABEL = {0: "both reasoning", 1: "stronger reasoning only",
              2: "weaker reasoning only", 3: "neither reasoning"}


def _flags_cache():
    import json as _json
    cache_p = Path("judge_results/judging_variations/cot_flags.json")
    return _json.load(open(cache_p)) if cache_p.exists() else {}


def _ptype_of(cache, bench, major, minor):
    a, b = cache.get(f"{bench}|{major}"), cache.get(f"{bench}|{minor}")
    if a is None or b is None:
        return None
    return 0 if a and b else 1 if a else 2 if b else 3


def _bench_conds():
    benches, conds = [], []
    for s_ in EVAL_COLORS:
        b_, _, c_ = s_.partition(" ")
        if b_ not in benches:
            benches.append(b_)
        c_ = c_ or "baseline"
        if c_ not in conds:
            conds.append(c_)
    return benches, conds


def _dot_grid(grp, models, out, name, xlab, xlim, chance):
    """grp: (type, bench, cond, judge) -> (p, lo, hi). Rows=types, cols=benches,
    judges on y, one cond-colored dot per condition."""
    types = sorted({k[0] for k in grp})
    benches = sorted({k[1] for k in grp})
    conds = list(dict.fromkeys(k[2] for k in sorted(grp)))
    judges = [j for j in JUDGE_ORDER if not j.endswith("-yarn")
              and any(k[3] == j for k in grp)]
    if not types or not judges:
        return
    ccolors = dict(zip(conds, SERIES_PALETTE))
    fig, axes = plt.subplots(len(types), len(benches),
                             figsize=(2.8 * len(benches), 0.55 * len(judges) * len(types) + 1.2),
                             sharex=True, squeeze=False)
    for ti, t in enumerate(types):
        for bi, bench in enumerate(benches):
            ax = axes[ti][bi]
            style(ax)
            ax.axvline(chance, ls="--", c="#8a8a86", lw=1)
            for ji, j in enumerate(judges):
                for k, c_ in enumerate(conds):
                    v = grp.get((t, bench, c_, j))
                    if not v:
                        continue
                    y = ji + (k - (len(conds) - 1) / 2) * 0.18
                    ax.plot([v[1], v[2]], [y, y], color=ccolors[c_], lw=1.6)
                    ax.scatter([v[0]], [y], color=ccolors[c_], s=26, zorder=5)
            ax.set_yticks(range(len(judges)))
            ax.set_yticklabels(judges if bi == 0 else [], fontsize=7)
            ax.set_xlim(*xlim)
            if ti == 0:
                ax.set_title(bench, fontsize=9, color="#52514e", loc="left")
            if bi == 0:
                ax.text(-0.34, 0.5, TYPE_LABEL.get(t, ""), transform=ax.transAxes,
                        rotation=90, va="center", fontsize=7, color="#52514e")
            if ti == len(types) - 1:
                ax.set_xlabel(xlab, fontsize=8)
    handles = [plt.Line2D([], [], marker="o", ls="", color=ccolors[c_], label=c_)
               for c_ in conds]
    fig.legend(handles=handles, frameon=False, fontsize=7, ncol=len(conds),
               loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    save(fig, out, name)


def fig_position_bias(rows, models, out):
    cache = _flags_cache()
    raw = defaultdict(list)
    for r in rows:
        if r["slot_a"] is None or r["is_self"]:
            continue
        b_, _, c_ = r["eval"].partition(" ")
        t = _ptype_of(cache, b_, r["major"], r["minor"])
        if t is None:
            continue
        raw[(t, b_, c_ or "baseline", r["judge"].removesuffix("-yarn"))].append(
            (r["slot_a"], max(r["n_items"], 1)))
    grp = {}
    for k, vals in raw.items():
        n = sum(w for _, w in vals) * 2
        p = float(np.average([v for v, _ in vals], weights=[w for _, w in vals]))
        lo, hi = wilson(p * n, n)
        grp[k] = (p, lo, hi)
    _dot_grid(grp, models, out, "9_position_bias_by_eval",
              "P(picked slot 1)", (0.15, 0.85), 0.5)


def _pick_longer_rate(bench):
    """(judge, pairname or "") -> [picked-longer, n]; nested lengths.json gives
    per-pair keys (typed figures), flat gives pair "" (legacy)."""
    per_judge = defaultdict(lambda: [0, 0])
    for d in DIRS[bench]:
        lj = d / "lengths.json"
        if lj.exists():
            for j, v in json.load(open(lj)).items():
                if isinstance(v, dict):
                    for pair, (c, n) in v.items():
                        per_judge[(j, pair)][0] += c
                        per_judge[(j, pair)][1] += n
                else:
                    per_judge[(j, "")][0] += v[0]
                    per_judge[(j, "")][1] += v[1]
            continue
        trans = {Path(f).stem: json.load(open(f)) for f in glob.glob(str(d / "transcripts/*.json"))
                 if "__ep1" not in f}
        for pf in glob.glob(str(d / "pairs/*__full.json")):
            major, minor, _ = Path(pf).stem.split("__")
            if major == minor or major in EXCLUDE or minor in EXCLUDE:
                continue
            for item, judges in json.load(open(pf)).items():
                for j, cell in judges.items():
                    for o in cell["orders"]:
                        if not o or o["pick"] not in ("1", "2"):
                            continue
                        lm, ln = len(trans[major].get(item, "")), len(trans[minor].get(item, ""))
                        if lm == ln:
                            continue
                        picked_len = lm if o["pick"] == o["slot_of_major"] else ln
                        other = ln if o["pick"] == o["slot_of_major"] else lm
                        per_judge[(j, f"{major}-vs-{minor}")][1] += 1
                        per_judge[(j, f"{major}-vs-{minor}")][0] += picked_len > other
    return per_judge


def fig_length_bias(models, out):
    cache = _flags_cache()
    grp = {}
    raw = defaultdict(lambda: [0, 0])
    for series in BENCHES:
        b_, _, c_ = series.partition(" ")
        for (j, pair), (c, n) in _pick_longer_rate(series).items():
            major, _, minor = pair.partition("-vs-")
            t = _ptype_of(cache, b_, major, minor) if pair else None
            if t is None:
                continue
            raw[(t, b_, c_ or "baseline", j.removesuffix("-yarn"))][0] += c
            raw[(t, b_, c_ or "baseline", j.removesuffix("-yarn"))][1] += n
    for k, (c, n) in raw.items():
        if n >= 30:
            p, lo, hi = c / n, *wilson(c, n)
            grp[k] = (p, lo, hi)
    _dot_grid(grp, models, out, "10_length_bias_by_eval",
              "P(picked the longer transcript)", (0.15, 0.85), 0.5)


def fig_capability_vs_length(models, out):
    fig, axes = plt.subplots(1, len(BENCHES), figsize=(3.0 * len(BENCHES), 3.1))
    axes = np.atleast_1d(axes)
    for ax, b in zip(axes, BENCHES):
        style(ax)
        lens, metas = defaultdict(list), {}
        for d in DIRS[b]:
            if not (d / "index.json").exists():
                continue
            for m, meta in json.load(open(d / "index.json"))["models"].items():
                f = d / f"transcripts/{m}.json"
                if meta.get(GAP_AXIS) is None or not f.exists():
                    continue
                tr = json.load(open(f))
                if tr:
                    lens[m].extend(len(t) for t in tr.values())
                    metas.setdefault(m, meta)
        for m, L in lens.items():
            meta = metas[m]
            cap = meta[GAP_AXIS]
            med = float(np.median(L))
            arm_b = meta["arm"] == "B"
            ax.scatter([cap], [med], s=60, color="#eda100" if arm_b else "#2a78d6",
                       marker="D" if arm_b else "o", edgecolors="white", zorder=5)
            ax.annotate(meta["label"], (cap, med), textcoords="offset points",
                        xytext=(4, 4), fontsize=6.5, color="#444")
        ax.set_xlabel(GAP_AXIS)
        ax.set_yscale("log")
        ax.set_title(b, fontsize=10)
    axes[0].set_ylabel("median judged-transcript length (chars, log)")
    fig.suptitle("Capability vs shown work — judged (both-correct, final-text) transcripts;"
                 " blue = grid, yellow ◇ = epoch imports", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, out, "11_capability_vs_length_by_eval")


def fig_item_difficulty(models, out):
    fig, axes = plt.subplots(1, len(BENCHES), figsize=(3.0 * len(BENCHES), 3.1), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, b in zip(axes, BENCHES):
        style(ax)
        acc = defaultdict(lambda: [0, 0])
        for d in DIRS[b]:
            if not (d / "index.json").exists():
                continue
            meta = json.load(open(d / "index.json"))["models"]
            grid = [m for m, mm in meta.items() if mm["arm"] == "A"]
            solved = defaultdict(int)
            for m in grid:
                f = d / f"transcripts/{m}.json"
                if f.exists():
                    for item in json.load(open(f)):
                        solved[item] += 1
            for pf in glob.glob(str(d / "pairs/*__full.json")):
                major, minor, _ = Path(pf).stem.split("__")
                if (major == minor or major not in grid or minor not in grid
                        or major in EXCLUDE or minor in EXCLUDE):
                    continue
                ca, cb = meta[major].get(GAP_AXIS), meta[minor].get(GAP_AXIS)
                if ca is None or cb is None:
                    continue
                flip = ca < cb  # picked_major tracks the first-named, not the stronger
                for item, judges in json.load(open(pf)).items():
                    k = solved.get(item, 0)
                    for j, cell in judges.items():
                        for o in cell["orders"]:
                            if o and o["picked_major"] is not None:
                                acc[k][1] += 1
                                acc[k][0] += (1 - o["picked_major"]) if flip else o["picked_major"]
        ks = sorted(acc)
        ax.axhline(0.5, ls="--", c="#8a8a86", lw=1)
        for k in ks:
            c, n = acc[k]
            lo, hi = wilson(c, n)
            ax.errorbar(k, c / n, yerr=[[c / n - lo], [hi - c / n]], marker="o",
                        color=EVAL_COLORS[b], capsize=3, lw=1.5, ms=6)
            ax.annotate(f"n={n}", (k, c / n), textcoords="offset points", xytext=(0, -14),
                        fontsize=6.5, color="#666", ha="center")
        ax.set_xlabel("item difficulty (# grid models solving it)")
        ax.set_title(b, fontsize=10)
    axes[0].set_ylabel("judge accuracy")
    fig.suptitle("Judge accuracy vs item difficulty — easy items (solved by many) vs hard, all judges pooled",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, out, "5_item_difficulty_by_eval")


def fig_fill_plan(rows, models, out):
    if "gpqa" not in models:
        return
    judges = JUDGE_ORDER + ["gpt-4.1", "o4-mini", "claude-sonnet-4.5"]
    members = [m for m in models["gpqa"] if models["gpqa"][m]["arm"] == "A"] + ["qwen3-32b"]
    done = defaultdict(set)
    for r in rows:
        if r["eval"] in BENCHES:
            done[(r["judge"], frozenset([r["major"], r["minor"]]))].add(r["eval"])
    pairs = sorted({frozenset([a, b]) for a in members for b in members if a != b}, key=str)
    M = np.zeros((len(pairs), len(judges)))
    for (j, p), evs in done.items():
        if j in judges and p in pairs:
            M[pairs.index(p), judges.index(j)] = len(evs)
    fig, ax = plt.subplots(figsize=(6.5, 0.22 * len(pairs) + 1.6))
    im = ax.imshow(M, cmap="Greens", vmin=0, vmax=len(BENCHES), aspect="auto")
    ax.set_xticks(range(len(judges)))
    ax.set_xticklabels(judges, rotation=30, ha="right", fontsize=7)
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels([" v ".join(sorted(p)) for p in pairs], fontsize=5.5)
    fig.colorbar(im, ax=ax, label=f"# evals judged (of {len(BENCHES)})", shrink=0.7)
    ax.set_title("Coverage: evals judged per (judge, pair) cell", fontsize=10)
    fig.tight_layout()
    save(fig, out, "heatmap_fill_plan")


def main():
    global BENCHES, DIRS, GAP_AXIS, GAP_LABEL
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--datasets", nargs="+", default=EXPANSION_DATASETS,
                    help="viewer dataset dirs under interactive_dashboards/jv/data")
    ap.add_argument("--gap", choices=["eci", "acc"], default="eci",
                    help="capability measure for the gap axis and pair orientation")
    ap.add_argument("--exclude-members", nargs="*", default=[],
                    help="drop every pair containing these members")
    args = ap.parse_args()
    EXCLUDE.update(args.exclude_members)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    BENCHES = list(dict.fromkeys(series_name(d) for d in args.datasets))
    EVAL_COLORS.update({b: SERIES_PALETTE[i % len(SERIES_PALETTE)] for i, b in enumerate(BENCHES)})
    DIRS = {s: [] for s in BENCHES}
    for d in args.datasets:
        DIRS[series_name(d)].append(DATA / d)
    if args.gap == "acc":
        GAP_AXIS, GAP_LABEL = "accuracy", "capability gap (accuracy diff, per bench)"
    rows, models = load()
    print(f"{len(rows)} runs loaded; writing to {args.out_dir}")
    try:
        fig_model_accuracy(models, args.out_dir)
    except Exception as e:
        print(f"SKIPPED fig_model_accuracy: {type(e).__name__}: {e}")
    try:
        fig_gap_curves(rows, models, args.out_dir)
    except Exception as e:
        print(f"SKIPPED fig_gap_curves: {type(e).__name__}: {e}")
    try:
        fig_weak_judge(rows, models, args.out_dir)
    except Exception as e:
        print(f"SKIPPED fig_weak_judge: {type(e).__name__}: {e}")
    try:
        fig_cap_release_plane(rows, models, args.out_dir)
    except Exception as e:
        print(f"SKIPPED fig_cap_release_plane: {type(e).__name__}: {e}")
    try:
        fig_control_pairs(rows, models, args.out_dir)
    except Exception as e:
        print(f"SKIPPED fig_control_pairs: {type(e).__name__}: {e}")
    for b in BENCHES:
        fig_judge_pair_heatmap(rows, models, args.out_dir, b)
    try:
        fig_position_bias(rows, models, args.out_dir)
    except Exception as e:
        print(f"SKIPPED fig_position_bias: {type(e).__name__}: {e}")
    try:
        fig_length_bias(models, args.out_dir)
    except Exception as e:
        print(f"SKIPPED fig_length_bias: {type(e).__name__}: {e}")
    try:
        fig_capability_vs_length(models, args.out_dir)
    except Exception as e:
        print(f"SKIPPED fig_capability_vs_length: {type(e).__name__}: {e}")
    try:
        fig_item_difficulty(models, args.out_dir)
    except Exception as e:
        print(f"SKIPPED fig_item_difficulty: {type(e).__name__}: {e}")
    try:
        fig_fill_plan(rows, models, args.out_dir)
    except Exception as e:
        print(f"SKIPPED fig_fill_plan: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
