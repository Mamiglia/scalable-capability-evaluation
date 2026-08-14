"""Figure suite over interactive_viewer exports.

Default: the data-expansion run with ECI gap axis (the run's design variable).
--datasets extends to any viewer datasets (e.g. gpqa-pilot-rd2); pilot models
have no ECI, so combined runs use --gap acc (per-bench accuracy diff).
Exported scores are P(pick first-named); load() reorients every pair so score
= P(pick the gap-axis-stronger member). Truncation runs are skipped.
--exclude-members drops answer models by name fragment (the deepseek family has
a terse-answer artefact, so its pairs get their own figure set).

Usage:
  uv run python scripts/judge/figures_expansion.py --out-dir judge_results/multi-eval-pilot/figures
  uv run python scripts/judge/figures_expansion.py --out-dir judge_results/combined/figures \
      --datasets data-expansion-gpqa data-expansion-aime2025 data-expansion-aime2026 gpqa-pilot-rd2 \
      --gap acc
  ... --out-dir judge_results/multi-eval-pilot/figures-no-deepseek --exclude-members deepseek
"""

import argparse
import csv
import glob
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import beta as beta_dist

DATA = Path("interactive_viewer/data")
MODEL_SET_CSV = Path("configs/comparisons/model_set.csv")
EXPANSION_DATASETS = ["data-expansion-gpqa", "data-expansion-aime2025", "data-expansion-aime2026"]
EVAL_COLORS = {"gpqa": "#2a78d6", "aime": "#eb6834", "aime (reasoning shown)": "#1baf7a"}

# set in main from --datasets / --gap
BENCHES = []       # series names, in dataset order (same bench pools across datasets)
DIRS = {}          # series -> list of viewer dataset dirs
GAP_AXIS = "eci"   # model-meta key the gap and pair orientation use
GAP_LABEL = "capability gap (ECI)"
EXCLUDE = set()    # lowercased name fragments; matching answer models are dropped
                   # (they can still judge — the artefact is in their transcripts)


def excluded(member):
    """True if the member name contains any --exclude-members fragment."""
    return any(frag in member.lower() for frag in EXCLUDE)


def series_name(dataset):
    if dataset == "gpqa-pilot-rd2":
        return "gpqa"
    s = dataset.replace("data-expansion-", "")
    if s.startswith("aime"):
        return "aime (reasoning shown)" if s.endswith("-ablation") else "aime"
    return s
JUDGE_ORDER = ["qwen3-8b", "qwen3-32b", "qwen3-vl-30b-a3b-instruct",
               "ling-2.6-flash", "glm-4.6", "deepseek-v4-flash-0731", "gemini-3.1-flash-lite"]
JUDGE_COLORS = dict(zip(JUDGE_ORDER, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                                      "#e87ba4", "#008300", "#8250c4"]))


def load():
    """rows: one per (eval, judge, pair run); models: per-eval metadata."""
    rows, models, dropped, kept = [], {}, set(), set()
    for s in BENCHES:
      for d in DIRS[s]:
        if not (d / "index.json").exists():
            continue
        idx = json.load(open(d / "index.json"))
        # excluded models stay in models{} so jlabel() can still caption them as
        # judges; the figures that plot members skip them
        for m, meta in idx["models"].items():
            (dropped if excluded(m) else kept).add(m)
            models.setdefault(s, {}).setdefault(m, meta)
        for r in idx["runs"]:
            if r.get("frac", "full") != "full":
                continue
            if excluded(r["major"]) or excluded(r["minor"]):
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
    print(f"excluded members: {', '.join(sorted(dropped)) or 'none'}")
    print(f"{len(kept)} members kept")
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


def style(ax):
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def save(fig, out, name):
    fig.savefig(out / f"{name}.png", dpi=150)
    fig.savefig(out / f"{name}.svg")
    plt.close(fig)
    print(" ", name)


JUDGE_ONLY = {"qwen3-32b": ("Qwen3 32B", 130.0)}  # off-grid judge, absent from member tables


def jlabel(models, j):
    if j in JUDGE_ONLY:
        return f"{JUDGE_ONLY[j][0]} ({JUDGE_ONLY[j][1]:.0f})"
    m = models.get("gpqa", {}).get(j) or next((mm[j] for mm in models.values() if j in mm), {})
    e = m.get("eci")
    return f"{m.get('label', j)} ({e:.0f})" if e else m.get("label", j)


# --- figures -------------------------------------------------------------


def fig_model_accuracy(models, out):
    grid, labels = [m for m in JUDGE_ORDER
                    if m in models.get("gpqa", {}) and not excluded(m)], {}
    for s in BENCHES:
        for m, meta in models.get(s, {}).items():
            if meta["arm"] == "A" and m not in grid and not excluded(m):
                grid.append(m)
            labels.setdefault(m, meta["label"])
    fig, ax = plt.subplots(figsize=(1.1 * len(grid) + 3, 4.5))
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


def _gap_curve(ax, rows, key, series, colors, labelfn, xlab, fit=True):
    style(ax)
    ax.axhline(0.5, ls="--", c="#8a8a86", lw=1)
    for s in series:
        rs = [r for r in rows if r["_series"] == s and not r["is_self"] and r.get(key) is not None
              and r["score"] is not None]
        if not rs:
            continue
        xs = np.array([r[key] for r in rs], float)
        ys = np.array([r["score"] for r in rs], float)
        ns = np.array([max(r["n_items"], 1) for r in rs], float)
        ax.scatter(xs, ys, s=8 + ns / 4, color=colors[s], alpha=0.55,
                   edgecolors="white", linewidths=0.5, label=labelfn(s))
        if fit and len(rs) >= 4:
            f = fit_sigmoid(xs, ys, ns)
            if f:
                gx = np.linspace(0, max(xs) * 1.02, 100)
                ax.plot(gx, sigmoid(gx, *f), color=colors[s], lw=1.8)
    ax.set_xlabel(xlab)
    ax.set_ylabel("judge accuracy")
    ax.set_ylim(0.05, 1.02)


def fig_gap_curves(rows, models, out):
    # 2a by judge (pooled evals), 2a-eval by eval; same pair by release gap (2b)
    specs = [("2a_gap_curve_by_capability", "gap", GAP_LABEL),
             ("2b_gap_curve_by_release", "rel_gap_months",
              "release gap (months, + = stronger member newer)")]
    if GAP_AXIS != "eci":
        specs.append(("2a_gap_curve_by_capability_eci", "eci_gap", "capability gap (ECI)"))
    ecis = {}
    for j in JUDGE_ORDER:
        if j in JUDGE_ONLY:
            ecis[j] = JUDGE_ONLY[j][1]
        else:
            m = models.get("gpqa", {}).get(j) or next((mm[j] for mm in models.values() if j in mm), {})
            ecis[j] = m.get("eci")
    ranked = sorted((j for j, e in ecis.items() if e), key=lambda j: ecis[j])
    jcolors = {j: plt.cm.viridis(i / max(len(ranked) - 1, 1)) for i, j in enumerate(ranked)}
    jcolors |= {j: "#999" for j, e in ecis.items() if not e}
    for name, key, xlab in specs:
        for split, series, colors, labelfn, suffix in [
            ("judge", JUDGE_ORDER, jcolors, lambda j: jlabel(models, j), ""),
            ("eval", list(EVAL_COLORS), EVAL_COLORS, lambda e: e, "_by_eval"),
        ]:
            rs = [dict(r, _series=(r["judge"] if split == "judge" else r["eval"]),
                       score=(r["score_eci"] if key == "eci_gap" else r["score"])) for r in rows]
            fig, ax = plt.subplots(figsize=(9, 5.5))
            _gap_curve(ax, rs, key, series, colors, labelfn, xlab)
            # self-pair strip at x=0 shown as open markers
            selfs = [r for r in rs if r["is_self"] and r["score"] is not None]
            for r in selfs:
                c = (jcolors if split == "judge" else EVAL_COLORS).get(r["_series"], "#999")
                ax.scatter([0], [r["score"]], marker="o", facecolors="none",
                           edgecolors=c, s=30, linewidths=1.2, zorder=5)
            ax.legend(frameon=False, fontsize=8, loc="lower right", ncol=2)
            split_lab = "judge" if split == "judge" else "eval"
            ax.set_title(f"Judge accuracy vs gap, colored by {split_lab} — lines: sigmoid fits ·"
                         " open circles at 0: self-pairs · dot area ∝ items", fontsize=10)
            fig.tight_layout()
            save(fig, out, name + suffix)


def fig_weak_judge(rows, models, out):
    rs = [dict(r, _series=r["eval"]) for r in rows if r["weak_judge"]]
    fig, ax = plt.subplots(figsize=(9, 5.5))
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
    for r in rs:
        pairs.setdefault((r["major"], r["minor"]), r["gap"] or 0)
    pair_order = sorted(pairs, key=lambda p: -pairs[p])
    judges = [j for j in JUDGE_ORDER if any(r["judge"] == j for r in rs)] + \
             sorted({r["judge"] for r in rs} - set(JUDGE_ORDER))
    M = np.full((len(pair_order), len(judges)), np.nan)
    for r in rs:
        M[pair_order.index((r["major"], r["minor"])), judges.index(r["judge"])] = r["score"]
    fig, ax = plt.subplots(figsize=(10, 0.42 * len(pair_order) + 2.2))
    im = ax.imshow(M, cmap="RdBu_r", vmin=0, vmax=1, aspect="auto")
    lab = models.get(bench) or next(iter(models.values()))
    def plabel(p):
        gm = lab.get(p[0], {}); gn = lab.get(p[1], {})
        return f"{gm.get('label', p[0])} v {gn.get('label', p[1])} · Δ{pairs[p]:.3g}"
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

    fig, ax = plt.subplots(figsize=(8.2, 6.8))
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
        fig, axs = plt.subplots(1, len(evals), figsize=(6.6 * len(evals), 6.4), squeeze=False)
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
    fig, axs = plt.subplots(1, 3, figsize=(15.5, 5.4))
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
        (ha, ta), (hb, tb) = design[grid[hi][2]], design[grid[lo][2]]
        if ha == hb:
            blocks["same half"].append(((hi, lo), p))
        elif ta == tb:
            blocks["same tier"].append(((hi, lo), p))
    blocks["same half"].sort(key=lambda kp: kp[1]["cap"])
    blocks["same tier"].sort(key=lambda kp: kp[1]["rel"])

    heights = [max(len(b), 1) for b in blocks.values()]
    fig, axs = plt.subplots(2, 1, figsize=(9, 0.55 * sum(heights) + 2.6), sharex=True,
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


def fig_position_bias(rows, models, out):
    fig, axes = plt.subplots(1, len(BENCHES), figsize=(4.4 * len(BENCHES), 4), sharex=True)
    for ax, b in zip(np.atleast_1d(axes), BENCHES):
        style(ax)
        ax.axvline(0.5, ls="--", c="#8a8a86", lw=1)
        ys, labels = [], []
        for i, j in enumerate(j for j in JUDGE_ORDER):
            rs = [r for r in rows if r["eval"] == b and r["judge"] == j and r["slot_a"] is not None]
            if not rs:
                continue
            n = sum(r["n_items"] * 2 for r in rs)
            p = float(np.average([r["slot_a"] for r in rs], weights=[max(r["n_items"], 1) for r in rs]))
            lo, hi = wilson(p * n, n)
            y = len(labels)
            ax.plot([lo, hi], [y, y], color=JUDGE_COLORS[j], lw=2)
            ax.scatter([p], [y], color=JUDGE_COLORS[j], s=60, zorder=5)
            labels.append(jlabel(models, j))
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("P(picked slot 1)")
        ax.set_title(b, fontsize=10)
        ax.set_xlim(0.15, 0.85)
    fig.suptitle("Position bias per judge and eval — all runs pooled, Wilson 95% CI", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save(fig, out, "9_position_bias_by_eval")


def _pick_longer_rate(bench):
    per_judge = defaultdict(lambda: [0, 0])
    for d in DIRS[bench]:
        trans = {Path(f).stem: json.load(open(f)) for f in glob.glob(str(d / "transcripts/*.json"))
                 if "__ep1" not in f}
        for pf in glob.glob(str(d / "pairs/*__full.json")):
            major, minor, _ = Path(pf).stem.split("__")
            if major == minor or excluded(major) or excluded(minor):
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
                        per_judge[j][1] += 1
                        per_judge[j][0] += picked_len > other
    return per_judge


def fig_length_bias(models, out):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    style(ax)
    ax.axhline(0.5, ls="--", c="#8a8a86", lw=1)
    W = 0.26
    js = JUDGE_ORDER
    for i, b in enumerate(BENCHES):
        rates = _pick_longer_rate(b)
        xs, ys, es = [], [], []
        for k, j in enumerate(js):
            if j in rates and rates[j][1] > 0:
                c, n = rates[j]
                xs.append(k + (i - 1) * W)
                ys.append(c / n)
                lo, hi = wilson(c, n)
                es.append((c / n - lo, hi - c / n))
        if xs:
            ax.bar(xs, ys, width=W, color=EVAL_COLORS[b], label=b, edgecolor="white", linewidth=0.5)
            ax.errorbar(xs, ys, yerr=np.array(es).T, fmt="none", ecolor="#555", lw=1, capsize=2)
    ax.set_xticks(range(len(js)))
    ax.set_xticklabels([jlabel(models, j) for j in js], rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("P(picked the longer transcript)")
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Length bias per judge and eval — cross pairs only, Wilson 95% CI", fontsize=11)
    fig.tight_layout()
    save(fig, out, "10_length_bias_by_eval")


def fig_capability_vs_length(models, out):
    fig, axes = plt.subplots(1, len(BENCHES), figsize=(4.4 * len(BENCHES), 4.2))
    axes = np.atleast_1d(axes)
    for ax, b in zip(axes, BENCHES):
        style(ax)
        lens, metas = defaultdict(list), {}
        for d in DIRS[b]:
            if not (d / "index.json").exists():
                continue
            for m, meta in json.load(open(d / "index.json"))["models"].items():
                f = d / f"transcripts/{m}.json"
                if excluded(m) or meta.get(GAP_AXIS) is None or not f.exists():
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
    fig, axes = plt.subplots(1, len(BENCHES), figsize=(4.4 * len(BENCHES), 4.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, b in zip(axes, BENCHES):
        style(ax)
        acc = defaultdict(lambda: [0, 0])
        for d in DIRS[b]:
            if not (d / "index.json").exists():
                continue
            meta = json.load(open(d / "index.json"))["models"]
            grid = [m for m, mm in meta.items() if mm["arm"] == "A" and not excluded(m)]
            solved = defaultdict(int)
            for m in grid:
                f = d / f"transcripts/{m}.json"
                if f.exists():
                    for item in json.load(open(f)):
                        solved[item] += 1
            for pf in glob.glob(str(d / "pairs/*__full.json")):
                major, minor, _ = Path(pf).stem.split("__")
                if (major == minor or major not in grid or minor not in grid
                        or excluded(major) or excluded(minor)):
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
    members = [m for m in models["gpqa"]
               if models["gpqa"][m]["arm"] == "A" and not excluded(m)] + ["qwen3-32b"]
    done = defaultdict(set)
    for r in rows:
        if r["eval"] in BENCHES:
            done[(r["judge"], frozenset([r["major"], r["minor"]]))].add(r["eval"])
    pairs = sorted({frozenset([a, b]) for a in members for b in members if a != b}, key=str)
    M = np.zeros((len(pairs), len(judges)))
    for (j, p), evs in done.items():
        if j in judges and p in pairs:
            M[pairs.index(p), judges.index(j)] = len(evs)
    fig, ax = plt.subplots(figsize=(9, 0.28 * len(pairs) + 2))
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
                    help="viewer dataset dirs under interactive_viewer/data")
    ap.add_argument("--gap", choices=["eci", "acc"], default="eci",
                    help="capability measure for the gap axis and pair orientation")
    ap.add_argument("--exclude-members", nargs="*", default=[],
                    help="drop members whose name contains any of these "
                         "(case-insensitive), and every pair they appear in")
    args = ap.parse_args()
    EXCLUDE.update(f.lower() for f in args.exclude_members)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    BENCHES = list(dict.fromkeys(series_name(d) for d in args.datasets))
    DIRS = {s: [] for s in BENCHES}
    for d in args.datasets:
        DIRS[series_name(d)].append(DATA / d)
    if args.gap == "acc":
        GAP_AXIS, GAP_LABEL = "accuracy", "capability gap (accuracy diff, per bench)"
    rows, models = load()
    print(f"{len(rows)} runs loaded; writing to {args.out_dir}")
    fig_model_accuracy(models, args.out_dir)
    fig_gap_curves(rows, models, args.out_dir)
    fig_weak_judge(rows, models, args.out_dir)
    fig_cap_release_plane(rows, models, args.out_dir)
    fig_control_pairs(rows, models, args.out_dir)
    for b in BENCHES:
        fig_judge_pair_heatmap(rows, models, args.out_dir, b)
    fig_position_bias(rows, models, args.out_dir)
    fig_length_bias(models, args.out_dir)
    fig_capability_vs_length(models, args.out_dir)
    fig_item_difficulty(models, args.out_dir)
    fig_fill_plan(rows, models, args.out_dir)


if __name__ == "__main__":
    main()
