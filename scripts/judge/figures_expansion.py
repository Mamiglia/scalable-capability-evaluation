"""Figure suite over interactive_viewer exports.

Default: the data-expansion run with ECI gap axis (the run's design variable).
--datasets extends to any viewer datasets (e.g. gpqa-pilot-rd2); pilot models
have no ECI, so combined runs use --gap acc (per-bench accuracy diff).
Exported scores are P(pick first-named); load() reorients every pair so score
= P(pick the gap-axis-stronger member). Truncation runs are skipped.
--exclude-members drops answer models by name fragment (the deepseek family has
a terse-answer artefact, so its pairs get their own figure set).
--reasoning-filter drops transcripts a judge cannot assess because they show no
work: per_transcript one at a time, per_model whole models by their median.
Reasoning-effort variants of a model go with either, so the pool holds one entry
per model. Defaults come from configs/reasoning_filter.yaml.

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
from functools import lru_cache
from pathlib import Path
from statistics import median

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.ticker import MaxNLocator
from scipy.optimize import curve_fit
from scipy.stats import beta as beta_dist, spearmanr, t as student_t, ttest_1samp
from sklearn.isotonic import IsotonicRegression

from judge.dataset import is_effort_variant, strip_answer
from mc_cost.gpqa import model_family

DATA = Path("interactive_viewer/data")
MODEL_SET_CSV = Path("configs/comparisons/model_set.csv")
REASONING_CONFIG = Path("configs/reasoning_filter.yaml")
REASONING_MODES = ("none", "per_transcript", "per_model")
EXPANSION_DATASETS = ["data-expansion-gpqa", "data-expansion-aime2025", "data-expansion-aime2026"]
EVAL_COLORS = {"gpqa": "#2a78d6", "aime": "#eb6834", "aime (reasoning shown)": "#1baf7a"}

# set in main from --datasets / --gap
BENCHES = []       # series names, in dataset order (same bench pools across datasets)
DIRS = {}          # series -> list of viewer dataset dirs
GAP_AXIS = "eci"   # model-meta key the gap and pair orientation use
GAP_LABEL = "capability gap (ECI)"
EXCLUDE = set()    # lowercased name fragments; matching answer models are dropped
EXCLUDE_EXACT = set()  # whole lowercased names the reasoning filter dropped
REASONING_MODE = "none"
MIN_REASONING_CHARS = 0


def excluded(model):
    """True if the model was dropped by name fragment or by the reasoning filter."""
    name = model.lower()
    return name in EXCLUDE_EXACT or any(frag in name for frag in EXCLUDE)


@lru_cache(maxsize=None)
def transcript_chars(d):
    """(model, item) -> characters of work, one viewer dataset."""
    chars = {}
    for f in glob.glob(str(d / "transcripts/*.json")):
        model = Path(f).stem
        if model.endswith("__ep1"):  # epoch-1 copies, not in the model table
            continue
        for item, text in json.load(open(f)).items():
            chars[(model, item)] = len(strip_answer(text))
    return chars


def reasoning_ok(d, model, item):
    """True if this transcript clears the floor.

    A model with transcripts exported but none for this item did not answer it
    correctly, so the item is not part of a both-correct pair and does not count.
    A model with no transcripts exported at all cannot be checked, so it is kept.
    """
    chars = transcript_chars(d)
    if model not in _models_with_transcripts(d):
        return True
    return chars.get((model, item), 0) >= MIN_REASONING_CHARS


@lru_cache(maxsize=None)
def _models_with_transcripts(d):
    return {model for model, _ in transcript_chars(d)}


def models_without_reasoning(dirs, suffixes):
    """Answer models the filter drops whole: effort variants, and under per_model
    those whose median transcript shows less work than the floor."""
    lengths = defaultdict(list)
    for d in dirs:
        for (model, _), chars in transcript_chars(d).items():
            lengths[model].append(chars)
    dropped = {m for m in lengths if is_effort_variant(m, suffixes)}
    if REASONING_MODE == "per_model":
        dropped |= {m for m, L in lengths.items() if median(L) < MIN_REASONING_CHARS}
    return dropped


def series_name(dataset):
    if dataset == "gpqa-pilot-rd2" or dataset.endswith("-gpqa"):
        return "gpqa"
    s = dataset.replace("data-expansion-", "")
    if s.startswith("aime"):
        return "aime (reasoning shown)" if s.endswith("-ablation") else "aime"
    return s
JUDGE_ORDER = ["qwen3-8b", "qwen3-32b", "qwen3-vl-30b-a3b-instruct",
               "ling-2.6-flash", "glm-4.6", "deepseek-v4-flash-0731", "gemini-3.1-flash-lite"]
JUDGE_COLORS = dict(zip(JUDGE_ORDER, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                                      "#e87ba4", "#008300", "#8250c4"]))


@lru_cache(maxsize=None)
def refit_runs(d):
    """Per-run verdict stats over the items whose two transcripts both show work.

    The exported run stats are over every item, so per-transcript filtering has to
    re-tally them from the item-level verdicts. Keys are (judge, major, minor) with
    major and minor as the pair file names them, before any reorientation.

    Self-pairs are left as exported: with one model on both sides there is no major
    to pick, so some exports leave picked_major unset. They are drawn as gap-0
    markers and enter no fit, so nothing measured depends on them.
    """
    stats = {}
    for pf in glob.glob(str(d / "pairs/*__full.json")):
        major, minor, _ = Path(pf).stem.split("__")
        if major == minor:
            continue
        tally = defaultdict(lambda: [0, 0, 0, 0])  # picked major, picked slot 1, parsed, items
        for item, judges in json.load(open(pf)).items():
            if not (reasoning_ok(d, major, item) and reasoning_ok(d, minor, item)):
                continue
            for j, cell in judges.items():
                tally[j][3] += 1
                for o in cell["orders"]:
                    if o and o["pick"] in ("1", "2"):
                        tally[j][2] += 1
                        tally[j][0] += o["picked_major"]
                        tally[j][1] += o["pick"] == "1"
        for j, (picked_major, slot_1, parsed, items) in tally.items():
            if not parsed:
                continue
            lo, hi = wilson(picked_major, parsed)
            stats[(j, major, minor)] = {
                "score": picked_major / parsed, "score_lo": lo, "score_hi": hi,
                "n_items": items, "slot_a": slot_1 / parsed,
                "noanswer_rate": 1 - parsed / (2 * items),
            }
    return stats


def load():
    """rows: one per (eval, judge, pair run); models: per-eval metadata."""
    rows, models, dropped, kept = [], {}, set(), set()
    for s in BENCHES:
      for d in DIRS[s]:
        if not (d / "index.json").exists():
            continue
        idx = json.load(open(d / "index.json"))
        for m, meta in idx["models"].items():
            (dropped if excluded(m) else kept).add(m)
            saved = models.setdefault(s, {}).setdefault(m, dict(meta))
            for key, value in meta.items():
                if saved.get(key) in (None, "") and value not in (None, ""):
                    saved[key] = value
        for r in idx["runs"]:
            if r.get("frac", "full") != "full":
                continue
            if excluded(r["major"]) or excluded(r["minor"]):
                continue
            r = dict(r)
            if REASONING_MODE == "per_transcript" and not r["is_self"]:
                refit = refit_runs(d).get((r["judge"], r["major"], r["minor"]))
                if refit is None:  # nothing left of this pair once the thin items go
                    continue
                r.update(refit)
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
            r["_major_meta"], r["_minor_meta"] = M, N
            r["_judge_meta"] = idx["models"].get(r["judge"])
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


def assert_nothing_excluded_plotted(rows):
    """Stop the run if an excluded name reached the answer-model pool."""
    members = {m for r in rows for m in (r["major"], r["minor"])}
    bad = sorted(n for n in members if excluded(n))
    assert not bad, f"excluded names reached the plotted answer models: {bad}"
    print(f"plotted answer models: {len(members)}, none matching {sorted(EXCLUDE) or 'nothing'}")
    print(f"plotted judges: {len({r['judge'] for r in rows})} (answer-model exclusions do not apply)")


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
    for s in ("left", "bottom"):
        ax.spines[s].set_position(("outward", 5))


def save(fig, out, name, dpi=150):
    fig.savefig(out / f"{name}.png", dpi=dpi)
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


MANUSCRIPT_DPI = 300
MEAN_COLOUR = "#222"     # neutral: the palette already encodes judge identity
GAP_CURVE_RC = {"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                "xtick.labelsize": 7, "ytick.labelsize": 7}


def _gap_curve(ax, rows, key, series, colors, fit=True):
    """Per-run scatter plus a sigmoid fit or connected empirical series."""
    style(ax)
    ax.axhline(0.5, ls="--", c="#8a8a86", lw=0.8)
    for s in series:
        rs = [r for r in rows if r["_series"] == s and not r["is_self"] and r.get(key) is not None
              and r["score"] is not None]
        if not rs:
            continue
        xs = np.array([r[key] for r in rs], float)
        ys = np.array([r["score"] for r in rs], float)
        ns = np.array([max(r["n_items"], 1) for r in rs], float)
        ax.scatter(xs, ys, s=3, color=colors[s], alpha=0.35, linewidths=0)
        if not fit:
            order = np.argsort(xs)
            ax.plot(xs[order], ys[order], color=colors[s], lw=1, marker="o", ms=2)
        elif len(rs) >= 4:
            f = fit_sigmoid(xs, ys, ns)
            if f:
                gx = np.linspace(min(xs.min(), 0), xs.max(), 100)
                ax.plot(gx, sigmoid(gx, *f), color=colors[s], lw=1)


def _gap_bin_edges(gaps, reference=None):
    """Automatic bins, optionally reusing a wider reference data set's width."""
    gaps = np.unique(np.asarray(gaps, float))
    reference = gaps if reference is None else np.unique(np.asarray(reference, float))
    edges = np.histogram_bin_edges(reference, bins="auto")
    used = np.flatnonzero((edges[:-1] <= gaps.max()) & (edges[1:] >= gaps.min()))
    return edges[used[0]:used[-1] + 2]


def _bootstrap_mean_ci(rates, draws=10_000):
    """Deterministic percentile CI over judges, bounded by observed rates."""
    rates = np.asarray(rates, float)
    samples = rates[np.random.default_rng(0).integers(len(rates), size=(draws, len(rates)))]
    return np.quantile(samples.mean(axis=1), [0.025, 0.975])


def _judge_mean_band(ax, rows, key, reference_gaps=None, include_self=False,
                     per_judge_colors=None, fit_mean=False):
    """Accuracy binned along the gap axis, averaged over judges.

    Judges are the replication unit: per-judge rate within each bin, then mean
    and percentile-bootstrap 95% CI across judges. Returns how many judges contributed.
    """
    style(ax)
    ax.axhline(0.5, ls="--", c="#8a8a86", lw=0.8)
    rs = [r for r in rows if (include_self or not r["is_self"])
          and r.get(key) is not None and r["score"] is not None]
    if not rs:
        return 0
    cx, mu, lo, hi, fit_weights, judges = [], [], [], [], [], set()
    per_judge_points = defaultdict(list)

    def append_bin(bin_rows, x):
        per_judge = defaultdict(lambda: [0.0, 0.0])
        for r in bin_rows:
            w = max(r["n_items"], 1)
            per_judge[r["judge"]][0] += r["score"] * w
            per_judge[r["judge"]][1] += w
        rate_by_judge = {j: c / n for j, (c, n) in per_judge.items()}
        rates = np.array(list(rate_by_judge.values()))
        if len(rates) < 2:
            return
        judges.update(per_judge)
        for judge, rate in rate_by_judge.items():
            per_judge_points[judge].append((x, rate))
        cx.append(x)
        mu.append(float(rates.mean()))
        fit_weights.append(len(rates))
        a_ci, b_ci = _bootstrap_mean_ci(rates)
        lo.append(a_ci)
        hi.append(b_ci)

    if include_self:
        append_bin([r for r in rs if r["is_self"]], 0.0)
    nonself = [r for r in rs if not r["is_self"]]
    if nonself:
        edges = _gap_bin_edges([r[key] for r in nonself], reference=reference_gaps)
        for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
            last = i == len(edges) - 2
            bin_rows = [r for r in nonself
                        if (a <= r[key] <= b if last else a <= r[key] < b)]
            if bin_rows:
                append_bin(bin_rows, float(np.mean(np.unique([r[key] for r in bin_rows]))))
    ax.fill_between(cx, lo, hi, color=MEAN_COLOUR, alpha=0.18, linewidth=0)
    for judge, points in per_judge_points.items():
        if per_judge_colors is None or judge not in per_judge_colors:
            continue
        xs, ys = zip(*points)
        color = per_judge_colors[judge]
        ax.plot(xs, ys, color=color, lw=0.7, marker="o", ms=2, alpha=0.8,
                clip_on=False, zorder=2)
        if xs[0] == 0:
            ax.scatter([0], [ys[0]], s=12, facecolors="white", edgecolors=color,
                       linewidths=0.6, clip_on=False, zorder=4)
    fitted = fit_sigmoid(np.asarray(cx), np.asarray(mu), np.asarray(fit_weights)) \
        if fit_mean and len(cx) >= 4 else None
    if fitted:
        ax.scatter(cx, mu, color=MEAN_COLOUR, s=9, clip_on=False, zorder=4)
        gx = np.linspace(min(cx), max(cx), 100)
        ax.plot(gx, sigmoid(gx, *fitted), color=MEAN_COLOUR, lw=1.2, zorder=3)
    else:
        ax.plot(cx, mu, color=MEAN_COLOUR, lw=1.2, marker="o", ms=3,
                clip_on=False, zorder=3)
    return len(judges)


def _zero_gap_self_comparison(rows):
    """Matched per-judge fitted accuracy at gap zero and observed self accuracy."""
    result = []
    judges = sorted({r["judge"] for r in rows if r["eval"] == "gpqa"})
    for judge in judges:
        cross = [r for r in rows if r["eval"] == "gpqa" and r["judge"] == judge
                 and not r["is_self"] and r.get("gap") is not None and r["score"] is not None]
        selfs = [r for r in rows if r["eval"] == "gpqa" and r["judge"] == judge
                 and r["is_self"] and r["score"] is not None]
        if len({r["gap"] for r in cross}) < 4 or not selfs:
            continue
        fit = fit_sigmoid(
            np.asarray([r["gap"] for r in cross]),
            np.asarray([r["score"] for r in cross]),
            np.asarray([max(r["n_items"], 1) for r in cross]))
        if fit is None:
            continue
        total = sum(max(r["n_items"], 1) for r in selfs)
        observed = sum(r["score"] * max(r["n_items"], 1) for r in selfs) / total
        fitted = float(sigmoid(0.0, *fit))
        result.append({"judge": judge, "fitted_gap0": fitted,
                       "self_accuracy": observed, "delta": fitted - observed,
                       "n_self_runs": len(selfs)})
    return result


def _family_gap_cells(rows, reference_gaps):
    """Within/across-family rates in matched judge × capability-gap bins."""
    cross = [r for r in rows if r["eval"] == "gpqa" and not r["is_self"]
             and r.get("gap") is not None and r["score"] is not None]
    within = [r for r in cross if model_family(r["major"]) == model_family(r["minor"])]
    if not within:
        return []
    common = [r for r in cross if r["gap"] <= max(r["gap"] for r in within)]
    edges = _gap_bin_edges([r["gap"] for r in common], reference=reference_gaps)
    cells = []
    for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        last = i == len(edges) - 2
        block = [r for r in common
                 if (a <= r["gap"] <= b if last else a <= r["gap"] < b)]
        if not block:
            continue
        x = float(np.mean(np.unique([r["gap"] for r in block])))
        tally = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0]))
        for r in block:
            kind = ("within" if model_family(r["major"]) == model_family(r["minor"])
                    else "across")
            w = max(r["n_items"], 1)
            tally[r["judge"]][kind][0] += r["score"] * w
            tally[r["judge"]][kind][1] += w
        for judge, rates in tally.items():
            if not {"within", "across"} <= set(rates):
                continue
            w = rates["within"][0] / rates["within"][1]
            a_rate = rates["across"][0] / rates["across"][1]
            cells.append({"bin": i, "gap": x, "judge": judge, "within": w,
                          "across": a_rate, "delta": w - a_rate})
    return sorted(cells, key=lambda r: (r["bin"], r["judge"]))


def _family_delta_slopes(cells):
    """Linear change in within-minus-across accuracy across bins, per judge."""
    slopes = []
    for judge in sorted({r["judge"] for r in cells}):
        rs = [r for r in cells if r["judge"] == judge]
        if len({r["gap"] for r in rs}) < 2:
            continue
        slope = float(np.polyfit([r["gap"] for r in rs], [r["delta"] for r in rs], 1)[0])
        slopes.append({"judge": judge, "slope": slope, "n_bins": len(rs)})
    return slopes


def _family_judge_means(cells):
    """Give every judge equal weight in the overall matched family contrast."""
    means = []
    for judge in sorted({r["judge"] for r in cells}):
        values = [r["delta"] for r in cells if r["judge"] == judge]
        means.append({"judge": judge, "delta": float(np.mean(values)),
                      "n_bins": len(values)})
    return means


def _fit_forecast_threshold(points):
    """Opponent accuracy where a judge is equally likely to choose the target."""
    ordered = sorted(points)
    x = np.asarray([p[0] for p in ordered], float)
    y = np.asarray([p[1] for p in ordered], float)
    w = np.asarray([p[2] for p in ordered], float)
    if len(x) < 3:
        return None
    fitted = IsotonicRegression(increasing=False).fit_transform(x, y, sample_weight=w)
    if fitted.min() > 0.5 or fitted.max() < 0.5:
        return None
    equal = np.flatnonzero(np.isclose(fitted, 0.5))
    if len(equal):
        return float(np.mean(x[equal]))
    above, below = np.flatnonzero(fitted > 0.5), np.flatnonzero(fitted < 0.5)
    if not len(above) or not len(below) or above[-1] >= below[0]:
        return None
    i, j = above[-1], below[0]
    return float(x[i] + (0.5 - fitted[i]) * (x[j] - x[i])
                 / (fitted[j] - fitted[i]))


def _forecast_rank_cells(rows, min_opponents=4):
    """Held-out target rank estimates from each judge's opponent comparisons."""
    pool, grouped = {}, {}
    for r in rows:
        if r.get("eval") != "gpqa" or r["is_self"] or r["score"] is None:
            continue
        for model, meta in ((r["major"], r["_major_meta"]),
                            (r["minor"], r["_minor_meta"])):
            if meta.get("accuracy") is not None:
                pool[model] = float(meta["accuracy"])
        judge_release = (r.get("_judge_meta") or {}).get("release")
        weight = max(1, 2 * r["n_items"] * (1 - r["noanswer_rate"]))
        for target, opponent, target_meta, opponent_meta, preference in (
            (r["major"], r["minor"], r["_major_meta"], r["_minor_meta"], r["score"]),
            (r["minor"], r["major"], r["_minor_meta"], r["_major_meta"], 1 - r["score"]),
        ):
            gap = _month_gap(target_meta.get("release"), judge_release)
            actual = target_meta.get("accuracy")
            opponent_accuracy = opponent_meta.get("accuracy")
            if (target == r["judge"] or gap is None or gap <= 0
                    or actual is None or opponent_accuracy is None):
                continue
            cell = grouped.setdefault((r["judge"], target), {
                "actual_accuracy": float(actual), "release_gap_months": gap,
                "opponents": defaultdict(lambda: [0.0, 0.0, 0.0]),
            })
            edge = cell["opponents"][opponent]
            edge[0] += preference * weight
            edge[1] += weight
            edge[2] = float(opponent_accuracy)

    estimates = []
    for (judge, target), cell in grouped.items():
        points = [(accuracy, chosen / weight, weight)
                  for chosen, weight, accuracy in cell["opponents"].values()]
        actual = cell["actual_accuracy"]
        n_stronger = sum(accuracy > actual for accuracy, _, _ in points)
        n_weaker = sum(accuracy < actual for accuracy, _, _ in points)
        if len(points) < min_opponents or not n_stronger or not n_weaker:
            continue
        estimated = _fit_forecast_threshold(points)
        if estimated is None:
            continue
        comparison_pool = [accuracy for model, accuracy in pool.items() if model != target]

        def rank(accuracy):
            return 1 + sum(a > accuracy for a in comparison_pool) + \
                   0.5 * sum(np.isclose(a, accuracy) for a in comparison_pool)

        actual_rank, estimated_rank = rank(actual), rank(estimated)
        absolute_error = abs(estimated_rank - actual_rank)
        loo = []
        for i in range(len(points)):
            threshold = _fit_forecast_threshold(points[:i] + points[i + 1:])
            if threshold is not None:
                loo.append(abs(rank(threshold) - actual_rank))
        estimates.append({
            "judge": judge, "target": target,
            "release_gap_months": cell["release_gap_months"],
            "n_opponents": len(points), "n_stronger": n_stronger,
            "n_weaker": n_weaker, "estimated_accuracy": estimated,
            "actual_accuracy": actual, "estimated_rank": estimated_rank,
            "actual_rank": actual_rank, "rank_error": estimated_rank - actual_rank,
            "absolute_rank_error": absolute_error,
            "loo_lo": min(loo) if loo else np.nan,
            "loo_hi": max(loo) if loo else np.nan,
        })
    return sorted(estimates, key=lambda r: (r["release_gap_months"], r["judge"], r["target"]))


def _mean_t_ci(values, min_n=3):
    """Mean and t-based 95% CI over replication units above the requested N."""
    values = np.asarray(values, float)
    mean = float(values.mean())
    if len(values) < min_n:
        return mean, np.nan, np.nan
    sem = float(values.std(ddof=1) / np.sqrt(len(values)))
    half = float(student_t.ppf(0.975, len(values) - 1) * sem)
    return mean, mean - half, mean + half


def _write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_model_inventory(rows, models, out):
    """Write the GPQA judge/evaluated-model roster in CSV and Markdown."""
    gpqa = [r for r in rows if r["eval"] == "gpqa"]
    judges = {r["judge"] for r in gpqa}
    evaluated = {m for r in gpqa for m in (r["major"], r["minor"])}
    inventory = []
    for model in sorted(judges | evaluated):
        accuracy = models.get("gpqa", {}).get(model, {}).get("accuracy")
        inventory.append({
            "model": model,
            "used_as_judge": "Y" if model in judges else "N",
            "evaluated": "Y" if model in evaluated else "N",
            "gpqa_accuracy": f"{accuracy:.3f}" if accuracy is not None else "—",
        })
    _write_csv(out / "0_model_inventory.csv", inventory)
    header = "| Model | Used as judge | Evaluated | GPQA accuracy |\n|---|:---:|:---:|---:|\n"
    body = "".join(f"| {r['model']} | {r['used_as_judge']} | {r['evaluated']} | "
                   f"{r['gpqa_accuracy']} |\n" for r in inventory)
    (out / "0_model_inventory.md").write_text(header + body)


def _gap_curve_panels(rows, key, xlab, series, colors, labelfn, left_tag,
                      self_pairs=True, fit=True, reference_gaps=None,
                      mean_self_pairs=False):
    """Per-series panel beside the judges-averaged panel, one legend below both."""
    present = [s for s in series
               if any(r["_series"] == s and not r["is_self"] and r.get(key) is not None
                      and r["score"] is not None for r in rows)]
    # both panels read the same rows, so the mean visibly summarises what is drawn
    drawn = [r for r in rows if r["_series"] in present]
    if len(drawn) < len(rows):
        print(f"    {len(rows) - len(drawn)} runs outside the plotted series, skipped")
    fig, axs = plt.subplots(1, 2, figsize=(5.5, 3.4), sharex=True, sharey=True)
    _gap_curve(axs[0], drawn, key, present, colors, fit=fit)
    if self_pairs:
        for r in drawn:
            if r["is_self"] and r["score"] is not None:
                axs[0].scatter([0], [r["score"]], marker="o", facecolors="none",
                               edgecolors=colors.get(r["_series"], "#999"),
                               s=12, linewidths=0.6, zorder=5)
    n_judges = _judge_mean_band(axs[1], drawn, key, reference_gaps=reference_gaps,
                                include_self=mean_self_pairs)
    for ax, tag in zip(axs, (left_tag, "judges averaged")):
        ax.set_xlabel(xlab)
        ax.set_ylim(0, 1.02)
        ax.set_yticks([0, 0.5, 1])
        ax.locator_params(axis="x", nbins=4)
        ax.set_title(tag, fontsize=7, loc="left", color="#555")
    axs[0].set_ylabel("judge accuracy")
    h = [plt.Line2D([], [], color=colors[s], lw=1, label=labelfn(s)) for s in present]
    if self_pairs:
        h.append(plt.Line2D([], [], ls="", marker="o", mfc="none", mec="#555", ms=4,
                            label="self-pairs"))
    h += [plt.Line2D([], [], ls="--", color="#8a8a86", lw=0.8, label="chance"),
          plt.Line2D([], [], color=MEAN_COLOUR, lw=1, marker="o", ms=3,
                     label="mean over judges"),
          plt.Rectangle((0, 0), 1, 1, color=MEAN_COLOUR, alpha=0.18,
                        label="95% bootstrap CI over judges")]
    fig.legend(handles=h, loc="lower center", ncol=2, frameon=False,
               handlelength=1.4, columnspacing=1.2, handletextpad=0.5)
    legend_rows = (len(h) + 1) // 2
    fig.tight_layout(rect=[0, 0.06 + 0.04 * legend_rows, 1, 1])
    return fig, n_judges


def _judge_gap_curve(rows, key, xlab, series, colors, labelfn):
    """One panel: binned judge lines over the fitted judge mean and CI."""
    present = [s for s in series
               if any(r["_series"] == s and not r["is_self"]
                      and r.get(key) is not None and r["score"] is not None for r in rows)]
    drawn = [r for r in rows if r["_series"] in present]
    if len(drawn) < len(rows):
        print(f"    {len(rows) - len(drawn)} runs outside the plotted series, skipped")
    fig, ax = plt.subplots(figsize=(1.8, 4.2))
    n_judges = _judge_mean_band(
        ax, drawn, key, include_self=True,
        per_judge_colors=colors, fit_mean=True)
    values = [r[key] for r in drawn if not r["is_self"] and r.get(key) is not None]
    xmin, xmax = min(0, min(values)), max(0, max(values))
    ticks = MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]).tick_values(xmin, xmax)
    ax.set(xlabel=xlab[:1].upper() + xlab[1:], ylabel="Judge accuracy",
           xlim=(ticks[0], ticks[-1]), ylim=(0, 1))
    ax.set_xticks(ticks)
    ax.set_yticks([0, 0.5, 1])
    compact = lambda label: (label.replace(" A3B Instruct", "")
                             .replace(" Flash-Lite", " FL")
                             .replace(" Instruct", ""))
    handles = [plt.Line2D([], [], color=colors[s], lw=0.7, marker="o", ms=2,
                          label=compact(labelfn(s))) for s in present]
    handles += [
        plt.Line2D([], [], ls="", marker="o", mfc="white", mec="#555", ms=4,
                   label="self-pairs"),
        plt.Line2D([], [], ls="--", color="#8a8a86", lw=0.8, label="chance"),
        plt.Line2D([], [], color=MEAN_COLOUR, lw=1.2, marker="o", ms=3,
                   label="mean"),
        plt.Rectangle((0, 0), 1, 1, color=MEAN_COLOUR, alpha=0.18,
                      label="95% CI"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=1, frameon=False,
               handlelength=1.4, handletextpad=0.5)
    fig.subplots_adjust(left=0.31, right=0.96, bottom=0.66, top=0.97)
    return fig, n_judges


def fig_gap_curves(rows, models, out):
    # 2a by judge (pooled evals), 2a-eval by eval; same pair by release gap (2b)
    gap_label = ("ECI gap" if GAP_AXIS == "eci" else
                 "GPQA accuracy gap" if BENCHES == ["gpqa"] else "Accuracy gap")
    specs = [("2a_gap_curve_by_capability", "gap", gap_label),
             ("2b_gap_curve_by_release", "rel_gap_months", "Release gap (months)")]
    if GAP_AXIS != "eci":
        specs.append(("2a_gap_curve_by_capability_eci", "eci_gap", "ECI gap"))
    with plt.rc_context(GAP_CURVE_RC):
        for name, key, xlab in specs:
            for split, series, colors, labelfn, suffix in [
                ("judge", JUDGE_ORDER, JUDGE_COLORS, lambda j: jlabel(models, j), ""),
                ("eval", list(EVAL_COLORS), EVAL_COLORS, lambda e: e, "_by_eval"),
            ]:
                rs = [dict(r, _series=(r["judge"] if split == "judge" else r["eval"]),
                           score=(r["score_eci"] if key == "eci_gap" else r["score"]))
                      for r in rows]
                if split == "judge":
                    fig, n_judges = _judge_gap_curve(
                        rs, key, xlab, series, colors, labelfn)
                else:
                    fig, n_judges = _gap_curve_panels(
                        rs, key, xlab, series, colors, labelfn, "per eval")
                save(fig, out, name + suffix, dpi=MANUSCRIPT_DPI)
                print(f"    CI unit = judges, n = {n_judges}")


def fig_weak_judge(rows, models, out):
    rs = [dict(r, _series=r["eval"]) for r in rows if r["weak_judge"]]
    with plt.rc_context(GAP_CURVE_RC):
        fig, n_judges = _gap_curve_panels(rs, "gap", GAP_LABEL, list(EVAL_COLORS),
                                          EVAL_COLORS, lambda e: e, "per eval",
                                          self_pairs=False)
        save(fig, out, "2d_gap_curve_weak_judge_by_eval", dpi=MANUSCRIPT_DPI)
        print(f"    CI unit = judges, n = {n_judges}")


def _oversight_rows(rows, models):
    """GPQA pairs with a newer, stronger-than-judge focal and older counterpart."""
    gpqa = models.get("gpqa", {})
    selected = []
    for r in rows:
        if r["eval"] != "gpqa" or r["is_self"] or r["score"] is None:
            continue
        a = r.get("_major_meta") or gpqa.get(r["major"])
        b = r.get("_minor_meta") or gpqa.get(r["minor"])
        judge = r.get("_judge_meta") or gpqa.get(r["judge"])
        if any(m is None or m.get("accuracy") is None or not m.get("release")
               for m in (a, b, judge)):
            continue

        def qualifies(focal, other):
            return (focal["accuracy"] > judge["accuracy"]
                    and _month_gap(focal["release"], judge["release"]) > 0
                    and _month_gap(focal["release"], other["release"]) > 0)

        if (a["accuracy"] == b["accuracy"]
                or not (qualifies(a, b) or qualifies(b, a))):
            continue
        selected.append(dict(r, gap=abs(a["accuracy"] - b["accuracy"]),
                             score=r["score"] if a["accuracy"] > b["accuracy"]
                             else 1 - r["score"], _series=r["judge"]))
    return selected


def fig_oversight(rows, models, out):
    rs = _oversight_rows(rows, models)
    if not rs:
        print("  SKIPPED 2e_gap_curve_oversight_by_judge: no eligible runs")
        return
    present = {r["judge"] for r in rs}
    controls = [dict(r, gap=0.0, _series=r["judge"]) for r in rows
                if r["eval"] == "gpqa" and r["is_self"] and r["score"] is not None
                and r["judge"] in present]
    judges = [j for j in JUDGE_ORDER if j in present] + sorted(present - set(JUDGE_ORDER))
    colors = {j: JUDGE_COLORS.get(j, plt.cm.tab10(i % 10)) for i, j in enumerate(judges)}
    labels = {m: meta.get("label", m) for m, meta in models.get("gpqa", {}).items()}
    all_gpqa_gaps = [r["gap"] for r in rows if r["eval"] == "gpqa" and not r["is_self"]
                     and r.get("gap") is not None and r["score"] is not None]
    with plt.rc_context(GAP_CURVE_RC):
        fig, ax = plt.subplots(figsize=(1.8, 4.2))
        n_judges = _judge_mean_band(
            ax, rs + controls, "gap", reference_gaps=all_gpqa_gaps,
            include_self=True, per_judge_colors=colors, fit_mean=True)
        ax.set(xlabel="GPQA accuracy gap", ylabel="Judge accuracy", ylim=(0, 1))
        ax.set_yticks([0, 0.5, 1])
        ax.set_xlim(0, 0.2)
        ax.set_xticks([0, 0.1, 0.2])
        compact = lambda label: (label.replace(" A3B Instruct", "")
                                 .replace(" Flash-Lite", " FL")
                                 .replace(" Instruct", ""))
        handles = [plt.Line2D([], [], color=colors[j], lw=0.7, marker="o", ms=2,
                              label=compact(labels.get(j, j))) for j in judges]
        handles += [
            plt.Line2D([], [], ls="", marker="o", mfc="white", mec="#555", ms=4,
                       label="self-pairs"),
            plt.Line2D([], [], ls="--", color="#8a8a86", lw=0.8, label="chance"),
            plt.Line2D([], [], color=MEAN_COLOUR, lw=1.2, marker="o", ms=3,
                       label="mean"),
            plt.Rectangle((0, 0), 1, 1, color=MEAN_COLOUR, alpha=0.18,
                          label="95% CI"),
        ]
        fig.legend(handles=handles, loc="lower center", ncol=1, frameon=False,
                   handlelength=1.4, handletextpad=0.5)
        fig.subplots_adjust(left=0.31, right=0.96, bottom=0.66, top=0.97)
        save(fig, out, "2e_gap_curve_oversight_by_judge", dpi=MANUSCRIPT_DPI)
    print(f"    oversight runs = {len(rs)}; self-pair controls = {len(controls)}; "
          f"CI unit = judges, n = {n_judges}")


def fig_zero_gap_self(rows, out):
    comparisons = _zero_gap_self_comparison(rows)
    if len(comparisons) < 3:
        print("  SKIPPED 2g_zero_gap_vs_self_pair: fewer than 3 matched judges")
        return
    _write_csv(out / "2g_zero_gap_vs_self_pair.csv", comparisons)
    fitted = np.asarray([r["fitted_gap0"] for r in comparisons])
    observed = np.asarray([r["self_accuracy"] for r in comparisons])
    deltas = fitted - observed
    delta_mean, delta_lo, delta_hi = _mean_t_ci(deltas)
    p = float(ttest_1samp(deltas, 0, alternative="greater").pvalue)

    with plt.rc_context(GAP_CURVE_RC):
        fig, ax = plt.subplots(figsize=(1.6, 3.8))
        style(ax)
        ax.axhline(0.5, ls="--", c="#8a8a86", lw=0.8)
        judges = [j for j in JUDGE_ORDER if any(r["judge"] == j for r in comparisons)] + \
                 sorted({r["judge"] for r in comparisons} - set(JUDGE_ORDER))
        colors = {j: JUDGE_COLORS.get(j, plt.cm.tab10(i % 10))
                  for i, j in enumerate(judges)}
        for row in comparisons:
            color = colors[row["judge"]]
            ax.plot([0, 1], [row["fitted_gap0"], row["self_accuracy"]],
                    color=color, lw=0.7, alpha=0.8, zorder=2)
            ax.scatter([0], [row["fitted_gap0"]], color=color, s=7,
                       clip_on=False, zorder=3)
            ax.scatter([1], [row["self_accuracy"]], facecolors="white",
                       edgecolors=color, s=9, linewidths=0.7, clip_on=False, zorder=3)
        means = []
        for x, values in enumerate((fitted, observed)):
            mean, lo, hi = _mean_t_ci(values)
            means.append(mean)
            ax.errorbar(x, mean, yerr=[[mean - lo], [hi - mean]], fmt="none",
                        color=MEAN_COLOUR, elinewidth=1, capsize=0,
                        clip_on=False, zorder=4)
        ax.plot([0, 1], means, color=MEAN_COLOUR, lw=1.6, marker="o", ms=3.5,
                clip_on=False, zorder=5)
        ax.set_xticks([0, 1], ["Fitted\ngap 0", "Self-pair"])
        ax.set(xlim=(0, 1), ylabel="Judge accuracy", ylim=(0, 1))
        ax.set_yticks([0, 0.5, 1])
        handles = [plt.Line2D([], [], color=colors[j], lw=0.7, marker="o", ms=2,
                              label=j) for j in judges]
        handles += [
            plt.Line2D([], [], ls="", marker="o", mfc="white", mec="#555", ms=4,
                       label="self-pairs"),
            plt.Line2D([], [], color=MEAN_COLOUR, lw=1.6, marker="o", ms=3.5,
                       label="mean ± 95% CI"),
        ]
        fig.legend(handles=handles, loc="lower center", ncol=1, frameon=False,
                   handlelength=1.3, handletextpad=0.4)
        fig.tight_layout(rect=[0, 0.40, 1, 1])
        save(fig, out, "2g_zero_gap_vs_self_pair", dpi=MANUSCRIPT_DPI)
    print(f"    fitted gap-0 minus self-pair = {delta_mean:+.3f} "
          f"[{delta_lo:+.3f}, {delta_hi:+.3f}], one-sided paired t p={p:.4g}, "
          f"n={len(comparisons)} judges")


def fig_family_gap(rows, out):
    reference = [r["gap"] for r in rows if r["eval"] == "gpqa" and not r["is_self"]
                 and r.get("gap") is not None and r["score"] is not None]
    cells = _family_gap_cells(rows, reference)
    if not cells:
        print("  SKIPPED 2h_family_gap: no matched judge × gap-bin cells")
        return
    _write_csv(out / "2h_family_gap.csv", cells)
    summaries = []
    for bin_id in sorted({r["bin"] for r in cells}):
        block = [r for r in cells if r["bin"] == bin_id]
        entry = {"bin": bin_id, "gap": block[0]["gap"], "n": len(block), "rows": block}
        for key in ("within", "across", "delta"):
            entry[key], entry[f"{key}_lo"], entry[f"{key}_hi"] = _mean_t_ci(
                [r[key] for r in block], min_n=2)
        summaries.append(entry)

    with plt.rc_context(GAP_CURVE_RC):
        fig, (ax, dx) = plt.subplots(2, 1, figsize=(2.8, 3.8), sharex=True,
                                     gridspec_kw={"height_ratios": [2, 1], "hspace": 0.12})
        style(ax)
        style(dx)
        ax.axhline(0.5, ls="--", c="#8a8a86", lw=0.8)
        dx.axhline(0, ls="--", c="#8a8a86", lw=0.8)
        x = np.asarray([s["gap"] for s in summaries])
        for key, color, label in (("across", "#777", "Across family"),
                                  ("within", MEAN_COLOUR, "Within family")):
            mean = np.asarray([s[key] for s in summaries])
            lo = np.clip(np.asarray([s[f"{key}_lo"] for s in summaries]), 0, 1)
            hi = np.clip(np.asarray([s[f"{key}_hi"] for s in summaries]), 0, 1)
            ci = np.isfinite(lo) & np.isfinite(hi)
            ax.errorbar(x[ci], mean[ci], yerr=[mean[ci] - lo[ci], hi[ci] - mean[ci]],
                        fmt="none", color=color, elinewidth=0.8, capsize=0, alpha=0.7)
            ax.plot(x, mean, color=color, lw=1, marker="o", ms=3, label=label)
        delta = np.asarray([s["delta"] for s in summaries])
        dlo = np.asarray([s["delta_lo"] for s in summaries])
        dhi = np.asarray([s["delta_hi"] for s in summaries])
        ci = np.isfinite(dlo) & np.isfinite(dhi)
        dx.errorbar(x[ci], delta[ci], yerr=[delta[ci] - dlo[ci], dhi[ci] - delta[ci]],
                    fmt="none", color=MEAN_COLOUR, elinewidth=0.8, capsize=0, alpha=0.7)
        dx.plot(x, delta, color=MEAN_COLOUR, lw=1, marker="o", ms=3)
        for s in summaries:
            if s["n"] >= 3:
                continue
            ax.scatter([s["gap"] - 0.002] * s["n"], [r["across"] for r in s["rows"]],
                       s=6, color="#777", alpha=0.65)
            ax.scatter([s["gap"] + 0.002] * s["n"], [r["within"] for r in s["rows"]],
                       s=6, color=MEAN_COLOUR, alpha=0.65)
            dx.scatter([s["gap"]] * s["n"], [r["delta"] for r in s["rows"]],
                       s=6, color=MEAN_COLOUR, alpha=0.65)
        ax.set(ylabel="Judge accuracy", ylim=(0, 1))
        ax.set_yticks([0, 0.5, 1])
        dx.set(xlabel="GPQA accuracy gap", ylabel="Within − across",
               xlim=(0, 0.3), ylim=(-0.4, 0.4))
        dx.set_xticks([0, 0.1, 0.2, 0.3])
        dx.set_yticks([-0.4, 0, 0.4])
        handles = [
            plt.Line2D([], [], color=MEAN_COLOUR, lw=1, marker="o", ms=3,
                       label="Within family"),
            plt.Line2D([], [], color="#777", lw=1, marker="o", ms=3,
                       label="Across family"),
            plt.Line2D([], [], color="#888", lw=0.8, label="95% CI over judges"),
        ]
        fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
                   handlelength=1.3, columnspacing=1, handletextpad=0.4)
        fig.subplots_adjust(left=0.30, right=0.97, top=0.98, bottom=0.23, hspace=0.1)
        save(fig, out, "2h_family_gap", dpi=MANUSCRIPT_DPI)

    judge_means = _family_judge_means(cells)
    mean, lo, hi = _mean_t_ci([r["delta"] for r in judge_means])
    p = float(ttest_1samp([r["delta"] for r in judge_means], 0).pvalue)
    print(f"    judge-balanced within-across difference = {mean:+.3f} "
          f"[{lo:+.3f}, {hi:+.3f}], two-sided t p={p:.4g}, "
          f"n={len(judge_means)} judges")
    slopes = _family_delta_slopes(cells)
    if len(slopes) >= 3:
        mean, lo, hi = _mean_t_ci([r["slope"] for r in slopes])
        p = float(ttest_1samp([r["slope"] for r in slopes], 0).pvalue)
        print(f"    within-across delta slope per 0.1 gap = {mean / 10:+.3f} "
              f"[{lo / 10:+.3f}, {hi / 10:+.3f}], "
              f"two-sided t p={p:.4g}, n={len(slopes)} judges")
    print("    matched judge counts by gap bin: " +
          ", ".join(f"{s['gap']:.3f}: {s['n']}" for s in summaries))


def fig_forecasting(rows, out):
    cells = _forecast_rank_cells(rows)
    if len(cells) < 3:
        print("  SKIPPED 2i_forecasting_rank_error: fewer than 3 bracketed cells")
        return
    _write_csv(out / "2i_forecasting_rank_error.csv", cells)
    judges = [j for j in JUDGE_ORDER if any(r["judge"] == j for r in cells)] + \
             sorted({r["judge"] for r in cells} - set(JUDGE_ORDER))
    colors = {j: JUDGE_COLORS.get(j, plt.cm.tab10(i % 10))
              for i, j in enumerate(judges)}
    xmax = max(3, int(np.ceil(max(r["release_gap_months"] for r in cells) / 3) * 3))
    upper = max(r["loo_hi"] if np.isfinite(r["loo_hi"])
                else r["absolute_rank_error"] for r in cells)
    ymax = max(5, int(np.ceil(upper / 5) * 5))

    with plt.rc_context(GAP_CURVE_RC):
        fig, ax = plt.subplots(figsize=(1.8, 4.0))
        style(ax)
        for judge in judges:
            rs = sorted((r for r in cells if r["judge"] == judge),
                        key=lambda r: r["release_gap_months"])
            x = np.asarray([r["release_gap_months"] for r in rs])
            y = np.asarray([r["absolute_rank_error"] for r in rs])
            ax.plot(x, y, color=colors[judge], lw=0.7, marker="o", ms=3,
                    alpha=0.8, zorder=3)
            finite = np.asarray([np.isfinite(r["loo_lo"]) and np.isfinite(r["loo_hi"])
                                 for r in rs])
            if finite.any():
                lo = np.asarray([r["loo_lo"] for r in rs])[finite]
                hi = np.asarray([r["loo_hi"] for r in rs])[finite]
                ax.errorbar(x[finite], y[finite], yerr=[y[finite] - lo, hi - y[finite]],
                            fmt="none", color=colors[judge], elinewidth=0.8,
                            capsize=0, alpha=0.7, zorder=2)
        x = np.asarray([r["release_gap_months"] for r in cells], float)
        y = np.asarray([r["absolute_rank_error"] for r in cells], float)
        slope, intercept = np.polyfit(x, y, 1)
        gx = np.asarray([x.min(), x.max()])
        ax.plot(gx, intercept + slope * gx, color=MEAN_COLOUR, lw=1.6, zorder=4)
        ax.set(xlabel="Release gap (months)", ylabel="Absolute rank error",
               xlim=(0, xmax), ylim=(0, ymax))
        ax.set_xticks(np.arange(0, xmax + 0.1, 3))
        ystep = 5 if ymax <= 20 else 10
        ax.set_yticks(np.append(np.arange(0, ymax, ystep), ymax))
        handles = [plt.Line2D([], [], color=colors[j], lw=0.7, marker="o", ms=3,
                              label=j) for j in judges]
        handles += [
            plt.Line2D([], [], color="#888", lw=0.8,
                       label="Leave-one-opponent-out range"),
            plt.Line2D([], [], color=MEAN_COLOUR, lw=1.6, label="Pooled linear fit"),
        ]
        fig.legend(handles=handles, loc="lower center", ncol=1, frameon=False,
                   handlelength=1.3, handletextpad=0.4)
        fig.subplots_adjust(left=0.31, right=0.96, bottom=0.66, top=0.97)
        save(fig, out, "2i_forecasting_rank_error", dpi=MANUSCRIPT_DPI)
    rho = spearmanr(x, y).statistic
    print(f"    forecasting cells = {len(cells)} across {len(judges)} judges; "
          f"pooled slope = {slope:+.3f} rank positions/month; Spearman rho = {rho:+.3f}")


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
    g, design = {}, _design_grid()
    for mm in models.values():
        for m, meta in mm.items():
            if (meta["arm"] == "A" and meta.get("eci") and meta.get("release")
                    and meta["label"] in design):
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
                if not (reasoning_ok(d, major, item) and reasoning_ok(d, minor, item)):
                    continue
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
                    if not (reasoning_ok(d, major, item) and reasoning_ok(d, minor, item)):
                        continue
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
    global BENCHES, DIRS, GAP_AXIS, GAP_LABEL, REASONING_MODE, MIN_REASONING_CHARS
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--datasets", nargs="+", default=EXPANSION_DATASETS,
                    help="viewer dataset dirs under interactive_viewer/data")
    ap.add_argument("--gap", choices=["eci", "acc"], default="eci",
                    help="capability measure for the gap axis and pair orientation")
    ap.add_argument("--exclude-members", nargs="*", default=[],
                    help="drop models whose name contains any of these "
                         "(case-insensitive) from the answer-model pool")
    ap.add_argument("--reasoning-config", type=Path, default=REASONING_CONFIG,
                    help="defaults for the transcript filter")
    ap.add_argument("--reasoning-filter", choices=REASONING_MODES,
                    help="drop transcripts, or whole answer models, that show no work")
    ap.add_argument("--min-reasoning-chars", type=int,
                    help="characters after the ANSWER line a transcript needs")
    args = ap.parse_args()
    EXCLUDE.update(f.lower() for f in args.exclude_members if f)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    BENCHES = list(dict.fromkeys(series_name(d) for d in args.datasets))
    DIRS = {s: [] for s in BENCHES}
    for d in args.datasets:
        DIRS[series_name(d)].append(DATA / d)
    if args.gap == "acc":
        GAP_AXIS, GAP_LABEL = "accuracy", "capability gap (accuracy diff, per bench)"

    filter_config = yaml.safe_load(args.reasoning_config.read_text())
    REASONING_MODE = args.reasoning_filter or filter_config["mode"]
    MIN_REASONING_CHARS = (args.min_reasoning_chars if args.min_reasoning_chars is not None
                           else filter_config["min_reasoning_chars"])
    suffixes = filter_config["effort_suffixes"] if filter_config["exclude_effort_variants"] else ()
    thin = models_without_reasoning([d for ds in DIRS.values() for d in ds], suffixes)
    EXCLUDE_EXACT.update(m.lower() for m in thin)
    print(f"reasoning filter: {REASONING_MODE}, floor {MIN_REASONING_CHARS} chars, "
          f"effort suffixes {list(suffixes) or 'none'}")
    print(f"dropped whole: {', '.join(sorted(thin)) or 'none'}")

    rows, models = load()
    assert_nothing_excluded_plotted(rows)
    print(f"{len(rows)} runs loaded; writing to {args.out_dir}")
    write_model_inventory(rows, models, args.out_dir)
    fig_model_accuracy(models, args.out_dir)
    fig_gap_curves(rows, models, args.out_dir)
    fig_weak_judge(rows, models, args.out_dir)
    fig_oversight(rows, models, args.out_dir)
    fig_zero_gap_self(rows, args.out_dir)
    fig_family_gap(rows, args.out_dir)
    fig_forecasting(rows, args.out_dir)
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
