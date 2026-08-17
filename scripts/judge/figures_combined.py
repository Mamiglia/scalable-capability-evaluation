"""Combined gpqa figures: gpqa-pilot-rd2 + data-expansion pooled.

Every pair is oriented stronger-first by gpqa accuracy (the only capability
measure both datasets share; pilot models have no ECI), and score = P(judge
picks the stronger member) — expansion scores are flipped from the exported
P(pick first-named) where needed. gpqa runs only; AIME has no pilot arm.

Judge controls classify each (judge, pair) run by whether the judge is weaker
than both members (gpqa accuracy) and/or older than both (release month).

Usage:
  uv run python scripts/judge/figures_combined.py --out-dir judge_results/combined/figures
"""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

PILOT = Path("judge_results/gpqa-pilot-rd2")
EXPANSION_IDX = Path("interactive_dashboards/transcript_viewer/data/data-expansion-gpqa/index.json")

# metadata missing from pilot models.csv / the viewer export:
# judge-only pilot models (accuracy from Epoch's CSV) and release dates of
# epoch-import members; qwen3-32b accuracy is from the pilot's own gpqa run
EXTRA = {
    "gpt-4.1-nano": {"acc": 0.489, "release": "2025-04", "label": "GPT-4.1-nano"},
    "qwen-2.5-72b-instruct": {"acc": 0.491, "release": "2024-09", "label": "Qwen2.5-72B"},
    "qwen3-32b": {"acc": 0.657, "release": "2025-04", "label": "Qwen3-32B"},
    "gpt-5.5-low": {"release": "2026-04"},
    "gemini-3.5-flash": {"release": "2026-05"},
    "qwen3.6-plus": {"release": "2026-03"},
    "claude-3-5-sonnet-0620": {"release": "2024-06"},
    "llama-3.1-70b": {"release": "2024-07"},
    "gemma-2-27b": {"release": "2024-06"},
}

CLASS_ORDER = ["judge stronger/newer or mixed", "weaker only", "older only", "weaker + older"]
CLASS_COLORS = dict(zip(CLASS_ORDER, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]))


def months(release):
    if not release or "-" not in str(release):
        return None
    y, m = str(release).split("-")[:2]
    return int(y) * 12 + int(m)


def wilson(c, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p, d = c / n, 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return center - half, center + half


def load():
    """rows: one per (dataset, judge, pair) with score = P(pick acc-stronger)."""
    meta = {"pilot": {}, "expansion": {}}
    for r in csv.DictReader(open(PILOT / "models.csv")):
        meta["pilot"][r["model"]] = {"acc": float(r["accuracy"]), "rel": months(r["release"]),
                                     "label": r["label"]}
    idx = json.load(open(EXPANSION_IDX))
    for m, info in idx["models"].items():
        meta["expansion"][m] = {"acc": info["accuracy"],
                                "rel": months(info.get("release") or EXTRA.get(m, {}).get("release")),
                                "label": info["label"]}
    for ds in meta.values():
        for m, e in EXTRA.items():
            if m not in ds and "acc" in e:
                ds[m] = {"acc": e["acc"], "rel": months(e["release"]), "label": e["label"]}

    rows = []
    for r in csv.DictReader(open(PILOT / "summary.csv")):
        if r["major"] == r["minor"] or r["truncate_frac"]:
            continue
        rows.append({"dataset": "pilot", "judge": r["judge"], "major": r["major"],
                     "minor": r["minor"], "score": float(r["score"]), "n": int(r["n_rows"])})
    for r in idx["runs"]:
        if r["is_self"] or r["score"] is None:
            continue
        rows.append({"dataset": "expansion", "judge": r["judge"], "major": r["major"],
                     "minor": r["minor"], "score": r["score"], "n": 2 * r["n_items"]})

    for r in rows:
        M = meta[r["dataset"]]
        a, b = M[r["major"]], M[r["minor"]]
        if a["acc"] < b["acc"]:
            r["major"], r["minor"], a, b = r["minor"], r["major"], b, a
            r["score"] = 1 - r["score"]
        j = M[r["judge"]]
        r["cap_gap"] = a["acc"] - b["acc"]
        r["rel_gap"] = a["rel"] - b["rel"] if a["rel"] and b["rel"] else None
        r["judge_weaker"] = j["acc"] < min(a["acc"], b["acc"])
        r["judge_older"] = (j["rel"] < min(a["rel"], b["rel"])
                           if j["rel"] and a["rel"] and b["rel"] else None)
        r["labels"] = (a["label"], b["label"], j["label"])
    return rows


def resid(y, x):
    b, a = np.polyfit(x, y, 1)
    return y - (a + b * x)


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


def fig_plane(rows, out):
    """3 panels: raw cap-gap x release-gap plane, then each axis residualized
    on the other; colour = pooled P(pick stronger), square = pilot pairs."""
    pairs = {}
    for r in rows:
        if r["rel_gap"] is None:
            continue
        p = pairs.setdefault((r["dataset"], r["major"], r["minor"]),
                             {"cap": r["cap_gap"], "rel": r["rel_gap"], "s": [], "w": [],
                              "labels": r["labels"]})
        p["s"].append(r["score"])
        p["w"].append(r["n"])
    keys = list(pairs)
    cap = np.array([pairs[k]["cap"] for k in keys])
    rel = np.array([pairs[k]["rel"] for k in keys])
    score = np.array([float(np.average(pairs[k]["s"], weights=pairs[k]["w"])) for k in keys])
    nrun = np.array([len(pairs[k]["s"]) for k in keys])
    runs = np.array([(r["score"], r["cap_gap"], r["rel_gap"]) for r in rows
                     if r["rel_gap"] is not None])

    r_cr = pearsonr(cap, rel)[0]
    pr_rel = pearsonr(resid(runs[:, 0], runs[:, 1]), resid(runs[:, 2], runs[:, 1]))[0]
    pr_cap = pearsonr(resid(runs[:, 0], runs[:, 2]), resid(runs[:, 1], runs[:, 2]))[0]

    fig, axs = plt.subplots(1, 3, figsize=(14.5, 4.8), layout="constrained")
    panels = [
        (cap, rel, "capability gap (gpqa accuracy diff)", "release gap (months, + = stronger newer)",
         "1 · raw axes", f"axes correlated: r = {r_cr:+.2f}"),
        (cap, resid(rel, cap), "capability gap (gpqa accuracy diff)", "release gap | capability (resid, months)",
         "2 · release gap, capability regressed out", f"partial r(acc, release | capability) = {pr_rel:+.2f}"),
        (resid(cap, rel), rel, "capability gap | release (resid)", "release gap (months, + = stronger newer)",
         "3 · capability gap, release regressed out", f"partial r(acc, capability | release) = {pr_cap:+.2f}"),
    ]
    for ax, (x, y, xl, yl, title, note) in zip(axs, panels):
        style(ax)
        ax.margins(x=0.1, y=0.15)
        ax.axhline(0, ls=":", c="#8a8a86", lw=0.7)
        for ds, marker in [("expansion", "o"), ("pilot", "s")]:
            sel = np.array([k[0] == ds for k in keys])
            ax.scatter(x[sel], y[sel], c=score[sel], cmap="RdBu_r", vmin=0, vmax=1,
                       s=20 + 7 * nrun[sel], marker=marker, edgecolors="white",
                       linewidths=0.8, zorder=3)
        for rank, i in enumerate(np.argsort(x)):
            up = rank % 2 == 0
            la, lb, _ = pairs[keys[i]]["labels"]
            ax.annotate(f"{la}\nv {lb}", (x[i], y[i]), xytext=(0, 8 if up else -18),
                        textcoords="offset points", ha="center",
                        va="bottom" if up else "top", fontsize=5, color="#666")
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(title, fontsize=9, loc="left", color="#333")
        ax.text(0.98, 0.02, note, transform=ax.transAxes, ha="right", va="bottom",
                fontsize=7, color="#666")
    h = [plt.Line2D([], [], marker="o", ls="", mfc="none", mec="#888", label="expansion pair"),
         plt.Line2D([], [], marker="s", ls="", mfc="none", mec="#888", label="pilot rd2 pair"),
         plt.Line2D([], [], marker="o", ls="", mfc="none", mec="#888", ms=10, label="size: judges pooled")]
    axs[0].legend(handles=h, frameon=False, fontsize=7, loc="upper left")
    sm = plt.cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap="RdBu_r")
    cb = fig.colorbar(sm, ax=axs, shrink=0.85, pad=0.01)
    cb.set_label("P(judge picks stronger member); 0.5 = chance")
    cb.ax.axhline(0.5, color="#222", lw=0.8)
    fig.suptitle("Judge accuracy over the capability-gap / release-gap plane — gpqa, pilot rd2 + expansion,"
                 " judges pooled per pair; stronger = higher gpqa accuracy", fontsize=10)
    save(fig, out, "heatmap_cap_vs_release_combined")


def judge_class(r):
    if r["judge_weaker"] and r["judge_older"]:
        return "weaker + older"
    if r["judge_weaker"]:
        return "weaker only"
    if r["judge_older"]:
        return "older only"
    return "judge stronger/newer or mixed"


def fig_judge_controls(rows, out):
    """P(pick stronger) vs capability gap, split by judge handicap class.
    'weaker only' isolates hypothesis 2 (weak judge, contemporary);
    'older only' isolates hypothesis 1 (old judge, capability-matched)."""
    rows = [r for r in rows if r["judge_older"] is not None]
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    style(ax)
    ax.axhline(0.5, ls="--", c="#8a8a86", lw=1)
    for cls in CLASS_ORDER:
        rs = [r for r in rows if judge_class(r) == cls]
        if not rs:
            continue
        xs = np.array([r["cap_gap"] for r in rs])
        ys = np.array([r["score"] for r in rs])
        ws = np.array([r["n"] for r in rs])
        c, n = float((ys * ws).sum()), float(ws.sum())
        lo, hi = wilson(c, n)
        ax.scatter(xs, ys, s=10 + ws / 8, color=CLASS_COLORS[cls], alpha=0.55,
                   edgecolors="white", linewidths=0.5,
                   label=f"{cls} — pooled {c/n:.2f} [{lo:.2f}, {hi:.2f}], {len(rs)} runs")
        ax.axhline(c / n, color=CLASS_COLORS[cls], lw=1.4, ls=":", alpha=0.9)
    ax.set_xlabel("capability gap (gpqa accuracy diff)")
    ax.set_ylabel("P(judge picks stronger member)")
    ax.set_ylim(0.0, 1.05)
    ax.legend(frameon=False, fontsize=8, loc="lower right",
              title="judge vs pair members (dotted line = class pooled mean)", title_fontsize=8)
    ax.set_title("Judge handicap controls — gpqa, pilot rd2 + expansion pooled; dot area ∝ verdicts\n"
                 "weaker = judge gpqa accuracy below both members · older = judge released before both",
                 fontsize=10, loc="left")
    fig.tight_layout()
    save(fig, out, "judge_controls_weak_old")


def write_controls_csv(rows, out):
    with open(out / "judge_controls.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "dataset", "judge", "stronger", "weaker",
                    "cap_gap", "rel_gap_months", "score", "n_verdicts"])
        for r in sorted(rows, key=lambda r: (judge_class(r), -r["cap_gap"])):
            if r["judge_older"] is None:
                continue
            la, lb, lj = r["labels"]
            w.writerow([judge_class(r), r["dataset"], lj, la, lb,
                        f"{r['cap_gap']:.3f}", r["rel_gap"], f"{r['score']:.3f}", r["n"]])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = load()
    from collections import Counter
    print(f"{len(rows)} runs ({Counter(r['dataset'] for r in rows)})")
    print("judge classes:", dict(Counter(judge_class(r) for r in rows if r["judge_older"] is not None)))
    fig_plane(rows, args.out_dir)
    fig_judge_controls(rows, args.out_dir)
    write_controls_csv(rows, args.out_dir)


if __name__ == "__main__":
    main()
