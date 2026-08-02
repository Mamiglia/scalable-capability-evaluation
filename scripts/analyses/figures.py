"""Generate the judge-pilot figures from the stored tables.

Reads verdicts.csv, models.csv, summary.csv, judge_fits.csv; writes numbered
png+svg figures to <dir>/figures/. Judges keep a fixed colour everywhere;
controls (chance line, length-only baseline, self-pair leakage) are marked on
the figures they qualify, not plotted separately.

Usage:
  uv run python scripts/analyses/figures.py
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from comparisons import COMPARISONS, MODELS
from plots import (COL, HEAT, INK, INK2, JRAMP, MUTED, PALETTE,
                   chance, dot_ci, legend, plt, save, style)
from stats import counterbalanced, sem_ci, sigmoid_fit, wilson

JUDGES = ["gemini-3.1-flash-lite", "qwen3.5-122b-a10b", "gemma-4-31b-it",
          "qwen3.6-35b-a3b", "qwen3-32b", "gemini-2.5-flash", "gpt-4o",
          "gemma-4-26b-a4b-it"]
JCOLOR = dict(zip(JUDGES, PALETTE))

LAB: dict = {}


def init_labels(models):
    """Model captions used everywhere: name (GPQA acc, release year)."""
    for m, meta in MODELS.items():
        acc = models.loc[m, "accuracy"] if m in models.index else np.nan
        rel = meta.get("release")
        LAB[m] = f"{meta['label']} ({acc:.2f}, '{rel[2:4]})" if rel else f"{meta['label']} ({acc:.2f})"


def lab(model):
    return LAB.get(model, model)


def release_year(model):
    d = pd.to_datetime(MODELS[model]["release"])
    return d.year + d.dayofyear / 366


def ordered(summary):
    """Summary rows in fixed judge order, by gap within judge; self-pairs first."""
    summary = summary.copy()
    summary["jorder"] = summary["judge"].map({j: i for i, j in enumerate(JUDGES)})
    return summary.sort_values(["jorder", "d_acc"]).reset_index(drop=True)


def row_positions(summary):
    """y position per row, with a gap between judge blocks (top row = y 0)."""
    ys, y = [], 0.0
    prev = None
    for _, r in summary.iterrows():
        if prev is not None and r["judge"] != prev:
            y += 1.2
        ys.append(y)
        y += 1.0
        prev = r["judge"]
    return np.array(ys)


def pair_label(r):
    if r["major"] == r["minor"]:
        return f"{lab(r['major'])} (self)"
    return f"{lab(r['major'])} v {lab(r['minor'])}"


# --- 1. per-run overview, with inline controls ---------------------------

def fig_run_overview(summary, out):
    s = ordered(summary)
    ys = row_positions(s)
    fig, ax = plt.subplots(figsize=(7.0, 0.16 * len(s) + 1.3))
    for y, (_, r) in zip(ys, s.iterrows()):
        is_self = r.major == r.minor
        note = f"{r.score:.2f} (n={r.n_items}) {r.stars}"
        if is_self:
            note += f" · slot-A {r.slot_a:.2f}"
        dot_ci(ax, y, r.score, r.score_lo, r.score_hi, JCOLOR[r.judge],
               note=note, open_marker=is_self)
    for j in s.judge.unique():
        block = ys[(s.judge == j).to_numpy()]
        ax.annotate(f"judge: {lab(j)}", xy=(-0.02, block.min() - 0.9),
                    xycoords=("axes fraction", "data"), ha="right", va="center",
                    color=INK2, fontweight="bold", annotation_clip=False)
    ax.set_yticks(ys)
    ax.set_yticklabels([pair_label(r) for _, r in s.iterrows()], fontsize=7)
    for tick, (_, r) in zip(ax.get_yticklabels(), s.iterrows()):
        if r.major == r.minor:
            tick.set_fontstyle("italic")
    ax.invert_yaxis()
    chance(ax, 0.5, axis="x")
    ax.set_xlabel("judge accuracy (counterbalanced over A/B orders)")
    ax.set_xlim(0, 1)
    style(ax, grid_axis="x")
    handles = [Line2D([], [], marker="o", ls="-", color=INK2, ms=5.5, label="judge accuracy, 95% CI"),
               Line2D([], [], marker="o", ls="", mfc="white", mec=INK2, ms=5.5,
                      label="self-pair control (should sit at chance)")]
    leg = ax.legend(handles=handles, frameon=False, loc="lower left",
                    bbox_to_anchor=(-0.35, 1.005), ncol=2,
                    handletextpad=0.4, columnspacing=1.2)
    for t in leg.get_texts():
        t.set_color(INK2)
    save(fig, out / "1_run_overview")


# --- 2. gap curve, judges coloured by a scalar ---------------------------

def pair_jitter(summary, width=0.006):
    """Tiny x offset per run within each (major, minor) pair, for visibility."""
    off = {}
    for (maj, mnr), block in summary.groupby(["major", "minor"]):
        for i, idx in enumerate(block.index):
            off[idx] = (i - (len(block) - 1) / 2) * width
    return off


def gap_curve(summary, out, name, jvalue,
              note="lines connect a judge's runs · self-pairs: 95% CI", fits=None,
              jitter=0.003):
    """Score vs raw-accuracy gap; judges on the red-grey-blue ramp (weak/old ->
    strong/new), self-pairs open in their own broken-axis section.

    Ramp steps are rank-spaced, not value-proportional: judge values cluster
    and proportional shades were indistinguishable.
    """
    order = sorted(JUDGES, key=lambda j: (jvalue[j], j))
    col = {j: JRAMP(i / (len(order) - 1)) for i, j in enumerate(order)}

    fig, (axs, ax) = plt.subplots(1, 2, sharey=True, figsize=(6.2, 3.4),
                                  gridspec_kw={"width_ratios": [1, 11], "wspace": 0.05})
    xmax = summary.d_acc.max() * 1.05
    selfs = summary[summary.major == summary.minor].reset_index()
    for i, r in selfs.iterrows():
        x = (i - (len(selfs) - 1) / 2) * (1.0 / max(len(selfs), 1))
        axs.errorbar(x, r.score, yerr=[[r.score - r.score_lo], [r.score_hi - r.score]],
                     fmt="o", ms=4.5, color=col[r.judge], ecolor=col[r.judge],
                     elinewidth=0.9, capsize=0, mfc="white", mew=1.1, zorder=3)
    cross = summary[summary.major != summary.minor]
    jit = pair_jitter(cross, width=jitter)
    connected = set()
    for j, block in cross.groupby("judge"):
        xy = sorted((r.d_acc + jit[idx], r.score) for idx, r in block.iterrows())
        if fits is None and len(xy) > 1:
            connected.add(j)
            ax.plot([p[0] for p in xy], [p[1] for p in xy], "-", color=col[j],
                    lw=1.2, alpha=0.75, zorder=2)
        ax.plot([p[0] for p in xy], [p[1] for p in xy], "o", ms=4.5, color=col[j],
                alpha=0.9, zorder=3)
    if fits is not None:
        x = np.linspace(0, xmax, 100)
        for _, f in fits.iterrows():
            connected.add(f.judge)
            ax.plot(x, 1 / (1 + np.exp(-(f.a + f.d * x))), color=col[f.judge],
                    lw=1.5, zorder=2)
    for a in (axs, ax):
        chance(a)
        a.set_ylim(0, 1)
        style(a)
    axs.set_xlim(-0.7, 0.7)
    axs.set_xticks([0])
    axs.set_xticklabels(["self\npairs"])
    axs.set_ylabel("judge accuracy")
    ax.set_xlim(0, xmax)
    ax.set_xlabel("capability gap (GPQA accuracy difference)")
    # Rotated pair labels under the axis, dodged apart where gaps cluster.
    min_sep = xmax * 0.045
    lx = None
    for x, m, n in sorted({(r.d_acc, r.major, r.minor) for _, r in cross.iterrows()}):
        lx = x if lx is None or x - lx >= min_sep else lx + min_sep
        ax.annotate(f"{MODELS[m]['label']}\n{MODELS[n]['label']}",
                    xy=(lx, 0), xycoords=("data", "axes fraction"),
                    xytext=(0, -30), textcoords="offset points", rotation=90,
                    ha="center", va="top", color=MUTED, fontsize=5.5,
                    annotation_clip=False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False)
    for a, x0 in ((axs, 1.0), (ax, 0.0)):
        a.plot([x0, x0], [-0.015, 0.015], transform=a.transAxes, color=MUTED,
               lw=0.9, clip_on=False, marker=[(-0.5, -1), (0.5, 1)], ms=5, mew=0.9)
    present = set(summary.judge)
    jh = [Line2D([], [], marker="o", ls="-" if j in connected else "", color=col[j],
                 ms=5, label=lab(j))
          for j in reversed(order) if j in present]
    leg = ax.legend(handles=jh, frameon=False, loc="center left",
                    bbox_to_anchor=(1.02, 0.5))
    for t in leg.get_texts():
        t.set_color(INK2)
    ax.text(0.98, 0.04, note, transform=ax.transAxes, ha="right", color=MUTED, fontsize=7)
    save(fig, out / name)


def fig_gap_curves(summary, fits, models, out):
    note = "lines: sigmoid fits · self-pairs: 95% CI"
    gap_curve(summary, out, "2a_gap_curve_by_capability",
              {j: models.loc[j, "accuracy"] for j in JUDGES}, note=note, fits=fits)
    gap_curve(summary, out, "2b_gap_curve_by_release",
              {j: release_year(j) for j in JUDGES}, note=note, fits=fits)


def filtered_gap_curves(summary, models, out, keep, stem, note):
    """Gap-curve pair (capability/release colouring) over a run subset."""
    sub = summary[summary.apply(keep, axis=1)].copy()
    gap_curve(sub, out, f"{stem}_by_capability",
              {j: models.loc[j, "accuracy"] for j in JUDGES}, note=note, jitter=0.0006)
    gap_curve(sub, out, f"{stem}_by_release",
              {j: release_year(j) for j in JUDGES}, note=note, jitter=0.0006)


def fig_gap_curves_filtered(summary, models, out):
    acc = models["accuracy"]
    filtered_gap_curves(
        summary, models, out,
        lambda r: acc[r.judge] < min(acc[r.major], acc[r.minor]),
        "2d_gap_curve_weak_judge",
        "both pair members stronger than the judge")
    filtered_gap_curves(
        summary, models, out,
        lambda r: release_year(r.judge) < min(release_year(r.major), release_year(r.minor)),
        "2e_gap_curve_old_judge",
        "both pair members released after the judge")


MATCHED_SETS = [
    ("Qwen pairs (×3)", [("qwen3.5-122b-a10b", "qwen3.6-35b-a3b"),
                         ("qwen3.5-122b-a10b", "qwen3-32b"),
                         ("qwen3.6-35b-a3b", "qwen3-32b")]),
    ("Gemma-26B pairs (×2)", [("gemma-4-31b-it", "gemma-4-26b-a4b-it"),
                              ("gemini-3.1-flash-lite", "gemma-4-26b-a4b-it")]),
    ("2024/25 temporal pairs (×2)", [("claude-3-5-sonnet-0620", "llama-3.1-70b"),
                                     ("gpt-4.1", "mistral-medium")]),
]


def fig_matched_judges(verdicts, models, out):
    """2c: judge sets that all judged identical pairs, so pair composition is
    controlled. x = release, colour = capability. Each panel is one such set."""
    norm = plt.Normalize(models["accuracy"].min(), models["accuracy"].max())
    panels = []
    for title, shared in MATCHED_SETS:
        members = {m for p in shared for m in p}
        pts = []
        for j in JUDGES:
            if j in members:
                continue
            runs = [verdicts[(verdicts.judge == j) & (verdicts.major == m) & (verdicts.minor == n)]
                    for m, n in shared]
            if any(not len(r) for r in runs):
                continue
            scores = np.concatenate([counterbalanced(r) for r in runs])
            pts.append((release_year(j), *sem_ci(scores), j))
        if pts:
            panels.append((title, sorted(pts)))
    fig, axes = plt.subplots(1, len(panels), sharey=True, sharex=True,
                             figsize=(2.2 * len(panels) + 0.6, 2.7), layout="constrained")
    for ax, (title, pts) in zip(np.atleast_1d(axes), panels):
        for i, (x, m, lo, hi, j) in enumerate(pts):
            c = JRAMP(norm(models.loc[j, "accuracy"]))
            ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], fmt="o", ms=6,
                        color=c, ecolor=c, elinewidth=1.0, capsize=0, zorder=3)
            crowded = i and abs(x - pts[i - 1][0]) < 0.3 and abs(m - pts[i - 1][1]) < 0.08
            ax.annotate(MODELS[j]["label"], xy=(x, m), xytext=(0, -12 if crowded else 8),
                        textcoords="offset points", ha="center", color=INK2, fontsize=6.5)
        chance(ax)
        ax.set_title(title, color=INK2, fontsize=8)
        ax.set_xlabel("judge release year")
        style(ax)
    ax0 = np.atleast_1d(axes)[0]
    ax0.set_ylabel("judge accuracy (shared pairs)")
    ax0.set_ylim(0.25, 0.9)
    ax0.set_xlim(2024.4, 2026.6)
    ax0.set_xticks([2025, 2026])
    ax0.set_xticklabels(["2025", "2026"])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=JRAMP), ax=axes,
                      pad=0.015, shrink=0.9)
    cb.set_label("judge GPQA accuracy", color=INK2)
    cb.ax.tick_params(colors=MUTED, labelcolor=INK2)
    save(fig, out / "2c_matched_judges")


# --- 3. judge strength ---------------------------------------------------

def fig_judge_strength(verdicts, models, out):
    """Judge x pair score heatmap over the neutral-judge panel."""
    js = COMPARISONS["judge_strength"]
    acc = models["accuracy"]
    pairs = sorted({(m, n) for _, m, n in js},
                   key=lambda p: acc[p[0]] - acc[p[1]], reverse=True)
    judges = sorted({j for j, _, _ in js}, key=lambda j: acc[j])
    M = np.full((len(pairs), len(judges)), np.nan)
    for j, m, n in js:
        run = verdicts[(verdicts.judge == j) & (verdicts.major == m) & (verdicts.minor == n)]
        if len(run):
            M[pairs.index((m, n)), judges.index(j)] = counterbalanced(run).mean()
    keep = np.isfinite(M).any(axis=1)
    M, pairs = M[keep], [p for p, k in zip(pairs, keep) if k]

    fig, ax = plt.subplots(figsize=(0.62 * len(judges) + 2.4, 0.30 * len(pairs) + 1.4))
    im = ax.imshow(M, cmap=HEAT, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(judges)))
    ax.set_xticklabels([lab(j) for j in judges], rotation=35, ha="right")
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels([f"{lab(m)} v {lab(n)}  ·  Δ{acc[m] - acc[n]:.2f}"
                        for m, n in pairs])
    ax.set_xlabel("judge")
    ax.set_ylabel("pair")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color=INK2 if 0.2 < M[i, j] < 0.8 else "white")
    ax.tick_params(colors=MUTED, labelcolor=INK2)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label("judge accuracy (0.5 = chance)", color=INK2)
    cb.ax.tick_params(colors=MUTED, labelcolor=INK2)
    ax.set_title("blank = not judged yet · CIs in 3b", color=MUTED, fontsize=7, loc="right")
    save(fig, out / "3_judge_strength")


def fig_judge_strength_scatter(verdicts, models, out):
    """The 3_judge_strength data as a scatter: gap on x, judge strength as colour."""
    js = COMPARISONS["judge_strength"]
    acc = models["accuracy"]
    judges = sorted({j for j, _, _ in js}, key=lambda j: acc[j])
    col = {j: JRAMP(i / (len(judges) - 1)) for i, j in enumerate(judges)}
    rows = []
    for j, m, n in js:
        run = verdicts[(verdicts.judge == j) & (verdicts.major == m) & (verdicts.minor == n)]
        if len(run):
            s, lo, hi = sem_ci(counterbalanced(run))
            rows.append(dict(judge=j, major=m, minor=n, d_acc=acc[m] - acc[n],
                             score=s, lo=lo, hi=hi))
    d = pd.DataFrame(rows)
    jit = pair_jitter(d, width=0.003)
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for idx, r in d.iterrows():
        ax.errorbar(r.d_acc + jit[idx], r.score, yerr=[[r.score - r.lo], [r.hi - r.score]],
                    fmt="o", ms=4.5, color=col[r.judge], ecolor=col[r.judge],
                    elinewidth=0.8, capsize=0, alpha=0.9, zorder=3)
    chance(ax)
    ax.set_xlabel("capability gap (GPQA accuracy difference)")
    ax.set_ylabel("judge accuracy")
    ax.set_ylim(0, 1)
    style(ax)
    jh = [Line2D([], [], marker="o", ls="", color=col[j], ms=5, label=lab(j))
          for j in reversed(judges)]
    leg = ax.legend(handles=jh, frameon=False, loc="center left",
                    bbox_to_anchor=(1.02, 0.5))
    for t in leg.get_texts():
        t.set_color(INK2)
    ax.text(0.98, 0.05, "95% CI", transform=ax.transAxes, ha="right",
            color=MUTED, fontsize=7)
    save(fig, out / "3b_judge_strength_scatter")


# --- 4. length confound --------------------------------------------------

def fig_length(verdicts, summary, out):
    fig, (left, right) = plt.subplots(1, 2, figsize=(6.8, 2.4),
                                      gridspec_kw={"width_ratios": [1, 1], "wspace": 0.28})
    parsed = verdicts.dropna(subset=["picked_major", "len_major", "len_minor"])
    parsed = parsed[parsed.len_major != parsed.len_minor]
    chosen = np.where(parsed.picked == parsed.slot_of_major, parsed.len_major, parsed.len_minor)
    other = np.where(parsed.picked == parsed.slot_of_major, parsed.len_minor, parsed.len_major)
    parsed = parsed.assign(picked_longer=chosen > other)
    for j in JUDGES:
        d = parsed[parsed.judge == j]
        if not len(d):
            continue
        k = int(d.picked_longer.sum())
        lo, hi = wilson(k, len(d))
        dot_ci(left, JUDGES.index(j), k / len(d), lo, hi, JCOLOR[j])
        left.annotate(f"n={len(d)}", xy=(hi + 0.008, JUDGES.index(j)),
                      va="center", ha="left", color=MUTED, fontsize=7)
    left.set_yticks(range(len(JUDGES)))
    left.set_yticklabels([lab(j) for j in JUDGES])
    left.invert_yaxis()
    chance(left, 0.5, axis="x")
    left.set_xlabel("P(picked the longer transcript), all verdicts pooled")
    style(left, grid_axis="x")

    per_run = (verdicts.groupby(["judge", "major", "minor"])
               .apply(lambda d: (d.len_major - d.len_minor).mean() / 1000, include_groups=False)
               .rename("dlen").reset_index())
    m = summary.merge(per_run, on=["judge", "major", "minor"])
    for _, r in m.iterrows():
        right.plot(r.dlen, r.score, "o", ms=4.5, color=JCOLOR[r.judge],
                   mfc="white" if r.major == r.minor else JCOLOR[r.judge], mew=1.1)
    chance(right)
    right.axvline(0, color=MUTED, lw=0.8)
    right.set_xlim(-10, 5)
    right.set_xlabel("stronger − weaker mean length (k chars)")
    right.set_ylabel("judge accuracy")
    right.text(0.02, 0.04, "one point per run · colours as left · open = self-pair",
               transform=right.transAxes, color=MUTED, fontsize=7)
    style(right)
    save(fig, out / "4_length_confound")


# --- 5. item difficulty and discriminability -----------------------------

def fig_item_difficulty(verdicts, out,
                        irt_root=Path("irt_results/epoch_gpqa")):
    cross = verdicts[verdicts.major != verdicts.minor].dropna(subset=["picked_major"])
    per_item = (cross.groupby(["judge", "major", "minor", "item"])["picked_major"].mean()
                .groupby("item").mean().rename("score").reset_index())
    dif = pd.read_csv(irt_root / "gpqa_difficulties.csv")[["item", "b_1pl"]]
    disc = pd.read_csv(irt_root / "gpqa_discrimination.csv")[["item", "a_2pl"]]
    d = per_item.merge(dif, on="item").merge(disc, on="item")

    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(6.0, 2.4))
    for ax, col, xlabel in [(axes[0], "b_1pl", "item difficulty b (1PL)"),
                            (axes[1], "a_2pl", "item discrimination a (2PL)")]:
        ax.scatter(d[col], d.score, s=12, color=MUTED, alpha=0.45, lw=0, zorder=2)
        for _, cell in d.groupby(pd.qcut(d[col], 5), observed=True):
            m, lo, hi = sem_ci(cell.score.to_numpy())
            ax.errorbar(cell[col].mean(), m, yerr=[[m - lo], [hi - m]], fmt="o", ms=6,
                        color=PALETTE[0], ecolor=PALETTE[0], elinewidth=1.2, capsize=0,
                        zorder=4, markeredgecolor="white", markeredgewidth=0.8)
        slope, intercept = np.polyfit(d[col], d.score, 1)
        xr = np.array([d[col].min(), d[col].max()])
        ax.plot(xr, intercept + slope * xr, color=PALETTE[0], lw=1.4, zorder=3)
        chance(ax)
        ax.set_xlabel(xlabel)
        ax.set_ylim(0, 1)
        style(ax)
    axes[0].set_ylabel("mean judge accuracy")
    save(fig, out / "5_item_difficulty")


# --- 6. temporal: judging across release-date distance -------------------

def fig_temporal(verdicts, out):
    rungs = list(dict.fromkeys((maj, mnr) for _, maj, mnr in COMPARISONS["temporal"]))
    judges = list(dict.fromkeys(j for j, _, _ in COMPARISONS["temporal"]))
    date = {m: pd.to_datetime(MODELS[m]["release"]) for m in
            {x for r in rungs for x in r} | set(judges)}
    rung_marker = dict(zip(rungs, ["o", "s", "^", "D", "v"]))

    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    for j in judges:
        pts = []
        for maj, mnr in rungs:
            run = verdicts[(verdicts.judge == j) & (verdicts.major == maj) & (verdicts.minor == mnr)]
            if not len(run):
                continue
            pair_date = date[maj] + (date[mnr] - date[maj]) / 2
            lead = (pair_date - date[j]).days / 30.44
            m, lo, hi = sem_ci(counterbalanced(run))
            pts.append((lead, m, lo, hi, rung_marker[(maj, mnr)]))
        pts.sort()
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "-", color=JCOLOR[j], lw=1.2,
                alpha=0.7, zorder=2)
        for lead, m, lo, hi, marker in pts:
            ax.errorbar(lead, m, yerr=[[m - lo], [hi - m]], fmt=marker, ms=5.5,
                        color=JCOLOR[j], ecolor=JCOLOR[j], elinewidth=0.9, capsize=0, zorder=3)
    chance(ax)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("pair release − judge release (months)")
    ax.set_ylabel("judge accuracy")
    ax.set_ylim(0, 1)
    style(ax)
    jh = [Line2D([], [], marker="o", ls="-", color=JCOLOR[j], ms=5,
                 label=f"judge: {lab(j)}") for j in judges]
    rh = [Line2D([], [], marker=rung_marker[r], ls="", color=MUTED, ms=5,
                 label=f"{lab(r[0])} v {lab(r[1])}") for r in rungs]
    pad = [Line2D([], [], ls="", label="")] * (len(rh) - len(jh))
    leg = ax.legend(handles=jh + pad + rh, frameon=False, loc="upper left",
                    bbox_to_anchor=(-0.02, -0.24), ncol=2, columnspacing=1.5)
    for t in leg.get_texts():
        t.set_color(INK2)
    ax.text(0.99, 0.97, "GPT-5.5 pair is length-confounded (fig 4)",
            transform=ax.transAxes, ha="right", va="top", color=MUTED, fontsize=7)
    save(fig, out / "6_temporal")


# --- 7. score vs release-date distance, every run ------------------------

def fig_release_gap(summary, out):
    """Fig 6's release-distance x-axis, but over every run, not just the rungs."""
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    for _, r in summary.iterrows():
        is_self = r.major == r.minor
        dyr = (release_year(r.major) + release_year(r.minor)) / 2 - release_year(r.judge)
        ax.plot(dyr, r.score, "o", ms=4.5, color=JCOLOR[r.judge],
                mfc="white" if is_self else JCOLOR[r.judge],
                mew=1.1, alpha=0.9, zorder=3)
    chance(ax)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("pair mean release − judge release (years)")
    ax.set_ylabel("judge accuracy")
    ax.set_ylim(0, 1)
    style(ax)
    jh = [Line2D([], [], marker="o", ls="", color=JCOLOR[j], ms=5, label=lab(j))
          for j in JUDGES]
    jh.append(Line2D([], [], marker="o", ls="", mfc="white", mec=INK2, ms=5,
                     label="self-pair (control)"))
    leg = ax.legend(handles=jh, frameon=False, loc="center left",
                    bbox_to_anchor=(1.02, 0.5), title="judge")
    for t in leg.get_texts():
        t.set_color(INK2)
    leg.get_title().set_color(INK2)
    ax.text(0.02, 0.03, "one point per run",
            transform=ax.transAxes, ha="left", va="bottom", color=MUTED, fontsize=7)
    save(fig, out / "7_release_gap")


# --- 0. model accuracy ---------------------------------------------------

def fig_model_accuracy(models, out):
    m = models.sort_values("accuracy").reset_index()
    arm_col = {"A": PALETTE[0], "B": PALETTE[1]}
    fig, ax = plt.subplots(figsize=(COL, 0.19 * len(m) + 0.9))
    for y, r in m.iterrows():
        k = int(round(r.accuracy * r.n_samples))
        lo, hi = wilson(k, int(r.n_samples))
        dot_ci(ax, y, r.accuracy, lo, hi, arm_col[r.arm])
        ax.annotate(f"{r.accuracy:.0%} ({r.n_epochs} epochs)", xy=(hi + 0.04, y),
                    va="center", ha="left", color=MUTED, fontsize=7)
    ax.set_yticks(range(len(m)))
    ax.set_yticklabels([lab(r.model) for _, r in m.iterrows()])
    chance(ax, 0.25, axis="x")
    ax.set_xlabel("GPQA Diamond accuracy")
    ax.set_xlim(0, 1)
    style(ax, grid_axis="x")
    handles = [Line2D([], [], marker="o", ls="", color=arm_col["A"], ms=5.5,
                      label="arm A (our collection)"),
               Line2D([], [], marker="o", ls="", color=arm_col["B"], ms=5.5,
                      label="arm B (Epoch logs)")]
    leg = ax.legend(handles=handles, frameon=False, loc="lower left",
                    bbox_to_anchor=(0, 1.005), ncol=2, handletextpad=0.4,
                    columnspacing=1.0)
    for t in leg.get_texts():
        t.set_color(INK2)
    ax.text(1.0, -0.24, "error bars: 95% CI · dashed: chance", transform=ax.transAxes,
            ha="right", va="top", color=MUTED, fontsize=7)
    save(fig, out / "0_model_accuracy")


def render_all(verdicts, summary, fits, models, out):
    out.mkdir(parents=True, exist_ok=True)
    for stale in list(out.glob("*.png")) + list(out.glob("*.svg")):
        stale.unlink()
    fig_model_accuracy(models, out)
    fig_run_overview(summary, out)
    fig_gap_curves(summary, fits, models, out)
    fig_gap_curves_filtered(summary, models, out)
    fig_matched_judges(verdicts, models, out)
    fig_judge_strength(verdicts, models, out)
    fig_judge_strength_scatter(verdicts, models, out)
    fig_length(verdicts, summary, out)
    fig_item_difficulty(verdicts, out)
    fig_temporal(verdicts, out)
    fig_release_gap(summary, out)
    print(f"{len(list(out.glob('*.png')))} figures -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, default=Path("judge_results/judge-pilot"))
    args = ap.parse_args()

    verdicts = pd.read_csv(args.dir / "verdicts.csv")
    models = pd.read_csv(args.dir / "models.csv").set_index("model")
    summary = pd.read_csv(args.dir / "summary.csv")
    fits = pd.read_csv(args.dir / "judge_fits.csv")

    init_labels(models)
    render_all(verdicts, summary, fits, models, args.dir / "figures")


if __name__ == "__main__":
    main()
