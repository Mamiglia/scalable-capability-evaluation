"""Generate the gpqa-pilot-rd2 figures from the stored tables.

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
from scipy.stats import pearsonr

from comparisons import COMPARISONS, MODELS
from plots import (COL, HEAT, INK, INK2, JRAMP, MUTED, PALETTE, SEQ,
                   chance, dot_ci, legend, plt, save, style)
from stats import counterbalanced, perm_p, sem_ci, sigmoid_fit, stars, wilson

JUDGES = ["gemini-3.1-flash-lite", "qwen3.5-122b-a10b", "gemma-4-31b-it",
          "qwen3.6-35b-a3b", "qwen3-32b", "gemini-2.5-flash", "gpt-4o",
          "gemma-4-26b-a4b-it", "gpt-4.1-nano", "qwen-2.5-72b-instruct"]
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
              note="lines connect a judge's runs · self-pairs: one point per judge",
              fits=None, jitter=0.003):
    """Score vs raw-accuracy gap; fixed judge colours, legend ordered by the
    variant's value (capability or release), self-pairs open in their own
    broken-axis section. A ramp stopped working at 10 judges: the middle
    shades were indistinguishable.
    """
    order = sorted(JUDGES, key=lambda j: (jvalue[j], j))
    col = JCOLOR

    fig, (axs, ax) = plt.subplots(1, 2, sharey=True, figsize=(6.2, 3.4),
                                  gridspec_kw={"width_ratios": [1, 11], "wspace": 0.05})
    xmax = summary.d_acc.max() * 1.05
    # One point per judge: its self-pair runs aggregated (CI over run means).
    selfs = summary[summary.major == summary.minor]
    sj = [(j, *sem_ci(block["score"].to_numpy()))
          for j, block in selfs.groupby("judge")]
    sj.sort(key=lambda t: order.index(t[0]))
    for i, (j, m, lo, hi) in enumerate(sj):
        x = (i - (len(sj) - 1) / 2) * (1.0 / max(len(sj), 1))
        axs.errorbar(x, m, yerr=[[m - lo], [hi - m]],
                     fmt="o", ms=4.5, color=col[j], ecolor=col[j],
                     elinewidth=0.9, capsize=0, mfc="white", mew=1.1, zorder=3)
    cross = summary[summary.major != summary.minor]
    jit = pair_jitter(cross, width=jitter)
    connected = set()
    for j, block in cross.groupby("judge"):
        xy = sorted((r.d_acc + jit[idx], r.score) for idx, r in block.iterrows())
        if fits is None and len(xy) > 1:
            connected.add(j)
            ax.plot([p[0] for p in xy], [p[1] for p in xy], "-", color=col[j],
                    lw=1.2, zorder=2)
        ax.plot([p[0] for p in xy], [p[1] for p in xy], "o", ms=4.5, color=col[j],
                zorder=3)
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
    # Rotated pair labels under the axis: top-aligned, staggered on two
    # levels, dodged apart where gaps cluster.
    min_sep = xmax * 0.022
    lx = None
    for i, (x, m, n) in enumerate(sorted({(r.d_acc, r.major, r.minor)
                                          for _, r in cross.iterrows()})):
        lx = x if lx is None or x - lx >= min_sep else lx + min_sep
        ax.annotate(f"{MODELS[m]['label']} v {MODELS[n]['label']}",
                    xy=(lx, 0), xycoords=("data", "axes fraction"),
                    xytext=(0, -30 - (i % 2) * 16), textcoords="offset points",
                    rotation=90, rotation_mode="anchor", ha="right", va="center",
                    color=MUTED, fontsize=5.5, annotation_clip=False)
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
    note = ("lines: sigmoid fits · self-pairs: 95% CI" if fits is not None
            else "lines connect a judge's runs · self-pairs: 95% CI")
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
            # Unit = comparison: CI over the shared runs' means, not pooled items.
            means = [counterbalanced(r).mean() for r in runs]
            pts.append((release_year(j), *sem_ci(means), j))
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

def fig_judge_strength(summary, models, out):
    """Judge x pair accuracy heatmap over every cross-model run."""
    acc = models["accuracy"]
    cross = summary[summary.major != summary.minor]
    pairs = sorted({(r.major, r.minor) for _, r in cross.iterrows()},
                   key=lambda p: acc[p[0]] - acc[p[1]], reverse=True)
    judges = sorted(set(cross.judge), key=lambda j: acc[j])
    M = np.full((len(pairs), len(judges)), np.nan)
    for _, r in cross.iterrows():
        M[pairs.index((r.major, r.minor)), judges.index(r.judge)] = r.score
    keep = np.isfinite(M).any(axis=1)
    M, pairs = M[keep], [p for p, k in zip(pairs, keep) if k]

    fig, ax = plt.subplots(figsize=(0.62 * len(judges) + 2.4, 0.30 * len(pairs) + 1.4))
    im = ax.imshow(M, cmap=HEAT, vmin=0, vmax=1, aspect="auto")
    for i, (m, n) in enumerate(pairs):
        for k, j in enumerate(judges):
            if j in (m, n) and np.isfinite(M[i, k]):
                ax.add_patch(plt.Rectangle((k - 0.5, i - 0.5), 1, 1, fill=False,
                                           edgecolor=INK, lw=1.4, zorder=4))
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
    ax.set_title("blank = not judged · outlined = judge is a pair member",
                 color=MUTED, fontsize=7, loc="right")
    save(fig, out / "3_judge_strength")


def fig_judge_strength_scatter(summary, models, out):
    """3b: the judge-strength heatmap with continuous axes -- judge strength
    on x, pair gap on y, accuracy as point colour. All cross-model runs;
    black edge marks judge-in-pair runs."""
    acc = models["accuracy"]
    cross = summary[summary.major != summary.minor]
    pts = [(acc[r.judge], r.d_acc, r.score, r.judge in (r.major, r.minor))
           for _, r in cross.iterrows()]
    if not pts:
        return
    xs, ys, cs, own = zip(*pts)
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    sc = ax.scatter(xs, ys, c=cs, cmap=HEAT, vmin=0, vmax=1, s=52,
                    edgecolors=[INK if o else "white" for o in own],
                    linewidths=0.8, zorder=3)
    ax.text(0.98, 0.97, "black edge: judge is a pair member",
            transform=ax.transAxes, ha="right", va="top", color=MUTED, fontsize=7)
    ax.set_xlabel("judge GPQA accuracy")
    ax.set_ylabel("pair gap (GPQA accuracy difference)")
    style(ax)
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label("judge accuracy (0.5 = chance)", color=INK2)
    cb.ax.tick_params(colors=MUTED, labelcolor=INK2)
    save(fig, out / "3b_judge_strength_scatter")


# --- 4. length confound --------------------------------------------------

def picked_longer_rows(verdicts):
    parsed = verdicts.dropna(subset=["picked_major", "len_major", "len_minor"])
    parsed = parsed[parsed.len_major != parsed.len_minor]
    chosen = np.where(parsed.picked == parsed.slot_of_major, parsed.len_major, parsed.len_minor)
    other = np.where(parsed.picked == parsed.slot_of_major, parsed.len_minor, parsed.len_major)
    return parsed.assign(picked_longer=chosen > other)


def fig_length(verdicts, summary, out):
    fig, ax = plt.subplots(figsize=(4.4, 2.9))
    per_run = (verdicts.groupby(["judge", "major", "minor"])
               .apply(lambda d: (d.len_major - d.len_minor).mean() / 1000, include_groups=False)
               .rename("dlen").reset_index())
    m = summary.merge(per_run, on=["judge", "major", "minor"])
    present = []
    for _, r in m.iterrows():
        if r.judge not in present:
            present.append(r.judge)
        ax.plot(r.dlen, r.score, "o", ms=4.5, color=JCOLOR[r.judge],
                mfc="white" if r.major == r.minor else JCOLOR[r.judge], mew=1.1)
    chance(ax)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_xlim(-10, 5)
    ax.set_xlabel("stronger − weaker mean length (k chars)")
    ax.set_ylabel("judge accuracy")
    ax.text(0.02, 0.04, "one point per run · open = self-pair",
            transform=ax.transAxes, color=MUTED, fontsize=7)
    style(ax)
    jh = [Line2D([], [], marker="o", ls="", color=JCOLOR[j], ms=5, label=lab(j))
          for j in JUDGES if j in present]
    leg = ax.legend(handles=jh, frameon=False, loc="center left",
                    bbox_to_anchor=(1.02, 0.5), title="judge")
    for t in leg.get_texts():
        t.set_color(INK2)
    leg.get_title().set_color(INK2)
    save(fig, out / "4_length_confound")


# --- 9-11. bias diagnostics ----------------------------------------------

def judge_rate_rows(ax, verdicts, value):
    """Per-judge dot+CI rows of a per-run rate; returns judges plotted."""
    present = []
    for j in JUDGES:
        d = verdicts[verdicts.judge == j]
        if not len(d):
            continue
        rates = [value(g) for _, g in d.groupby(["major", "minor"])]
        m, lo, hi = sem_ci(rates)
        dot_ci(ax, len(present), m, lo, hi, JCOLOR[j])
        ax.annotate(f"n={len(rates)} runs", xy=(hi + 0.008, len(present)),
                    va="center", ha="left", color=MUTED, fontsize=7)
        present.append(j)
    ax.set_yticks(range(len(present)))
    ax.set_yticklabels([lab(j) for j in present])
    ax.invert_yaxis()
    chance(ax, 0.5, axis="x")
    style(ax, grid_axis="x")
    return present


def fig_position_bias(verdicts, out):
    """9: first-slot pick rate per judge, over all its runs."""
    parsed = verdicts[verdicts.picked.isin(["1", "2"])]
    if not len(parsed):
        return
    fig, ax = plt.subplots(figsize=(COL, 2.6))
    judge_rate_rows(ax, parsed, lambda g: (g.picked == "1").mean())
    ax.set_xlabel("P(picked slot 1)")
    ax.text(1.0, -0.22, "95% CI over runs", transform=ax.transAxes,
            ha="right", va="top", color=MUTED, fontsize=7)
    save(fig, out / "9_position_bias")


def fig_length_bias(verdicts, out):
    """10: P(picked the longer transcript) per judge, over all its runs."""
    parsed = picked_longer_rows(verdicts)
    if not len(parsed):
        return
    fig, ax = plt.subplots(figsize=(COL, 2.6))
    judge_rate_rows(ax, parsed, lambda g: g.picked_longer.mean())
    ax.set_xlabel("P(picked the longer transcript)")
    ax.text(1.0, -0.22, "95% CI over runs · equal-length items excluded",
            transform=ax.transAxes, ha="right", va="top", color=MUTED, fontsize=7)
    save(fig, out / "10_length_bias")


def fig_capability_vs_length(verdicts, models, out):
    """11: model GPQA accuracy vs its mean transcript length as judged."""
    lens = pd.concat([
        verdicts[["major", "len_major"]].rename(columns={"major": "model", "len_major": "len"}),
        verdicts[["minor", "len_minor"]].rename(columns={"minor": "model", "len_minor": "len"}),
    ]).dropna()
    if not len(lens):
        return
    mean_len = lens.groupby("model")["len"].mean() / 1000
    arm_col = {"A": PALETTE[0], "B": PALETTE[1]}
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    pts = sorted((models.loc[m, "accuracy"], v, m) for m, v in mean_len.items())
    offsets = [(0, 8), (0, -12), (0, 20), (0, -24)]
    for i, (x, y, m) in enumerate(pts):
        ax.plot(x, y, "o", ms=5.5, color=arm_col[models.loc[m, "arm"]], zorder=3)
        ax.annotate(MODELS[m]["label"], xy=(x, y), xytext=offsets[i % 4],
                    textcoords="offset points", ha="center", color=INK2, fontsize=6)
    ax.set_yscale("log")
    ax.set_yticks([0.5, 1, 2, 4, 8, 12])
    ax.set_yticklabels(["0.5", "1", "2", "4", "8", "12"])
    ax.set_xlabel("GPQA Diamond accuracy")
    ax.set_ylabel("mean transcript length (k chars)")
    ax.margins(x=0.08, y=0.18)
    style(ax)
    handles = [Line2D([], [], marker="o", ls="", color=arm_col["A"], ms=5.5,
                      label="Collected data"),
               Line2D([], [], marker="o", ls="", color=arm_col["B"], ms=5.5,
                      label="Epoch data")]
    leg = ax.legend(handles=handles, frameon=False, loc="lower left",
                    bbox_to_anchor=(0, 1.005), ncol=2, handletextpad=0.4,
                    columnspacing=1.0)
    for t in leg.get_texts():
        t.set_color(INK2)
    ax.text(1.0, 1.02, "length as the judge saw it", transform=ax.transAxes,
            ha="right", va="bottom", color=MUTED, fontsize=7)
    save(fig, out / "11_capability_vs_length")


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
        r, p = pearsonr(d[col], d.score)
        ax.text(0.03, 0.05, f"r={r:+.2f}, p={p:.2g}, n={len(d)} items\n"
                "line: fit on all items (grey)",
                transform=ax.transAxes, color=MUTED, fontsize=7)
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

    series = []
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
        if pts:
            series.append((j, sorted(pts)))
    if not series:
        return

    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    for j, pts in series:
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


# --- 8. truncation control -----------------------------------------------

def truncation_series(trunc, verdicts):
    """(judge, major, minor) -> {shown %: per-item scores}; 100% is the
    untruncated run of the same judge and pair."""
    series = {}
    for (j, m, n), block in trunc.groupby(["judge", "major", "minor"]):
        d = {int(round(f * 100)): counterbalanced(fb)
             for f, fb in block.groupby("truncate_frac")}
        full = verdicts[(verdicts.judge == j) & (verdicts.major == m) & (verdicts.minor == n)]
        if len(full):
            d[100] = counterbalanced(full)
        series[(j, m, n)] = d
    return series


def fig_truncation(trunc, verdicts, out):
    """Accuracy vs fraction of the shorter transcript shown (both sides cut
    to the same length, answers removed)."""
    series = truncation_series(trunc, verdicts)
    if not series:
        return
    fracs = sorted({f for d in series.values() for f in d})

    fig, ax = plt.subplots(figsize=(5.8, 3.2))
    pairs = sorted({(m, n) for _, m, n in series})
    pcol = {p: PALETTE[i % len(PALETTE)] for i, p in enumerate(pairs)}
    # One line per pair, averaged over the judges that judged it.
    by_pair = {}
    for (j, m, n), d in series.items():
        for f, s in d.items():
            by_pair.setdefault(((m, n), f), []).append(s.mean())
    for p in pairs:
        xs = sorted(f for (pp, f) in by_pair if pp == p)
        ys = [np.mean(by_pair[(p, f)]) for f in xs]
        ax.plot(xs, ys, "-o", color=pcol[p], ms=3.5, lw=1.0, zorder=2)
    # The unit is the comparison (judge x pair), not the item: run means only.
    run_means = {f: [d[f].mean() for d in series.values() if f in d] for f in fracs}
    mid, lo, hi = zip(*(sem_ci(run_means[f]) for f in fracs))
    ax.fill_between(fracs, lo, hi, color=INK, alpha=0.10, lw=0)
    ax.plot(fracs, mid, "-o", color=INK, lw=2.0, ms=5, zorder=4)
    chance(ax)
    ax.set_xticks(fracs)
    ax.set_xticklabels([f"{f}%" for f in fracs])
    ax.set_xlabel("fraction of the shorter transcript shown (both sides cut equal)")
    ax.set_ylabel("judge accuracy")
    ax.set_ylim(0, 1.0)
    style(ax)
    handles = [Line2D([], [], marker="o", ls="-", color=INK, lw=2, ms=5, label="all runs")]
    handles += [Line2D([], [], marker="o", ls="-", color=pcol[p], ms=4,
                       label=f"{lab(p[0])} v {lab(p[1])}") for p in pairs]
    leg = ax.legend(handles=handles, frameon=False, loc="upper left",
                    bbox_to_anchor=(-0.02, -0.18), ncol=2, columnspacing=1.4)
    for t in leg.get_texts():
        t.set_color(INK2)
    ax.text(0.99, 0.03, "one line per pair, averaged over its judges · 100% = untruncated\n"
            f"band: 95% CI over judge+pair runs (n={len(series)})",
            transform=ax.transAxes, ha="right", color=MUTED, fontsize=7)
    save(fig, out / "8_truncation_control")


def fig_truncation_gap(trunc, verdicts, models, out):
    """Gap-curve view of the truncation control: accuracy vs pair gap, pooled
    over judges, one line per truncation level."""
    series = truncation_series(trunc, verdicts)
    if not series:
        return
    acc = models["accuracy"]
    levels = sorted({f for d in series.values() for f in d})
    pooled = {}
    for (j, m, n), d in series.items():
        for f, s in d.items():
            pooled.setdefault(((m, n), f), []).append(s)
    pairs = sorted({p for p, _ in pooled}, key=lambda p: acc[p[0]] - acc[p[1]])

    trunc_levels = [f for f in levels if f < 100]
    col = {f: SEQ(i / max(len(trunc_levels) - 1, 1)) for i, f in enumerate(trunc_levels)}
    col[100] = INK

    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    for f in levels:
        xs, ys = [], []
        for p in pairs:
            if (p, f) in pooled:
                xs.append(acc[p[0]] - acc[p[1]])
                # Unit = comparison: mean of the judges' run means.
                ys.append(np.mean([s.mean() for s in pooled[(p, f)]]))
        label = f"{f}%" if f < 100 else "100% (full, unequal)"
        ax.plot(xs, ys, "-o", color=col[f], ms=4.5, lw=1.6 if f == 100 else 1.3,
                label=label, zorder=4 if f == 100 else 3)
    chance(ax)
    ax.set_xlabel("capability gap (GPQA accuracy difference)")
    ax.set_ylabel("judge accuracy")
    ax.set_ylim(0, 1)
    style(ax)
    leg = ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5),
                    title="fraction of shorter\ntranscript shown")
    for t in leg.get_texts():
        t.set_color(INK2)
    leg.get_title().set_color(INK2)
    min_sep = (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.03
    lx = None
    for i, p in enumerate(pairs):
        x = acc[p[0]] - acc[p[1]]
        lx = x if lx is None or x - lx >= min_sep else lx + min_sep
        ax.annotate(f"{MODELS[p[0]]['label']} v {MODELS[p[1]]['label']}",
                    xy=(lx, 0), xycoords=("data", "axes fraction"),
                    xytext=(0, -30 - (i % 2) * 16), textcoords="offset points",
                    rotation=90, rotation_mode="anchor", ha="right", va="center",
                    color=MUTED, fontsize=5.5, annotation_clip=False)
    ax.text(0.99, 0.03, "pooled over judges",
            transform=ax.transAxes, ha="right", color=MUTED, fontsize=7)
    save(fig, out / "8a_truncation_gap_curve")


# --- 0. model accuracy ---------------------------------------------------

def fig_model_accuracy(models, out):
    m = models[models.arm.isin(["A", "B"])].sort_values("accuracy").reset_index()
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
                      label="Collected data"),
               Line2D([], [], marker="o", ls="", color=arm_col["B"], ms=5.5,
                      label="Epoch data")]
    leg = ax.legend(handles=handles, frameon=False, loc="lower left",
                    bbox_to_anchor=(0, 1.005), ncol=2, handletextpad=0.4,
                    columnspacing=1.0)
    for t in leg.get_texts():
        t.set_color(INK2)
    ax.text(1.0, -0.24, "error bars: 95% CI · dashed: chance", transform=ax.transAxes,
            ha="right", va="top", color=MUTED, fontsize=7)
    save(fig, out / "0_model_accuracy")


def render_all(verdicts, trunc, summary, fits, models, out):
    out.mkdir(parents=True, exist_ok=True)
    for stale in list(out.glob("*.png")) + list(out.glob("*.svg")):
        stale.unlink()
    fig_model_accuracy(models, out)
    fig_run_overview(summary, out)
    fig_gap_curves(summary, fits, models, out)
    fig_gap_curves_filtered(summary, models, out)
    fig_matched_judges(verdicts, models, out)
    fig_judge_strength(summary, models, out)
    fig_judge_strength_scatter(summary, models, out)
    fig_length(verdicts, summary, out)
    fig_item_difficulty(verdicts, out)
    fig_temporal(verdicts, out)
    fig_release_gap(summary, out)
    fig_truncation(trunc, verdicts, out)
    fig_truncation_gap(trunc, verdicts, models, out)
    fig_position_bias(verdicts, out)
    fig_length_bias(verdicts, out)
    fig_capability_vs_length(verdicts, models, out)
    print(f"{len(list(out.glob('*.png')))} figures -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, default=Path("judge_results/gpqa-pilot-rd2"))
    args = ap.parse_args()

    verdicts = pd.read_csv(args.dir / "verdicts.csv")
    models = pd.read_csv(args.dir / "models.csv").set_index("model")
    summary = pd.read_csv(args.dir / "summary.csv")
    fits = pd.read_csv(args.dir / "judge_fits.csv")

    # Judge-only models (no transcripts) get rows from comparisons.py metadata.
    for m, meta in MODELS.items():
        if m not in models.index and "accuracy" in meta:
            models.loc[m, ["arm", "accuracy", "logit"]] = [
                "judge", meta["accuracy"],
                np.log(meta["accuracy"] / (1 - meta["accuracy"]))]

    trunc = verdicts[verdicts.truncate_frac.notna()]
    verdicts = verdicts[verdicts.truncate_frac.isna()]
    summary_full = summary[summary.truncate_frac.isna()]

    init_labels(models)
    render_all(verdicts, trunc, summary_full, fits, models, args.dir / "figures")

    # The same figures over our own transcripts only (no Epoch pairs). The
    # sigmoid fits carry over: the panel runs they use are all arm A.
    arm = models["arm"]
    own = lambda df: df[df.major.map(arm).eq("A") & df.minor.map(arm).eq("A")]
    render_all(own(verdicts), own(trunc), own(summary_full), fits, models,
               args.dir / "figures" / "own_data")

    # And per truncation level: the whole suite on that level's runs alone
    # (connected lines instead of sigmoid fits; no nested truncation figures).
    none = trunc.iloc[0:0]
    for f in sorted(trunc.truncate_frac.unique()):
        render_all(trunc[trunc.truncate_frac == f], none,
                   summary[summary.truncate_frac == f], None, models,
                   args.dir / "figures" / f"trunc-{int(f * 100)}")


if __name__ == "__main__":
    main()
