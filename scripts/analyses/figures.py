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
from plots import (INK, INK2, MUTED, PALETTE, SEQ_BLUE, SEQ_ORANGE, chance, dot_ci,
                   legend, plt, save, style)
from stats import counterbalanced, sem_ci, wilson

JUDGES = ["gemini-3.1-flash-lite", "qwen3.5-122b-a10b", "gemma-4-31b-it",
          "qwen3.6-35b-a3b", "qwen3-32b", "gemini-2.5-flash", "gpt-4o"]
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
    fig, ax = plt.subplots(figsize=(5.2, 0.16 * len(s) + 1.4))
    for y, (_, r) in zip(ys, s.iterrows()):
        is_self = r.major == r.minor
        note = f"{r.score:.2f} (n={r.n_items}) {r.stars}"
        if is_self:
            note = f"slot-A {r.slot_a:.2f}" + ("  ok" if r.slot_ok else "  position bias")
        dot_ci(ax, y, r.score, r.score_lo, r.score_hi, JCOLOR[r.judge],
               note=note, open_marker=is_self)
    for j in s.judge.unique():
        block = ys[(s.judge == j).to_numpy()]
        ax.annotate(f"judge: {lab(j)}", xy=(-0.02, block.min() - 0.9),
                    xycoords=("axes fraction", "data"), ha="right", va="center",
                    color=INK2, fontsize=7.5, fontweight="bold", annotation_clip=False)
    ax.set_yticks(ys)
    ax.set_yticklabels([pair_label(r) for _, r in s.iterrows()], fontsize=6.5)
    ax.invert_yaxis()
    chance(ax, 0.5, axis="x")
    ax.set_xlabel("judge score (counterbalanced over A/B orders)")
    ax.set_xlim(0, 1)
    style(ax, grid_axis="x")
    handles = [Line2D([], [], marker="o", ls="-", color=INK2, ms=5.5, label="judge score, 95% CI"),
               Line2D([], [], marker="o", ls="", mfc="white", mec=INK2, ms=5.5,
                      label="self-pair control (should sit at chance)")]
    leg = ax.legend(handles=handles, frameon=False, loc="lower left",
                    bbox_to_anchor=(-0.35, 1.005), fontsize=6.5, ncol=2,
                    handletextpad=0.4, columnspacing=1.2)
    for t in leg.get_texts():
        t.set_color(INK2)
    save(fig, out / "1_run_overview")


# --- 2. gap curve, judges coloured by a scalar ---------------------------

def gap_curve(summary, fits, out, name, jvalue, cmap):
    """Score vs raw-accuracy gap; one colour ramp over judges, self-pairs open.

    Ramp steps are rank-spaced, not value-proportional: judge values cluster
    (three fitted judges within .72-.85 accuracy, five judges all 2026), and
    proportional shades made their lines indistinguishable.
    """
    order = sorted(JUDGES, key=lambda j: (jvalue[j], j))
    col = {j: cmap(i / (len(order) - 1)) for i, j in enumerate(order)}
    fitted = {f.judge: f for _, f in fits.iterrows()}
    # Solid dark = strongest/newest fitted judge, dashes lighten down the
    # ranking; each pattern repeats in its legend entry.
    ls = {j: s for j, s in zip([j for j in reversed(order) if j in fitted],
                               ["-", (0, (5, 2)), (0, (2, 1.5))])}

    # Broken x axis: self-pairs live in their own section on the left.
    fig, (axs, ax) = plt.subplots(1, 2, sharey=True, figsize=(5.0, 3.2),
                                  gridspec_kw={"width_ratios": [1, 11], "wspace": 0.05})
    xmax = summary.d_acc.max() * 1.05
    selfs = summary[summary.major == summary.minor].reset_index()
    for i, r in selfs.iterrows():
        x = (i - (len(selfs) - 1) / 2) * 0.25
        axs.errorbar(x, r.score, yerr=[[r.score - r.score_lo], [r.score_hi - r.score]],
                     fmt="o", ms=4.5, color=col[r.judge], ecolor=col[r.judge],
                     elinewidth=0.9, capsize=0, zorder=3)
    for _, r in summary[summary.major != summary.minor].iterrows():
        c = col[r.judge]
        ax.errorbar(r.d_acc, r.score, yerr=[[r.score - r.score_lo], [r.score_hi - r.score]],
                    fmt="o", ms=4.5, color=c, ecolor=c, elinewidth=0.9, capsize=0,
                    alpha=0.9, zorder=3)
    x = np.linspace(0, xmax, 100)
    for f in fitted.values():
        ax.plot(x, 1 / (1 + np.exp(-(f.a + f.d * x))), color=col[f.judge],
                ls=ls[f.judge], lw=1.5, zorder=2)
    for a in (axs, ax):
        chance(a)
        a.set_ylim(0, 1)
        style(a)
    axs.set_xlim(-0.6, 0.6)
    axs.set_xticks([0])
    axs.set_xticklabels(["self\npairs"])
    axs.set_ylabel("judge score")
    ax.set_xlim(0, xmax)
    ax.set_xlabel("capability gap (GPQA accuracy difference)")
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False)
    # break marks at the join
    for a, x0 in ((axs, 1.0), (ax, 0.0)):
        a.plot([x0, x0], [-0.015, 0.015], transform=a.transAxes, color=MUTED,
               lw=0.9, clip_on=False, marker=[(-0.5, -1), (0.5, 1)], ms=5, mew=0.9)
    jh = [Line2D([], [], marker="o", ls=ls.get(j, ""), color=col[j], ms=5,
                 label=lab(j))
          for j in order]
    leg = ax.legend(handles=jh, frameon=False, loc="lower right", fontsize=6)
    for t in leg.get_texts():
        t.set_color(INK2)
    ax.text(0.98, 0.06, "lines: sigmoid fit on gap-ladder runs · error bars: 95% CI",
            transform=ax.transAxes, ha="right", color=MUTED, fontsize=6)
    save(fig, out / name)


def fig_gap_curves(summary, fits, models, out):
    gap_curve(summary, fits, out, "2a_gap_curve_by_capability",
              {j: models.loc[j, "accuracy"] for j in JUDGES}, SEQ_BLUE)
    gap_curve(summary, fits, out, "2b_gap_curve_by_release",
              {j: release_year(j) for j in JUDGES}, SEQ_ORANGE)


# --- 3. judge strength ---------------------------------------------------

def fig_judge_strength(verdicts, models, out):
    pairs = list(dict.fromkeys((maj, mnr) for _, maj, mnr in COMPARISONS["judge_strength"]))
    fig, axes = plt.subplots(1, len(pairs), sharey=True, figsize=(2.3 * len(pairs), 2.7))
    for ax, (maj, mnr) in zip(np.atleast_1d(axes), pairs):
        pts = []
        for j, m2, n2 in COMPARISONS["judge_strength"]:
            if (m2, n2) != (maj, mnr):
                continue
            run = verdicts[(verdicts.judge == j) & (verdicts.major == maj) & (verdicts.minor == mnr)]
            if not len(run):
                continue
            m, lo, hi = sem_ci(counterbalanced(run))
            pts.append((models.loc[j, "accuracy"], m, lo, hi, j))
        pts.sort()
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "-", color=MUTED, lw=1.0, zorder=2)
        for acc, m, lo, hi, j in pts:
            ax.errorbar(acc, m, yerr=[[m - lo], [hi - m]], fmt="o", ms=5.5, color=JCOLOR[j],
                        ecolor=JCOLOR[j], elinewidth=0.9, capsize=0, zorder=3)
            ax.annotate(lab(j), xy=(acc, m), xytext=(0, -10), textcoords="offset points",
                        ha="center", color=INK2, fontsize=5.5, rotation=0)
        chance(ax)
        ax.set_title(f"{lab(maj)} v {lab(mnr)}", color=INK2, fontsize=7.5)
        ax.set_xlim(0.6, 0.9)
        ax.set_ylim(0.3, 1.0)
        style(ax)
    np.atleast_1d(axes)[0].set_ylabel("judge score")
    fig.supxlabel("judge's own GPQA accuracy", color=INK2, fontsize=8.5)
    fig.text(0.99, 0.01, "every judge that judged each pair in the pilot · error bars: 95% CI",
             ha="right", color=MUTED, fontsize=6)
    save(fig, out / "3_judge_strength")


# --- 4. length confound --------------------------------------------------

def fig_length(verdicts, summary, out):
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.0, 2.8),
                                      gridspec_kw={"width_ratios": [1.2, 1]})
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
        dot_ci(left, JUDGES.index(j), k / len(d), lo, hi, JCOLOR[j], note=f"n={len(d)}")
    left.set_yticks(range(len(JUDGES)))
    left.set_yticklabels([lab(j) for j in JUDGES], fontsize=7)
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
    right.set_xlabel("stronger − weaker mean length (k chars)")
    right.set_ylabel("judge score")
    right.text(0.02, 0.04, "one point per run · colour = judge (see left) · open = self-pair\n"
               "error bars: 95% CI",
               transform=right.transAxes, color=MUTED, fontsize=6)
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

    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(6.4, 2.7))
    for ax, col, xlabel in [(axes[0], "b_1pl", "item difficulty b (1PL)"),
                            (axes[1], "a_2pl", "item discrimination a (2PL)")]:
        ax.scatter(d[col], d.score, s=12, color=MUTED, alpha=0.45, lw=0, zorder=2)
        for _, cell in d.groupby(pd.qcut(d[col], 5), observed=True):
            m, lo, hi = sem_ci(cell.score.to_numpy())
            ax.errorbar(cell[col].mean(), m, yerr=[[m - lo], [hi - m]], fmt="o", ms=6,
                        color=PALETTE[0], ecolor=PALETTE[0], elinewidth=1.2, capsize=0,
                        zorder=4, markeredgecolor="white", markeredgewidth=0.8)
        chance(ax)
        ax.set_xlabel(xlabel)
        ax.set_ylim(0, 1)
        style(ax)
    axes[0].set_ylabel("mean judge score")
    axes[0].text(0.03, 0.05, f"n={len(d)} items · grey dots pooled over cross-model runs · "
                 "blue: quintile means, 95% CI", transform=axes[0].transAxes, color=MUTED, fontsize=6)
    save(fig, out / "5_item_difficulty")


# --- 6. temporal: judging across release-date distance -------------------

def fig_temporal(verdicts, out):
    rungs = list(dict.fromkeys((maj, mnr) for _, maj, mnr in COMPARISONS["temporal"]))
    judges = list(dict.fromkeys(j for j, _, _ in COMPARISONS["temporal"]))
    date = {m: pd.to_datetime(MODELS[m]["release"]) for m in
            {x for r in rungs for x in r} | set(judges)}
    rung_marker = dict(zip(rungs, ["o", "s", "^"]))

    fig, ax = plt.subplots(figsize=(5.2, 3.2))
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
    ax.annotate("pair from judge's own era", xy=(0, 0.02), xycoords=("data", "axes fraction"),
                rotation=90, ha="right", va="bottom", color=MUTED, fontsize=6)
    ax.set_xlabel("pair release − judge release (months; + = judging newer models)")
    ax.set_ylabel("judge score")
    ax.set_ylim(0, 1)
    style(ax)
    jh = [Line2D([], [], marker="o", ls="-", color=JCOLOR[j], ms=5, label=lab(j))
          for j in judges]
    rh = [Line2D([], [], marker=rung_marker[r], ls="", color=MUTED, ms=5,
                 label=f"{lab(r[0])} v {lab(r[1])}") for r in rungs]
    l1 = ax.legend(handles=jh, frameon=False, loc="upper right", fontsize=6)
    ax.add_artist(l1)
    l2 = ax.legend(handles=rh, frameon=False, loc="lower left", fontsize=6)
    for leg in (l1, l2):
        for t in leg.get_texts():
            t.set_color(INK2)
    ax.text(0.02, 0.97, "2026 pair is length-confounded — see 4_length_confound\n"
            "error bars: 95% CI",
            transform=ax.transAxes, ha="left", va="top", color=MUTED, fontsize=6)
    save(fig, out / "6_temporal")


# --- 7. score vs release-date distance, every run ------------------------

def fig_release_gap(summary, out):
    """Fig 6's release-distance x-axis, but over every run, not just the rungs."""
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    for _, r in summary.iterrows():
        is_self = r.major == r.minor
        dyr = (release_year(r.major) + release_year(r.minor)) / 2 - release_year(r.judge)
        ax.errorbar(dyr, r.score, yerr=[[r.score - r.score_lo], [r.score_hi - r.score]],
                    fmt="o", ms=4.5, color=JCOLOR[r.judge], ecolor=JCOLOR[r.judge],
                    elinewidth=0.9, capsize=0, mfc="white" if is_self else JCOLOR[r.judge],
                    mew=1.1, alpha=0.9, zorder=3)
    chance(ax)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.annotate("pair from judge's own era", xy=(0, 0.02), xycoords=("data", "axes fraction"),
                rotation=90, ha="right", va="bottom", color=MUTED, fontsize=6)
    ax.set_xlabel("pair mean release − judge release (years; + = judging newer models)")
    ax.set_ylabel("judge score")
    ax.set_ylim(0, 1)
    style(ax)
    jh = [Line2D([], [], marker="o", ls="", color=JCOLOR[j], ms=5, label=lab(j))
          for j in JUDGES]
    jh.append(Line2D([], [], marker="o", ls="", mfc="white", mec=INK2, ms=5,
                     label="self-pair (control)"))
    leg = ax.legend(handles=jh, frameon=False, loc="lower left", fontsize=5.8,
                    title="judge", title_fontsize=6)
    for t in leg.get_texts():
        t.set_color(INK2)
    leg.get_title().set_color(INK2)
    ax.text(0.99, 0.97, "one point per run, 95% CI · capability gap varies freely here —\n"
            "read with 2a/2b before attributing anything to age",
            transform=ax.transAxes, ha="right", va="top", color=MUTED, fontsize=6)
    save(fig, out / "7_release_gap")


# --- 0. model accuracy ---------------------------------------------------

def fig_model_accuracy(models, out):
    m = models.sort_values("accuracy").reset_index()
    arm_col = {"A": PALETTE[0], "B": PALETTE[1]}
    fig, ax = plt.subplots(figsize=(4.2, 0.22 * len(m) + 1.0))
    for y, r in m.iterrows():
        k = int(round(r.accuracy * r.n_samples))
        lo, hi = wilson(k, int(r.n_samples))
        dot_ci(ax, y, r.accuracy, lo, hi, arm_col[r.arm],
               note=f"{r.accuracy:.0%} (n={r.n_samples})")
    ax.set_yticks(range(len(m)))
    ax.set_yticklabels([lab(r.model) for _, r in m.iterrows()], fontsize=7)
    chance(ax, 0.25, axis="x")
    ax.set_xlabel("GPQA Diamond accuracy (all epochs; 95% CI; chance 0.25)")
    ax.set_xlim(0, 1)
    style(ax, grid_axis="x")
    handles = [Line2D([], [], marker="o", ls="", color=arm_col["A"], ms=5.5,
                      label="arm A (our collection)"),
               Line2D([], [], marker="o", ls="", color=arm_col["B"], ms=5.5,
                      label="arm B (Epoch logs)")]
    leg = ax.legend(handles=handles, frameon=False, loc="upper left",
                    bbox_to_anchor=(0.02, 0.35), fontsize=6.5)
    for t in leg.get_texts():
        t.set_color(INK2)
    save(fig, out / "0_model_accuracy")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, default=Path("judge_results/judge-pilot"))
    args = ap.parse_args()

    verdicts = pd.read_csv(args.dir / "verdicts.csv")
    models = pd.read_csv(args.dir / "models.csv").set_index("model")
    summary = pd.read_csv(args.dir / "summary.csv")
    fits = pd.read_csv(args.dir / "judge_fits.csv")
    out = args.dir / "figures"
    out.mkdir(parents=True, exist_ok=True)
    for stale in list(out.glob("*.png")) + list(out.glob("*.svg")):
        stale.unlink()

    init_labels(models)
    fig_model_accuracy(models, out)
    fig_run_overview(summary, out)
    fig_gap_curves(summary, fits, models, out)
    fig_judge_strength(verdicts, models, out)
    fig_length(verdicts, summary, out)
    fig_item_difficulty(verdicts, out)
    fig_temporal(verdicts, out)
    fig_release_gap(summary, out)
    print(f"8 figures -> {out}")


if __name__ == "__main__":
    main()
