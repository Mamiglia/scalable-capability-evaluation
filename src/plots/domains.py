"""Figures of a frozen domain (scripts/figures/domain_figures.py): theta by semester and by release date, item
difficulty per bench, theta-vs-b map per semester, accuracy heatmap, per-bench accuracy."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "svg.fonttype": "none"})
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7", "#7f7f7f"]  # 7: math has 7 benches
GREY = "#9a9a9a"
SEMESTERS = ["2023", "2024-I", "2024-II", "2025-I", "2025-II", "2026-I"]
SEMESTER_COLOR = dict(zip(SEMESTERS, PALETTE))  # one colour per semester, the same in every domain


def offset(ax, x=True, y=True):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if x:
        ax.spines["bottom"].set_position(("outward", 5))
    if y:
        ax.spines["left"].set_position(("outward", 5))


def save(fig, stub: Path):
    fig.savefig(stub.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stub.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stub.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("wrote", stub.with_suffix(".png"))


def theta_figure(models: pd.DataFrame, semesters: list[str], out: Path):
    """a: global theta of every model by semester (roster coloured, others grey). b: local theta* vs global."""
    color = SEMESTER_COLOR
    fig, (a, b) = plt.subplots(1, 2, figsize=(3.6, 2.1), gridspec_kw={"width_ratios": [1.3, 1]})
    rng = np.random.default_rng(0)
    for i, semester in enumerate(semesters):
        rows = models[models.semester == semester]
        x = i + rng.uniform(-0.18, 0.18, len(rows))
        a.errorbar(x, rows.global_theta, yerr=rows.global_theta_sd, fmt="o", ms=3, lw=0.8, capsize=0,
                   color=color[semester], label=semester)
    a.set_xticks(range(len(semesters)), semesters)
    a.tick_params(axis="x", rotation=90)
    a.set_ylabel("Global θ")
    tasks = [c.removesuffix("_role") for c in models.columns if c.endswith("_role")]
    for task in tasks:
        role = models[f"{task}_role"]
        for marker, which in (("o", "target"), ("s", "reference")):
            rows = models[role == which]
            b.errorbar(rows.global_theta, rows[f"{task}_theta"], yerr=rows[f"{task}_theta_sd"], fmt=marker, ms=3,
                       lw=0.8, capsize=0, color=color[task], mfc=color[task] if which == "target" else "white")
    roster = models[models.semester.notna()]  # coding's 2025-I roster goes below -3; theta* intervals may spill over
    lo, hi = min(-3, np.floor(roster.global_theta.min())), max(3, np.ceil(roster.global_theta.max()))
    ticks = [lo, 0, hi]
    b.plot([lo, hi], [lo, hi], ls="--", lw=0.8, color=GREY)
    b.set_xlim(lo, hi), b.set_ylim(lo, hi), b.set_xticks(ticks), b.set_yticks(ticks)
    a.set_ylim(lo, hi), a.set_yticks(ticks)
    b.set_xlabel("Global θ"), b.set_ylabel("θ* (HARD items)")
    b.plot([], [], "o", color="k", ms=3, label="target"), b.plot([], [], "s", mfc="white", color="k", ms=3, label="reference")
    for ax in (a, b):
        offset(ax)
    a.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, handletextpad=0.2, columnspacing=0.8)
    b.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, handletextpad=0.2)
    fig.tight_layout(w_pad=1.5)
    save(fig, out / "theta")


def difficulty_figure(items: pd.DataFrame, out: Path):
    """Item difficulty b per bench (step histograms), with the discrimination a underneath."""
    benches = sorted(items.bench.unique())
    color = dict(zip(benches, PALETTE))
    fig, (a, b) = plt.subplots(2, 1, figsize=(2.3, 2.9), sharex=True)
    edges = np.linspace(np.floor(items.b.min()), np.ceil(items.b.max()), 25)
    for bench in benches:
        rows = items[items.bench == bench]
        a.hist(rows.b, bins=edges, histtype="step", lw=1, color=color[bench], label=f"{bench} ({len(rows)})")
        b.scatter(rows.b, rows.a, s=4, lw=0, alpha=0.6, color=color[bench])
    a.set_ylabel("Items")
    b.set_xlabel("Difficulty b"), b.set_ylabel("Discrim. a")
    b.set_xlim(edges[0], edges[-1]), b.set_xticks([edges[0], 0, edges[-1]])
    for ax in (a, b):
        offset(ax)
    a.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, handletextpad=0.4, columnspacing=0.8, fontsize=6)
    fig.tight_layout(h_pad=0.5)
    save(fig, out / "difficulty")


def wright_figure(models: pd.DataFrame, items: pd.DataFrame, out: Path):
    """Person-item map per semester task: roster global theta (left strip) against EASY/HARD item difficulty b."""
    tasks = [c.removesuffix("_role") for c in models.columns if c.endswith("_role")]
    roles = {"target": ("o", "k"), "reference": ("s", "white"), "historical": ("^", "white")}
    sets = {"easy": PALETTE[0], "hard": PALETTE[1]}
    rostered = models[models[[f"{t}_role" for t in tasks]].notna().any(axis=1)]
    lo = np.floor(min(items.b.min(), rostered.global_theta.min()))
    hi = np.ceil(max(items.b.max(), rostered.global_theta.max()))
    edges = np.linspace(lo, hi, int(2 * (hi - lo)) + 1)  # half-logit bins
    top = max(np.histogram(items.b[items[f"{t}_set"] == s], edges)[0].max() for t in tasks for s in sets)
    top = int(np.ceil(top / 10) * 10)
    fig, axes = plt.subplots(1, len(tasks), figsize=(0.8 * len(tasks) + 0.5, 1.9), sharey=True, squeeze=False)
    rng = np.random.default_rng(0)
    for ax, task in zip(axes[0], tasks):
        for name, c in sets.items():
            ax.hist(items.b[items[f"{task}_set"] == name], bins=edges, orientation="horizontal", histtype="stepfilled",
                    lw=1, ec=c, fc=c + "33", label=name.upper())
        for role, (marker, mfc) in roles.items():
            rows = models[models[f"{task}_role"] == role]
            if len(rows):
                x = -top * rng.uniform(0.12, 0.48, len(rows))  # models sit in a strip left of the item counts
                ax.errorbar(x, rows.global_theta, yerr=rows.global_theta_sd, fmt=marker, ms=3, lw=0.6, capsize=0,
                            color="k", mfc=mfc, mew=0.6, label=role)
        ax.axvline(0, color=GREY, lw=0.8, zorder=3)
        ax.set_title(task, loc="left", fontsize=7, color=GREY)
        ax.set_xlim(-0.6 * top, top), ax.set_xticks([0, top]), ax.set_xlabel("Items")
        ax.set_ylim(lo, hi), ax.set_yticks([lo, 0, hi])
        offset(ax)
    axes[0, 0].set_ylabel("θ, b (logit)")
    rows = 1 if len(tasks) > 2 else 2  # a narrow figure stacks the legend instead of widening
    handles = {l: h for ax in axes[0] for h, l in zip(*ax.get_legend_handles_labels())}
    fig.legend(handles.values(), handles.keys(), loc="lower center", ncol=-(-len(handles) // rows), frameon=False,
               handletextpad=0.3, columnspacing=0.8, borderaxespad=0)
    fig.tight_layout(w_pad=0.6, rect=(0, 0.08 * rows + 0.02, 1, 1))
    save(fig, out / "wright")


def release_figure(domains: dict[str, pd.DataFrame], dates: pd.Series, out: Path):
    """Global theta of the roster against release date (`dates`: model -> release date), one panel per domain,
    coloured by semester."""
    fig, axes = plt.subplots(1, len(domains), figsize=(1.75 * len(domains) + 0.4, 2.1), sharex=True, squeeze=False)
    for ax, (domain, models) in zip(axes[0], domains.items()):
        roster = models[models.semester.notna()]
        for semester, rows in roster.groupby("semester"):
            ax.errorbar(dates.reindex(rows.index), rows.global_theta, yerr=rows.global_theta_sd, fmt="o", ms=3,
                        lw=0.8, capsize=0, color=SEMESTER_COLOR[semester], label=semester)
        lo, hi = min(-3, np.floor(roster.global_theta.min())), max(3, np.ceil(roster.global_theta.max()))
        ax.set_ylim(lo, hi), ax.set_yticks([lo, 0, hi])
        ax.xaxis.set_major_locator(mdates.YearLocator()), ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.set_xlabel("Release date")
        if len(domains) > 1:
            ax.set_title(domain, loc="left", fontsize=7, color=GREY)
        offset(ax)
    shown = dates.reindex(pd.concat([m[m.semester.notna()] for m in domains.values()]).index)  # shared x: whole years over every panel, so even a
    axes[0, 0].set_xlim(pd.Timestamp(shown.min().year, 1, 1), pd.Timestamp(shown.max().year + 1, 1, 1))  # short roster spans two ticks
    axes[0, 0].set_ylabel("Global θ")
    handles = {l: h for ax in axes[0] for h, l in zip(*ax.get_legend_handles_labels())}
    order = [l for l in SEMESTERS if l in handles]
    ncol = len(order) if len(domains) > 1 else 2  # a narrow figure stacks the legend instead of widening
    rows = -(-len(order) // ncol)
    fig.legend([handles[l] for l in order], order, loc="lower center", ncol=ncol, frameon=False,
               handletextpad=0.2, columnspacing=0.8, borderaxespad=0)
    fig.tight_layout(w_pad=1.0, rect=(0, 0.07 * rows + 0.02, 1, 1))
    save(fig, out / "theta_release")


def accuracy_figure(models: pd.DataFrame, responses: pd.DataFrame, out: Path):
    """Roster models (rows, by global theta, labels coloured by semester) x bench: mean accuracy over every repeat.
    Blank = never run."""
    roster = models[models.semester.notna()].sort_values("global_theta", ascending=False)
    acc = responses[responses.model.isin(roster.index)].groupby(["model", "bench"]).correct.mean().unstack()
    acc = acc.loc[roster.index]
    # the row labels take ~1.7in, the cells 0.3in per bench, the colorbar and semester legend 0.8in
    label_w, cell_w, cbar_w = 1.7, 0.3 * acc.shape[1], 0.8
    width = label_w + cell_w + cbar_w
    fig = plt.figure(figsize=(width, 0.085 * len(acc) + 0.55))
    ax = fig.add_axes([label_w / width, 0.1, cell_w / width, 0.88])
    im = ax.imshow(acc.to_numpy(float), aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(acc.shape[1]), acc.columns, rotation=90, ha="center")
    ax.set_yticks(range(len(acc)), acc.index, fontsize=5)
    for label, semester in zip(ax.get_yticklabels(), roster.semester):
        label.set_color(SEMESTER_COLOR[semester])
    ax.tick_params(length=0)
    for i, row in enumerate(acc.to_numpy(float)):
        for j, v in enumerate(row):
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=4.5, color="white" if v > 0.6 else "black")
    cbar = fig.colorbar(im, cax=fig.add_axes([(label_w + cell_w + 0.1) / width, 0.1, 0.02, 0.3]), ticks=[0, 0.5, 1])
    cbar.set_label("Accuracy")
    cbar.outline.set_visible(False)
    used = [s for s in SEMESTERS if s in set(roster.semester)]
    fig.legend([plt.Line2D([], [], marker="s", ls="", ms=4, color=SEMESTER_COLOR[s]) for s in used], used, frameon=False,
               loc="upper left", bbox_to_anchor=((label_w + cell_w + 0.05) / width, 0.98), handletextpad=0.2,
               borderaxespad=0, fontsize=6)
    save(fig, out / "accuracy")


def benchmarks_figure(domains: dict[str, tuple[pd.DataFrame, pd.DataFrame]], out: Path, ncols: int = 5):
    """Every bench of every domain as a small panel: accuracy of each roster model by semester, semester mean as a bar."""
    panels = []
    for domain, (models, responses) in domains.items():
        semester = models.semester.dropna()
        acc = responses[responses.model.isin(semester.index)].groupby(["bench", "model"]).correct.mean()
        panels += [(domain, bench, rows.droplevel("bench")) for bench, rows in acc.groupby("bench")]
    nrows = -(-len(panels) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.4 * ncols, 1.35 * nrows + 0.5), sharex=True, sharey=True, squeeze=False)
    rng = np.random.default_rng(0)
    for ax, (domain, bench, acc) in zip(axes.flat, panels):
        semester = domains[domain][0].semester.reindex(acc.index)
        for i, s in enumerate(SEMESTERS):
            rows = acc[semester == s]
            if len(rows):
                ax.plot(i + rng.uniform(-0.22, 0.22, len(rows)), rows, "o", ms=2.5, mew=0, alpha=0.75, color=SEMESTER_COLOR[s])
                ax.plot([i - 0.32, i + 0.32], [rows.mean()] * 2, lw=1.2, color=SEMESTER_COLOR[s])
        ax.set_title(f"{domain}\n{bench}", loc="left", fontsize=6.5, color=GREY)
        ax.set_xticks(range(len(SEMESTERS)), SEMESTERS)
        ax.tick_params(axis="x", labelrotation=90)
        ax.set_xlim(-0.6, len(SEMESTERS) - 0.4), ax.set_ylim(0, 1), ax.set_yticks([0, 0.5, 1])
        offset(ax)
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)
    for ax in axes.flat[max(0, len(panels) - ncols):len(panels)]:  # the last panel of each column keeps its labels
        ax.tick_params(labelbottom=True)
    for ax in axes[:, 0]:
        ax.set_ylabel("Accuracy")
    fig.tight_layout(w_pad=0.6, h_pad=0.8)
    save(fig, out / "benchmarks")
