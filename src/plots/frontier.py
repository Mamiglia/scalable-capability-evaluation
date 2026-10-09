"""Figures of src/challenges/frontier/plot_scores.py: prediction scores of every protocol on the frontier targets."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from plots.palette import PROTOCOL
from plots.style import INK2, MUTED, save, style

import matplotlib.pyplot as plt

# protocol file stem -> (label, colour); order = legend order. Colour is the only encoding.
SHOW = {
    "judgelin_kimi-k2-thinking": ("Pairwise judge", PROTOCOL["LLM Matchup"]),
    "direct_judge_kimi-k2-thinking": ("Absolute judge", PROTOCOL["LLM Rater"]),
    "embedding": ("Embeddings", PROTOCOL["Embedding"]),
    "ridge": ("Deterministic features", PROTOCOL["Length+regex"]),
}
# vendor -> [(data-dir name, label)], newest first. "*" = Epoch AI SWE-bench import: swe_bench_mini only, Epoch's scaffold
# (integrate.py EPOCH). union-alpha left out.
VENDORS = {
    "Anthropic": [("claude-fable-5", "Fable 5"), ("claude-opus-4.8", "Opus 4.8"), ("claude-opus-4.7", "Opus 4.7")],
    "OpenAI": [("gpt-5.6-sol", "GPT-5.6 Sol"), ("gpt-5.5-pre-release_xhigh", "GPT-5.5*"), ("gpt-5.4-2026-03-05_high", "GPT-5.4*"),
               ("gpt-5.3-codex_high", "GPT-5.3 Codex*"), ("gpt-5.2-2025-12-11_high", "GPT-5.2*"),
               ("gpt-5.1-2025-11-13_high", "GPT-5.1*"), ("gpt-5", "GPT-5")],
    "Google": [("gemini-3.5-flash", "Gemini 3.5 Flash")],
}
XLIM = (-8, 8)  # scores beyond are drawn as a caret at the edge (a near-exact hit sends -log2 to +14)


OPEN_SHORT = {"nvidia-nemotron-3-ultra-550b-a55b-bf16": "Nemotron 3 Ultra", "gemma-4-31b-it": "Gemma 4 31B",
              "qwen3.6-27b": "Qwen3.6 27B", "kimi-k2.5": "Kimi K2.5", "qwen3.5-397b-a17b": "Qwen3.5 397B",
              "kimi-k2.6": "Kimi K2.6", "glm-5.2": "GLM-5.2", "glm-5.1": "GLM-5.1"}


def scores_figure(table: pd.DataFrame, reference: pd.DataFrame, out: Path) -> None:
    """Every protocol's score per frontier target (rows in VENDORS order, one lane per protocol, carets beyond XLIM)
    and the open-weight targets' mean with its t-based 95% CI as the last row."""
    shown = [m for rows in VENDORS.values() for m, _ in rows]
    # rows top to bottom: models in vendor order (a gap between vendors, no vendor names), then the open-weight reference
    rows, y = [], 0.0
    for models in VENDORS.values():
        for model, label in models:
            rows.append((y, label, model)); y += 1
        y += 0.5
    rows.append((y, "Open models (mean)", "open"))
    lane = np.linspace(-0.27, 0.27, len(SHOW))  # one fixed lane per protocol inside a row, so dots never cover each other

    fig, ax = plt.subplots(figsize=(2.3, 0.175 * len(rows) + 0.75), layout="constrained")
    ax.axvline(0, color=MUTED, lw=0.8, ls="--", zorder=1)
    clip = lambda v: float(np.clip(v, *XLIM))
    for k, (stem, (label, color)) in enumerate(SHOW.items()):
        t = table[table.protocol == label].set_index("model").score
        for ypos, _, model in rows:
            if model is None:
                continue
            if model == "open":
                r = reference[reference.protocol == label].score  # mean, t-based 95% CI over the open targets
                half = stats.t.ppf(0.975, len(r) - 1) * r.std(ddof=1) / np.sqrt(len(r))
                ax.errorbar(r.mean(), ypos + lane[k], xerr=half, fmt="o", color=color, ms=2.8, mec="none", lw=1, capsize=0, zorder=3)
                continue
            v = t.get(model, np.nan)
            if np.isnan(v):
                continue
            inside = XLIM[0] <= v <= XLIM[1]
            ax.plot(clip(v), ypos + lane[k], "o" if inside else (">" if v > 0 else "<"), color=color, ms=2.8 if inside else 3.2,
                    mec="none", zorder=3, clip_on=False, label=label if model == shown[0] else None)
    style(ax)
    ax.grid(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_position(("outward", 5))
    ax.set_yticks([r[0] for r in rows], [r[1] for r in rows])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(rows[-1][0] + 0.7, -0.6)
    ax.set_xlim(*XLIM)
    ax.set_xticks([-8, -4, 0, 4, 8])
    ax.set_xlabel("Prediction score")
    leg = fig.legend(loc="outside lower center", ncol=2, frameon=False, columnspacing=0.8, handlelength=1.0,
                     handletextpad=0.4, labelspacing=0.3, fontsize=6.5)
    for t in leg.get_texts():
        t.set_color(INK2)
    save(fig, str(out))


def pairwise_boxes(table: pd.DataFrame, reference: pd.DataFrame, out: Path) -> None:
    """Pairwise judge only: closed vs open models as two horizontal boxplots; every model a dot on the box's line,
    named right above it (names stack upward where scores are close, a thin tick joins name and dot)."""
    label, color = SHOW["judgelin_kimi-k2-thinking"]
    names = {m: l for rows in VENDORS.values() for m, l in rows}
    groups = [("Open models", reference[reference.protocol == label].assign(name=lambda d: d.model.map(OPEN_SHORT))),
              ("Closed models", table[table.protocol == label].assign(name=lambda d: d.model.map(names)))]
    lo = 2 * np.floor(min(g.score.min() for _, g in groups) / 2)
    hi = 2 * np.ceil(max(g.score.max() for _, g in groups) / 2)
    rows = [0.0, 1.9]  # y of each group's box line; the name stack sits above it
    step, base = 0.27, 0.3
    fig, ax = plt.subplots(figsize=(3.3, 2.0))
    fig.subplots_adjust(left=0.21, right=0.99, top=0.99, bottom=0.2)
    fig.set_layout_engine("none")  # freeze the axes geometry: name widths are measured in it, and save() must not re-layout
    ax.set_xlim(lo, hi)
    ax.set_ylim(-0.45, rows[-1] + base + 3 * step + 0.2)
    fig.canvas.draw()
    to_data = ax.transData.inverted()
    ax.axvline(0, color=MUTED, lw=0.6, ls="--", zorder=1)
    for y0, (_, g) in zip(rows, groups):
        ax.boxplot(g.score, positions=[y0], vert=False, widths=0.32, showfliers=False, showcaps=False,
                   boxprops=dict(color=MUTED, lw=0.7), medianprops=dict(color=INK2, lw=1),
                   whiskerprops=dict(color=MUTED, lw=0.7), zorder=2)
        ax.plot(g.score, np.full(len(g), y0), "o", color=color, ms=2.8, mec="none", zorder=3)
        right = []  # per stack level: right edge (data x) of the last name placed
        for x, name in sorted(zip(g.score, g.name)):
            t = ax.text(x, 0, name, fontsize=5.5, color=INK2, ha="center", va="bottom", zorder=4)
            bb = t.get_window_extent(fig.canvas.get_renderer()).transformed(to_data)
            half = (bb.x1 - bb.x0) / 2 + 0.1
            level = next((k for k, r in enumerate(right) if x - half > r), len(right))
            if level == len(right):
                right.append(x + half)
            else:
                right[level] = x + half
            y = y0 + base + level * step
            t.set_position((x, y))
            ax.plot([x, x], [y0 + 0.07, y - 0.03], color="#cfceC7", lw=0.5, zorder=2)
    style(ax)
    ax.grid(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_position(("outward", 5))
    ax.set_yticks(rows, [name for name, _ in groups])
    ax.tick_params(axis="y", length=0)
    ax.set_xticks(np.arange(lo, hi + 1, 2))
    ax.set_xlabel("Prediction score")
    save(fig, str(out.with_name("pairwise_boxplot")))
    print("wrote", out.with_name("pairwise_boxplot.png"))
