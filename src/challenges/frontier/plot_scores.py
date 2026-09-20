"""Prediction scores of every protocol on the frontier targets (coding_frontier, 2026-I).

    PYTHONPATH=src:scripts/analyses .venv/bin/python src/challenges/frontier/plot_scores.py

Score of protocol P on target m: S_m(P) = -log2(e_m(P) / e_m(base)), e_m = (theta_hat - theta*)^2, base = the
constant forecast at the references' mean theta* (`mean_ref`: with saturated items, accuracy carries no
information, so predicting the reference mean is the no-information baseline). Positive = better than the
baseline, +1 = squared error halved. Reads data/meta/<domain>/eval_<protocol>.csv (written by `uv run evaluate`),
writes analysis/frontier/prediction_scores.{csv,png,svg}.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, "scripts/analyses")
from palette import PROTOCOL  # noqa: E402
from plots import INK2, MUTED, save, style  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

META = Path("data/meta/coding_frontier")
OPEN = Path("data/meta/coding")  # the same protocols on this semester's open-weight targets: the reference group
OUT = Path("analysis/frontier/prediction_scores")
BASE = "mean_ref"
SEMESTER = "2026-I"
EXISTING_DIRECT = Path("logs/direct/coding/2026-I/kimi-k2-thinking/1a2c14b2")  # swe + terminal ratings, 16 models
# protocol file stem -> (label, colour); order = legend order. Colour is the only encoding.
SHOW = {
    "judgelin_kimi-k2-thinking": ("Pairwise judge", PROTOCOL["LLM Matchup"]),
    "direct_judge_kimi-k2-thinking": ("Absolute judge", PROTOCOL["LLM Rater"]),
    "embedding": ("Embeddings", PROTOCOL["Embedding"]),
    "ridge": ("Deterministic features", PROTOCOL["Length+regex"]),
}
# vendor -> [(data-dir name, label)], newest first. "*" = Epoch AI SWE-bench import: swe_bench_mini only, Epoch's scaffold
# (integrate.py EPOCH). union-alpha left out (user, 2026-09-19).
VENDORS = {
    "Anthropic": [("claude-fable-5", "Fable 5"), ("claude-opus-4.8", "Opus 4.8"), ("claude-opus-4.7", "Opus 4.7")],
    "OpenAI": [("gpt-5.6-sol", "GPT-5.6 Sol"), ("gpt-5.5-pre-release_xhigh", "GPT-5.5*"), ("gpt-5.4-2026-03-05_high", "GPT-5.4*"),
               ("gpt-5.3-codex_high", "GPT-5.3 Codex*"), ("gpt-5.2-2025-12-11_high", "GPT-5.2*"),
               ("gpt-5.1-2025-11-13_high", "GPT-5.1*"), ("gpt-5", "GPT-5")],
    "Google": [("gemini-3.5-flash", "Gemini 3.5 Flash")],
}
XLIM = (-8, 8)  # scores beyond are drawn as a caret at the edge (a near-exact hit sends -log2 to +14)


def existing_direct() -> pd.DataFrame:
    """Coding's absolute-judge forecasts from the run already collected on 2026-09-14 (EXISTING_DIRECT), not a new one:
    its hash predates agent_bench_os joining the domain, so `evaluate --direct-judge` would re-rate everything."""
    from capability.evaluate import forecast
    from capability.task import Task
    from configs.domain import DomainConfig
    from configs.judge import DirectJudgeConfig
    from protocols.direct_judge.judge import JudgeDirect, from_logs

    class Existing(JudgeDirect):
        def scores(self, train, test):
            return from_logs(EXISTING_DIRECT)

    domain = DomainConfig.from_cli(Path("configs/domain/coding.yml"))
    protocol = Existing(DirectJudgeConfig.from_cli(Path("configs/judges/direct/kimi-k2-thinking.yml")), domain)
    return forecast(Task.from_semester(SEMESTER, domain), protocol)


def scores(meta: Path) -> pd.DataFrame:
    """One row per (protocol, model): theta_hat, theta, e, e_base, score."""
    base = pd.read_csv(meta / f"eval_{BASE}.csv").query("semester == @SEMESTER").set_index("model")
    rows = []
    for stem, (label, _) in SHOW.items():
        path = meta / f"eval_{stem}.csv"
        f = (pd.read_csv(path) if path.exists() else existing_direct()).query("semester == @SEMESTER").set_index("model")
        for m, r in f.iterrows():
            e = (r.theta_hat - r.theta) ** 2
            e_base = (base.loc[m, "theta_hat"] - base.loc[m, "theta"]) ** 2
            rows.append(dict(protocol=label, model=m, theta_hat=r.theta_hat, theta=r.theta,
                             e=e, e_base=e_base, score=-np.log2(e / e_base)))
    return pd.DataFrame(rows)


OPEN_SHORT = {"nvidia-nemotron-3-ultra-550b-a55b-bf16": "Nemotron 3 Ultra", "gemma-4-31b-it": "Gemma 4 31B",
              "qwen3.6-27b": "Qwen3.6 27B", "kimi-k2.5": "Kimi K2.5", "qwen3.5-397b-a17b": "Qwen3.5 397B",
              "kimi-k2.6": "Kimi K2.6", "glm-5.2": "GLM-5.2", "glm-5.1": "GLM-5.1"}


def spread(ys: np.ndarray, gap: float) -> np.ndarray:
    """Label heights: the sorted values pushed apart to at least `gap`, centred on where they started."""
    order = np.argsort(ys)
    out = ys[order].astype(float).copy()
    for i in range(1, len(out)):
        out[i] = max(out[i], out[i - 1] + gap)
    out += (ys[order].mean() - out.mean())
    placed = np.empty_like(out)
    placed[order] = out
    return placed


def pairwise_boxes(table: pd.DataFrame, reference: pd.DataFrame) -> None:
    """Pairwise judge only: closed vs open models as two boxplots, every model a labelled dot."""
    label, color = SHOW["judgelin_kimi-k2-thinking"]
    names = {m: l for rows in VENDORS.values() for m, l in rows}
    groups = [("Closed models", table[table.protocol == label].assign(name=lambda d: d.model.map(names))),
              ("Open models", reference[reference.protocol == label].assign(name=lambda d: d.model.map(OPEN_SHORT)))]
    lo = 2 * np.floor(min(g.score.min() for _, g in groups) / 2)
    hi = 2 * np.ceil(max(g.score.max() for _, g in groups) / 2)
    xs = [0.0, 2.6]
    fig, ax = plt.subplots(figsize=(3.0, 2.2), layout="constrained")
    ax.axhline(0, color=MUTED, lw=0.6, ls="--", zorder=1)
    for x, (_, g) in zip(xs, groups):
        ax.boxplot(g.score, positions=[x], widths=0.28, showfliers=False, showcaps=False,
                   boxprops=dict(color=MUTED, lw=0.7), medianprops=dict(color=INK2, lw=1),
                   whiskerprops=dict(color=MUTED, lw=0.7), zorder=2)
        ax.plot(np.full(len(g), x), g.score, "o", color=color, ms=2.8, mec="none", zorder=3)
        # labels alternate right / left of the box by rank, each side spread just enough not to collide
        g = g.sort_values("score", ascending=False).reset_index(drop=True)
        for side, sign in ((0, 1), (1, -1)):
            part = g.iloc[side::2]
            for y, y_text, name in zip(part.score, spread(part.score.to_numpy(), 0.075 * (hi - lo)), part.name):
                if abs(y_text - y) > 0.12:
                    ax.plot([x + sign * 0.17, x + sign * 0.22], [y, y_text], color="#cfceC7", lw=0.5, zorder=2)
                ax.text(x + sign * 0.24, y_text, name, fontsize=5.5, color=INK2, va="center",
                        ha="left" if sign > 0 else "right", zorder=4, bbox=dict(facecolor="white", edgecolor="none", pad=0.15))
    style(ax)
    ax.grid(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_position(("outward", 5))
    ax.set_xticks(xs, [name for name, _ in groups])
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(xs[0] - 1.15, xs[1] + 0.95)
    ax.set_ylim(lo, hi)
    ax.set_yticks(np.arange(lo, hi + 1, 2))
    ax.set_ylabel("Prediction score")
    save(fig, str(OUT.with_name("pairwise_boxplot")))
    print("wrote", OUT.with_name("pairwise_boxplot.png"))


def main() -> int:
    table = scores(META)
    shown = [m for rows in VENDORS.values() for m, _ in rows]
    table = table[table.model.isin(shown)]
    reference = scores(OPEN)  # 8 open-weight 2026-I targets
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([table.assign(group="frontier"), reference.assign(group="open")]).round(4).to_csv(OUT.with_suffix(".csv"), index=False)

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
    save(fig, str(OUT))
    pairwise_boxes(table, reference)
    print(table.pivot(index="model", columns="protocol", values="score").loc[shown].round(2).to_string())
    print("wrote", OUT.with_suffix(".png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
