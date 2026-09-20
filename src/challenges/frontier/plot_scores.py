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


def main() -> int:
    table = scores(META)
    shown = [m for rows in VENDORS.values() for m, _ in rows]
    table = table[table.model.isin(shown)]
    reference = scores(OPEN)  # 8 open-weight 2026-I targets
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([table.assign(group="frontier"), reference.assign(group="open")]).round(4).to_csv(OUT.with_suffix(".csv"), index=False)

    # rows top to bottom: a grey header per vendor, its models, then the open-weight reference
    rows, y = [], 0.0
    for vendor, models in VENDORS.items():
        rows.append((y, vendor, None)); y += 0.8
        for model, label in models:
            rows.append((y, label, model)); y += 1
        y += 0.5
    rows.append((y, "Open-weight", None)); y += 0.8
    rows.append((y, "Mean of 8 targets", "open"))

    fig, ax = plt.subplots(figsize=(2.5, 0.135 * len(rows) + 0.75), layout="constrained")
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
                ax.errorbar(r.mean(), ypos, xerr=half, fmt="o", color=color, ms=3.2, mec="none", lw=1, capsize=0, zorder=3)
                continue
            v = t.get(model, np.nan)
            if np.isnan(v):
                continue
            inside = XLIM[0] <= v <= XLIM[1]
            ax.plot(clip(v), ypos, "o" if inside else (">" if v > 0 else "<"), color=color, ms=3.2 if inside else 3.6,
                    mec="none", zorder=3, clip_on=False, label=label if model == shown[0] else None)
    for ypos, _, model in rows:  # faint connector: ties a row's four dots together now that there is no grid
        if model and model != "open":
            v = table[table.model == model].score.dropna()
            ax.plot([clip(v.min()), clip(v.max())], [ypos, ypos], color="#d9d8d2", lw=0.8, zorder=2, solid_capstyle="round")

    style(ax)
    ax.grid(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_position(("outward", 5))
    ax.set_yticks([r[0] for r in rows], [r[1] for r in rows])
    for tick, (_, _, model) in zip(ax.get_yticklabels(), rows):
        if model is None:  # vendor header
            tick.set_color(MUTED); tick.set_fontsize(6.5); tick.set_fontstyle("italic")
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
    print(table.pivot(index="model", columns="protocol", values="score").loc[shown].round(2).to_string())
    print("wrote", OUT.with_suffix(".png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
