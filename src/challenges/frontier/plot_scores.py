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
SHORT = {"claude-fable-5": "Fable 5", "gpt-5.6-sol": "GPT-5.6 Sol", "claude-opus-4.8": "Opus 4.8",
         "claude-opus-4.7": "Opus 4.7", "gemini-3.5-flash": "Gemini 3.5 Flash",
         # Epoch AI SWE-bench import: swe_bench_mini only, Epoch's scaffold (integrate.py EPOCH); "*" marks them
         "gpt-5.5-pre-release_xhigh": "GPT-5.5*", "gpt-5.4-2026-03-05_high": "GPT-5.4*", "gpt-5.3-codex_high": "GPT-5.3 Codex*",
         "gpt-5.2-2025-12-11_high": "GPT-5.2*", "gpt-5.1-2025-11-13_high": "GPT-5.1*", "gpt-5": "GPT-5"}
# union-alpha left out (user, 2026-09-19)


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
    table = table[table.model.isin(SHORT)]
    reference = scores(OPEN)  # 8 open-weight 2026-I targets
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([table.assign(group="frontier"), reference.assign(group="open")]).round(4).to_csv(OUT.with_suffix(".csv"), index=False)
    order = table.drop_duplicates("model").sort_values("theta", ascending=False).model.tolist()
    labels = [SHORT[m] for m in order] + ["Open models"]
    x = np.arange(len(labels))
    dodge = np.linspace(-0.12, 0.12, len(SHOW))  # small: the user wants the points of a model close together

    fig, ax = plt.subplots(figsize=(1.2 + 0.17 * len(labels), 2.1), layout="constrained")
    ax.axhline(0, color=MUTED, lw=0.8, ls="--", zorder=1)
    for k, (stem, (label, color)) in enumerate(SHOW.items()):
        t = table[table.protocol == label].set_index("model").reindex(order)
        ax.plot(x[:-1] + dodge[k], t.score, "o", color=color, ms=3.2, mec="none", ls="", label=label, zorder=3)
        r = reference[reference.protocol == label].score  # mean and t-based 95% CI over the open targets
        half = stats.t.ppf(0.975, len(r) - 1) * r.std(ddof=1) / np.sqrt(len(r))
        ax.errorbar(x[-1] + dodge[k], r.mean(), yerr=half, fmt="o", color=color, ms=3.2, mec="none", lw=1, capsize=0, zorder=3)
    ax.axvline(len(order) - 0.5, color=MUTED, lw=0.6, ls=":", zorder=1)
    style(ax)
    ax.grid(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_position(("outward", 5))
    ax.set_xticks(x, labels, rotation=90, ha="center")
    lo, hi = 4 * np.floor(table.score.min() / 4), 4 * np.ceil(table.score.max() / 4)  # sparse ticks, nothing clipped
    ax.set_ylim(lo, hi)
    ax.set_yticks(np.arange(lo, hi + 1, 4))
    ax.set_ylabel("Prediction score")
    leg = fig.legend(loc="outside lower center", ncol=2, frameon=False, columnspacing=0.8, handlelength=1.0,
                     handletextpad=0.4, labelspacing=0.3, fontsize=6.5)
    for t in leg.get_texts():
        t.set_color(INK2)
    save(fig, str(OUT))
    print(table.pivot(index="model", columns="protocol", values="score").loc[order].round(2).to_string())
    print("\nfrontier mean:", table.groupby("protocol").score.mean().round(2).to_dict())
    print(f"open models (n={reference.model.nunique()}) mean:", reference.groupby("protocol").score.mean().round(2).to_dict())
    print("wrote", OUT.with_suffix(".png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
