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

sys.path.insert(0, "scripts/analyses")
from palette import MATCHUP, PROTOCOL  # noqa: E402
from plots import INK2, MUTED, save, style  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

META = Path("data/meta/coding_frontier")
OUT = Path("analysis/frontier/prediction_scores")
BASE = "mean_ref"
# protocol file stem -> (label, colour, marker); order = legend order
SHOW = {
    "judge_kimi-k2-thinking": ("Matchup BT", PROTOCOL["LLM Matchup"], "o"),
    "judgecal_kimi-k2-thinking": ("Matchup calibrated", MATCHUP["Mixed"], "s"),
    "judgelin_kimi-k2-thinking": ("Matchup linear", MATCHUP["Ref pairs"], "D"),
    "ridge": ("Ridge features", PROTOCOL["Length+regex"], "^"),
    "chars": ("Length", PROTOCOL["Length"], "v"),
    "mean_frontier": ("Top-3 ref. mean", MUTED, "x"),
}
SHORT = {"claude-fable-5": "Fable 5", "gpt-5.6-sol": "GPT-5.6 Sol", "claude-opus-4.8": "Opus 4.8",
         "union-alpha": "Union Alpha", "claude-opus-4.7": "Opus 4.7", "gemini-3.5-flash": "Gemini 3.5 Flash"}


def scores() -> pd.DataFrame:
    """One row per (protocol, model): theta_hat, theta, e, e_base, score."""
    base = pd.read_csv(META / f"eval_{BASE}.csv").set_index("model")
    rows = []
    for stem, (label, _, _) in SHOW.items():
        f = pd.read_csv(META / f"eval_{stem}.csv").set_index("model")
        for m, r in f.iterrows():
            e = (r.theta_hat - r.theta) ** 2
            e_base = (base.loc[m, "theta_hat"] - base.loc[m, "theta"]) ** 2
            rows.append(dict(protocol=label, model=m, theta_hat=r.theta_hat, theta=r.theta,
                             e=e, e_base=e_base, score=-np.log2(e / e_base)))
    return pd.DataFrame(rows)


def main() -> int:
    table = scores()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(OUT.with_suffix(".csv"), index=False)
    order = table.drop_duplicates("model").sort_values("theta", ascending=False).model.tolist()
    labels = [SHORT.get(m, m) for m in order] + ["Mean"]
    x = np.arange(len(labels))
    n = len(SHOW)
    dodge = np.linspace(-0.3, 0.3, n)

    fig, ax = plt.subplots(figsize=(2.1, 2.3), layout="constrained")
    ax.axhline(0, color=MUTED, lw=0.8, ls="--", zorder=1)
    for k, (stem, (label, color, marker)) in enumerate(SHOW.items()):
        t = table[table.protocol == label].set_index("model").loc[order]
        ys = list(t.score) + [t.score.mean()]
        ax.plot(x + dodge[k], ys, marker, color=color, ms=3, mec=color, mew=0.8, ls="", label=label, zorder=3,
                mfc=color if marker != "x" else color)
    ax.axvline(len(order) - 0.5, color=MUTED, lw=0.6, ls=":", zorder=1)
    style(ax)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_position(("outward", 5))
    ax.set_xticks(x, labels, rotation=90, ha="center")
    lo, hi = np.floor(table.score.min()), np.ceil(table.score.max())
    lo, hi = min(lo, -1), max(hi, 1)
    ax.set_ylim(lo, hi)
    ax.set_yticks(np.arange(lo, hi + 1, 1))
    ax.set_ylabel("Prediction score (bits)")
    ax.set_xlabel("")
    leg = fig.legend(loc="outside lower center", ncol=2, frameon=False, columnspacing=0.8, handlelength=1.0,
                     handletextpad=0.4, labelspacing=0.3, fontsize=6.5)
    for t in leg.get_texts():
        t.set_color(INK2)
    save(fig, str(OUT))
    print(table.pivot(index="model", columns="protocol", values="score").loc[order].round(2).to_string())
    print("\nmean over targets:", table.groupby("protocol").score.mean().round(2).sort_values(ascending=False).to_dict())
    print("wrote", OUT.with_suffix(".png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
