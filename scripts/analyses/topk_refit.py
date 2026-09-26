"""Saturation-boundary refit (manuscript Appendix D.11, Figure 25, Table 10). No judging is run.

    PYTHONPATH=src uv run python scripts/analyses/topk_refit.py [--pairwise-only] [--refit]

Per domain, task 2025-II / 2026-I and level (3..8, "all"): theta* refit on the level's HARD items, every protocol
refit on the train items still EASY, the judge chosen by reference leave-one-out error. --pairwise-only skips the
protocols that read transcripts or embeddings (Baseline + Pairwise judge only). theta* per level is cached
in analysis/topk_refit/theta.csv (--refit redoes it). 
"""

import copy
import logging
import os
import sys
import warnings
from pathlib import Path

import pandas as pd

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").exists())
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

from analysis import calibration, loading, matching, stats  # noqa: E402
from analysis.baselines import EasyItemPrediction  # noqa: E402
from analysis.challenges import saturation_boundary as sb  # noqa: E402
from configs.logs import logger  # noqa: E402
from protocols import PROTOCOLS  # noqa: E402
from protocols.embeddings import EmbeddingRegression  # noqa: E402

warnings.filterwarnings("ignore")
logger.setLevel(logging.WARNING)
OUT = Path("analysis/topk_refit")
DOMAINS = ["science", "math", "coding", "cyber"]
SEMESTERS = ["2025-II", "2026-I"]
# (protocol, variant) of the main analysis
SHOWN = {"easy_items": "-", "glm_regex": "-", "embedding": "full", "direct": "raw", "pairwise": "linear"}
PAIRWISE_ONLY = "--pairwise-only" in sys.argv


def cut(run: loading.JudgeRun, items: pd.DataFrame) -> loading.JudgeRun:
    """A copy of a matched run keeping only the rows on `items`."""
    out = copy.copy(run)
    out.table = run.table.merge(items[["bench", "item"]])
    return out


def match_pairwise_rows(run: loading.JudgeRun, task) -> loading.JudgeRun:
    """matching.match_pairwise without the `missing` count (which reads transcripts): train items, one verdict
    per (bench, item, left, right), the run's own verdict over an imported one."""
    v = run.table.merge(task.references.items[["bench", "item"]]).sort_values("imported", kind="stable")
    run.table = v.drop_duplicates(["bench", "item", "model_left", "model_right"], keep="first")
    return run


def compute() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(theta, forecasts): theta* per domain, semester, level, model; forecast rows with a `level` column."""
    cache = OUT / "theta.csv"
    cached = pd.read_csv(cache, dtype={"level": str}) if cache.exists() and "--refit" not in sys.argv else None
    thetas, frames = [], []
    for name in DOMAINS:
        dom = loading.domain(name)
        protocols = [(EasyItemPrediction(), "-")] if PAIRWISE_ONLY else \
            [(EasyItemPrediction(), "-"), (PROTOCOLS["glm_regex"], "-"), (EmbeddingRegression("full"), "full")]
        for task in loading.tasks(dom, SEMESTERS):
            runs = loading.judge_runs(dom, task)
            if PAIRWISE_ONLY:
                runs = [match_pairwise_rows(r, task) for r in runs if r.protocol == "pairwise"]
            else:
                runs = [matching.match(r, task) for r in runs]
            for level in sb.LEVELS:
                if level != "all" and level > len(task.references.models):  # fewer references than the level: no task
                    continue
                hard, observed = sb.level_split(dom, task, level)
                key = (name, task.semester, str(level))
                if cached is not None:
                    theta = cached[(cached.domain == name) & (cached.semester == task.semester) & (cached.level == str(level))].set_index("model").theta
                else:
                    theta = sb.refit_theta(dom, task, hard)
                thetas.append(pd.DataFrame({"domain": name, "semester": task.semester, "level": str(level),
                                            "model": theta.index, "theta": theta.to_numpy(), "n_hard": len(hard),
                                            "n_observed": len(observed)}))
                lt = sb.level_task(task, theta, observed)
                frames.append(calibration.forecasts(dom, lt, protocols, [cut(r, observed) for r in runs]).assign(level=str(level)))
                print(*key, len(hard), "hard", len(observed), "observed", file=sys.stderr)
    return pd.concat(thetas, ignore_index=True), pd.concat(frames, ignore_index=True)


def ratios(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Per level x domain (and domain "all") x shown protocol: MSE / Baseline MSE with a bootstrap band over targets.
    Judged protocols are the judge with the lowest reference leave-one-out error at that level."""
    rows = []
    for level, f in forecasts.groupby("level", sort=False):  # every protocol on the full train set (same items)
        f = stats.one_per_target(f.drop(columns="level"))
        f = f[f.protocol.map(SHOWN).eq(f.variant)].assign(err=lambda d: (d.theta_hat - d.theta) ** 2)
        base = f[f.protocol == stats.BASELINE][["domain", "semester", "model", "err"]].rename(columns={"err": "err_baseline"})
        f = f.merge(base).dropna(subset=["err", "err_baseline"])
        for (domain, protocol), g in [*f.groupby(["domain", "protocol"]), *(( ("all", p), g) for p, g in f.groupby("protocol"))]:
            rows.append(dict(level=level, domain=domain, protocol=protocol, judge=",".join(sorted(set(g.judge) - {"-"})), **sb.ratio(g)))
    return pd.DataFrame(rows)


PAPER = {"pairwise": ("Pairwise judge", "#2a78d6"), "direct": ("Absolute scoring", "#1baf7a"),
         "glm_regex": ("Deterministic features", "#eb6834"), "embedding": ("Embedding", "#4a3aa7")}  # CLAUDE.md
TITLES = {"science": "Science", "math": "Maths", "coding": "Coding", "cyber": "Cyber"}
FIGURE = Path("analysis/results/figures/paper/topk_refit")


def plot(table: pd.DataFrame, out: Path = FIGURE) -> None:
    """2 x 2 domain panels: each protocol's error ratio to the Baseline per level, bootstrap band over targets."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "svg.fonttype": "none", "font.family": "sans-serif",
                         "font.sans-serif": ["Arial"], "axes.linewidth": 0.8, "xtick.major.width": 0.8,
                         "ytick.major.width": 0.8, "axes.spines.top": False, "axes.spines.right": False})
    levels = [str(l) for l in sb.LEVELS]
    x = range(len(levels))
    fig, axes = plt.subplots(2, 2, figsize=(4.2, 3.4), sharex=True, sharey=True, gridspec_kw={"hspace": 0.35, "wspace": 0.15})
    for ax, domain in zip(axes.flat, DOMAINS):
        ax.axhline(1, color="#9a9a9a", lw=0.8, ls="--")
        for protocol, (label, color) in PAPER.items():
            t = table[(table.domain == domain) & (table.protocol == protocol)].set_index("level").reindex(levels)
            ax.fill_between(x, t.lo, t.hi, color=color, alpha=0.15, lw=0)
            ax.plot(x, t.ratio, color=color, lw=1, marker="o", ms=3, clip_on=False, label=label)
        ax.set_title(TITLES[domain], loc="left", fontsize=8, color="#7a7a7a")
        ax.set_yscale("log"), ax.set_ylim(0.03, 10)
        ax.set_yticks([0.03, 0.1, 0.3, 1, 3, 10], ["0.03×", "0.1×", "0.3×", "1×", "3×", "10×"])
        ax.yaxis.set_minor_locator(plt.NullLocator())
        ax.set_xticks(list(x), levels), ax.set_xlim(-0.3, len(levels) - 0.7)
        ax.spines["bottom"].set_bounds(0, len(levels) - 1)
    fig.supylabel("Squared error vs baseline", fontsize=8, x=0.02)
    fig.supxlabel("Models that must solve an item", fontsize=8, y=0.0)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.93))
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "pdf", "png"):
        fig.savefig(out.with_suffix(f".{ext}"), bbox_inches="tight", **({"dpi": 300} if ext == "png" else {}))


def main() -> int:
    if "--plot" in sys.argv:  # rescore and redraw from the cached forecasts
        table = ratios(pd.read_csv(OUT / "forecasts.csv", dtype={"level": str}))
        table.to_csv(OUT / "ratios.csv", index=False, float_format="%.4f")
        print(table[table.protocol == "pairwise"].pivot(index="domain", columns="level", values="ratio")[[str(l) for l in sb.LEVELS]].round(2).to_string())
        plot(table)
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    tag = "_pairwise" if PAIRWISE_ONLY else ""
    theta, forecasts = compute()
    theta.to_csv(OUT / "theta.csv", index=False, float_format="%.4f")
    forecasts.to_csv(OUT / f"forecasts{tag}.csv", index=False, float_format="%.4f")
    table = ratios(forecasts)
    table.to_csv(OUT / f"ratios{tag}.csv", index=False, float_format="%.4f")
    order = {d: i for i, d in enumerate([*DOMAINS, "all"])}
    for protocol, t in table.groupby("protocol"):
        print(protocol)
        print(t.pivot(index="domain", columns="level", values="ratio")[[str(l) for l in sb.LEVELS]]
              .sort_index(key=lambda i: i.map(order)).round(2).to_string())
    n = theta.drop_duplicates(["domain", "semester", "level"]).groupby(["domain", "level"]).n_observed.sum().unstack()
    print(n[[str(l) for l in sb.LEVELS]].to_string())
    base = table[table.protocol == stats.BASELINE].pivot(index="domain", columns="level", values="baseline_mse")
    print("Baseline MSE\n", base[[str(l) for l in sb.LEVELS]].round(2).to_string())
    if not PAIRWISE_ONLY:
        plot(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
