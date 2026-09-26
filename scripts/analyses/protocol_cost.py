"""Cost-error operating curves per domain: the reference-LOO judge's pairwise and direct runs subsampled by judged items,
length+prose and embedding as points (no judging is run).

    uv run python scripts/analyses/protocol_cost.py [--recompute]

Judge = per (domain, semester, protocol) the core-condition run with the lowest reference leave-one-out MSE
(analysis.calibration.core_runs / judge_forecast; pairwise linear, direct raw, as results.py shows). For each fraction of its judged
train items, DRAWS random item subsets feed the same conversion (analysis.judges); a target's error at that fraction is
its squared error averaged over draws. Cost = the run's logged tokens at OpenRouter list prices (openrouter.ai/api/v1/models,
fetched at --recompute; logs carry no total_cost; uncached input + cache reads at the listed cache price + output) times
the fraction, averaged over the domain's semesters. Embedding: Qwen3-Embedding-8B ($0.01/M) on the direct LOO run's input
tokens (each train transcript read once, plus the ratings prompt). Length+prose: regex on CPU, $0. Runpod judges
(science 2024-I direct nous-hermes-2-yi-34b) are priced as a same-size OpenRouter model (PROXY).
y = MSE over every target of every semester of the domain; band / bars = 95% percentile bootstrap over targets
(the replication unit of analysis.stats.mse_row). Dashed line: the reference-mean forecast.
Writes analysis/judge/protocol_cost{,_points}.csv and protocol_cost.{png,svg,pdf}.
"""

import json
import logging
import sys
import urllib.request
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from inspect_ai.log import read_eval_log

from analysis import calibration, judges, loading, matching
from capability.evaluate import forecast
from configs.logs import logger
from protocols import PROTOCOLS

warnings.filterwarnings("ignore")
logger.setLevel(logging.ERROR)
OUT = Path("analysis/judge/protocol_cost")
DOMAINS = ["science", "math", "coding", "cyber"]
VARIANT = {"pairwise": "linear", "direct": "raw", "embedding": "full", "glm_regex": "-"}
SERIES = {"pairwise": ("Pairwise judging", "#2a78d6"), "direct": ("Absolute scoring", "#eb6834"),
          "glm_regex": ("Deterministic features", "#1baf7a"), "embedding": ("Embedding", "#4a3aa7")}
FRACTIONS = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
DRAWS = 30
EMBED = 0.01  # $/M tokens, qwen/qwen3-embedding-8b
# runpod-served judges have no list price: a similar-size OpenRouter model stands in
PROXY = {"openai-api/runpod/NousResearch/Nous-Hermes-2-Yi-34B": "qwen/qwen3-32b"}
FREE = 1e-3  # where $0 sits on the log axis
YMAX = 3


def prices() -> dict[str, tuple[float, float, float]]:
    """OpenRouter id -> $/M (input, cached input, output); cached = input where no cache price is listed."""
    data = json.load(urllib.request.urlopen("https://openrouter.ai/api/v1/models"))["data"]
    out = {}
    for m in data:
        p = {k: float(v) * 1e6 for k, v in m["pricing"].items() if isinstance(v, str) and v}
        out[m["id"]] = (p["prompt"], p.get("input_cache_read", p["prompt"]), p["completion"])
    return out


def run_cost(run_dir: Path, table: dict) -> tuple[float, int]:
    """(dollars, input tokens) over every log of the run dir."""
    dollars, tokens = 0.0, 0
    files = Path(run_dir).parent.glob("*/*.eval") if "logs/direct/coding" in str(run_dir) else Path(run_dir).glob("*.eval")  # CORRECTION
    for f in files:
        for model, u in (read_eval_log(str(f), header_only=True).stats.model_usage or {}).items():
            inp, cached, out = table[PROXY.get(model, model.removeprefix("openrouter/"))]
            cache = u.input_tokens_cache_read or 0
            dollars += ((u.input_tokens - cache) * inp + cache * cached + u.output_tokens * out) / 1e6
            tokens += u.input_tokens
    return dollars, tokens


def loo_pick(runs: list, task, protocol: str):
    """The matched core-condition run of `protocol` with the lowest reference leave-one-out MSE, or None."""
    core = [matching.match(r, task) for r in calibration.core_runs(runs) if r.protocol == protocol]
    return min(core, key=lambda r: calibration.judge_forecast(r, task, VARIANT[protocol])[1], default=None)


def compute() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(curves: one row per target x protocol x fraction, points: one row per target x transcript protocol),
    both with the cost per semester of that row's run."""
    table, rng = prices(), np.random.default_rng(0)
    f = pd.read_csv("analysis/results/forecasts.csv")
    curves, points = [], []
    for name in DOMAINS:
        dom = loading.domain(name)
        for task in loading.tasks(dom):
            sem, refs, targets = task.semester, task.references.theta, list(task.truth.index)
            runs = loading.judge_runs(dom, task)
            base = forecast(task, PROTOCOLS["mean_ref"]).set_index("model").theta_hat.reindex(targets)
            embed_tokens = 0
            for protocol in ("pairwise", "direct"):
                if (run := loo_pick(runs, task, protocol)) is None:
                    print("no", protocol, "run", name, sem, file=sys.stderr)
                    continue
                dollars, tokens = run_cost(run.harness.log_dir(sem), table)
                embed_tokens = tokens if protocol == "direct" else embed_tokens
                items = run.table[["bench", "item"]].drop_duplicates().reset_index(drop=True)
                for frac in FRACTIONS:
                    n = max(3, round(frac * len(items)))
                    err = []
                    for _ in range(1 if frac == 1 else DRAWS):
                        sub = run.table.merge(items.iloc[rng.choice(len(items), n, replace=False)])
                        theta_hat = (judges.pairwise_theta(sub, refs, targets, "linear") if protocol == "pairwise"
                                     else judges.direct_theta(sub, refs, targets, "raw"))
                        err.append((theta_hat.reindex(targets) - task.truth) ** 2)
                    curves.append(pd.DataFrame({"model": targets, "se": pd.concat(err, axis=1).mean(axis=1).to_numpy(),
                                                "baseline_se": ((base - task.truth) ** 2).to_numpy()})
                                  .assign(domain=name, semester=sem, protocol=protocol, judge=run.judge,
                                          condition=run.condition, fraction=frac, n_items=n, cost=frac * dollars))
            for protocol in ("glm_regex", "embedding"):
                g = f[(f.domain == name) & (f.semester == sem) & (f.protocol == protocol) & (f.variant == VARIANT[protocol])]
                points.append(g.assign(se=(g.theta_hat - g.theta) ** 2, cost=0.0 if protocol == "glm_regex" else embed_tokens * EMBED / 1e6)
                              [["domain", "semester", "protocol", "model", "se", "cost"]])
            print(name, sem, file=sys.stderr)
    return pd.concat(curves, ignore_index=True), pd.concat(points, ignore_index=True)


def boot(se: pd.Series, draws: int = 2000) -> tuple[float, float, float]:
    """(mse, lo, hi): 95% percentile bootstrap over targets, as analysis.stats.mse_row."""
    err = se.dropna().to_numpy()
    b = np.random.default_rng(0).choice(err, size=(draws, len(err))).mean(axis=1)
    return err.mean(), *np.percentile(b, [2.5, 97.5])


def summarise(curves: pd.DataFrame, points: pd.DataFrame) -> pd.DataFrame:
    """One row per domain x protocol x fraction: mean cost per semester, pooled MSE and its interval."""
    rows = []
    for (name, protocol, frac), g in curves.groupby(["domain", "protocol", "fraction"]):
        cost = g.drop_duplicates("semester").cost.mean()
        rows.append(dict(domain=name, protocol=protocol, fraction=frac, cost=cost, **dict(zip(["mse", "lo", "hi"], boot(g.se)))))
    for (name, protocol), g in points.groupby(["domain", "protocol"]):
        rows.append(dict(domain=name, protocol=protocol, fraction=np.nan, cost=g.drop_duplicates("semester").cost.mean(),
                         **dict(zip(["mse", "lo", "hi"], boot(g.se)))))
    for name, g in curves[curves.fraction == 1].drop_duplicates(["domain", "semester", "model"]).groupby("domain"):
        rows.append(dict(domain=name, protocol="mean_ref", fraction=np.nan, cost=0.0, **dict(zip(["mse", "lo", "hi"], boot(g.baseline_se)))))
    return pd.DataFrame(rows)


def plot(s: pd.DataFrame) -> None:
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "svg.fonttype": "none"})
    fig, axes = plt.subplots(2, 2, figsize=(3.4, 2.6), sharex=True, sharey=True, gridspec_kw={"wspace": 0.25, "hspace": 0.45})
    for ax, name in zip(axes.flat, DOMAINS):
        d = s[s.domain == name]
        ax.axhline(d[d.protocol == "mean_ref"].mse.iloc[0], color="#9a9a9a", lw=0.8, ls="--", zorder=0)
        for key, (_, color) in SERIES.items():
            g = d[d.protocol == key].sort_values("fraction")
            x = g.cost.clip(lower=FREE)
            if key in ("pairwise", "direct"):
                ax.fill_between(x, g.lo, g.hi, color=color, alpha=0.18, lw=0)
                ax.plot(x, g.mse, color=color, lw=1, marker="o", ms=2.5, mfc="white", mew=0.7)
            else:
                if (g.mse > YMAX).any():  # off the chart (math/science length+prose blow up): a triangle at the edge + its value
                    ax.plot(x, [YMAX] * len(g), "^", color=color, ms=3.5, clip_on=False)
                    for xi, m in zip(x, g.mse):
                        ax.text(xi * 1.6, YMAX, f"{m:.0f}", fontsize=5, color=color, va="center")
                else:
                    ax.errorbar(x, g.mse, yerr=[g.mse - g.lo, g.hi - g.mse], fmt="o", color=color, ms=3.5, lw=1, capsize=0)
        ax.text(0, 1.04, name, transform=ax.transAxes, fontsize=7, color="#9a9a9a")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_position(("outward", 4))
    ax = axes[0, 0]
    ax.set_xscale("log"), ax.set_yscale("log")
    ax.set_xlim(FREE / 1.5, 1000)
    ax.set_xticks([FREE, 0.01, 0.1, 1, 10, 100, 1000], ["0", ".01", ".1", "1", "10", "100", "1k"])
    ax.set_ylim(0.03, YMAX)
    ax.set_yticks([0.03, 0.1, 0.3, 1, 3], ["0.03", "0.1", "0.3", "1", "3"])
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.xaxis.set_minor_locator(plt.NullLocator())
    for a in axes[1]:
        a.tick_params(axis="x", labelrotation=90)
        a.set_xlabel("Cost per semester ($)")
    for a in axes[:, 0]:
        a.set_ylabel("Squared error (θ²)")
    handles = [plt.Line2D([], [], color=c, lw=1, marker="o", ms=2.5, mfc="white", mew=0.7, label=l) for k, (l, c) in SERIES.items() if k in ("pairwise", "direct")]
    handles += [plt.Line2D([], [], color=c, ls="", marker="o", ms=3.5, label=l) for k, (l, c) in SERIES.items() if k not in ("pairwise", "direct")]
    handles.append(plt.Line2D([], [], color="#9a9a9a", lw=0.8, ls="--", label="Baseline"))
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.97),
               handletextpad=0.3, columnspacing=0.8, handlelength=1.4)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(OUT.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")


def main() -> int:
    points_csv = OUT.with_name(OUT.name + "_points.csv")
    if "--recompute" in sys.argv or not OUT.with_suffix(".csv").exists():
        curves, points = compute()
        curves.to_csv(OUT.with_suffix(".csv"), index=False, float_format="%.5f")
        points.to_csv(points_csv, index=False, float_format="%.5f")
    curves, points = pd.read_csv(OUT.with_suffix(".csv")), pd.read_csv(points_csv)
    print(curves[curves.fraction == 1].drop_duplicates(["domain", "semester", "protocol"])
          [["domain", "semester", "protocol", "judge", "condition", "n_items", "cost"]].round(2).to_string(index=False))
    s = summarise(curves, points)
    print(s.round(3).to_string(index=False))
    plot(s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
