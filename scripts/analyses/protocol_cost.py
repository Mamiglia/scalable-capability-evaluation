"""Cost-error operating curves per domain: the reference-LOO judge's pairwise and direct runs subsampled by judged items,
length+prose and embedding as points (no judging is run).

    uv run python scripts/analyses/protocol_cost.py [--recompute]

Judge = per (domain, semester, protocol) the core-condition run with the lowest reference leave-one-out MSE
(analysis.calibration.core_runs / judge_forecast; pairwise linear, direct raw, as results.py shows). For each fraction of its judged
train items, DRAWS random item subsets feed the same conversion (protocols.judges); a target's error at that fraction is
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

import numpy as np
import pandas as pd
from inspect_ai.log import read_eval_log

from analysis import calibration, loading, matching
from protocols import judges
from capability.evaluate import forecast
from configs.logs import logger
from plots.protocol_cost import plot
from protocols import PROTOCOLS

warnings.filterwarnings("ignore")
logger.setLevel(logging.ERROR)
OUT = Path("analysis/judge/protocol_cost")
DOMAINS = ["science", "math", "coding", "cyber"]
VARIANT = {"pairwise": "linear", "direct": "raw", "embedding": "full", "glm_regex": "-"}
FRACTIONS = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
DRAWS = 30
EMBED = 0.01  # $/M tokens, qwen/qwen3-embedding-8b
# runpod-served judges have no list price: a similar-size OpenRouter model stands in
PROXY = {"openai-api/runpod/NousResearch/Nous-Hermes-2-Yi-34B": "qwen/qwen3-32b"}


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
    for f in run_dir.glob("*.eval"):
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
                dollars, tokens = map(sum, zip(*(run_cost(d, table) for d in (run.harness.log_dirs(sem) if run.protocol == "direct" else [run.harness.log_dir(sem)]))))
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
    plot(s, DOMAINS, OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
