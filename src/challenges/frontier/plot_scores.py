"""Prediction scores of every protocol on the frontier targets (coding_frontier, 2026-I).

    PYTHONPATH=src:scripts/analyses .venv/bin/python src/challenges/frontier/plot_scores.py

Score of protocol P on target m: S_m(P) = -log2(e_m(P) / e_m(base)), e_m = (theta_hat - theta*)^2, base = the
constant forecast at the references' mean theta* (`mean_ref`: with saturated items, accuracy carries no
information, so predicting the reference mean is the no-information baseline). Positive = better than the
baseline, +1 = squared error halved. Reads results/<domain>/eval_<protocol>.csv (written by `uv run evaluate`),
writes analysis/frontier/prediction_scores.{csv,png,svg}.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts/analyses")
from plots.frontier import SHOW, VENDORS, pairwise_boxes, scores_figure  # noqa: E402

META = Path("results/coding_frontier")
OPEN = Path("results/coding")  # the same protocols on this semester's open-weight targets: the reference group
OUT = Path("analysis/frontier/prediction_scores")
BASE = "mean_ref"
SEMESTER = "2026-I"
EXISTING_DIRECT = Path("logs/direct/coding/2026-I/kimi-k2-thinking/1a2c14b2")  # swe + terminal ratings, 16 models


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

    scores_figure(table, reference, OUT)
    pairwise_boxes(table, reference, OUT)
    print(table.pivot(index="model", columns="protocol", values="score").loc[shown].round(2).to_string())
    print("wrote", OUT.with_suffix(".png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
