"""Score protocols against the frozen theta* of a domain's semester tasks.

    uv run evaluate configs/domain/science.yml [--protocol chars ridge ...] [--semester 2026-I ...]

--protocol selects entries of protocols.PROTOCOLS (default: all). --judge adds pairwise
judging; --direct-judge adds direct ratings calibrated on reference + historical capability.
Both take judge YAMLs and collect missing judgments on demand. --semester restricts the tasks.

Scores are per semester and pooled over all semesters ("all"; valid because every local fit
shares the global item scale): MSE in theta units. The per-target forecasts go to
data/meta/<domain>/eval_<name>.csv (gitignored).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from capability.task import Task
from configs.domain import DomainConfig
from configs.judge import JudgeConfig, DirectJudgeConfig
from protocols import PROTOCOLS, Protocol
from protocols.pairwise import JudgeBT
from protocols.pairwise.calibrated import JudgeCalibrated
from protocols.pairwise.linear import JudgeLinear
from protocols.direct_judge import JudgeDirect
from reference import tables


def score(pred: pd.Series, truth: pd.Series) -> dict:
    """MSE of theta-hat against theta* over the models both series index; NaN forecasts are dropped.

    Every protocol forecasts on the task's theta* scale, so the raw
    squared error is the metric: no shift/scale-free variant.
    """
    pred, truth = pred.align(truth, join="inner")
    known = pred.notna()
    pred, truth = pred[known], truth[known]
    if len(pred) == 0:
        return {"n": 0, "mse": np.nan}
    return {"n": len(pred), "mse": float(((pred - truth) ** 2).mean())}


def semester_names(domain: DomainConfig) -> list[str]:
    columns = tables.load(domain)[0].columns
    return [c.removesuffix("_role") for c in columns if c.endswith("_role")]


def forecast(task: Task, protocol: Protocol) -> pd.DataFrame:
    """One row per target: semester, model, theta_hat, theta."""
    theta_hat = protocol(task).reindex(task.truth.index)
    return pd.DataFrame({"semester": task.semester, "model": task.truth.index,
                         "theta_hat": theta_hat.to_numpy(float), "theta": task.truth.to_numpy()})


def evaluate(domain: DomainConfig, protocol: Protocol, name: str = "", semester_list: list[str] | None = None) -> pd.DataFrame:
    """Run the protocol on every semester task; one score row per semester plus "all"."""
    tasks = [Task.from_semester(s, domain) for s in semester_list or semester_names(domain)]
    forecasts = pd.concat([forecast(task, protocol) for task in tasks], ignore_index=True)
    if name:
        forecasts.to_csv(tables.frozen_dir(domain) / f"eval_{name}.csv", index=False, float_format="%.4f")

    scores = []
    groups = list(forecasts.groupby("semester", sort=False)) + [("all", forecasts)]
    for semester, group in groups:
        group = group.set_index("model")
        scores.append({"semester": semester, **score(group.theta_hat, group.theta)})
    return pd.DataFrame(scores)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("domain", type=Path)
    parser.add_argument("--protocol", nargs="+", default=[], choices=list(PROTOCOLS), help="default: all")
    parser.add_argument("--judge", nargs="+", default=[], type=Path, help="JudgeConfig yml(s) -> judge_<name> protocols")
    parser.add_argument("--direct-judge", nargs="+", default=[], type=Path,
                        help="judge yml(s) -> direct_judge_<name> protocols (1–10 ratings, reasoning shown)")
    parser.add_argument("--semester", nargs="+", default=None, help="default: every semester task")
    args = parser.parse_args()
    domain = DomainConfig.from_cli(args.domain)

    # JudgeBT, JudgeCalibrated and JudgeLinear read the same run dir, so the judging happens once.
    judges = [cls(JudgeConfig.from_cli(path), domain) for path in args.judge for cls in (JudgeBT, JudgeCalibrated, JudgeLinear)]
    judges += [JudgeDirect(DirectJudgeConfig.from_cli(path), domain) for path in args.direct_judge]
    protocols = [PROTOCOLS[name] for name in args.protocol or PROTOCOLS] + judges

    scores = [evaluate(domain, protocol, protocol.name, args.semester).assign(protocol=protocol.name) for protocol in protocols]
    table = pd.concat(scores).pivot(index="semester", columns="protocol", values=["n", "mse"])
    print(table.round(2).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
