"""Score protocols against the frozen theta* of a domain's semester tasks.

    uv run evaluate configs/domain/science.yml [--protocol chars ridge ...] [--judge logs/rank/science/<judge>/<hash> ...]

--protocol names entries of protocols.PROTOCOLS (default: all of them). --judge adds one
PairwiseJudge protocol per judge: the dirs are grouped by their <judge> directory, so one judge's
semester runs pool into one protocol named judge_<judge>.

Scores are per semester and pooled over all semesters ("all"; valid because every local fit
shares the global item scale): Pearson r, MSE, affine-fit MSE. The per-target forecasts go to
data/meta/<domain>/eval_<name>.csv (gitignored).
"""

import argparse
from pathlib import Path

import pandas as pd

from capability import semesters
from capability.task import tasks
from configs.domain import DomainConfig
from protocols import PROTOCOLS, Protocol
from protocols.judge import PairwiseJudge


def evaluate(domain: DomainConfig, protocol: Protocol, name: str = "") -> pd.DataFrame:
    """Run the protocol on every semester task; one score row per semester plus "all"."""
    frozen_models = semesters.load(domain)[0]
    forecasts = []
    for task in tasks(domain):
        theta_hat = protocol(task)
        for target in task.targets:
            forecasts.append({
                "semester": task.semester,
                "model": target,
                "theta_hat": float(theta_hat[target]),
                "theta": frozen_models.loc[target, f"{task.semester}_theta"],
            })
    forecasts = pd.DataFrame(forecasts)
    if name:
        forecasts.to_csv(semesters.frozen_dir(domain) / f"eval_{name}.csv", index=False, float_format="%.4f")

    scores = []
    groups = list(forecasts.groupby("semester", sort=False)) + [("all", forecasts)]
    for semester, group in groups:
        group = group.set_index("model")
        scores.append({"semester": semester, **semesters.score(group.theta_hat, group.theta)})
    return pd.DataFrame(scores)


def judge_protocols(log_dirs: list[Path]) -> list[PairwiseJudge]:
    """One PairwiseJudge per judge: log dirs are logs/rank/<domain>/<judge>/<hash>, grouped by <judge>."""
    dirs_by_judge: dict[str, list[Path]] = {}
    for log_dir in log_dirs:
        dirs_by_judge.setdefault(log_dir.parent.name, []).append(log_dir)
    return [PairwiseJudge(dirs) for dirs in dirs_by_judge.values()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("domain", type=Path)
    parser.add_argument("--protocol", nargs="+", default=[], choices=list(PROTOCOLS))
    parser.add_argument("--judge", nargs="+", default=[], type=Path, help="judge-run log dirs -> pairwise-judge protocols")
    args = parser.parse_args()
    domain = DomainConfig.from_cli(args.domain)

    judges = judge_protocols(args.judge)
    protocols = {**PROTOCOLS, **{judge.name: judge for judge in judges}}
    selected = args.protocol + [judge.name for judge in judges] or list(protocols)

    tables = [evaluate(domain, protocols[name], name).assign(protocol=name) for name in selected]
    table = pd.concat(tables).pivot(index="semester", columns="protocol", values=["n", "r", "affine_mse"])
    print(table.round(2).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
