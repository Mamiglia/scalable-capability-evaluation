"""Score protocols against the frozen theta* of a domain's semester tasks.

    uv run evaluate configs/domain/science.yml [--protocol chars ridge ...] [--judge logs/rank/science/<judge>/<hash> ...]

--judge adds one PairwiseJudge protocol per judge (dirs grouped by their <judge> directory, so one
judge's semester runs pool into one protocol), named judge_<judge>; --protocol defaults to every protocol.

Per semester and pooled over all semesters ("all"; valid because every local fit shares the
global item scale): Pearson r, MSE, affine-fit MSE. theta-hat per target -> data/meta/<domain>/eval_<name>.csv.
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
    models = semesters.load(domain)[0]
    rows = [dict(semester=t.semester, model=t.target, theta_hat=float(protocol(t)),
                 theta=models.loc[t.target, f"{t.semester}_theta"]) for t in tasks(domain)]
    pred = pd.DataFrame(rows)
    if name:
        pred.to_csv(semesters.frozen_dir(domain) / f"eval_{name}.csv", index=False, float_format="%.4f")
    groups = [(sem, g) for sem, g in pred.groupby("semester", sort=False)] + [("all", pred)]
    return pd.DataFrame([{"semester": sem, **semesters.score(g.set_index("model").theta_hat, g.set_index("model").theta)}
                         for sem, g in groups])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("domain", type=Path)
    parser.add_argument("--protocol", nargs="+", default=[], choices=list(PROTOCOLS))
    parser.add_argument("--judge", nargs="+", default=[], type=Path, help="judge-run log dirs -> pairwise-judge protocols")
    a = parser.parse_args()
    domain = DomainConfig.from_cli(a.domain)
    by_judge = {d.parent.name: [x for x in a.judge if x.parent.name == d.parent.name] for d in a.judge}
    judges = [PairwiseJudge(dirs) for dirs in by_judge.values()]
    protocols = {**PROTOCOLS, **{p.name: p for p in judges}}
    names = a.protocol + [p.name for p in judges] or list(protocols)
    table = pd.concat([evaluate(domain, protocols[n], n).assign(protocol=n) for n in names])
    print(table.pivot(index="semester", columns="protocol", values=["n", "r", "affine_mse"]).round(2).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
