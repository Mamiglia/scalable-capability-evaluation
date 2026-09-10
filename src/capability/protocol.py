"""A protocol forecasts a target model's theta* from the frozen semester tables (capability.semesters).

    uv run evaluate configs/domain/science.yml [--protocol chars]

A protocol is a callable Task -> theta-hat. A Task is one (semester, target): the 8 reference
models with their local theta* (anchors), the target, the train items, and correct repeat-0
transcripts of the train items for any of these models. evaluate() scores every task's forecast
against theta* per semester and pooled over all semesters ("all"; valid because every local fit
shares the global item scale).
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import pandas as pd

from capability import semesters
from configs.domain import DomainConfig
from judge.dataset import Transcript, read_transcripts


@dataclass
class Task:
    domain: DomainConfig
    semester: str
    references: list[str]
    theta_ref: pd.Series  # local theta* of the references, indexed by model
    target: str
    items: pd.DataFrame  # train items: bench, item, guess, a, b
    _models: pd.DataFrame  # frozen models.csv (for the log paths)

    def transcripts(self, model: str, include_reasoning: bool = True) -> dict[tuple[str, str], list[Transcript]]:
        """Correct repeat-0 transcripts of the train items: (bench, item) -> [Transcript]."""
        row = self._models.set_index("model").loc[model]
        out = {}
        for bench, ids in self.items.groupby("bench").item:
            path = row.get(f"path_{bench}", "")
            if not path:
                continue
            read = read_transcripts(semesters.DATA / path, require_correct=True,
                                    include_reasoning=include_reasoning, items=set(ids))
            out.update({(bench, i): t for i, t in read.items()})
        return out


Protocol = Callable[[Task], float]


def tasks(domain: DomainConfig) -> Iterator[Task]:
    models, items = semesters.load(domain)
    for sem in [c[: -len("_role")] for c in models.columns if c.endswith("_role")]:
        role, theta = models[f"{sem}_role"], models.set_index("model")[f"{sem}_theta"]
        ref = models.model[role == "reference"].tolist()
        train = items[items[f"{sem}_train"]][["bench", "item", "guess", "a", "b"]].reset_index(drop=True)
        for target in models.model[role == "target"]:
            yield Task(domain, sem, ref, theta.loc[ref], target, train, models)


def evaluate(domain: DomainConfig, protocol: Protocol, name: str = "") -> pd.DataFrame:
    """Per-semester and pooled scores of `protocol`; theta-hat per target -> data/meta/<domain>/eval_<name>.csv."""
    models = semesters.load(domain)[0].set_index("model")
    rows = [dict(semester=t.semester, model=t.target, theta_hat=float(protocol(t)),
                 theta=models.loc[t.target, f"{t.semester}_theta"]) for t in tasks(domain)]
    pred = pd.DataFrame(rows)
    if name:
        pred.to_csv(semesters.frozen_dir(domain) / f"eval_{name}.csv", index=False, float_format="%.4f")
    groups = [(sem, g) for sem, g in pred.groupby("semester", sort=False)] + [("all", pred)]
    return pd.DataFrame([{"semester": sem, **semesters.score(g.set_index("model").theta_hat, g.set_index("model").theta)}
                         for sem, g in groups])


def chars(task: Task) -> float:
    """Trivial protocol: mean length of a model's correct train transcripts, affine-mapped on the references."""
    def mean_chars(model):
        lens = [len(t[0].text) for t in task.transcripts(model).values()]
        return float(np.mean(lens)) if lens else np.nan
    x = pd.Series({m: mean_chars(m) for m in task.references}).dropna()
    beta, alpha = np.polyfit(x, task.theta_ref.loc[x.index], 1)
    return alpha + beta * mean_chars(task.target)


PROTOCOLS: dict[str, Protocol] = {"chars": chars}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("domain", type=Path)
    parser.add_argument("--protocol", default="chars", choices=sorted(PROTOCOLS))
    a = parser.parse_args()
    table = evaluate(DomainConfig.from_cli(a.domain), PROTOCOLS[a.protocol], a.protocol)
    print(table.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
