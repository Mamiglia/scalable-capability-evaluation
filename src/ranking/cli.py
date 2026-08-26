"""Rank the models judged in a log dir (Bradley-Terry) and score the ranking against accuracy.

    uv run rank logs/rank/science/<judge>/<hash> --domain configs/domain/science.yml [--n_resamples 200]

Kendall tau (or Spearman) between ability and accuracy on each difficulty quartile of the domain
and on all its items. Writes ranking.csv, scores.csv, ranking.json (and bootstrap.csv) into the
log dir unless --out_dir is given.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import draccus
import pandas as pd

from configs.domain import DomainConfig
from ranking.bradley_terry import RIDGE, rank
from ranking.scoring import bootstrap, score
from ranking.verdicts import from_logs

QUARTILES = (1, 2, 3, 4)


@dataclass
class RankConfig:
    logs: str  # judge log dir: one <judge, condition>, every pair judged there
    domain: str  # domain YAML; its responses/items are the validation data
    out_dir: str = ""  # default: the log dir
    drop_ties: bool = False
    ridge: float = RIDGE
    correlation: str = "kendall"  # or spearman
    n_resamples: int = 0  # bootstrap draws; 0 = skip
    seed: int = 0


def rank_logs(log_dir: Path, domain: DomainConfig, out_dir: Path | None = None, drop_ties: bool = False,
              ridge: float = RIDGE, correlation: str = "kendall", n_resamples: int = 0, seed: int = 0) -> dict:
    """Rank every verdict in the log dir; score against accuracy on Q1..Q4 and all items."""
    verdicts = from_logs(log_dir)
    if verdicts.empty:
        raise ValueError(f"no verdicts in {log_dir}")
    ranking = rank(verdicts, drop_ties=drop_ties, ridge=ridge)
    validation = domain.load_responses().merge(domain.load_items()[["bench", "item", "quartile"]])
    subsets = {f"Q{q}": validation[validation.quartile == q] for q in QUARTILES} | {"all": validation}

    rows = []
    for name, subset in subsets.items():
        row = {"quartile": name} | score(ranking, subset, correlation=correlation)
        if n_resamples:
            draws = bootstrap(verdicts, subset, n_resamples=n_resamples, seed=seed, drop_ties=drop_ties,
                              ridge=ridge, correlation=correlation)
            row |= {f"correlation_{q}": v for q, v in draws["correlation"].items()}
            if name == "all":
                draws["ability"].to_csv((out_dir or log_dir) / "bootstrap.csv", index=False)
        rows.append(row)
    scores = pd.DataFrame(rows)

    out = Path(out_dir or log_dir)
    out.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(out / "ranking.csv", index=False)
    scores.to_csv(out / "scores.csv", index=False)
    result = {"ranking": ranking.to_dict("records"), "scores": scores.to_dict("records"),
              "n_verdicts": len(verdicts), "judge": sorted(verdicts.judge.unique())}
    (out / "ranking.json").write_text(json.dumps(result, indent=2))
    print(ranking.to_string(index=False))
    print(scores.round(3).to_string(index=False))
    return result


def main() -> int:
    cfg = draccus.parse(RankConfig)
    rank_logs(Path(cfg.logs), DomainConfig.from_cli(cfg.domain), Path(cfg.out_dir) if cfg.out_dir else None,
              drop_ties=cfg.drop_ties, ridge=cfg.ridge, correlation=cfg.correlation,
              n_resamples=cfg.n_resamples, seed=cfg.seed)
    return 0


if __name__ == "__main__":
    main()
