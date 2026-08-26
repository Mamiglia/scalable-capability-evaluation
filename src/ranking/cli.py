"""Rank models per benchmark from a verdict table, score against accuracy, optionally bootstrap."""

from dataclasses import dataclass, field
from pathlib import Path

import draccus
import pandas as pd

from ranking.bradley_terry import RIDGE, rank
from ranking.scoring import bootstrap, score


@dataclass
class RankConfig:
    verdicts: str  # parquet: bench, judge, item, model_left, model_right, pick_left
    correctness: str  # parquet: bench, model, item, correct
    out_dir: str
    benches: list[str] = field(default_factory=list)  # empty list = keep all
    judges: list[str] = field(default_factory=list)
    models: list[str] = field(default_factory=list)  # both models of a verdict must be listed
    items_csv: str = ""  # file with an item column; empty = keep all
    drop_ties: bool = False
    ridge: float = RIDGE
    correlation: str = "kendall"  # or spearman
    n_resamples: int = 0  # bootstrap draws; 0 = skip
    seed: int = 0


def select(verdicts: pd.DataFrame, cfg: RankConfig) -> pd.DataFrame:
    """Keep the verdicts matching every non-empty filter."""
    keep = pd.Series(True, index=verdicts.index)
    if cfg.benches:
        keep &= verdicts.bench.isin(cfg.benches)
    if cfg.judges:
        keep &= verdicts.judge.isin(cfg.judges)
    if cfg.models:
        keep &= verdicts.model_left.isin(cfg.models) & verdicts.model_right.isin(cfg.models)
    if cfg.items_csv:
        keep &= verdicts.item.isin(pd.read_csv(cfg.items_csv).item)
    return verdicts[keep]


def main() -> int:
    cfg = draccus.parse(RankConfig)
    verdicts = select(pd.read_parquet(cfg.verdicts), cfg)
    correctness = pd.read_parquet(cfg.correctness)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    scores, intervals = [], []
    for bench, judged in verdicts.groupby("bench"):
        ranking = rank(judged, drop_ties=cfg.drop_ties, ridge=cfg.ridge)
        ranking.to_csv(out / f"ranking_{bench}.csv", index=False)
        result = score(ranking, correctness, judged, correlation=cfg.correlation)
        scores.append({"bench": bench, **result})
        print(f"{bench}: {result['n_models']} models, {result['n_items']} items, "
              f"{cfg.correlation} {result['correlation']:.3f}")
        if cfg.n_resamples:
            draws = bootstrap(judged, correctness, n_resamples=cfg.n_resamples, seed=cfg.seed,
                              drop_ties=cfg.drop_ties, ridge=cfg.ridge, correlation=cfg.correlation)
            intervals.append(draws["ability"].assign(bench=bench))
            scores[-1].update({f"correlation_{q}": v for q, v in draws["correlation"].items()})
    pd.DataFrame(scores).to_csv(out / "scores.csv", index=False)
    if intervals:
        pd.concat(intervals).to_csv(out / "bootstrap.csv", index=False)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    main()
