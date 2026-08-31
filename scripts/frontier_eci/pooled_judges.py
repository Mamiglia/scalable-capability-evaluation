"""Pool raw verdicts from every judge released strictly before each frontier target."""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.frontier_eci.canonical_analysis import (  # noqa: E402
    filter_protocol_verdicts, merge_rollout_sources,
)
from scripts.frontier_eci.canonical_coverage import (  # noqa: E402
    _load_config, _path, historical_rollouts,
)
from scripts.frontier_eci.rank_regression import (  # noqa: E402
    fit_pooled_judges, write_pooled_judge_output,
)
from scripts.judge.verdict_tables import read_verdicts  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path,
                        default=ROOT / "configs/frontier_eci_canonical_anchors/experiment.yaml")
    parser.add_argument("--logs-root", type=Path,
                        default=ROOT / "logs/frontier-eci-canonical-anchors")
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    config = _load_config(args.config)
    quartiles = pd.read_csv(_path(config["quartiles"]), dtype={"item": str})
    canonical = pd.read_parquet(
        _path(config["canonical_identity"]),
        columns=["bench", "model", "item", "epoch", "correct", "is_canonical"])
    frontier = pd.read_parquet(_path(config["frontier_rollouts"])).copy()
    historical = historical_rollouts(config.get("historical_backfill", {}))
    historical["epoch"] = 0
    supplements = historical_rollouts(config.get("actual_supplements", {}))
    supplements["epoch"] = 0
    rollouts = merge_rollout_sources(canonical, frontier, historical, supplements)
    rollouts["item"] = rollouts.item.astype(str)
    hard = rollouts.merge(quartiles[["bench", "item", "item_quartile"]],
                          on=["bench", "item"], how="inner")
    hard = hard.rename(columns={"item_quartile": "target_quartile"})

    judges = [judge.split("/")[-1] for group in config["judges"].values() for judge in group]
    old = [pd.read_parquet(_path(path)) for path in config["verdicts"]]
    paid, _ = read_verdicts([args.logs_root])
    columns = ["bench", "judge", "item", "model_left", "model_right",
               "epoch_left", "epoch_right", "judge_epoch", "pick_left"]
    verdicts = pd.concat([table[columns] for table in old] + [paid[columns]], ignore_index=True)
    verdicts = (verdicts[verdicts.judge.isin(judges)]
                .drop_duplicates(columns[:-1], keep="last").reset_index(drop=True))
    protocol = filter_protocol_verdicts(verdicts, rollouts, quartiles)
    abilities, fits, predictions = fit_pooled_judges(
        protocol, hard, list(config["candidates"]), judges, config["releases"],
        resamples=args.resamples, seed=args.seed, workers=args.workers, pair_limit=8)
    output = _path(config["results"]) / "analysis/rank_regression/pooled_judges"
    summary = write_pooled_judge_output(
        abilities, fits, predictions, output, figures=not args.no_figures)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
