"""Pool raw verdicts from every judge released strictly before each frontier target."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.frontier_eci.canonical_coverage import _path  # noqa: E402
from scripts.frontier_eci.rank_regression import (  # noqa: E402
    fit_pooled_judges, load_rank_analysis_inputs, write_pooled_judge_output,
)


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

    config, hard, protocol, _, _ = load_rank_analysis_inputs(args.config, args.logs_root)
    judges = [judge.split("/")[-1] for group in config["judges"].values() for judge in group]
    abilities, fits, predictions = fit_pooled_judges(
        protocol, hard, list(config["candidates"]), judges, config["releases"],
        resamples=args.resamples, seed=args.seed, workers=args.workers, pair_limit=8)
    output = _path(config["results"]) / "analysis/rank_regression/pooled_judges"
    summary = write_pooled_judge_output(
        abilities, fits, predictions, output, figures=not args.no_figures)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
