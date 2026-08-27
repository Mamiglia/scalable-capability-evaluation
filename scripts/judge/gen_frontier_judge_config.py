#!/usr/bin/env python3
"""Turn configs/frontier_pairs.yaml into run_judging.py configs, one per benchmark.

Resolves each model's transcript log (our pool, or the Epoch log / its trimmed copy),
checks the release-date rule (a model released after the judge only meets models released
before itself) and puts the more accurate model first in every comparison.

Usage:
  uv run python scripts/judge/gen_frontier_judge_config.py --pairs configs/frontier_pairs.yaml \\
      --log-root logs/frontier --out-prefix configs/judge_frontier
"""

import argparse
import glob
from pathlib import Path

import pandas as pd
import yaml
from inspect_ai.log import read_eval_log

MODELS_DIR = Path("configs/models")


def release_dates() -> dict[str, pd.Timestamp]:
    dates = {}
    for path in MODELS_DIR.glob("*/*.yml"):
        spec = yaml.safe_load(path.read_text())
        if spec.get("release_date"):
            dates[path.stem] = pd.Timestamp(str(spec["release_date"])[:10])
    return dates


def transcript_paths(cfg: dict, bench: str) -> dict[str, str]:
    """Short name -> eval log for every model that has one on this benchmark."""
    paths = {}
    for directory in sorted(glob.glob(f"{cfg['pool'][bench]}/*")):
        logs = sorted(glob.glob(f"{directory}/0/*.eval"))
        if logs:
            paths[Path(directory).name] = logs[0]
    for short, directory in cfg["pool_dirs"].items():
        if directory in paths:
            paths[short] = paths.pop(directory)
    epoch = yaml.safe_load(Path(cfg["epoch_logs"][bench]).read_text())
    trimmed = cfg["epoch_trimmed_root"].get(bench)
    for epoch_name, short in epoch["models"].items():
        paths[short] = (f"{trimmed}/{short}/0/{short}.eval" if trimmed
                        else f"{epoch['out_dir']}/{epoch_name}.eval")
    return paths


def accuracy(path: str) -> float:
    metrics = read_eval_log(path, header_only=True).results.scores[0].metrics
    return next(metrics[key].value for key in ("accuracy", "mean") if key in metrics)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, default=Path("configs/frontier_pairs.yaml"))
    parser.add_argument("--log-root", default="logs/frontier")
    parser.add_argument("--out-prefix", default="configs/judge_frontier")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.pairs.read_text())
    dates = release_dates()
    judge_release = pd.Timestamp(cfg["judge_release"])

    for bench, pairs in cfg["pairs"].items():
        paths = transcript_paths(cfg, bench)
        used = sorted({model for model, opponents in pairs.items() for model in [model, *opponents]})
        missing = [m for m in used if m not in paths or not Path(paths[m]).exists()]
        if missing:
            raise SystemExit(f"{bench}: no transcript log for {missing}")
        accuracies = {m: accuracy(paths[m]) for m in used}

        comparisons = []
        for model, opponents in pairs.items():
            for opponent in opponents:
                if dates[model] > judge_release and dates[opponent] >= dates[model]:
                    raise SystemExit(f"{bench}: {model} ({dates[model].date()}) released after the judge "
                                     f"may only meet earlier models; {opponent} is {dates[opponent].date()}")
                major, minor = sorted([model, opponent], key=lambda m: -accuracies[m])
                comparisons.extend([judge, major, minor] for judge in cfg["judges"])

        out = {"log_root": f"{args.log_root}/{bench}",
               "transcripts": {m: paths[m] for m in used},
               "comparisons": comparisons}
        target = Path(f"{args.out_prefix}_{bench}.yaml")
        target.write_text(yaml.safe_dump(out, sort_keys=False, width=200))
        print(f"{target}: {len(comparisons)} comparisons ({len(cfg['judges'])} judges) over {len(used)} models")


if __name__ == "__main__":
    main()
