#!/usr/bin/env python3
"""Turn configs/frontier_pairs.yaml into run_judging.py configs, one per benchmark.

Resolves each model's transcript log (our pool, or the Epoch log imported into the pool layout),
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

from configs.paths import is_successful_eval

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
        logs = [log for log in sorted(glob.glob(f"{directory}/0/*.eval")) if is_successful_eval(Path(log))]
        if logs:
            paths[Path(directory).name] = logs[0]
    for short, directory in cfg["pool_dirs"].items():
        if directory in paths:
            paths[short] = paths.pop(directory)
    epoch = yaml.safe_load(Path(cfg["epoch_logs"][bench]).read_text())
    for short in epoch["models"].values():
        logs = [log for log in sorted(glob.glob(f"{cfg['epoch_pool_root'][bench]}/{short}/0/*.eval"))
                if is_successful_eval(Path(log))]
        if logs:
            paths[short] = logs[0]
    return paths


def judge_spec(slug: str) -> dict:
    """The model YAML under configs/models whose inspect slug is ``slug``."""
    found = [spec for path in MODELS_DIR.glob("*/*.yml")
             if (spec := yaml.safe_load(path.read_text())).get("slug") == slug]
    if len(found) != 1:
        raise SystemExit(f"judge {slug}: need exactly one model YAML, found {len(found)}")
    return found[0]


def accuracy(path: str) -> float:
    metrics = read_eval_log(path, header_only=True).results.scores[0].metrics
    return next(metrics[key].value for key in ("accuracy", "mean") if key in metrics)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, default=Path("configs/frontier_pairs.yaml"))
    parser.add_argument("--log-root", default="logs/frontier")
    parser.add_argument("--out-prefix", default="configs/judge_frontier")
    parser.add_argument("--shards", type=int, default=1, help="split each per-judge config into this many files "
                        "(pairs dealt round-robin) so several runners can work side by side")
    parser.add_argument("--per-judge-dir", type=Path,
                        help="also write one config per judge here (<bench>/<judge>.yaml) with the judge's "
                             "endpoint pin and generate args inlined, the schema earth's runner reads")
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
        if args.per_judge_dir:
            for judge in cfg["judges"]:
                spec = judge_spec(judge)
                rows = [row for row in comparisons if row[0] == judge]
                for shard in range(args.shards):
                    per_judge = {"log_root": out["log_root"], "transcripts": out["transcripts"],
                                 "comparisons": rows[shard::args.shards],
                                 "judge_model_args": {judge: spec.get("model_args") or {}}}
                    if spec.get("generate_args"):
                        per_judge["generate_args"] = spec["generate_args"]
                    suffix = f"-{shard}" if args.shards > 1 else ""
                    path = args.per_judge_dir / bench / f"{judge.split('/')[-1]}{suffix}.yaml"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(yaml.safe_dump(per_judge, sort_keys=False, width=200))


if __name__ == "__main__":
    main()
