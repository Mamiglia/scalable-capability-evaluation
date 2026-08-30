"""Run frontier judging one pair at a time under an OpenRouter spend cap."""

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import yaml


def can_afford(*, spent: float, reserve: float, limit: float) -> bool:
    return spent + reserve <= limit


def remaining_budget(*, total_limit: float, historical_spend: float) -> float:
    return round(total_limit - historical_spend, 2)


def key_usage() -> float:
    request = urllib.request.Request(
        "https://openrouter.ai/api/v1/key",
        headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return float(json.load(response)["data"]["usage"])


def reserves(path: Path) -> dict[tuple[str, str, str], float]:
    with path.open(newline="") as handle:
        return {
            (row["config"], row["target"], row["opponent"]): float(row["upper_cost"])
            for row in csv.DictReader(handle)
        }


def record(path: Path, **values) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps({"time": datetime.now(timezone.utc).isoformat(), **values}) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", nargs="+", type=Path)
    parser.add_argument("--baseline", required=True, type=float)
    parser.add_argument("--total-limit", type=float, default=200)
    parser.add_argument("--historical-spend", type=float, default=133.4)
    parser.add_argument("--reserve-multiplier", type=float, default=1.5)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--reserve-usd", type=float, default=5.0,
                        help="per-pair reserve when no preflight table is supplied")
    parser.add_argument("--ledger", required=True, type=Path)
    args = parser.parse_args()
    upper = reserves(args.preflight) if args.preflight else {}
    allowance = remaining_budget(
        total_limit=args.total_limit, historical_spend=args.historical_spend)
    record(args.ledger, event="header", historical_spend=args.historical_spend,
           total_limit=args.total_limit, incremental_allowance=allowance)

    for config_path in args.configs:
        config_key = config_path.as_posix()
        config = yaml.safe_load(config_path.read_text())
        for judge, target, opponent in config["comparisons"]:
            reserve = args.reserve_multiplier * upper.get(
                (config_key, target, opponent), args.reserve_usd)
            usage = key_usage()
            spent = usage - args.baseline
            if not can_afford(spent=spent, reserve=reserve, limit=allowance):
                record(args.ledger, event="budget_stop", usage=usage, spent=spent,
                       reserve=reserve, limit=allowance, config=config_key,
                       target=target, opponent=opponent)
                print(f"BUDGET STOP: ${spent:.2f} spent; ${reserve:.2f} reserved; ${allowance:.2f} cap")
                return 2

            one_pair = config | {"comparisons": [[judge, target, opponent]]}
            with tempfile.TemporaryDirectory() as directory:
                shard = Path(directory) / config_path.name
                shard.write_text(yaml.safe_dump(one_pair, sort_keys=False))
                result = subprocess.run([
                    sys.executable,
                    "scripts/judge/run_judging.py",
                    "--config", str(shard),
                ])
            usage_after = key_usage()
            record(args.ledger, event="pair", status=result.returncode,
                   usage=usage_after, spent=usage_after - args.baseline,
                   reserve=reserve, config=config_key, judge=judge,
                   target=target, opponent=opponent)
            if result.returncode:
                return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
