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
from judge.logs import has_success_log


def can_afford(*, spent: float, reserve: float, limit: float) -> bool:
    return spent + reserve <= limit


def remaining_budget(*, total_limit: float, historical_spend: float) -> float:
    return round(total_limit - historical_spend, 2)


def config_key(path: str | Path) -> str:
    path = Path(path)
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def reserve_key(config: str | Path, target: str, opponent: str) -> tuple[str, str, str]:
    return config_key(config), *sorted((target, opponent))


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
            reserve_key(row["config"], row["target"], row["opponent"]): float(row["upper_cost"])
            for row in csv.DictReader(handle)
        }


def reserve_for(upper: dict[tuple[str, str, str], float], config: Path, target: str, opponent: str,
                *, fallback: float, strict: bool) -> float:
    key = reserve_key(config, target, opponent)
    if strict and key not in upper:
        raise ValueError(f"preflight missing reserve for {key}")
    return upper.get(key, fallback)


def record(path: Path, **values) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps({"time": datetime.now(timezone.utc).isoformat(), **values}) + "\n")


def ledger_header(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open() as handle:
        for line in handle:
            event = json.loads(line)
            if event.get("event") == "header":
                return event
    return None


def pair_successful(config: dict, judge: str, target: str, opponent: str) -> bool:
    """Match run_judging.py's success-log layout for a one-pair config."""
    pair_dir = Path(config["log_root"]) / judge.split("/")[-1] / f"{target}-vs-{opponent}"
    if has_success_log(pair_dir):
        return True
    if "only_items" not in config:
        return False
    phase1_items = config.get("phase1_items") or 20
    if len(config["only_items"]) <= phase1_items:
        return has_success_log(pair_dir / "phase-1")
    return has_success_log(pair_dir / "phase-1") and has_success_log(pair_dir / "phase-2")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", nargs="+", type=Path)
    parser.add_argument("--baseline", required=True, type=float)
    parser.add_argument("--limit", type=float,
                        help="incremental spend cap for this key/run")
    parser.add_argument("--total-limit", type=float, default=200)
    parser.add_argument("--historical-spend", type=float, default=133.4)
    parser.add_argument("--reserve-multiplier", type=float, default=1.5)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--reserve-usd", type=float, default=5.0,
                        help="per-pair reserve when no preflight table is supplied")
    parser.add_argument("--ledger", required=True, type=Path)
    args = parser.parse_args()
    upper = reserves(args.preflight) if args.preflight else {}
    available = remaining_budget(
        total_limit=args.total_limit, historical_spend=args.historical_spend)
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    allowance = args.limit if args.limit is not None else available
    header = ledger_header(args.ledger)
    if header:
        if header["baseline"] != args.baseline:
            raise ValueError("ledger baseline conflicts with supplied baseline")
        if (header["historical_spend"], header["total_limit"]) != (
                args.historical_spend, args.total_limit):
            raise ValueError("ledger budget conflicts with supplied budget")
        if header["incremental_allowance"] != allowance:
            raise ValueError("ledger incremental allowance conflicts with supplied limit")
        baseline = header["baseline"]
    else:
        if args.ledger.exists() and args.ledger.stat().st_size:
            raise ValueError("ledger has entries but no header")
        baseline = args.baseline
        record(args.ledger, event="header", baseline=baseline,
               historical_spend=args.historical_spend, total_limit=args.total_limit,
               incremental_allowance=allowance)

    for config_path in args.configs:
        config = yaml.safe_load(config_path.read_text())
        for judge, target, opponent in config["comparisons"]:
            if pair_successful(config, judge, target, opponent):
                continue
            reserve = args.reserve_multiplier * reserve_for(
                upper, config_path, target, opponent,
                fallback=args.reserve_usd, strict=args.preflight is not None)
            usage = key_usage()
            spent = usage - baseline
            if not can_afford(spent=spent, reserve=reserve, limit=allowance):
                record(args.ledger, event="budget_stop", usage=usage, spent=spent,
                       reserve=reserve, limit=allowance, config=config_key(config_path),
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
                   usage=usage_after, spent=usage_after - baseline,
                   reserve=reserve, config=config_key(config_path), judge=judge,
                   target=target, opponent=opponent)
            if result.returncode:
                return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
