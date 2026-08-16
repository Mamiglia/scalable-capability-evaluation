#!/usr/bin/env python3
"""Build the small static data file used by the upload calculator."""

import argparse
import json
import math
import os
import tempfile
from pathlib import Path

import pandas as pd

from mc_cost.calculator import build_calculator_payload
from mc_cost.catalogue import normalized_name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, default=Path("analysis/mc_cost"))
    parser.add_argument(
        "--out", type=Path,
        default=Path("interactive_viewer/data/mc_cost_calculator.json"),
    )
    parser.add_argument(
        "--model-pool", type=Path, default=Path("model-sampler/model_pool.json"),
    )
    args = parser.parse_args()
    pool = {
        normalized_name(row["name"]): float(row["eci"])
        for row in json.loads(args.model_pool.read_text())
        if row.get("eci") is not None and math.isfinite(row["eci"])
    }
    catalogue = pd.read_csv(args.analysis / "catalogue.csv")
    catalogue["eci"] = catalogue["display_name"].map(
        lambda value: pool.get(normalized_name(value)),
    )
    payload = build_calculator_payload(
        catalogue,
        pd.read_parquet(args.analysis / "token_candidate.parquet"),
        pd.read_parquet(args.analysis / "token_judge.parquet"),
        pd.read_parquet(args.analysis / "judgments.parquet"),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=args.out.parent, delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, args.out)
    finally:
        if temporary and temporary.exists():
            temporary.unlink()


if __name__ == "__main__":
    main()
