#!/usr/bin/env python3
"""Run one stage of the GPQA judge GLM pipeline.

The stage modules remain directly executable. This command is the supported
entry point and keeps their options intact. Run ``pipeline.py STAGE --help`` for
stage-specific arguments.
"""
from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Sequence


STAGES = {
    "check-features": ("transcript_features", ["--self-check"]),
    "prepare": ("prepare_gpqa", []),
    "generate-fits": ("fit_features", []),
    "evaluate": ("evaluation_power", []),
    "figures": ("make_figures", []),
}

REQUIRED_OUTPUTS = {
    "prepare": ("--out", "--out-correctness"),
    "generate-fits": ("--out-dir",),
    "evaluate": ("--out-dir",),
    "figures": ("--out-dir",),
}


def has_option(values: Sequence[str], option: str) -> bool:
    return option in values or any(value.startswith(f"{option}=") for value in values)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=[*STAGES, "collect"])
    values = list(sys.argv[1:] if argv is None else argv)
    if not values or values[0] in {"-h", "--help"}:
        parser.print_help()
        return 0
    args = parser.parse_args(values[:1])
    rest = values[1:]

    if not any(value in {"-h", "--help"} for value in rest):
        missing = [option for option in REQUIRED_OUTPUTS.get(args.stage, ())
                   if not has_option(rest, option)]
        if missing:
            parser.error(f"{args.stage} requires explicit {' and '.join(missing)}")

    if args.stage == "collect":
        if rest and rest[0] in {"-h", "--help"}:
            print("usage: pipeline.py collect GENERATED_FIT_DIRECTORY")
            return 0
        if not rest or rest[0].startswith("-"):
            parser.error("collect requires the generated-fit directory")
        module_name, stage_args = "fit_features", ["--bias-map", rest[0], *rest[1:]]
    else:
        module_name, prefix = STAGES[args.stage]
        stage_args = [*prefix, *rest]

    module = importlib.import_module(module_name)
    result = module.main(stage_args)
    return 0 if result is None else int(result)


if __name__ == "__main__":
    raise SystemExit(main())
