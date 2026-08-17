"""Move judge logs from the old <root>/<bench>/<judge>/<pair>/ layout into
<root>/<bench>/<judge>/<condition hash>/, where eval_set resumes them (2026-08-17 change).

The condition is read from each log's task metadata; the pair dir's eval-set bookkeeping
files are dropped. Dry run unless --apply is given.

    uv run python scripts/judge/relocate_judge_logs.py logs/aime logs/pilot --apply
"""

import argparse
import shutil
from pathlib import Path

from inspect_ai.log import read_eval_log

from configs.experiment import ExperimentConfig, condition_hash

FIELDS = ("system_prompt", "include_reasoning", "truncate_frac", "max_length", "require_both_correct", "include_reverse")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--apply", action="store_true", help="Move files (default: only print).")
    args = parser.parse_args()

    defaults = {f.name: f.default for f in ExperimentConfig.__dataclass_fields__.values()}
    for root in args.roots:
        for pair_dir in sorted(p for p in root.glob("*/*/*") if p.is_dir()):
            evals = list(pair_dir.glob("*.eval"))
            if not evals:
                continue
            for path in evals:
                meta = read_eval_log(str(path), header_only=True).eval.metadata or {}
                if "include_reasoning" not in meta:
                    print(f"skip (not a judge log): {path}")
                    continue
                values = [meta.get(f) if meta.get(f) is not None else defaults[f] for f in FIELDS]
                target = pair_dir.parent / condition_hash(*values) / path.name
                print(f"{path} -> {target}")
                if args.apply:
                    target.parent.mkdir(exist_ok=True)
                    shutil.move(path, target)
            if args.apply and not list(pair_dir.glob("*.eval")):
                shutil.rmtree(pair_dir)  # only eval-set.json / logs.json / .eval-set-id left


if __name__ == "__main__":
    main()
