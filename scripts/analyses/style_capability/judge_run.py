"""Judge the style_capability conditions under two judge prompts (default +
style-blind), writing verdict eval logs. Reuses judge.task.judge_task; does not
touch the pilot runner. Config: configs/style_capability_judge.yaml.

  --dry-run   print both-correct counts + verdict/cost estimate, no spend
  (default)   run judging

Run from the MAIN checkout (for .env + data):
  .venv/bin/python <worktree>/scripts/analyses/style_capability/judge_run.py --dry-run
"""

import argparse
import glob
from pathlib import Path

import yaml
from inspect_ai import eval_set

from configs.experiment import SYSTEM_PROMPT
from judge.task import judge_task

HERE = Path(__file__).resolve().parent
STYLE_BLIND = SYSTEM_PROMPT.rstrip() + (
    "\n\nImportant: ignore writing style, tone, confidence, formatting, verbosity, "
    "and any surface presentation. Judge ONLY which response shows more correct and "
    "capable reasoning."
)


def resolve(evals_dir, name):
    hits = sorted(glob.glob(f"{evals_dir}/{name}/*.eval"))
    if not hits:
        raise SystemExit(f"no eval log for condition {name}")
    return hits[-1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path,
                    default=HERE.parents[2] / "configs/style_capability_judge.yaml")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    evals, comps, judges = cfg["evals_dir"], cfg["comparisons"], cfg["judges"]
    jargs = cfg.get("judge_model_args", {})
    variants = {"default": SYSTEM_PROMPT, "styleblind": STYLE_BLIND}

    if args.dry_run:
        total = 0
        for major, minor in comps:
            t = judge_task(resolve(evals, major), resolve(evals, minor), include_reasoning=False)
            n = len(t.dataset) // 2  # include_reverse doubles each item into 2 orders
            total += n
            print(f"  {n:3d} both-correct items : {major}  vs  {minor}")
        runs = len(variants) * len(judges) * len(comps)
        calls = len(variants) * len(judges) * total * 2
        print(f"\n{runs} judge runs | ~{calls} verdicts "
              f"({len(variants)} prompts x {len(judges)} judges x both orders)")
        print("rough cost: cheap judges, ~2k input tok each -> a few dollars")
        return 0

    failures = []
    for variant, sp in variants.items():
        for judge in judges:
            js = judge.split("/")[-1]
            for major, minor in comps:
                task = judge_task(resolve(evals, major), resolve(evals, minor),
                                  system_prompt=sp, include_reasoning=False,
                                  task_name=f"{js}_{major}-vs-{minor}",
                                  experiment_name="style_capability")
                ok, _ = eval_set(
                    tasks=[task], model=judge, model_args=jargs.get(judge, {}),
                    log_dir=f"{cfg['log_root']}/{variant}/{js}/{major}-vs-{minor}",
                    max_connections=40)
                if not ok:
                    failures.append(f"{variant}/{js}/{major}-vs-{minor}")
                print(f"done {variant} {js} {major}-vs-{minor}", flush=True)
    if failures:
        print(f"\n{len(failures)} FAILED:")
        for f in failures:
            print(" ", f)
        return 1
    print("\nall judge runs complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
