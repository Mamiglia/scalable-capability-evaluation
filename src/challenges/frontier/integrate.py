"""Add the `frontier` cohort to the frozen coding tables: the frontier models as extra 2026-I targets.

    PYTHONPATH=src .venv/bin/python src/challenges/frontier/integrate.py [--write]

coding's own fit stays untouched (a refreeze would reshuffle the train set, and drops this cohort:
rerun this after one). Rows written to the long tables (reference.tables), domain coding:
- tasks       2026-I targets with cohort = frontier: unrostered coding models named by --extra (default
              claude-opus-4.7, gemini-3.5-flash, gpt-5) on their full-bench logs, and the frontier models,
              theta* from `reference.freeze.local_fit` on the 2026-I HARD items they answered (a, b, guess
              pinned to the global fit, so theta* lands on the same scale as everyone else's).
- models      one row per frontier model, no theta_global (it is not in coding's fit).
- samples     the frontier models' scored responses on their subset logs; unscored samples (refusals,
              errors, cancelled) are dropped, not counted wrong (`evals.responses` would count them wrong).
- coding/fit.yml  `cohorts.frontier`: provenance.

The domain name coding_frontier reads coding with this cohort as the 2026-I targets (tables.COHORTS), so
`uv run evaluate configs/domain/coding_frontier.yml --semester 2026-I ...` scores protocols on the frontier
targets alone, against the same references and items as the coding 2026-I task.
"""

import argparse
import datetime
import glob
from pathlib import Path

import pandas as pd
import yaml

from configs.paths import is_successful_eval
from evals.sample import extract_score
from inspect_ai.log import read_eval_log_sample_summaries
from reference import tables
from reference.freeze import local_fit

SEMESTER = "2026-I"
SUBSETS = {"terminal_bench_2_1": "0af2ba2c", "swe_bench_mini": "2d1e487f"}
# GPT-5.x from the Epoch AI SWE-bench import (data/collection_runs.md): swe_bench_mini only, 46 of the 50 items
# (23 of the 27 HARD), Epoch's bash scaffold rather than ours -- theta* rests on those 23 items and on item
# parameters fitted under our scaffold. Partial by design.
EPOCH = ("swe_bench_mini", "6d369262", ["gpt-5.1-2025-11-13_high", "gpt-5.2-2025-12-11_high", "gpt-5.3-codex_high",
                                        "gpt-5.4-2026-03-05_high", "gpt-5.5-pre-release_xhigh"])


def frontier_logs() -> dict[str, dict[str, Path]]:
    """model -> bench -> its repeat-0 log under the subset hashes."""
    found: dict[str, dict[str, Path]] = {}
    for bench, h in SUBSETS.items():
        for p in sorted(glob.glob(f"data/{bench}/{h}/*/0/*.eval")):
            path = Path(p)
            if is_successful_eval(path):
                found.setdefault(path.parts[3], {})[bench] = path
    bench, h, models = EPOCH
    for model in models:
        (path,) = [Path(p) for p in glob.glob(f"data/{bench}/{h}/{model}/0/*.eval") if is_successful_eval(Path(p))]
        found[model] = {bench: path}
    return found


def responses_of(model: str, logs: dict[str, Path]) -> pd.DataFrame:
    rows = []
    for bench, path in logs.items():
        for s in read_eval_log_sample_summaries(str(path)):
            if not s.scores:  # refusal / error / cancelled: not scored, not wrong
                continue
            rows.append(dict(bench=bench, item=str(s.id), model=model, repeat=0, correct=bool(extract_score(s))))
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--samples", type=int, default=1000)
    ap.add_argument("--extra", nargs="*", default=["claude-opus-4.7", "gemini-3.5-flash", "gpt-5"],
                    help="unrostered coding models (full-bench logs) to add as targets too, theta* from coding's responses")
    args = ap.parse_args()

    models, items = tables.load("coding")
    responses = tables.load_responses("coding")
    settings = tables.fit_settings("coding")

    logs = frontier_logs()
    new = pd.concat([responses_of(m, l) for m, l in logs.items()], ignore_index=True)
    hard = items[items[f"{SEMESTER}_set"] == "hard"][["bench", "item", "a", "b", "guess"]]
    n_hard = new.merge(hard[["bench", "item"]]).groupby("model").size()
    print("frontier models:", {m: sorted(l) for m, l in logs.items()})
    print("scored responses per model:", new.groupby("model").size().to_dict(), "| on HARD:", n_hard.to_dict())

    extra = [m for m in args.extra if m in models.index and pd.isna(models.loc[m, f"{SEMESTER}_theta"])]
    fit_input = pd.concat([new, responses[responses.model.isin(extra)]], ignore_index=True)
    theta = local_fit(fit_input, hard, sorted(logs) + extra, warmup=args.warmup, samples=args.samples,
                      seed=settings.get("seed", 0))
    refs = models[models[f"{SEMESTER}_role"] == "reference"][f"{SEMESTER}_theta"]
    print(f"\n{SEMESTER} theta* (references span {refs.min():.2f} .. {refs.max():.2f}):")
    print(theta.round(3).to_string())

    targets = sorted(logs) + extra
    cohort = pd.DataFrame(dict(semester=SEMESTER, model=targets, role="target", cohort="frontier",
                               theta=theta.loc[targets, "theta"].values, theta_sd=theta.loc[targets, "theta_sd"].values))
    added = pd.DataFrame(dict(model=sorted(logs), semester=None, theta_global=float("nan"), theta_global_sd=float("nan")))
    paths = pd.DataFrame([dict(bench=b, model=m, repeat=0, log=str(p.relative_to("data"))) for m, l in logs.items() for b, p in l.items()])
    settings["cohorts"] = {"frontier": {
        "derived_at": datetime.date.today().isoformat(), "frontier_targets": sorted(logs), "extra_targets": extra,
        "epoch_import_targets": EPOCH[2], "frontier_hashes": SUBSETS,
        "note": "frontier models as extra 2026-I targets (theta* = local_fit on their scored 2026-I HARD responses); "
                "coding_frontier reads them in place of the roster targets; unscored (refused) samples excluded"}}
    if not args.write:
        print("\ndry run; pass --write to add the frontier cohort to the coding tables")
        return 0
    old = tables.read("models", "coding")
    tables.write("tasks", "coding", pd.concat([tables.read("tasks", "coding").query("cohort != 'frontier'"), cohort]))
    tables.write("models", "coding", pd.concat([old[old.theta_global.notna()], added]))
    kept = tables.read("samples", "coding").query("model not in @added.model")
    tables.write("samples", "coding", pd.concat([kept, new[["bench", "item", "model", "repeat", "correct"]].merge(paths)]))
    (tables.frozen_dir("coding") / "fit.yml").write_text(yaml.safe_dump(settings, sort_keys=False))
    tables.write_paper()
    print("wrote the frontier cohort:", len(cohort), "targets,", len(new), "responses")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
